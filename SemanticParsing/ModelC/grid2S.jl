using ArgParse
using JLD
using CUDArt
device(0)
using Knet: stack_isempty
using Knet
using DataFrames
using Base.LinAlg: axpy!, scale!


@knet function generic_layer(x; f1=:dot, f2=:relu, wdims=(), bdims=(), winit=Xavier(), binit=Constant(0))
	w = par(init=winit, dims=wdims)
	y = f1(w,x)
	b = par(init=binit, dims=bdims)
	z = add(b,y)
	return f2(z)
end

@knet function relu_layer(x; input=0, output=0)
	return generic_layer(x; f1=:dot, f2=:relu, wdims=(output,input), bdims=(output,1))
end

@knet function conv_layer(x; cwindow=0, cinput=0, coutput=0)
	w = par(init=Xavier(), dims=(cwindow,cwindow,22, 2,coutput))
	c = conv(w,x)
	return c
end

@knet function softmax_layer(x; input=0, output=0)
	return generic_layer(x; f1=:dot, f2=:soft, wdims=(output,input), bdims=(output,1))
end

function get_worldf(predtype, o)
	if predtype == "id" || predtype == "grid"
		@knet function output_layer(x; output=20)
			return wbf(x; out=output, f=:soft)
		end
	else
		@knet function output_layer(x; output=3)
			return wb(x; out=output)
		end
	end
	@knet function cnn(x; cwin1=2, cout1=20, output=20)
		c1 = conv_layer(x; cwindow=cwin1, coutput=cout1)
		return output_layer(c1; output=output)
	end

	@knet function cnn2(x; cwin1=2, cout1=20, hidden=100, output=20)
		c1 = conv_layer(x; cwindow=cwin1, coutput=cout1)
		h = relu_layer(c1, output=hidden)
		return output_layer(h; output=output)
	end

	@knet function cnn3(x; cwin1=2, cout1=20, hidden=100, pdrop=0.5, output=20)
		c1 = conv_layer(x; cwindow=cwin1, coutput=cout1)
		h = relu_layer(c1, output=hidden)
		hd = drop(h; pdrop=pdrop)
		h2 = relu_layer(hd, output=hidden)
		return output_layer(h2; output=output)
	end
	
	outdim = predtype == "id" ? 20 : 18*18
	outdim = predtype == "loc" ? 3 : outdim
	worldf = nothing
	if o[:chidden] == 0
		worldf = compile(:cnn; cwin1=o[:cwin], cout1=o[:cout], output=outdim)
	elseif o[:chidden] != 0 && o[:dropout] == 0
		worldf = compile(:cnn2; cwin1=o[:cwin], cout1=o[:cout], hidden=o[:chidden], output=outdim)
	else
		worldf = compile(:cnn3; cwin1=o[:cwin], cout1=o[:cout], hidden=o[:chidden], pdrop=o[:dropout], output=outdim)
	end
	return worldf
end

#=====================
@knet function cnn3(x; cwin1=2, cout1=20, hidden=100, output=20, pdrop=0.5)
	c1 = conv_layer(x; cwindow=cwin1, coutput=cout1)
	h = relu_layer(c1, output=hidden)
	hd = drop(h; pdrop=pdrop)
	return softmax_layer(hd; output=output)
end
======================#

function train(worldf, worlddata, loss; gclip=0, dropout=false)
	sumloss = numloss = 0
	reset!(worldf)
	i = 1
	for (x,y) in worlddata
		if dropout
			ypred = forw(worldf, x, dropout=true)
		else
			ypred = forw(worldf, x)
		end

		sumloss += size(y, 1) == 3 ? quadloss(ypred, y)*size(y,2) : zeroone(ypred, y)*size(y,2)
		numloss += size(y,2)

		back(worldf, y, loss)

		update!(worldf; gclip=gclip)
		reset!(worldf)
	end
	sumloss / numloss
end

function test(worldf, worlddata, loss, loc=false)
	sumloss = numloss = 0
	reset!(worldf)
	i = 1
	for (x,y) in worlddata
		ypred = forw(worldf, x)
		sumloss += loss(ypred, y)*size(y,2)
		numloss += size(y,2)
		reset!(worldf)
	end
	sumloss / numloss
end

flat(A) = mapreduce(x->isa(x,Array)? flat(x): x, vcat, [], A)

function predict(worldf, worlddata; ftype=Float32, xsparse=false)
	reset!(worldf)
	ypreds = Any[]
	for (x,y) in worlddata
		ypred = forw(worldf, x)
		push!(ypreds, ypred)
	end
	println(flat(ypred))
end

#predtype = id | grid | loc
function get_worlds(rawdata; batchsize=100, predtype = "id", ftype=Float32)
	#data = map(x -> (rawdata[x,1:end-64], rawdata[x,end-63:end]),1:size(rawdata,1));
	worlds = zeros(ftype, 18, 18, 22, 2, size(rawdata, 1))

	ydim = 0
	if predtype == "id"
		ydim = 20
	elseif predtype == "loc"
		ydim = 3
	else
		ydim = 18 * 18
	end
	y = zeros(ftype, ydim, size(rawdata, 1))
	
	for indx=1:size(rawdata, 1)
		data = rawdata[indx, :]
		before = zeros(ftype, 18, 18, 22, 1)
		before[2:17,2:17, 22, 1] = 1#set empty

		#borders
		before[[1,18],:,21, 1] = 1
		before[:,[1,18],21, 1] = 1

		after = copy(before)

		blocs = map(x -> (data[1,x], data[1, x+1], data[1, x+2]), 1:3:60)
		alocs = map(x -> (data[1, x], data[1, x+1], data[1, x+2]), 61:3:120)
		
		if predtype == "id"
			source = round(Int, data[1, 222])
			y[source, indx] = 1
		elseif predtype == "loc"
			source = round(Int, data[1, 222])
			y[:, indx] = data[1, (source*3 - 2):(source*3)]
		else
			source = round(Int, data[1, 222])
			x = round(Int, data[1, (source*3 - 2)] / 0.1524) + 10
			z = round(Int, data[1, (source*3)] / 0.1524) + 10
			source = (z-1)*18 + x
			y[source, indx] = 1
		end
		
		for (locs, world) in [(blocs, before), (alocs, after)]
			for i=1:length(locs)
				loc = locs[i]
				if !(loc[1] == -1 && loc[2] == -1 && loc[3] == -1)
					x = round(Int, loc[1] / 0.1524) + 10
					z = round(Int, loc[3] / 0.1524) + 10
					#println((i, x, z))
					world[z, x, i, 1] = 1
					world[z, x, 22, 1] = 0
				end
			end
		end
		#println(world)
		worlds[:,:,:,1,indx] = before
		worlds[:,:,:,2,indx] = after
	end
	return minibatch(worlds, y, batchsize)
end

function main(args)
	s = ArgParseSettings()
	s.exc_handler=ArgParse.debug_handler
	@add_arg_table s begin
		("--worlddatafiles"; nargs='+'; default=["../../BlockWorld/SimpleCombined/Train.WWT-STRPLocGrid.data", "../../BlockWorld/SimpleCombined/Dev.WWT-STRPLocGrid.data"])
		("--loadfile"; help="initialize model from file")
		("--savefile"; help="save final model to file")
		("--bestfile"; help="save best model to file")
		("--hidden"; arg_type=Int; default=100; help="hidden layer size")
		("--chidden"; arg_type=Int; default=0; help="hidden layer size")
		("--cwin"; arg_type=Int; default=2; help="filter size")
		("--cout"; arg_type=Int; default=20; help="number of filters")
		("--epochs"; arg_type=Int; default=10; help="number of epochs to train")
		("--batchsize"; arg_type=Int; default=10; help="minibatch size")
		("--lr"; arg_type=Float64; default=1.0; help="learning rate")
		("--gclip"; arg_type=Float64; default=5.0; help="gradient clipping threshold")
		("--dropout"; arg_type=Float64; default=0.0; help="dropout probability")
		("--fdropout"; arg_type=Float64; default=0.5; help="dropout probability for the first one")
		("--ldropout"; arg_type=Float64; default=0.5; help="dropout probability for the last one")
		("--decay"; arg_type=Float64; default=0.9; help="learning rate decay if deverr increases")
		("--nogpu"; action = :store_true; help="do not use gpu, which is used by default if available")
		("--loc"; action = :store_true; help="predict location")
		("--seed"; arg_type=Int; default=42; help="random number seed")
		("--nx"; arg_type=Int; default=101; help="number of input columns in data")
		("--ny"; arg_type=Int; default=3; help="number of target columns in data")
		("--target"; arg_type=Int; default=1; help="which target to predict: 1:source,2:target,3:direction")
		("--xvocab"; arg_type=Int; default=622; help="vocab size for input columns (all columns assumed equal)")
		("--yvocab"; arg_type=Int; nargs='+'; default=[20,20,8]; help="vocab sizes for target columns (all columns assumed independent)")
		("--xsparse"; action = :store_true; help="use sparse inputs, dense arrays used by default")
		("--ftype"; default = "Float32"; help="floating point type to use: Float32 or Float64")
		("--patience"; arg_type=Int; default=0; help="patience")
		("--nlayers"; arg_type=Int; default=2; help="number of layers")
		("--logfile"; help="csv file for log")
		("--predict"; action = :store_true; help="load net and give predictions")
		("--predtype"; default = "id"; help="prediction type: id, loc, grid")
	end

	isa(args, AbstractString) && (args=split(args))
	o = parse_args(args, s; as_symbols=true); println(o)
	o[:seed] > 0 && setseed(o[:seed])
	o[:ftype] = eval(parse(o[:ftype]))

	#global rawdata = map(f->readdlm(f,Int), o[:datafiles])

	# Minibatch data: data[1]:train, data[2]:dev
	xrange = 1:o[:nx]
	yrange = (o[:nx] + o[:target]):(o[:nx] + o[:target])
	yvocab = o[:yvocab][o[:target]]

	#===========
	global data = map(rawdata) do d
		minibatch(d, xrange, yrange, o[:batchsize]; xvocab=o[:xvocab], yvocab=yvocab, ftype=o[:ftype], xsparse=o[:xsparse])
	end
	==========#

	rawworlddata = map(f->readdlm(f,Float32), o[:worlddatafiles])

	global worlddata = map(rawworlddata) do d
		get_worlds(d, batchsize=o[:batchsize], predtype=o[:predtype])
	end
	
	worldf = get_worldf(o[:predtype], o)

	setp(worldf, adam=true)
	setp(worldf, lr=o[:lr])

	lasterr = besterr = 1e6
	best_epoch = 0
	anger = 0
	stopcriterion = false
	df = DataFrame(epoch = Int[], lr = Float64[], trn_err = Float64[], dev_err = Float64[], best_err = Float64[], best_epoch = Int[])
	loss1 = o[:predtype] == "loc" ? quadloss : softloss
	loss2 = o[:predtype] == "loc" ? quadloss : zeroone

	for epoch=1:o[:epochs]      # TODO: experiment with pretraining
		drop = o[:dropout] != 0
		
		@date trnerr = train(worldf, worlddata[1], loss1; gclip=o[:gclip], dropout=drop)
		@date deverr = test(worldf, worlddata[2], loss2)

		if deverr < besterr
			besterr=deverr
			best_epoch = epoch
			o[:bestfile]!=nothing && save(o[:bestfile], "net", clean(net))
			anger = 0
		else
			anger += 1
		end
		if o[:patience] != 0 && anger == o[:patience]
			stopcriterion = true
			o[:lr] *= o[:decay]
			#setp(sentf, lr=o[:lr])
			#setp(worldf, lr=o[:lr])
			anger = 0
		end
		push!(df, (epoch, o[:lr], trnerr, deverr, besterr, best_epoch))
		println(df[epoch, :])
		if stopcriterion
			break
		end
		lasterr = deverr
	end

	#o[:savefile]!=nothing && save(o[:savefile], "net", clean(net))
	#@date devpred = predict(sentf, worldf, rawdata[2], get_worlds(rawworlddata[2]; batchsize=1); loc=o[:loc], xrange=xrange, xvocab=o[:xvocab], ftype=o[:ftype], xsparse=o[:xsparse])
	#println(devpred)
	o[:logfile]!=nothing && writetable(o[:logfile], df)
end
!isinteractive() && main(ARGS)
