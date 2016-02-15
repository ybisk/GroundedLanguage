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

@knet function cnn(x; cwin1=2, cout1=20, output=20)
	c1 = conv_layer(x; cwindow=cwin1, coutput=cout1)
	return softmax_layer(c1; output=output)
end

@knet function cnn2(x; cwin1=2, cout1=20, hidden=100, output=20)
	c1 = conv_layer(x; cwindow=cwin1, coutput=cout1)
	h = relu_layer(c1, output=hidden)
	return softmax_layer(h; output=output)
end


function train(worldf, worlddata, loss; gclip=0)
	sumloss = numloss = 0
	reset!(worldf)
	i = 1
	for (x,y) in worlddata
		ypred = forw(worldf, x)

		sumloss += zeroone(ypred, y)*size(y,2)
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


function get_worlds(rawdata; batchsize=100, ftype=Float32)
	data = map(x -> (rawdata[x,1:end-64], rawdata[x,end-63:end]),1:size(rawdata,1));
	worlds = zeros(ftype, 18, 18, 22, 2, size(rawdata, 1))
	y = zeros(ftype, 20, size(rawdata, 1))
	indx=1
	for item in data
		world = zeros(ftype, 18, 18, 22, 1)
		world[2:17,2:17, 22, 1] = 1#set empty

		#borders
		world[[1,18],:,21, 1] = 1
		world[:,[1,18],21, 1] = 1

		after = copy(world)

		state = item[2]
		locs = map(x -> (state[x], state[x+1], state[x+2]), 1:3:60)
		source = round(Int, state[61])
		y[source, indx] = 1

		for i=1:length(locs)
			loc = locs[i]
			if !(loc[1] == -1 && loc[2] == -1 && loc[3] == -1)
				x = round(Int, loc[1] / 0.1524) + 10
				z = round(Int, loc[3] / 0.1524) + 10
				#println((i, x, z))
				world[z, x, i, 1] = 1
				world[z, x, 22, 1] = 0

				if i != source
					after[z,x,i, 1] = 1
					after[z,x, 22, 1] = 0
				end
			end
		end
		#println(world)
		worlds[:,:,:,1,indx] = world

		x = round(Int, state[62] / 0.1524) + 10
		z = round(Int, state[64] / 0.1524) + 10
		#println((i, x, z))
		after[z, x, source, 1] = 1
		after[z, x, 22, 1] = 0
		worlds[:,:,:,2,indx] = after
		indx += 1

	end
	return minibatch(worlds, y, batchsize)
end

function main(args)
	s = ArgParseSettings()
	s.exc_handler=ArgParse.debug_handler
	@add_arg_table s begin
		("--worlddatafiles"; nargs='+'; default=["../../BlockWorld/MNIST/SimpleActions/logos/Train.SP.data", "../../BlockWorld/MNIST/SimpleActions/logos/Dev.SP.data"])
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
		("--dropout"; arg_type=Float64; default=0.5; help="dropout probability")
		("--fdropout"; arg_type=Float64; default=0.5; help="dropout probability for the first one")
		("--ldropout"; arg_type=Float64; default=0.5; help="dropout probability for the last one")
		("--decay"; arg_type=Float64; default=0.9; help="learning rate decay if deverr increases")
		("--nogpu"; action = :store_true; help="do not use gpu, which is used by default if available")
		("--loc"; action = :store_true; help="predict location")
		("--seed"; arg_type=Int; default=42; help="random number seed")
		("--nx"; arg_type=Int; default=79; help="number of input columns in data")
		("--ny"; arg_type=Int; default=3; help="number of target columns in data")
		("--target"; arg_type=Int; default=1; help="which target to predict: 1:source,2:target,3:direction")
		("--xvocab"; arg_type=Int; default=326; help="vocab size for input columns (all columns assumed equal)")
		("--yvocab"; arg_type=Int; nargs='+'; default=[20,20,8]; help="vocab sizes for target columns (all columns assumed independent)")
		("--xsparse"; action = :store_true; help="use sparse inputs, dense arrays used by default")
		("--ftype"; default = "Float32"; help="floating point type to use: Float32 or Float64")
		("--patience"; arg_type=Int; default=0; help="patience")
		("--nlayers"; arg_type=Int; default=2; help="number of layers")
		("--logfile"; help="csv file for log")
		("--predict"; action = :store_true; help="load net and give predictions")
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
	rawmerged = Any[]
	if length(rawworlddata) > 2
		push!(rawmerged, vcat(rawworlddata[1:2:end]))
		push!(rawmerged, vcat(rawworlddata[2:2:end]))
	else
		rawmerged = rawworlddata[:]
	end


	global worlddata = map(rawmerged) do d
		get_worlds(d, batchsize=o[:batchsize])
	end

	global worldf = o[:chidden] != 0 ? compile(:cnn2; cwin1=o[:cwin], cout1=o[:cout], hidden=o[:chidden], output=yvocab) : compile(:cnn; cwin1=o[:cwin], cout1=o[:cout], output=yvocab)

	setp(worldf, adam=true)
	setp(worldf, lr=o[:lr])

	lasterr = besterr = 1.0
	anger = 0
	stopcriterion = false
	df = DataFrame(epoch = Int[], lr = Float64[], trn_err = Float64[], dev_err = Float64[], best_err = Float64[])

	for epoch=1:o[:epochs]      # TODO: experiment with pretraining
		@date trnerr = train(worldf, worlddata[1], softloss; gclip=o[:gclip])
		@date deverr = test(worldf, worlddata[2], zeroone)

		if deverr < besterr
			besterr=deverr
			o[:bestfile]!=nothing && save(o[:bestfile], "net", clean(net))
			anger = 0
		else
			anger += 1
		end
		if o[:patience] != 0 && anger == o[:patience]
			stopcriterion = true
			o[:lr] *= o[:decay]
			setp(sentf, lr=o[:lr])
			setp(worldf, lr=o[:lr])
			anger = 0
		end
		push!(df, (epoch, o[:lr], trnerr, deverr, besterr))
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
