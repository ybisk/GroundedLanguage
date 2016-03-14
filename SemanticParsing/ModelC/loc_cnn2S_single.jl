using ArgParse
using JLD
using CUDArt
device(0)
using Knet: stack_isempty
using Knet
using DataFrames
using Base.LinAlg: axpy!, scale!

#=
h100, relu, 524288,0.08
h200, relu, 524288,0.07
h400, relu, 524288,0.12
h800, relu, 524288,0.194
h400, tanh, 524288,0.32
=#


function get_worldf(predtype, o)
	@knet function cnn(x, xb; cwin1=1, cout1=1)
		w = par(init=Xavier(), dims=(2, cwin1, 3, cout1))
		c = conv(w, x)
		t1 = wbf(c, out=200, f=:relu)
		t2 = wbf(t1, out=20, f=:sigm)
		h = xb * t2
		return h
	end
	
	worldf = compile(:cnn; cwin1=o[:cwin], cout1=o[:cout])
	return worldf
end

function pretrain(f; N=2^19, dims=(16, 1, 16), nblocks=20, ndims=length(dims), nbatch=128)
	sloss = zloss = 0
	nextn = 50000
	ncells = prod(dims)
	global worlds = zeros(Float32, 2, nblocks, ndims)
	global ygold = zeros(Float32, 3, 1)
	for n=1:N
		locations = randperm(ncells)
		for b=1:nblocks
			worlds[1, b, :] = worlds[2, b, :] = 2*([ind2sub(dims, locations[b])...] / 16 - 0.5) # fill blocks with random locations
		end
		mblock = rand(1:nblocks)
		worlds[2, mblock, :] = 2*([ind2sub(dims, locations[nblocks+1])...] / 16 - 0.5) # move block to an empty location
		ygold[:] = worlds[1, mblock, :]
		
		global ypred = forw(f, reshape(worlds, 2, 20, 3, 1), transpose(reshape(worlds[1,:,:], 20, 3)))

		sl = quadloss(ypred,ygold); sloss = (n==1 ? sl : 0.99 * sloss + 0.01 * sl)
		n==nextn && (println((n,sloss)); nextn += 50000)
		back(f, ygold, quadloss)
		update!(f, gclip=5.0)
	end
end

function train(worldf, worlddata, loss; gclip=0, dropout=false)
	sumloss = numloss = 0
	reset!(worldf)
	i = 1
	for (x, y, xb) in worlddata
		ypred = forw(worldf, x, xb)
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
	for (x, y, xb) in worlddata
		ypred = forw(worldf, x, xb)
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
	instances = Any[]

	ydim = 0
	if predtype == "id"
		ydim = 20
	elseif predtype == "loc"
		ydim = 3
	else
		ydim = 18 * 18
	end
	count = 0
	for indx=1:size(rawdata, 1)
		data = rawdata[indx, :]
		worlds = zeros(ftype, 2, 20, 3)
		y = zeros(ftype, ydim, 1)
		for i=1:20
			if !(data[1, (i-1)*3 + 1] == -1 && data[1, (i-1)*3 + 2] == -1 && data[1, (i-1)*3 + 3] == -1)
				for j=1:3
					worlds[1, i, j] = data[1, (i-1)*3 + j]
					worlds[2, i, j] = data[1, (i+19)*3 + j]
				end
			else
				count += 1
			end
		end
		
		if predtype == "id"
			source = round(Int, data[1, 222])
			y[source, 1] = 1
		elseif predtype == "loc"
			source = round(Int, data[1, 222])
			y[:, 1] = data[1, (source*3 - 2):(source*3)]
		else
			source = round(Int, data[1, 222])
			x = round(Int, data[1, (source*3 - 2)] / 0.1524) + 10
			z = round(Int, data[1, (source*3)] / 0.1524) + 10
			source = (z-1)*18 + x
			y[source, 1] = 1
		end

		#push!(instances, (reshape(worlds, 2, 20, 3, 1), y, reshape(worlds[1,:,:], 1, 20, 3, 1)))
		push!(instances, (reshape(worlds, 2, 20, 3, 1), y, transpose(reshape(worlds[1,:,:], 20, 3))))
	end

	return instances
end

#julia loc_cnn2S_single.jl --lr 0.001 --epoch 20 --predtype loc --pretrain
#pretrain: 0.00163596 quadloss / 0.456 block size
#nopretrain
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
		("--cwin"; arg_type=Int; default=1; help="filter size")
		("--cout"; arg_type=Int; default=1; help="number of filters")
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
		("--pretrain"; action = :store_true; help="pretraining")
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

	rawworlddata = map(f->readdlm(f,Float32), o[:worlddatafiles])

	global worlddata = map(rawworlddata) do d
		r = randperm(size(d, 1))
		get_worlds(d[r, :], batchsize=o[:batchsize], predtype=o[:predtype])
	end
	
	worldf = get_worldf(o[:predtype], o)
	setp(worldf, adam=true, lr=o[:lr])
	if o[:pretrain]
		pretrain(worldf)
		#setp(worldf, adam=true, lr=0.00001)
		setp(worldf, adam=true, lr=0.00001)
	end

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

	Knet.netprint(worldf)
	println("")
	
	println("Filter:")
	println(to_host(worldf.reg[3].out0))


	#o[:savefile]!=nothing && save(o[:savefile], "net", clean(net))
	#@date devpred = predict(sentf, worldf, rawdata[2], get_worlds(rawworlddata[2]; batchsize=1); loc=o[:loc], xrange=xrange, xvocab=o[:xvocab], ftype=o[:ftype], xsparse=o[:xsparse])
	#println(devpred)
	o[:logfile]!=nothing && writetable(o[:logfile], df)
end
!isinteractive() && main(ARGS)
