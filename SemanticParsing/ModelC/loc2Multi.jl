#=
Predicting the reference blocks

World Representation:
- Coordinates
- x,y,z coordinates of each block
  
Models:
Input:
- W = 60x1 before state
- X = 3x1 target location
  
Output: ID(20x19) - unique id for the two reference blocks
  
Feed forward:
- a relu layer
=#

using ArgParse
using JLD
using CUDArt
device(0)
using Knet: stack_isempty
using Knet
using DataFrames
using Base.LinAlg: axpy!, scale!

function get_worldf(predtype, o)
	@knet function fnn(w, x; hidden=800, output=190, winit=Gaussian(0,0.05))
		h = wbf2(w, x; out=hidden, f=:relu, winit=winit)
		#h2 = wbf(h; out=hidden, f=:relu, winit=winit)
		return wbf(h; out=output, f=:soft, winit=winit)
	end
	outdim = 190
	worldf = compile(:fnn; hidden=o[:hidden], output=outdim)
	return worldf
end

#p is a point
#s is a segment (p1, p2)
function distPoint2Segment(p, s)
	p1, p2 = s
	v = p2 - p1
	w = p - p1

	c1 = sum(w.*v)
	if c1 <= 0
		return norm(p - p1)
	end

	c2 = sum(v.*v)
	if c2 <= c1
		return norm(p - p2)
	end

	b = c1 / c2
	pb = p1 + b .* v
	return norm(p - pb)
end

#chech whether target location is around r1 and r2
function validate(r1, r2, t, threshold=2*0.1254)
	distance = distPoint2Segment(t, (r1,r2))
	distance <= threshold
end

function pretraining(f; N=2^15, dims=(16, 1, 16), nblocks=20, lr=0.001, adam=true, nbatch=128, winit=Gaussian(0,0.05))
	sloss = zloss = 0
	nextn = 2500
	ncells = prod(dims)
	global world = zeros(Float32, length(dims), nblocks)
	global target = zeros(Float32, length(dims), 1)
	global ygold = zeros(Float32, 190,1)
	global world2 = zeros(Float32, length(dims)*nblocks, nbatch)
	global target2 = zeros(Float32, length(dims), nbatch)
	global ygold2 = zeros(Float32, 190,nbatch)

	for n=1:N
		for m=1:nbatch
			_t1 = 0
			_t2 = 0
			notvalidated = true
			while notvalidated
				locations = randperm(ncells)
				for b=1:nblocks
					world[:,b] = 2*([ind2sub(dims, locations[b])...] / 16 - 0.5) # fill blocks with random locations
					world[2,b] = 0.1
				end
				target[:,1] = 2*([ind2sub(dims, locations[nblocks+1])...] / 16 - 0.5) # pick target at an empty location
				target[2,1] = 0.1
				d1 = world .- target
				d2 = sum(d1 .* d1, 1)
				l = collect(enumerate(d2))
				sort!(l, by=(x -> x[2]))
				#pick two of the closest
				_t1 = l[1][1]
				_t2 = l[2][1]
				notvalidated = !validate(world[:,_t1], world[:,_t2], target)
				#println("Validated: $(!notvalidated)")
			end
			
			t1 = min(_t1, _t2)
			t2 = max(_t1, _t2)
			
			indx = (t1-1)*nblocks - ((t1-1)*t1/2) + (t2 - t1)
			indx = round(Int, indx)
			ygold[:] = 0; ygold[indx] = 1
			world2[:,m] = vec(world) # fill the minibatch matrices
			target2[:,m] = target
			ygold2[:,m] = ygold
		end

		global ypred = forw(f, world2, target2)
		
		sl = softloss(ypred,ygold2); sloss = (n==1 ? sl : 0.99 * sloss + 0.01 * sl)
		zl = zeroone(ypred,ygold2);  zloss = (n==1 ? zl : 0.99 * zloss + 0.01 * zl)
		n==nextn && (println((n,sloss,1-zloss)); nextn+=2500)
		back(f, ygold2, softloss)
		update!(f)
		reset!(f)
	end
end

function train(worldf, worlddata, loss; gclip=0, dropout=false)
	sumloss = numloss = 0
	reset!(worldf)
	wybatches = worlddata[1]
	wposbatches = worlddata[2]

	for i=1:length(wybatches)
		x,y = wybatches[i]
		_,pos = wposbatches[i]

		if dropout
			ypred = forw(worldf, x, pos, dropout=true)
		else
			ypred = forw(worldf, x, pos)
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
	
	wybatches = worlddata[1]
	wposbatches = worlddata[2]

	for i=1:length(wybatches)
		x,y = wybatches[i]
		_,pos = wposbatches[i]
		ypred = forw(worldf, x, pos)
		#println("Gold: $(map(a -> indmax(y[:,a]), 1:size(y,2)))")
		#println("Pred: $(map(a -> indmax(to_host(ypred)[:,a]), 1:size(y,2)))")
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
	worlds = zeros(ftype, 60, size(rawdata, 1))

	ydim = 190
	y = zeros(ftype, ydim, size(rawdata, 1))
	targpos = zeros(ftype, 3, size(rawdata, 1))

	println("Number of instances: $(size(rawdata, 1))")
	idsaresorted = true
	nblocks=20
	
	for i=1:size(rawdata, 1)
		data = rawdata[i, :]

		source = round(Int, data[1, 121])
		worlds[:, i] = data[1, 1:60]'
		strt = (source-1)*3 + 61
		targpos[:, i] = data[1, strt:(strt+2)]'

		_t1 = round(Int, data[1, 122])
		_t2 = round(Int, data[1, 123])

		t1 = min(_t1, _t2)
		t2 = max(_t1, _t2)

		(t1 != _t1 || t2 != _t2) && (idsaresorted = false)
		
		indx = (t1-1)*nblocks - ((t1-1)*t1/2) + (t2 - t1)
		indx = round(Int, indx)
		#println("t1: $(t1), t2: $(t2), indx: $(indx)")
		#if i == 1
		#	println("World:\n$(reshape(worlds[:, i], 3, 20))")
		#	println("\nTargpos:\n$(targpos[:,i])")
		#	println("T1: $(t1), T2: $(t2), Indx: $(indx)")
		#end
		y[indx, i] = 1.0		
		
	end

	#!idsaresorted && (println("IDs are not sorted!!!"))
	return (minibatch(worlds, y, batchsize), minibatch(worlds, targpos, batchsize))
end

#Threshold 2*block_size
#julia loc2Multi.jl --lr 0.001 --epoch 50 --hidden 200 --pretrain
#dev best: 0.630597 
#julia loc2Multi.jl --lr 0.001 --epoch 50 --hidden 400 --pretrain
#dev best: 0.626866
#julia loc2Multi.jl --lr 0.001 --epoch 50 --hidden 800 --pretrain
#dev best: 0.63806

function main(args)
	s = ArgParseSettings()
	s.exc_handler=ArgParse.debug_handler
	@add_arg_table s begin
		("--worlddatafiles"; nargs='+'; default=["../../BlockWorld/Scene_Data/scene_data/Combined/Multi.train", "../../BlockWorld/Scene_Data/scene_data/Combined/Multi.dev"])
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
		("--pretrain"; action = :store_true; help="pretraining")
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

	rawworlddata = map(f->readdlm(f,Float32), o[:worlddatafiles])

	global worlddata = map(rawworlddata) do d
		r = randperm(size(d,1))
		get_worlds(d[r,:], batchsize=o[:batchsize], predtype=o[:predtype])
	end
	
	worldf = get_worldf(o[:predtype], o)

	drop = o[:dropout] != 0
	setp(worldf, adam=true)
	setp(worldf, lr=o[:lr])
	if o[:pretrain]
		pretraining(worldf)
		if !drop
			setp(worldf, adam=false)
			#setp(worldf, lr=0.001)
		end
	end
	
	lasterr = besterr = 1e6
	best_epoch = 0
	anger = 0
	stopcriterion = false
	df = DataFrame(epoch = Int[], lr = Float64[], trn_err = Float64[], dev_err = Float64[], best_err = Float64[], best_epoch = Int[])
	loss1 = o[:predtype] == "loc" ? quadloss : softloss
	loss2 = o[:predtype] == "loc" ? quadloss : zeroone

	for epoch=1:o[:epochs]      # TODO: experiment with pretraining
		
		@date trnerr = train(worldf, worlddata[1], loss1; gclip=o[:gclip], dropout=drop)
		@date deverr = test(worldf, worlddata[2], loss2)

		if deverr < besterr
			besterr=deverr
			best_epoch = epoch
			o[:bestfile]!=nothing && save(o[:bestfile], "net", clean(worldf))
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
