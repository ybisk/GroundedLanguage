using ArgParse
using JLD
using CUDArt
device(0)
using Knet: stack_isempty
using Knet
using DataFrames
using Base.LinAlg: axpy!, scale!

function get_worldf(predtype, o)
	if predtype == "id" || predtype == "grid"
		@knet function output_layer(x; output=180)
			return wbf(x; out=output, f=:soft)
		end
	else
		@knet function output_layer(x; output=3)
			return wb(x; out=output)
		end
	end

	@knet function fnn(w, x; hidden=400, output=180, winit=Gaussian(0,0.05))
		h = wbf2(w, x; out=hidden, f=:relu, winit=winit)
		return wbf(h; out=output, f=:soft, winit=winit)
	end

	@knet function droprelu(x; hidden=100, pdrop=0.5)
		d = drop(x, pdrop=pdrop)
		return wbf(d; out=hidden, f=:relu)
	end

	@knet function fnn2(w, x; hidden=400, pdrop=0.5, output=180, nlayers=2)
		h = wbf2(w, x; out=hidden, f=:relu, winit=Gaussian(0,0.05))
		hl = repeat(h; frepeat=:droprelu, nrepeat=nlayers-1, hidden=hidden, pdrop=pdrop)
		return wbf(hl; out=output, f=:soft)
	end
	
	outdim = predtype == "id" ? 180 : 18*18
	outdim = predtype == "loc" ? 3 : outdim
	worldf = nothing
	if o[:dropout] == 0
		worldf = compile(:fnn; hidden=o[:hidden], output=outdim)
	else
		worldf = compile(:fnn2; hidden=o[:hidden], pdrop=o[:dropout], output=outdim, nlayers=o[:nlayers])
	end
	return worldf
end

function pretraining(f; N=2^15, dims=(16, 1, 16), nblocks=20, ndims= length(dims), ndirs=9, lr=0.001, adam=true, nbatch=128, winit=Gaussian(0,0.05))
	sloss = zloss = 0
	nextn = 1
	ncells = prod(dims)
	global world = zeros(Float32, ndims, nblocks)
	global target = zeros(Float32, ndims, 1)
	global ygold = zeros(Float32, ndirs, nblocks)
	global world2 = zeros(Float32, length(world), nbatch)
	global target2 = zeros(Float32, length(target), nbatch)
	global ygold2 = zeros(Float32, length(ygold), nbatch)

	mapping = Dict()
	mapping[1] = 5
	mapping[2] = 4
	mapping[3] = 3
	mapping[4] = 6
	mapping[5] = 9
	mapping[6] = 2
	mapping[7] = 7
	mapping[8] = 8
	mapping[9] = 1

	tstcorrect = 0
	tstmissclassified = 0
	for n=1:N
	#for n=1:(N+10)
		for m=1:nbatch
			locations = randperm(ncells)
			rnumblocks = rand(10:20)#need missing blocks
			for b=1:nblocks
				if b <= rnumblocks
					world[:,b] = 2*([ind2sub(dims, locations[b])...] / 16 - 0.5) # fill blocks with random locations
					world[2,b] = 0.1
				else
					world[:,b] = -1
				end
			end
			target[:,1] = 2*([ind2sub(dims, randperm(ncells)[1])...] / 16 - 0.5) # pick target at an empty location
			target[2,1] = 0.1
			d1 = world .- target
			d2 = sum(d1 .* d1, 1)
			rblock = rand(find(d2 .== minimum(d2))) # pick one of the closest blocks randomly as reference
			d  = sign(target - world[:,rblock])    # a direction like [-1,0]
			d += 2
			d9 = sub2ind((3,3), round(Int, d[1]), round(Int, d[3]))
			d9 = mapping[d9]

			#d8 = d8 == 5 ? 9 : d8 > 5 ? d8 - 1 : d8
			#println("d8: $d8")
			#d8 = (d[1]==0 ? 0 : d[1]==1 ? 1 : 2) + 3*(d[3]==0 ? 0 : d[3]==1 ? 1 : 2) # map dir to [1:8]
			#@assert (d8 >= 1 && d8 <= 8)
			ygold[:] = 0; ygold[d9,rblock] = 1
			# fill the minibatch matrices
			world2[:,m] = vec(world) 
			target2[:,m] = vec(target)
			ygold2[:,m] = vec(ygold)
		end
		global ypred = forw(f, world2, target2)
		sl = softloss(ypred,ygold2); sloss = (n==1 ? sl : 0.99 * sloss + 0.01 * sl)
		zl = zeroone(ypred,ygold2);  zloss = (n==1 ? zl : 0.99 * zloss + 0.01 * zl)
		n==nextn && (println((n,sloss,1-zloss)); nextn*=2)
		
		id_ypred = map(a -> indmax(to_host(ypred)[:,a]), 1:size(ygold2,2))
		id_ygold = map(a -> indmax(ygold2[:,a]), 1:size(ygold2,2))
		
		if n > 2^15
			for j=1:length(id_ypred)
				if id_ypred[j] != id_ygold[j]
					gold_sub = ind2sub((9, 20), id_ygold[j])
					println("\n*************")
					println("Gold: $(gold_sub)")

					pred_sub = ind2sub((9, 20), id_ypred[j])
					println("Pred: $(pred_sub)")

					world = reshape(world2[:,j], 3, 20)
					println("World:\n$(world)")
					println("Relative Block's Loc:\n$(world[:, gold_sub[2]])\n")

					println("Target Loc:\n$(target2[:,j])\n")

					tstmissclassified += 1.0
				else
					tstcorrect += 1.0
				end
			end
		else
			back(f, ygold2, softloss)
			update!(f, gclip=5.0)
			reset!(f)

		end
	end

	println("Acc on held out data: $(tstcorrect / (tstcorrect + tstmissclassified))\n")
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
		id_ypred = map(a -> indmax(to_host(ypred)[:,a]), 1:size(y,2))
		id_ygold = map(a -> indmax(y[:,a]), 1:size(y,2))
		#===
		for j=1:length(id_ypred)
			if id_ypred[j] != id_ygold[j]
				gold_sub = ind2sub((9, 20), id_ygold[j])
				println("\n*************")
				println("Gold: $(gold_sub)")

				pred_sub = ind2sub((9, 20), id_ypred[j])
				println("Pred: $(pred_sub)")
				
				world = reshape(x[:,j], 3, 20)
				println("World:\n$(world)")
				println("Relative Block's Loc:\n$(world[:, gold_sub[2]])\n")

				println("Target Loc:\n$(pos[:,j])\n")
			end
		end
		===#
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
function get_worlds(rawdata; batchsize=100, predtype = "id", ftype=Float32, target=2)
	worlds = zeros(ftype, 60, size(rawdata, 1))

	ydim = 0
	if predtype == "id"
		ydim = target == 3 ? 20 * 9: 20
	elseif predtype == "loc"
		ydim = 3
	else
		ydim = 18 * 18
	end
	y = zeros(ftype, ydim, size(rawdata, 1))
	targpos = zeros(ftype, 3, size(rawdata, 1))

	println("Number of instances: $(size(rawdata, 1))")
	
	rinstance = rand(1:size(rawdata, 1))
	for indx=1:size(rawdata, 1)
		#===========
		Data Format
		World_t  World_t+1 Text S T RP Loc Grid
		  60       60      101  1 1 1   3   1     == 228

		  nx 101
		  xvocab 622
		===========#
		data = rawdata[indx, :]
		
		s = round(Int, data[1, 222])

		worlds[:, indx] = data[1, 1:60]'
		targpos[:, indx] = data[1, 225:227]'
		_y = zeros(Float32, 9, 20)

		if predtype == "id"
			source = 0
			if target == 3

				t = round(Int, data[1, 223])
				rp = round(Int, data[1, 224])
				#======debug============
				if indx == rinstance
					println("Data:\n$(data[1, 1:60])\n")
					println("Target:\n$(data[1, 225:227])\n")
					println("t: $t")
					println("rp: $rp")
				end
				=========================#
				#source = (rp - 1) * 20 + t
				_y[rp, t] = 1
			else
				source = round(Int, data[1, 222+target-1])
			end
			y[:, indx] = vec(_y)
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
		
	end
	return (minibatch(worlds, y, batchsize), minibatch(worlds, targpos, batchsize))
end

#julia loc2TP.jl --lr 0.001 --epoch 20 --target 3 --hidden 800 --pretrain
#with missing blocks: 74.4 (acc)
#without missing blocks: 72 (acc)
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
		get_worlds(d[r,:], batchsize=o[:batchsize], predtype=o[:predtype], target=o[:target])
	end
	
	worldf = get_worldf(o[:predtype], o)

	setp(worldf, adam=true)
	setp(worldf, lr=o[:lr])
	loss1 = o[:predtype] == "loc" ? quadloss : softloss
	loss2 = o[:predtype] == "loc" ? quadloss : zeroone

	if o[:pretrain]
		pretraining(worldf)

		trnerr = test(worldf, worlddata[1], loss2)
		deverr = test(worldf, worlddata[2], loss2)

		println("After Pretraining:\nTrn Err: $trnerr, Dev Err: $deverr\n")
		setp(worldf, adam=false)
		setp(worldf, lr=0.0001)
	end

	lasterr = besterr = 1e6
	best_epoch = 0
	anger = 0
	stopcriterion = false
	df = DataFrame(epoch = Int[], lr = Float64[], trn_err = Float64[], dev_err = Float64[], best_err = Float64[], best_epoch = Int[])
	
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
