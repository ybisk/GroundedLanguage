using ArgParse
using JLD
using CUDArt
device(0)
using Knet: stack_isempty
using Knet
using DataFrames

#==============================================================
CURRENT MODEL DEFINITIONS

A+B -> S
@knet function model_abS(x; cwin1=1, cout1=1, hidden=100, output=20)
	w = par(init=Gaussian(0, 0.1), dims=(2, cwin1, 3, cout1))
	c = conv(w,x)
	h = wbf(c, out=hidden, f=:relu)
	return wbf(h; out=output, f=:soft)
end

A+B ->
@knet function model_abT(x, xb; cwin1=1, cout1=1)
	w = par(init=Xavier(), dims=(2, cwin1, 3, cout1))
	c = conv(w, x)
	t1 = wbf(c, out=200, f=:relu)
	t2 = wbf(t1, out=20, f=:sigm)
	h = xb * t2
	return h
end

A+T -> N+R
@knet function model_atNRP(w, x; hidden=800, output=180, winit=Gaussian(0,0.05))
	h = wbf2(w, x; out=hidden, f=:relu, winit=winit)
	return wbf(h; out=output, f=:soft, winit=winit)
end
======================================================#

function testS(worldf, worlddata, loss, loc=false)
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

function testT(worldf, worlddata, loss)
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

function testNRP(worldf, worlddata, loss, loc=false)
	sumloss = numloss = 0
	reset!(worldf)

	wybatches = worlddata[1]
	wposbatches = worlddata[2]

	for i=1:length(wybatches)
		x,y = wybatches[i]
		_,pos = wposbatches[i]
		ypred = forw(worldf, x, pos)
		#println("\nGold: $(ind2sub((9, 20), indmax(y)))")
		#println("\nPred: $(ind2sub((9, 20), indmax(to_host(ypred))))")
		sumloss += loss(ypred, y)*size(y,2)
		numloss += size(y,2)
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
		sumloss += loss(ypred, y)*size(y,2)
		numloss += size(y,2)
		reset!(worldf)
	end
	sumloss / numloss
end

flat(A) = mapreduce(x->isa(x,Array)? flat(x): x, vcat, [], A)

function predictS(worldf, worlddata)
	reset!(worldf)
	ypreds = Any[]
	for (x,y) in worlddata
		ypred = forw(worldf, x)
		push!(ypreds, indmax(to_host(ypred)[:,1]))
	end
	return flat(ypreds)
end

function predictT(worldf, worlddata)
	reset!(worldf)
	ypreds = Any[]
	for (x, y, xb) in worlddata
		ypred = forw(worldf, x, xb)
		push!(ypreds, to_host(ypred))
	end
	return ypreds
end

function predictNRP(worldf, worlddata)
	reset!(worldf)
	ypreds = Any[]
	wybatches = worlddata[1]
	wposbatches = worlddata[2]

	for i=1:length(wybatches)
		x,y = wybatches[i]
		_,pos = wposbatches[i]
		ypred = forw(worldf, x, pos)
		push!(ypreds, to_host(ypred)[:,1])
		reset!(worldf)
	end
	return ypreds
end


#predtype = id | grid | loc
function get_worlds(rawdata; batchsize=100, predtype = "id", ftype=Float32, target=1, oldstyle=true)
	worlds = zeros(ftype, 2, 20, 3, size(rawdata, 1))

	ydim = 20
	y = zeros(ftype, ydim, size(rawdata, 1))
	
	for indx=1:size(rawdata, 1)
		data = rawdata[indx, :]

		for i=1:20
			for j=1:3
				worlds[1, i, j, indx] = data[1, (i-1)*3 + j]
				worlds[2, i, j, indx] = data[1, (i+19)*3 + j]
			end
		end
		
		
		source = oldstyle ? round(Int, data[1, 222]) : round(Int, data[1, 121])
		y[source, indx] = 1
		
	end
	return minibatch(worlds, y, batchsize)
end

function get_worldsT(rawdata; batchsize=100, predtype = "id", ftype=Float32, oldstyle=true)
	#data = map(x -> (rawdata[x,1:end-64], rawdata[x,end-63:end]),1:size(rawdata,1));
	instances = Any[]

	ydim = 3
	for indx=1:size(rawdata, 1)
		data = rawdata[indx, :]
		worlds = zeros(ftype, 2, 20, 3)
		y = zeros(ftype, ydim, 1)
		for i=1:20
			for j=1:3
				worlds[1, i, j] = data[1, (i-1)*3 + j]
				worlds[2, i, j] = data[1, (i+19)*3 + j]
			end
		end

		source = oldstyle ? round(Int, data[1, 222]) : round(Int, data[1, 121])
		y[:, 1] = data[1, (61+(source-1)*3):(63+(source-1)*3)]

		push!(instances, (reshape(worlds, 2, 20, 3, 1), y, transpose(reshape(worlds[2,:,:], 20, 3))))
	end

	return instances
end

function get_worldsNRP(rawdata, targs; batchsize=100, predtype = "id", ftype=Float32, oldstyle=true)
	worlds = zeros(ftype, 60, size(rawdata, 1))

	ydim = 20 * 9
	y = zeros(ftype, ydim, size(rawdata, 1))
	targpos = zeros(ftype, 3, size(rawdata, 1))

	#println("Number of instances: $(size(rawdata, 1))")

	rinstance = rand(1:size(rawdata, 1))
	for indx=1:size(rawdata, 1)
		data = rawdata[indx, :]

		s = oldstyle ? round(Int, data[1, 222]) : round(Int, data[1, 121])

		worlds[:, indx] = data[1, 1:60]'
		strt = (s-1)*3 + 61
		#targpos[:, indx] = data[1, strt:strt+2]'
		targpos[:, indx] = targs[indx]
		_y = zeros(Float32, 9, 20)


		t = oldstyle ? round(Int, data[1, 223]) :  round(Int, data[1, 122])
		rp = oldstyle ? round(Int, data[1, 224]) : round(Int, data[1, 123])
		_y[rp, t] = 1
		y[:, indx] = vec(_y)

	end
	return (minibatch(worlds, y, batchsize), minibatch(worlds, targpos, batchsize))
end

function get_worldsMulti(rawdata, targs; batchsize=100, predtype = "id", ftype=Float32)
	worlds = zeros(ftype, 60, size(rawdata, 1))

	ydim = 190
	y = zeros(ftype, ydim, size(rawdata, 1))
	targpos = zeros(ftype, 3, size(rawdata, 1))

	idsaresorted = true
	nblocks=20

	for i=1:size(rawdata, 1)
		data = rawdata[i, :]

		source = round(Int, data[1, 121])
		worlds[:, i] = data[1, 1:60]'
		strt = (source-1)*3 + 61
		targpos[:, i] = targs[i]

		_t1 = round(Int, data[1, 122])
		_t2 = round(Int, data[1, 123])

		t1 = min(_t1, _t2)
		t2 = max(_t1, _t2)

		(t1 != _t1 || t2 != _t2) && (idsaresorted = false)

		indx = (t1-1)*nblocks - ((t1-1)*t1/2) + (t2 - t1)
		indx = round(Int, indx)
		y[indx, i] = 1.0

	end

	#!idsaresorted && (println("IDs are not sorted!!!"))
	return (minibatch(worlds, y, batchsize), minibatch(worlds, targpos, batchsize))
end


function main(args)
	s = ArgParseSettings()
	s.exc_handler=ArgParse.debug_handler
	@add_arg_table s begin
		("--file"; default="../BlockWorld/Scene_Data/scene_data/GoldDigitAll/STRP.test")
		("--modelABS"; default= "ModelAB-S.jld"; help="A+B => S")
		("--modelABT"; default= "ModelAB-T.jld"; help="A+B => T")
		("--modelATNR"; default= "ModelAB-NR.jld"; help="A+T => N+R")
		("--modelMulti"; default= "Multi.jld"; help="A+T => R1+R2")
		("--writefile"; action = :store_true; help="produces inputfile.predicted")
		("--probs"; action = :store_true; help="give some predictions")
		("--multi"; action = :store_true; help="multi")
	end

	isa(args, AbstractString) && (args=split(args))
	o = parse_args(args, s; as_symbols=true); println(o)

	net1 = load(o[:modelABS], "net")
	net2 = load(o[:modelABT], "net")
	net3 = load(o[:modelATNR], "net")

	rawdata = readdlm(o[:file],Float32)
	worlddata = get_worlds(rawdata, batchsize=1, predtype="id", target=1, oldstyle=false)

	testerr = testS(net1, worlddata, zeroone)
	println("\n****RESULTS****")
	println("Accuracy of predicting the source block id: $(1-testerr)")

	preds = predictS(net1, worlddata)
	rawdata[:,121] = preds
	
	worlddata = get_worldsT(rawdata, batchsize=1, oldstyle=false)

	testerr = testT(net2, worlddata, quadloss)
	println("Error of predicting the target loc (in terms of average block sizes): $(sqrt(2*testerr)/0.1254)")
	preds = predictT(net2, worlddata)

	if o[:multi]
		net4 = load(o[:modelMulti], "net")
		testMulti = testNRP
		predictMulti = predictNRP
		worlddata = get_worldsMulti(rawdata, preds, batchsize=1)
		testerr = testMulti(net4, worlddata, zeroone)
		println("Accuracy of predicting the R1+R2: $(1-testerr)")
		
		mapping = Dict()
		indx = 1
		for i=1:19
			for j=(i+1):20
				mapping[indx] = (i,j)
				indx += 1
			end
		end

		preds = predictNRP(net4, worlddata)
		for i=1:length(preds)
			r1, r2 = mapping[round(Int, indmax(preds[i]))]
			rawdata[i, 122] = r1
			rawdata[i, 123] = r2
		end

		if o[:probs]
			println("\n**** Examples ****")

			for i=1:length(preds)
				println("")
				println(preds[i])
				l = collect(enumerate(preds[i]))
				sort!(l, by=(x -> x[2]), rev=true)
				
				println("Probs: 1: $(l[1][2]), 2: $(l[2][2])")
				println("(R1, R2): 1: $(mapping[l[1][1]]), 2: $(mapping[l[2][1]])")

				println("")
			end
		end


	else	
		worlddata = get_worldsNRP(rawdata, preds, batchsize=1, oldstyle=false)
	
		testerr = testNRP(net3, worlddata, zeroone)
		println("Accuracy of predicting the N+RP: $(1-testerr)")

		preds = predictNRP(net3, worlddata)
		for i=1:length(preds)
			rp, t = ind2sub((9, 20), indmax(preds[i]))
			rawdata[i, 122] = round(Int, t)
			rawdata[i, 123] = round(Int, rp)
		end
		if o[:probs]
			println("\n**** Examples ****")

			for i=1:length(preds)
				println("")
				println(preds[i])
				l = collect(enumerate(preds[i]))
				sort!(l, by=(x -> x[2]), rev=true)

				println("Probs: 1: $(l[1][2]), 2: $(l[2][2])")
				println("(Direction, Block): 1: $(ind2sub((9,20), l[1][1])), 2: $(ind2sub((9,20), l[2][1]))")

				println("")
			end
		end

	end

	o[:writefile] && writedlm("$(o[:file]).predicted", rawdata, ' ')

		
end
!isinteractive() && main(ARGS)
