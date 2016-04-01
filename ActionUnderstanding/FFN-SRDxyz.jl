# A linear model which predicts Source and Target XYZ location given SRD ids and the world
# Input File: JSONReader/data/2016-NAACL/SRDxyz/*.mat

using ArgParse
using JLD
using CUDArt
using Knet

@knet function srdxyz(srd, coor)
    a = par(init=Constant(0), dims=(0,0))
    b = par(init=Constant(0), dims=(0,0))
    return coor * (a * srd) + b * srd
end

function main(args)
    s = ArgParseSettings()
    s.exc_handler=ArgParse.debug_handler
    @add_arg_table s begin
        ("--datafiles"; nargs='+'; default=["JSONReader/data/2016-NAACL/SRDxyz/Train.mat",
                                            "JSONReader/data/2016-NAACL/SRDxyz/Dev.mat",
                                            "JSONReader/data/2016-NAACL/SRDxyz/Test.mat"])
        ("--loadfile"; help="initialize model from file")
        ("--savefile"; help="save final model to file")
        ("--bestfile"; help="save best model to file")
        ("--target"; arg_type=Int; default=1; help="which location to predict: 1:source,2:target")
        ("--epochs"; arg_type=Int; default=60; help="number of epochs to train")
        ("--batchsize"; arg_type=Int; default=9; help="number of examples with the same world")
        ("--nogpu"; action = :store_true; help="do not use gpu, which is used by default if available")
        ("--ftype"; default = "Float32"; help="floating point type to use: Float32 or Float64")
    end
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true); println(o)
    o[:ftype] = eval(parse(o[:ftype]))
    Knet.gpu(!o[:nogpu])

    # A minibatch consists of a location (source or target), a world, and 9 S/R/D triples that share the same location/world.
    info("Reading data files...")
    global trange = (o[:target]==1 ? (1:3) : o[:target]==2 ? (4:6) : error("Bad target $(o[:target])"))
    global srdmax = zeros(Int,3)
    global data = map(o[:datafiles]) do f
        d = readdlm(f)
        minibatches = Any[]
        for i1=1:o[:batchsize]:size(d,1)
            target = reshape(convert(Array{o[:ftype]}, d[i1,trange]), (3,1))
            world = reshape(convert(Array{o[:ftype]}, d[i1,10:69]), (3,20))
            triples = Any[]
            for i=i1:(i1+o[:batchsize]-1)
                @assert target == reshape(convert(Array{o[:ftype]}, d[i1,trange]), (3,1))
                @assert world == reshape(convert(Array{o[:ftype]}, d[i1,10:69]), (3,20))
                srd = 1+Int[d[i,7:9]...]
                for j=1:3; srd[j] > srdmax[j] && (srdmax[j]=srd[j]); end
                push!(triples, srd)
            end
            push!(minibatches, (target, world, triples))
        end
        return minibatches
    end

    info("Initializing model...")
    global net = (o[:loadfile]!=nothing ? load(o[:loadfile], "net") : compile(:srdxyz))
    setp(net, lr=0.001, adam=true)
    lastloss = bestloss = Inf
    for epoch=1:o[:epochs]
        trnloss = train(net, data[1], quadloss; srdmax=srdmax, ftype=o[:ftype], batchsize=o[:batchsize])
        devloss = test(net, data[2], quadloss;  srdmax=srdmax, ftype=o[:ftype], batchsize=o[:batchsize])
        tstloss = test(net, data[3], quadloss;  srdmax=srdmax, ftype=o[:ftype], batchsize=o[:batchsize])
        println((epoch, trnloss, devloss, tstloss)); flush(STDOUT)
        if devloss < bestloss
            bestloss=devloss
            o[:bestfile]!=nothing && save(o[:bestfile], "net", clean(net))
        end
        lastloss = devloss
    end
    o[:savefile]!=nothing && save(o[:savefile], "net", clean(net))
end

function train(f, data, loss; srdmax=[20,20,9], batchsize=9, update=true, ftype=Float32)
    sumloss = numloss = 0
    x = zeros(ftype, sum(srdmax), batchsize)
    ygold = zeros(ftype, 3, batchsize)
    for (target, world, sents) in data
        length(sents) == batchsize || error("Bad length")
        x[:] = 0
        for j=1:batchsize
            (s,r,d) = sents[j]
            x[s,j] = 1
            x[r+srdmax[1],j] = 1
            x[d+srdmax[1]+srdmax[2],j] = 1
        end
        ypred = forw(f, x, world)
        ygold[:] = 0
        ygold .+= target
        sumloss += loss(ypred, ygold); numloss += 1
        if update
            back(f, ygold, loss)
            update!(f)
        end
    end
    return sumloss / numloss
end

function test(f, data, loss; o...)
    train(f, data, loss; o..., update=false)
end

main(ARGS)
