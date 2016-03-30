# Recurrent neural network which predicts Source and predicts XYZ final location
# Input File: JSONReader/data/2016-NAACL/Sxyz/*.mat
# DY: we can use RNN-SRD.jl for Source prediction, this model will only do XYZ target location.

using ArgParse
using JLD
using CUDArt
device(0)
using Knet: stack_isempty
using Knet

function main(args)
    s = ArgParseSettings()
    s.exc_handler=ArgParse.debug_handler
    @add_arg_table s begin
        ("--datafiles"; nargs='+'; default=["JSONReader/data/2016-NAACL/Sxyz/Train.mat",
                                            "JSONReader/data/2016-NAACL/Sxyz/Dev.mat",
                                            "JSONReader/data/2016-NAACL/Sxyz/Test.mat"])
        ("--loadfile"; help="initialize model from file")
        ("--savefile"; help="save final model to file")
        ("--bestfile"; help="save best model to file")
        ("--epochs"; arg_type=Int; default=40; help="number of epochs to train")

        ("--hidden"; arg_type=Int; default=256; help="hidden layer size")
        ("--embedding"; arg_type=Int; default=0; help="word embedding size (default same as hidden)")
        ("--ndirs"; arg_type=Int; default=9; help="number of direction offsets to learn")

        ("--gclip"; arg_type=Float64; default=5.0; help="gradient clipping threshold")
        ("--dropout"; arg_type=Float64; default=0.5; help="dropout probability")
        ("--nogpu"; action = :store_true; help="do not use gpu, which is used by default if available")
        ("--seed"; arg_type=Int; default=20160427; help="random number seed")

        # DY: this is set to 9 because I can only minibatch with one world
        # ("--batchsize"; arg_type=Int; default=10; help="minibatch size")
        # DY: not sure if we need these with adam
        # ("--lr"; arg_type=Float64; default=1.0; help="learning rate")
        # ("--decay"; arg_type=Float64; default=0.9; help="learning rate decay if deverr increases")

        # DY: only one target for now:
        # ("--target"; arg_type=Int; default=1; help="which target to predict: 1:source,2:target,3:direction")
        # DY: this is read from the data now
        # ("--yvocab"; arg_type=Int; nargs='+'; default=[20,20,9]; help="vocab sizes for target columns (all columns assumed independent)")
        # DY: put these back later if we need them
        # ("--xsparse"; action = :store_true; help="use sparse inputs, dense arrays used by default")
        # ("--ftype"; default = "Float32"; help="floating point type to use: Float32 or Float64")
    end
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true); println(o)
    o[:seed] > 0 && setseed(o[:seed])
    # o[:ftype] = eval(parse(o[:ftype]))
    o[:embedding] == 0 && (o[:embedding] = o[:hidden])
    Knet.gpu(!o[:nogpu])
    global xvocab = 0

    info("Reading data files...")
    # A minibatch consists of a target location, a world, and 9 sentences that share the same target/world.
    global data = map(o[:datafiles]) do f
        d = readdlm(f)
        minibatches = Any[]
        for i1=1:9:size(d,1)
            target = reshape(convert(Array{Float32}, d[i1,2:4]), (3,1))
            world = reshape(convert(Array{Float32}, d[i1,5:64]), (3,20))
            sentences = Any[]
            for i=i1:(i1+8)
                @assert target == reshape(convert(Array{Float32}, d[i,2:4]), (3,1))
                @assert world == reshape(convert(Array{Float32}, d[i,5:64]), (3,20))
                sent = Int[]
                for j=65:size(d,2)
                    d[i,j]=="" && break
                    push!(sent, d[i,j])
                    d[i,j] > xvocab && (xvocab = d[i,j])
                end
                push!(sentences, sent)
            end
            push!(minibatches, (target, world, sentences))
        end
        return minibatches
    end

    info("Initializing model...")
    global net = (o[:loadfile]!=nothing ? load(o[:loadfile], "net") :
                  compile(:rnnxyz; hidden=o[:hidden], embedding=o[:embedding], ndirs=o[:ndirs], pdrop=o[:dropout]))
    # setp(net, lr=o[:lr])
    setp(net, lr=0.001, adam=true)
    lastloss = bestloss = Inf
    for epoch=1:o[:epochs]      # TODO: experiment with pretraining
        trnloss = train(net, data[1], quadloss; gclip=o[:gclip], xvocab=xvocab)
        devloss = test(net, data[2], quadloss)
        tstloss = test(net, data[3], quadloss)
        println((epoch, trnloss, devloss, tstloss))
        if devloss < bestloss
            bestloss=devloss
            o[:bestfile]!=nothing && save(o[:bestfile], "net", clean(net))
        end
        # if devloss > lastloss
        #     o[:lr] *= o[:decay]
        #     setp(net, lr=o[:lr])
        # end
        lastloss = devloss
    end
    o[:savefile]!=nothing && save(o[:savefile], "net", clean(net))
end

@knet function rnnxyz(word, coor; hidden=256, embedding=hidden, nblocks=20, ndims=3, ndirs=9, pdrop=0.5)
    wvec = wdot(word; out=embedding)                 # TODO: try different embedding dimension than hidden
    hvec = lstm(wvec; out=hidden)                    # TODO: try more layers
    if predict                                       # TODO: try dropout between wdot and lstm
        dvec = drop(hvec; pdrop=pdrop)
        refblock = wbf(dvec; out=nblocks, f=:soft73) # soft does much better than sigm here
        refxyz = coor * refblock
        direction = wbf(dvec; out=ndirs, f=:soft73) # TODO: try more dirs?
        offset = wdot(direction; out=ndims)
        return refxyz + offset
    end
end


function train(f, data, loss; gclip=0, xvocab=658, batchsize=9, update=true)
    sumloss = numloss = 0
    reset!(f)
    x = zeros(Float32, xvocab, batchsize)
    y = zeros(Float32, 3, batchsize)
    mask = zeros(Cuchar, batchsize)
    for (target, world, sents) in data
        length(sents) == batchsize || error("Bad length")
        T = maximum(map(length, sents))
        for t = 1:T
            x[:] = mask[:] = 0
            for j = 1:length(sents)
                s = sents[j]
                spos = t - T + length(s)
                if spos > 0
                    i = s[spos]
                    x[i,j] = 1
                    mask[j] = 1
                end
            end
            ypred = sforw(f, x, world, predict=(t==T), dropout=true)
            if t == T
                y[:] = 0
                y .+= target
                sumloss += loss(ypred, y); numloss += 1
                if update
                    sback(f, y, loss) # no need for mask?
                    while !stack_isempty(f); sback(f); end
                    update!(f, gclip=gclip)
                end
                reset!(f)
            end
        end
    end
    return sumloss / numloss
end

function test(f, data, loss; o...)
    train(f, data, loss; o..., update=false)
end

#!isinteractive() && main(ARGS)
main(ARGS)
