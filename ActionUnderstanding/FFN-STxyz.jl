# Feed-forward neural network which predicts Source and Target XYZ location
# Input File: JSONReader/data/2016-NAACL/STxyz/*.mat

using ArgParse
using JLD
using CUDArt
using Knet

function main(args)
    s = ArgParseSettings()
    s.exc_handler=ArgParse.debug_handler
    @add_arg_table s begin
        ("--datafiles"; nargs='+'; default=["JSONReader/data/2016-NAACL/STxyz/Train.mat",
                                            "JSONReader/data/2016-NAACL/STxyz/Dev.mat",
                                            "JSONReader/data/2016-NAACL/STxyz/Test.mat"])
        ("--loadfile"; help="initialize model from file")
        ("--savefile"; help="save final model to file")
        ("--bestfile"; help="save best model to file")
        ("--epochs"; arg_type=Int; default=40; help="number of epochs to train")
        ("--target"; arg_type=Int; default=2; help="which location to predict: 1:source,2:target")
        # DY: this is set to 9 because I can only minibatch with one world
        ("--batchsize"; arg_type=Int; default=9; help="minibatch size, all instances in a minibatch must share the same world")

        ("--hidden"; arg_type=Int; default=256; help="hidden layer size")
        ("--ndirs"; arg_type=Int; default=9; help="number of direction offsets to learn")
        ("--actf"; default=":relu"; help="activation function")
        ("--layers"; default=1; help="number of hidden layers")

        ("--dropout"; arg_type=Float64; default=0.5; help="dropout probability")
        ("--winit"; default="GlorotUniform(sqrt(6.0))")

        ("--nogpu"; action = :store_true; help="do not use gpu, which is used by default if available")
        ("--seed"; arg_type=Int; default=20160427; help="random number seed")
        ("--ftype"; default = "Float32"; help="floating point type to use: Float32 or Float64")

        # DY: not sure if we need these with adam
        # ("--lr"; arg_type=Float64; default=1.0; help="learning rate")
        # ("--decay"; arg_type=Float64; default=0.9; help="learning rate decay if deverr increases")

        # DY: this is read from the data now
        # ("--yvocab"; arg_type=Int; nargs='+'; default=[20,20,9]; help="vocab sizes for target columns (all columns assumed independent)")
        # DY: sparse not implemented yet
        # ("--xsparse"; action = :store_true; help="use sparse inputs, dense arrays used by default")
    end
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true); println(o)
    o[:seed] > 0 && setseed(o[:seed])
    o[:ftype] = eval(parse(o[:ftype]))
    o[:winit] = eval(parse(o[:winit]))
    o[:actf] = eval(parse(o[:actf]))
    Knet.gpu(!o[:nogpu])
    global xvocab = 0
    global maxlen = 0

    info("Reading data files...")
    trange = (o[:target]==1 ? (1:3) : o[:target]==2 ? (4:6) : error("Bad target $(o[:target])"))
    # A minibatch consists of a target location, a world, and 9 sentences that share the same target/world.
    global data = map(o[:datafiles]) do f
        d = readdlm(f)
        minibatches = Any[]
        for i1=1:o[:batchsize]:size(d,1)
            target = reshape(convert(Array{o[:ftype]}, d[i1,trange]), (3,1))
            world = reshape(convert(Array{o[:ftype]}, d[i1,7:66]), (3,20))
            sentences = Any[]
            for i=i1:(i1+o[:batchsize]-1)
                @assert target == reshape(convert(Array{o[:ftype]}, d[i1,trange]), (3,1))
                @assert world == reshape(convert(Array{o[:ftype]}, d[i1,7:66]), (3,20))
                sent = Int[]
                for j=67:size(d,2)
                    d[i,j]=="" && break
                    push!(sent, d[i,j])
                    d[i,j] > xvocab && (xvocab = d[i,j])
                end
                length(sent) > maxlen && (maxlen = length(sent))
                push!(sentences, sent)
            end
            push!(minibatches, (target, world, sentences))
        end
        return minibatches
    end

    info("Initializing model...")
    global net = (o[:loadfile]!=nothing ? load(o[:loadfile], "net") :
                  compile(:ffnxyz; hidden=o[:hidden], ndirs=o[:ndirs], pdrop=o[:dropout], winit=o[:winit],
                          f=o[:actf], layers=o[:layers]))
    setp(net, lr=0.001, adam=true)
    lastloss = bestloss = Inf
    for epoch=1:o[:epochs]
        trnloss = train(net, data[1], quadloss; xvocab=xvocab, maxlen=maxlen, ftype=o[:ftype], batchsize=o[:batchsize])
        devloss = test(net, data[2], quadloss;  xvocab=xvocab, maxlen=maxlen, ftype=o[:ftype], batchsize=o[:batchsize])
        tstloss = test(net, data[3], quadloss;  xvocab=xvocab, maxlen=maxlen, ftype=o[:ftype], batchsize=o[:batchsize])
        println((epoch, trnloss, devloss, tstloss)); flush(STDOUT)
        if devloss < bestloss
            bestloss=devloss
            o[:bestfile]!=nothing && save(o[:bestfile], "net", clean(net))
        end
        lastloss = devloss
    end
    o[:savefile]!=nothing && save(o[:savefile], "net", clean(net))
end

@knet function ffnxyz(sent, coor; hidden=256, nblocks=20, ndims=3, ndirs=9, pdrop=0.5, winit=Xavier(), f=:relu, layers=1, o...)
    hvec = repeat(sent; frepeat=:wbf, nrepeat=layers, f=f, out=hidden, winit=winit)
    dvec = drop(hvec; pdrop=pdrop)
    refblock = wbf(dvec; out=nblocks, f=:soft73, winit=winit) # soft does much better than sigm here
    refxyz = coor * refblock
    direction = wbf(dvec; out=ndirs, f=:soft73, winit=winit) # TODO: try more dirs?
    offset = wdot(direction; out=ndims, winit=winit)
    return refxyz + offset
end


function train(f, data, loss; gclip=0, xvocab=658, maxlen=80, batchsize=9, update=true, ftype=Float32, ndims=3)
    sumloss = numloss = 0
    x = zeros(ftype, xvocab*maxlen, batchsize)
    ygold = zeros(ftype, ndims, batchsize)
    for (target, world, sents) in data
        length(sents) == batchsize || error("Bad length")
        fill!(x,0)
        for i=1:batchsize
            for j=1:length(sents[i])
                x[sents[i][j]+xvocab*(j-1),i]=1
            end
        end
        ypred = forw(f, x, world; dropout=update)
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

# temp workaround: prevents error in running finalizer: ErrorException("auto_unbox: unable to determine argument type")
@gpu atexit(()->(for r in net.reg; r.out0!=nothing && Main.CUDArt.free(r.out0); end))
#!isinteractive() && main(ARGS)
main(ARGS)
