# Recurrent neural network which predicts Source, Target, and Relative Position independenantly
# Input File: JSONReader/data/2016-NAACL/STRP/*.mat

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
        ("--datafiles"; nargs='+'; default=["BlockWorld/logos/Train.STRP.data", "BlockWorld/logos/Dev.STRP.data"])
        ("--loadfile"; help="initialize model from file")
        ("--savefile"; help="save final model to file")
        ("--bestfile"; help="save best model to file")
        ("--hidden"; arg_type=Int; default=100; help="hidden layer size")
        ("--epochs"; arg_type=Int; default=10; help="number of epochs to train")
        ("--batchsize"; arg_type=Int; default=10; help="minibatch size")
        ("--lr"; arg_type=Float64; default=1.0; help="learning rate")
        ("--gclip"; arg_type=Float64; default=5.0; help="gradient clipping threshold")
        ("--dropout"; arg_type=Float64; default=0.5; help="dropout probability")
        ("--decay"; arg_type=Float64; default=0.9; help="learning rate decay if deverr increases")
        ("--nogpu"; action = :store_true; help="do not use gpu, which is used by default if available")
        ("--seed"; arg_type=Int; default=20160427; help="random number seed")
        ("--target"; arg_type=Int; default=1; help="which target to predict: 1:source,2:target,3:direction")
        ("--yvocab"; arg_type=Int; nargs='+'; default=[20,20,9]; help="vocab sizes for target columns (all columns assumed independent)")
        ("--xsparse"; action = :store_true; help="use sparse inputs, dense arrays used by default")
        ("--ftype"; default = "Float32"; help="floating point type to use: Float32 or Float64")
    end
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true); println(o)
    o[:seed] > 0 && setseed(o[:seed])
    o[:ftype] = eval(parse(o[:ftype]))
    Knet.gpu(!o[:nogpu])

    # Read data files: Limit to length 80 sentence (+3 for predictions)
    global rawdata = map(f->readdlm(f)[:,1:83], o[:datafiles])
    rawdata[rawdata.==""]=1
    xvocab = maximum(data[:,4:end])

    # Minibatch data: data[1]:train, data[2]:dev
    xrange = 4:80
    yrange = 1:3
    yvocab = o[:yvocab][o[:target]]
    global data = map(rawdata) do d
        minibatch(d, xrange, yrange, o[:batchsize]; xvocab=o[:xvocab], yvocab=yvocab, ftype=o[:ftype], xsparse=o[:xsparse])
    end

    # Load or create the model:
    global net = (o[:loadfile]!=nothing ? load(o[:loadfile], "net") :
                  compile(:rnnmodel; hidden=o[:hidden], output=yvocab, pdrop=o[:dropout]))
    setp(net, lr=o[:lr])
    lasterr = besterr = 1.0
    for epoch=1:o[:epochs]      # TODO: experiment with pretraining
        @date trnerr = train(net, data[1], softloss; gclip=o[:gclip])
        @date deverr = test(net, data[2], zeroone)
        println((epoch, o[:lr], trnerr, deverr))
        if deverr < besterr
            besterr=deverr
            o[:bestfile]!=nothing && save(o[:bestfile], "net", clean(net))
        end
        if deverr > lasterr
            o[:lr] *= o[:decay]
            setp(net, lr=o[:lr])
        end
        lasterr = deverr
    end
    o[:savefile]!=nothing && save(o[:savefile], "net", clean(net))
    @date devpred = predict(net, rawdata[2]; xrange=xrange, xvocab=o[:xvocab], ftype=o[:ftype], xsparse=o[:xsparse])
    println(devpred)
end

@knet function rnnmodel(word; hidden=100, embedding=hidden, output=20, pdrop=0.5)
    wvec = wdot(word; out=embedding)                 # TODO: try different embedding dimension than hidden
    hvec = lstm(wvec; out=hidden)                    # TODO: try more layers
    if predict                                       # TODO: try dropout between wdot and lstm
        dvec = drop(hvec; pdrop=pdrop)
        return wbf(dvec; out=output, f=:soft)
    end
end

### Minibatched data format:
# data is an array of (x,y,mask) triples
# x[xvocab+1,batchsize] contains one-hot word columns for the n'th word of batchsize sentences
# xvocab+1=eos is used for end-of-sentence
# sentences in a batch are padded at the beginning and get an eos at the end
# mask[batchsize] indicates whether i'th column of x is padding or not
# y is nothing until the very last token of a sentence batch
# y[yvocab,batchsize] contains one-hot target columns with the last token (eos) of a sentence batch

function train(f, data, loss; gclip=0)
    sumloss = numloss = 0
    reset!(f)
    for (x,y,mask) in data
        ypred = sforw(f, x, predict=(y!=nothing), dropout=true)
        y==nothing && continue
        sumloss += zeroone(ypred, y)*size(y,2)
        numloss += size(y,2)
        sback(f, y, loss; mask=mask)
        while !stack_isempty(f); sback(f); end
        update!(f; gclip=gclip)
        reset!(f)
    end
    sumloss / numloss
end

function test(f, data, loss)
    sumloss = numloss = 0
    reset!(f)
    for (x,y,mask) in data
        ypred = forw(f, x, predict=(y!=nothing))
        y==nothing && continue
        sumloss += loss(ypred, y)*size(y,2)
        numloss += size(y,2)
        reset!(f)
    end
    sumloss / numloss
end

function predict(f, data; xrange=4:83, padding=1, xvocab=326, ftype=Float32, xsparse=false)
    reset!(f)
    sentences = extract(data, xrange; padding=1)	# sentences[i][j] = j'th word of i'th sentence
    ypred = Any[]
    eos = xvocab + 1
    x = (xsparse ? sponehot : zeros)(ftype, eos, 1)
    for s in sentences
        for i = 1:length(s)
            setrow!(x, s[i], 1)
            forw(f, x, predict=false)
        end
        setrow!(x, eos, 1)
        y = forw(f, x, predict=true)
        push!(ypred, indmax(to_host(y)))
        reset!(f)
    end
    println(ypred)
end

function minibatch(data, xrange, yrange, batchsize; o...)
    x = extract(data, xrange; padding=1)	# x[i][j] = j'th word of i'th sentence
    y = extract(data, yrange)                   # y[i][j] = j'th class of i'th sentence
    s = sortperm(x, by=length)
    batches = Any[]
    for i=1:batchsize:length(x)
        j=min(i+batchsize-1,length(x))
        xx,yy = x[s[i:j]],y[s[i:j]]
        batchsentences(xx, yy, batches; o...)
    end
    return batches
end

function extract(data, xrange; padding=nothing)
    inst = Any[]
    for i=1:size(data,1)
        s = vec(data[i,xrange])
        if padding != nothing
            while s[end]==padding; pop!(s); end
        end
        push!(inst,s)
    end
    return inst
end

function batchsentences(x, y, batches; xvocab=326, yvocab=20, ftype=Float32, xsparse=false)
    @assert maximum(map(maximum,x)) <= xvocab
    @assert maximum(map(maximum,y)) <= yvocab
    eos = xvocab + 1
    batchsize = length(x)                       # number of sentences in batch
    maxlen = maximum(map(length,x))
    for t=1:maxlen+1                            # pad sentences in the beginning and add eos at the end
        xbatch = (xsparse ? sponehot : zeros)(ftype, eos, batchsize)
        mask = zeros(Cuchar, batchsize)         # mask[i]=0 if xbatch[:,i] is padding
        for s=1:batchsize                       # set xbatch[word][s]=1 if x[s][t]=word
            sentence = x[s]
            position = t - maxlen + length(sentence)
            if position < 1
                mask[s] = 0
            elseif position <= length(sentence)
                word = sentence[position]
                setrow!(xbatch, word, s)
                mask[s] = 1
            elseif position == 1+length(sentence)
                word = eos
                setrow!(xbatch, word, s)
                mask[s] = 1
            else
                error("This should not be happening")
            end
        end
        if t <= maxlen
            ybatch = nothing
        else
            ybatch = zeros(ftype, yvocab, batchsize)
            for s=1:batchsize
                answer = y[s][1]
                setrow!(ybatch, answer, s)
            end
        end
        push!(batches, (xbatch, ybatch, mask))
    end
end

# These assume one hot columns:
setrow!(x::SparseMatrixCSC,i,j)=(i>0 ? (x.rowval[j] = i; x.nzval[j] = 1) : (x.rowval[j]=1; x.nzval[j]=0); x)
setrow!(x::Array,i,j)=(x[:,j]=0; i>0 && (x[i,j]=1); x)

!isinteractive() && main(ARGS)

# TODO: write minibatching versions, this is too slow, even slower on gpu, so run with gpu(false)
# For minibatching:
# Sentences will have to be padded in the beginning, so they end together.
# They need to be sorted by length.
# Each minibatch will be accompanied with a mask.
# Possible minibatch sizes to get the whole data:
# 6003 = 9 x 23 x 29
# 855 = 9 x 5 x 19
