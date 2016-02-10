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
    return generic_layer(x; f1=:conv, f2=:relu, wdims=(cwindow,cwindow,cinput,coutput), bdims=(1,1,coutput,1))
end

#===============
@knet function conv_pool_layer(x; cwindow=0, cinput=0, coutput=0, pwindow=0)
    y = generic_layer(x; f1=:conv, f2=:relu, wdims=(cwindow,cwindow,cinput,coutput), bdims=(1,1,coutput,1))
    return pool(y; window=pwindow)
end
===============#

@knet function conv_pool_layer(x; cwindow=0, cinput=0, coutput=0, pwindow=2, padding=0)
    w = par(init=Xavier(), dims=(cwindow,cwindow,cinput,coutput))
    c = conv(w,x)
    b = par(init=Constant(0), dims=(1,1,coutput,1))
    a = add(b,c)
    r = relu(a)
    return pool(r; window=pwindow)
end

@knet function softmax_layer(x; input=0, output=0)
    return generic_layer(x; f1=:dot, f2=:soft, wdims=(output,input), bdims=(output,1))
end

@knet function cnn(x; cwin1=5, cout1=20, cwin2=5, cout2=20, pwin1=2, pwin2=2, hidden=100)
    a = conv_pool_layer(x; cwindow=cwin1, coutput=cout1, pwindow=pwin1)
    b = conv_pool_layer(a; cwindow=cwin2, coutput=cout2, pwindow=pwin2)
    c = relu_layer(b; output=hidden)
    return c
end

#====================
@knet function cnn(x; cwin1=2, cout1=20, pwin1=2, padding=0, hidden=100)
    a = conv_pool_layer(x; cwindow=cwin1, coutput=cout1, pwindow=pwin1)
    return relu_layer(a; output=hidden)
end
===================#

@knet function droplstm(x; hidden=100, pdrop=0.5)
	d = drop(x, pdrop=pdrop)
	return lstm(d, out=hidden)
end

@knet function wdot2(x, y; out=0, winit=Xavier(), o...)
    w1 = par(; o..., init=winit, dims=(out,0))
    w2 = par(; o..., init=winit, dims=(out,0))
    r1 = w1*x
    r2 = w2*y
    return add(r1, r2)
end

@knet function rnnmodel(word, state; hidden=100, embedding=hidden, output=20, pdrop=0.5, fpdrop=0.5, lpdrop=0.5, nlayers=2)
	#wvec = wdot(word; out=embedding)                 # TODO: try different embedding dimension than hidden
    wsvec = wdot2(word, state; out=embedding)
	#hd = drop(wsvec, pdrop=fpdrop)
	lh = lstm(wsvec, out=hidden)
	h = repeat(lh; frepeat=:droplstm, nrepeat=nlayers-1, hidden=hidden, pdrop=pdrop)
	if predict                                       # TODO: try dropout between wdot and lstm
		dvec = drop(h; pdrop=lpdrop)
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

function train(sentf, worldf, data, worlddata, locs, loss, loc=false; gclip=0)
    sumloss = numloss = 0
    reset!(sentf)
    reset!(worldf)
    i = 1
    getstate = true
    state = nothing
    for (x,y,mask) in data
        if getstate
            world = worlddata[i]
            i += 1
            getstate = false
            state = forw(worldf, world)
        end
        
        #ypred = sforw(sentf, x, state, predict=(y!=nothing), dropout=true, loc=loc)
        ypred = sforw(sentf, x, state, predict=(y!=nothing), dropout=true)
        y==nothing && continue
        y = loc ? locs[i-1] : y
        
        if i <= 6
            println("y: $(map(x-> indmax(y[:,x]), 1:size(y,2)))")
            ypredh = to_host(ypred)
            println("ypredh: $(map(x-> indmax(ypredh[:,x]), 1:size(ypredh,2)))")
        end

        #errloss = loc ? quadloss : zeroone
        errloss = zeroone
        sumloss += errloss(ypred, y)*size(y,2)
        numloss += size(y,2)
        (_, worldgrad) = sback(sentf, y, loss; mask=mask, getdx=true)
        while !stack_isempty(sentf); (_, wgrad) = sback(sentf; getdx=true); axpy!(1, wgrad, worldgrad); end
        update!(sentf; gclip=gclip)
        
        back(worldf, worldgrad)
        update!(worldf; gclip=gclip)
                        
        reset!(sentf)
        reset!(worldf)
        getstate = true
    end
    sumloss / numloss
end

function test(sentf, worldf, data, worlddata, locs, loss, loc=false)
    sumloss = numloss = 0
    reset!(sentf)
    reset!(worldf)
    i = 1
    getstate = true
    state = nothing
    for (x,y,mask) in data
        if getstate
            world = worlddata[i]
            i += 1
            getstate = false
            state = forw(worldf, world) 
        end
        #ypred = forw(sentf, x, state, predict=(y!=nothing), loc=loc)
        ypred = forw(sentf, x, state, predict=(y!=nothing))
        y==nothing && continue
        y = loc ? locs[i-1] : y
        sumloss += loss(ypred, y)*size(y,2)
        numloss += size(y,2)
        getstate = true
        reset!(sentf)
        reset!(worldf)
    end
    sumloss / numloss
end

function predict(sentf, worldf, data, worlddata; loc=false, xrange=1:79, padding=1, xvocab=326, ftype=Float32, xsparse=false)
    reset!(sentf)
    reset!(worldf)
    sentences = extract(data, xrange; padding=1)	# sentences[i][j] = j'th word of i'th sentence
    ypred = Any[]
    eos = xvocab + 1
    x = (xsparse ? sponehot : zeros)(ftype, eos, 1)
    state = nothing
    getstate = true
    ci = 1
    for s in sentences
        state = forw(worldf, worlddata[ci])
        ci += 1

        for i = 1:length(s)
            setrow!(x, s[i], 1)
            forw(sentf, x, state, predict=false, loc=loc)
        end
        setrow!(x, eos, 1)
        y = forw(sentf, x, state, predict=true, loc=loc)
        if loc
            push!(ypred, to_host(y))
        else
            push!(ypred, indmax(to_host(y)))
        end
        reset!(sentf)
        reset!(worldf)
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

function get_worlds(rawdata; batchsize=100, ftype=Float32)
    data = map(x -> (rawdata[x,1:end-64], rawdata[x,end-63:end]),1:size(rawdata,1));
    worlds = zeros(ftype, 18, 18, 22, size(rawdata, 1))
    indx=1
    for item in data
        world = zeros(18, 18, 22)
        world[2:17,2:17, 22] = 1#set empty
        
        #borders
        world[[1,18],:,21] = 1
        world[:,[1,18],21] = 1

        state = item[2]
        locs = map(x -> (state[x], state[x+1], state[x+2]), 1:3:60)
        #println("Loc size: $(length(locs))")
        #println(locs)
        for i=1:length(locs)
            loc = locs[i]
            if !(loc[1] == -1 && loc[2] == -1 && loc[3] == -1)
                x = round(Int, loc[1] / 0.1524) + 10
                z = round(Int, loc[3] / 0.1524) + 10
                #println((i, x, z))
                world[z, x, i] = 1
                world[z, x, 22] = 0
            end
        end
        worlds[:,:,:,indx] = world
        indx += 1
    end

    return minibatchworlds(worlds; batchsize=batchsize)
end
    
function minibatchworlds(worlds; batchsize=100)
    batches = Any[]
    for i=1:batchsize:size(worlds)[end]
        j = min(i+batchsize-1,size(worlds)[end])
        push!(batches, worlds[:,:,:,i:j])
    end
    return batches
end

function get_locs(locdata, ftype=Float32)
    data = zeros(ftype, 18*18, size(locdata, 2))

    for i=1:size(data,2)
        x = round(Int, locdata[1, i] / 0.1524) + 10
        z = round(Int, locdata[3, i] / 0.1524) + 10
        indx = (z-1)*18 + x
        data[indx,i] = 1
    end
    return data
end

function minibatchlocs(locs, batchsize=100)
    batches = Any[]
    for i=1:batchsize:size(locs)[end]
        j = min(i+batchsize-1,size(locs)[end])
        push!(batches, locs[:,i:j])
    end
    return batches
end

function main(args)
    s = ArgParseSettings()
    s.exc_handler=ArgParse.debug_handler
    @add_arg_table s begin
        ("--worlddatafiles"; nargs='+'; default=["../../BlockWorld/MNIST/SimpleActions/logos/Train.SP.data", "../../BlockWorld/MNIST/SimpleActions/logos/Dev.SP.data"])
        ("--datafiles"; nargs='+'; default=["../../BlockWorld/MNIST/SimpleActions/logos/Train.STRP.data", "../../BlockWorld/MNIST/SimpleActions/logos/Dev.STRP.data"])
        ("--loadfile"; help="initialize model from file")
        ("--savefile"; help="save final model to file")
        ("--bestfile"; help="save best model to file")
        ("--hidden"; arg_type=Int; default=100; help="hidden layer size")
        ("--chidden"; arg_type=Int; default=100; help="hidden layer size")
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
    
    global rawdata = map(f->readdlm(f,Int), o[:datafiles])
    
    # Minibatch data: data[1]:train, data[2]:dev
    xrange = 1:o[:nx]
    yrange = (o[:nx] + o[:target]):(o[:nx] + o[:target])
    yvocab = o[:yvocab][o[:target]]
    
    global data = map(rawdata) do d
        minibatch(d, xrange, yrange, o[:batchsize]; xvocab=o[:xvocab], yvocab=yvocab, ftype=o[:ftype], xsparse=o[:xsparse])
    end
    
    global rawworlddata = map(f->readdlm(f,Float32), o[:worlddatafiles])
    global worlddata = map(rawworlddata) do d
        get_worlds(d, batchsize=o[:batchsize])
    end

    rawlocs = Any[]
    push!(rawlocs, get_locs(rawworlddata[1][:,(end-2):end]'))
    push!(rawlocs, get_locs(rawworlddata[2][:,(end-2):end]'))

    global locs = map(rawlocs) do dat
        minibatchlocs(dat, o[:batchsize])
    end
    
    output = o[:loc] ? 18*18 : yvocab
    # Load or create the model:
    global sentf = (o[:loadfile]!=nothing ? load(o[:loadfile], "net") :
        compile(:rnnmodel; hidden=o[:hidden], output=output, pdrop=o[:dropout], fpdrop=o[:fdropout], lpdrop=o[:ldropout], nlayers=o[:nlayers]))
    
    global worldf = (o[:loadfile]!=nothing ? load(o[:loadfile], "net") :
                  compile(:cnn; hidden=100, cwin1=o[:cwin], cout1=o[:cout], cwin2=o[:cwin], cout2=o[:cout]))
    
    if o[:predict] && o[:loadfile] != nothing
        @date devpred = predict(net, combined[2]; xrange=xrange, xvocab=o[:xvocab], ftype=o[:ftype], xsparse=o[:xsparse])
        return
    end

    for net in (sentf, worldf); setp(net, adam=true); setp(net, lr=o[:lr]); end

    lasterr = besterr = 1.0
    anger = 0
    stopcriterion = false
    df = DataFrame(epoch = Int[], lr = Float64[], trn_err = Float64[], dev_err = Float64[], best_err = Float64[])

    #=========================
    loss1 = o[:loc] ? quadloss : softloss
    loss2 = o[:loc] ? quadloss : zeroone
    ===================#

    loss1 = softloss
    loss2 = zeroone

    for epoch=1:o[:epochs]      # TODO: experiment with pretraining
        @date trnerr = train(sentf, worldf, data[1], worlddata[1], locs[1], loss1, o[:loc]; gclip=o[:gclip])
        @date deverr = test(sentf, worldf, data[2], worlddata[2], locs[2], loss2, o[:loc])
	
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
    #o[:logfile]!=nothing && writetable(o[:logfile], df)
end
!isinteractive() && main(ARGS)
