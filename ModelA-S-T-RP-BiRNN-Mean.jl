using Knet
using ArgParse
using JLD
using CUDArt
using Knet: stack_isempty
using DataFrames
using Base.LinAlg: axpy!, scale!
#gpu(false)

# RNN model:

@knet function droplstm(x; hidden=100, pdrop=0.5)
    d = drop(x, pdrop=pdrop)
    return lstm(d, out=hidden)
end

@knet function rnnmodel(word; hidden=100, embedding=hidden, pdrop=0.5, nlayers=2)
    wvec = wdot(word; out=embedding)                 # TODO: try different embedding dimension than hidden
    hd = drop(wvec, pdrop=0.5)
    lh = lstm(hd, out=hidden)
    h = repeat(lh; frepeat=:droplstm, nrepeat=nlayers-1, hidden=hidden, pdrop=pdrop)
end

@knet function outlayer(x, y; output=20, pdrop=0.5)
    fdrop = drop(x, pdrop=pdrop)
    bdrop = drop(y, pdrop=pdrop)
    pred = wbf2(fdrop, bdrop; f=:soft, out=output)
    return pred
end

# Train and test scripts expect data to be an array of (s,p) pairs
# where s is an integer array of word indices
# and p is an integer class to be predicted

function train(forwnet, backnet, prednet, data, loss, hidden; xvocab=326, yvocab=20, gclip=0)
    for f in (forwnet,backnet); reset!(f); end
    x = zeros(Float32, xvocab, 1)
    y = zeros(Float32, yvocab, 1)
    hforw = CudaArray(Float32, hidden, 1)
    hback = CudaArray(Float32, hidden, 1)
    err = 0
    for (s,p) in data
        fill!(hforw,0)
        for i in 1:length(s)
            x[s[i]] = 1
            sforw(forwnet, x; dropout=true)
            hidden = Knet.reg(forwnet, :h).out # Assuming the register name is "hidden"
            axpy!(1, hidden, hforw)
            x[s[i]] = 0
        end
        scale!(1/length(s), hforw)
        fill!(hback,0)
        for i in length(s):-1:1
            x[s[i]] = 1
            sforw(backnet, x; dropout=true)
            hidden = Knet.reg(backnet, :h).out
            axpy!(1, hidden, hback)
            x[s[i]] = 0
        end
        scale!(1/length(s), hback)
        ypred = forw(prednet, hforw, hback; dropout=true)
        indmax(to_host(ypred))!=p && (err += 1)
        y[p] = 1
        (fgrad, bgrad) = back(prednet, y, loss; getdx=true)
        update!(prednet)
        y[p] = 0
        for g in (fgrad, bgrad); scale!(1/length(s), g); end
        sback(forwnet, fgrad); while !stack_isempty(forwnet); sback(forwnet); end
        sback(backnet, bgrad); while !stack_isempty(backnet); sback(backnet); end
        for f in (forwnet, backnet); update!(f; gclip=gclip); reset!(f); end
    end
    err / length(data)
end

function test(forwnet, backnet, prednet, data, loss, hidden; xvocab=326, yvocab=20)
    for f in (forwnet,backnet); reset!(f); end
    x = zeros(Float32, xvocab, 1)
    y = zeros(Float32, yvocab, 1)
    hforw = CudaArray(Float32, hidden, 1)
    hback = CudaArray(Float32, hidden, 1)
    
    sumloss = numloss = 0
    for (s,p) in data
        for i in 1:length(s)
            x[s[i]] = 1
            forw(forwnet, x; dropout=true)
            hidden = Knet.reg(forwnet, :h).out # Assuming the register name is "hidden"
            axpy!(1, hidden, hforw)
            x[s[i]] = 0
        end
        scale!(1/length(s), hforw)
        fill!(hback,0)
        for i in length(s):-1:1
            x[s[i]] = 1
            forw(backnet, x; dropout=true)
            hidden = Knet.reg(backnet, :h).out
            axpy!(1, hidden, hback)
            x[s[i]] = 0
        end
        scale!(1/length(s), hback)
        ypred = forw(prednet, hforw, hback; dropout=true)
        y[p] = 1; sumloss += loss(ypred, y); y[p] = 0
        numloss += 1
        for f in (forwnet,backnet); reset!(f); end
    end
    sumloss / numloss
end

function pred(forwnet, backnet, prednet, data, hidden; xvocab=326, yvocab=20)
    for f in (forwnet,backnet); reset!(f); end
    x = zeros(Float32, xvocab, 1)
    y = zeros(Float32, yvocab, 1)
    hforw = CudaArray(Float32, hidden, 1)
    hback = CudaArray(Float32, hidden, 1)
    preds = Any[]
    for (s,p) in data
        for i in 1:length(s)
            x[s[i]] = 1
            forw(forwnet, x; dropout=true)
            hidden = Knet.reg(forwnet, :h).out # Assuming the register name is "hidden"
            axpy!(1, hidden, hforw)
            x[s[i]] = 0
        end
        scale!(1/length(s), hforw)
        fill!(hback,0)
        for i in length(s):-1:1
            x[s[i]] = 1
            forw(backnet, x; dropout=true)
            hidden = Knet.reg(backnet, :h).out
            axpy!(1, hidden, hback)
            x[s[i]] = 0
        end
        scale!(1/length(s), hback)
        ypred = forw(prednet, hforw, hback; dropout=true)
        push!(preds, indmax(to_host(ypred)))
        for f in (forwnet,backnet); reset!(f); end
    end
    println(preds)
end


# Extract sentences, get rid of padding
function data2sent(data, nx)
    sent = Any[]
    for i=1:size(data,1)
        s = vec(data[i,1:nx])
        while s[end]==1; pop!(s); end
        push!(sent,s)
    end
    return sent
end

function main(args)
    s = ArgParseSettings()
    s.exc_handler=ArgParse.debug_handler
    @add_arg_table s begin
        ("--datafiles"; nargs='+'; default=["BlockWorld/logos/Train.STRP.data", "BlockWorld/logos/Dev.STRP.data"])
        ("--loadfile"; help="initialize model from file")
        ("--savefile"; help="save final model to file")
        ("--bestfile"; help="save best model to file")
        ("--target"; arg_type=Int; default=1; help="1:source,2:target,3:direction")
        ("--hidden"; arg_type=Int; default=100; help="hidden layer size")
        ("--epochs"; arg_type=Int; default=100; help="number of epochs to train")
        ("--lr"; arg_type=Float64; default=1.0; help="learning rate")
        ("--gclip"; arg_type=Float64; default=5.0; help="gradient clipping threshold")
        ("--dropout"; arg_type=Float64; default=0.5; help="dropout probability")
        ("--decay"; arg_type=Float64; default=0.9; help="learning rate decay if deverr increases")
        ("--gpu"; action = :store_true; help="use gpu, which is not used by default")
        ("--seed"; arg_type=Int; default=42; help="random number seed")
        ("--logfile")
    end
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true); println(o)
    o[:seed] > 0 && setseed(o[:seed])
    Knet.gpu(o[:gpu])

    # TODO: These are a bit hard-coded right now, probably should infer from data or make options.
    (nx, ny, xvocab, tvocab) = (79, 3, 326, [20, 20, 8])

    # Read data files: 6003x82, 855x82
    rawdata = map(f->readdlm(f,Int), o[:datafiles])

    # Extract sentences, get rid of padding
    sentences = map(d->data2sent(d,nx), rawdata)

    # Construct data sets
    # data[1]: source, data[2]: target, data[3]: direction
    # data[i][1]: train, data[i][2]: dev
    # data[i][j][k]: (sentence, answer) pair
    global data = Any[]
    for i = 1:3
        answers = map(d->vec(d[:,nx+i]), rawdata)
        d = map(zip, sentences, answers)
        push!(data, d)
    end

    # Train model for a dataset:
    epochs = 10
    yvocab = tvocab[o[:target]]
    mydata = data[o[:target]]
    #global net = (o[:loadfile]!=nothing ? load(o[:loadfile], "net") :
    #              compile(:birnnmodel; hidden=o[:hidden], output=yvocab, pdrop=o[:dropout]))

    global fnet = compile(:rnnmodel; hidden=o[:hidden], pdrop=o[:dropout])
    global bnet = compile(:rnnmodel; hidden=o[:hidden], pdrop=o[:dropout])
    global prednet = compile(:outlayer; output=yvocab, pdrop=o[:dropout])
    for net in [fnet, bnet, prednet]; setp(net, lr=o[:lr]); setp(net, adam=true); end
    lasterr = besterr = 1.0
    df = DataFrame(epoch = Int[], lr = Float64[], trn_err = Float64[], dev_err = Float64[], best_err = Float64[])
    for epoch=1:o[:epochs]      # TODO: experiment with pretraining
        @date trnerr = train(fnet, bnet, prednet, mydata[1], softloss, o[:hidden]; xvocab=xvocab, yvocab=yvocab, gclip=o[:gclip])
        @date deverr =  test(fnet, bnet, prednet, mydata[2], zeroone, o[:hidden];  xvocab=xvocab, yvocab=yvocab)
        if deverr < besterr
            besterr=deverr
            o[:bestfile]!=nothing && save(o[:bestfile], "net", clean(net))
        end
        push!(df, (epoch, o[:lr], trnerr, deverr, besterr))
        println(df[epoch, :])

        #===
        if deverr > lasterr
            o[:lr] *= o[:decay]
            setp(net, lr=o[:lr])
        end
        ===#
    end
    o[:savefile]!=nothing && save(o[:savefile], "net", clean(net))
    pred(fnet, bnet, prednet, mydata[2], o[:hidden])
    o[:logfile]!=nothing && writetable(o[:logfile], df)
end

!isinteractive() && main(ARGS)

# TODO: write minibatching versions, this is too slow, even slower on gpu, so run with gpu(false)
# For minibatching:
# Sentences will have to be padded in the beginning, so they end together.
# They need to be sorted by length.
# Each minibatch will be accompanied with a mask.
# Possible minibatch sizes to get the whole data:
# 6003 = 9 x 23 x 29
# 855 = 9 x 5 x 19

