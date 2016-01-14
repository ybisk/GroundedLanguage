using Knet
using ArgParse
using JLD
using CUDArt

# turns each row of data into a column of a one-hot sparse matrix
# data should have a one based index for each attribute
function sparsify(data, offset)
    I = Int[]
    J = Int[]
    V = Float32[]
    for inst=1:size(data,1)
        for attr=1:size(data,2)
            push!(V,1)
            push!(J,inst)
            push!(I, data[inst,attr] + (attr-1)*offset)
        end
    end
    sparse(I,J,V,size(data,2)*offset,size(data,1))
end

## Load Data, compute vocabulary and matrix dimensions ##
traindir = "BlockWorld/logos/Train.SP.data"
data     = readdlm(traindir, Float32);
V        = maximum(data[:,1:end-64])
indim    = Int(size(data[:,1:end-64],2)*V)
outdim   = 20
RPoutdim = 8

## Input:  Text, World (dim 60), Source, (x,y,z) ##
X  = sparsify(data[:,1:end-64], V);
W  = data[:,end-64+1:end-4]';
S  = sparsify(data[:,end-3], outdim);
Locs = data[:,end-2:end]';

testdir = "BlockWorld/logos/Dev.SP.data"
test_data = readdlm(testdir, Float32);
X_t  = sparsify(test_data[:,1:end-64], V);
W_t  = test_data[:,end-64+1:end-4]';
S_t  = sparsify(test_data[:,end-3], outdim);
Locs_t = test_data[:,end-2:end]';


function train(f, data, loss; loc=false) # we should use softloss, not quadloss when loc=false
    for (x, w, y) in data
        # @show loc,map(summary, (x, w, y))
        forw(f, x, w; loc=loc)
        back(f, y, loss)
        update!(f)
    end
end

function test(f, data, loss; loc=false)
    sumloss = numloss = 0
    for (x, w, y) in data
        ypred = forw(f, x, w; loc=loc)
        sumloss += loss(ypred, y)
        numloss += 1
    end
    sumloss / numloss
end

function trainloop(net, epochs, lrate, decay, world, X, W, Y, X_t, W_t, Y_t)
  batchsize=100;
  lasterr = 1.0;

  setp(net; lr=lrate, loc=world)
  global trn = minibatch(X, W, Y, batchsize)
  global tst = minibatch(X_t, W, Y_t, batchsize)
  for epoch=1:epochs
      train(net, trn, (world ? quadloss : softloss); loc=world)
      trnerr = test(net, trn, (world ? quadloss : zeroone); loc=world)
      tsterr = test(net, tst, (world ? quadloss : zeroone); loc=world)

      println((epoch, lrate, trnerr, tsterr))
      if tsterr > lasterr
          lrate = decay*lrate
          setp(net; lr=lrate)
      end
      lasterr = tsterr
  end
end

function minibatch(x,w,y, batchsize)
  @show map(summary, (x,w,y)),batchsize
  data = Any[]
  for i=1:batchsize:size(x,2)-batchsize+1
    j=i+batchsize-1
    push!(data, (x[:,i:j], w[:,i:j], y[:,i:j]))
  end
  return data
end

# indmax(ypred[:,i])
function predict(f, data)
  P = Int[]
  for i = 1:size(data,2)
    push!(P,indmax(to_host(forw(f,data[:,i]))))
  end
  println(P)
end

@knet function SM_Reg(x, world; dropout=0.5, outdim=20)
    h = wbf(x; out=100, f=:relu)
    hdrop = drop(h, pdrop=dropout)      ## Prob of dropping
    if loc
        h2 = wbf2(hdrop, world; out=100, f=:relu)
        h2drop = drop(h2, prdop=dropout)
        return wb(h2drop; out=3)
    else
        return wbf(hdrop; out=outdim, f=:soft)
    end
end

function main(args=ARGS)
  s = ArgParseSettings()
    @add_arg_table s begin
        ("--lrate"; arg_type=Float64; default=0.1)
        ("--decay"; arg_type=Float64; default=0.9)
        ("--dropout"; arg_type=Float64; default=0.5)
        ("--seed"; arg_type=Int; default=20160113)
        ("--epochs"; arg_type=Int; default=3)
  end
  isa(args, AbstractString) && (args=split(args))
  o = parse_args(args, s; as_symbols=true); println(o)
  o[:seed] > 0 && setseed(o[:seed]) 

  lrate = o[:lrate]
  decay = o[:decay]
  dropout = o[:dropout]
  epochs = o[:epochs]

  ### Train Source ###
  global net = compile(:SM_Reg, dropout=dropout, outdim=outdim)

  trainloop(net, epochs, lrate, decay, false, X, W, S, X_t, W_t, S_t)
  trainloop(net, epochs, lrate, decay, true, X, W, Locs, X_t, W_t, Locs_t)

  # Get Predictions
  predict(Snet, X_t)
  predict(Tnet, X_t)
  predict(RPnet, X_t)

  # Save net and parameterize
  JLD.save("Models/ModelA-$lrate-$decay-$dropout-$epochs.jld", "model", clean(net));
  # net = JLD.load("ModelA.jld", "model")
end

main(ARGS)
