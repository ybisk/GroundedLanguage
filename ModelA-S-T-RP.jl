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
traindir = "BlockWorld/logos/Train.STRP.data"
data     = readdlm(traindir);
V        = maximum(data[:,1:end-3])
indim    = Int(size(data[:,1:end-3],2)*V)
outdim   = 20
RPoutdim = 8

X  = sparsify(data[:,1:end-3], V);
S  = sparsify(data[:,end-2], outdim);
T  = sparsify(data[:,end-1], outdim);
RP = sparsify(data[:,end], RPoutdim);

testdir = "BlockWorld/logos/Dev.STRP.data"
test_data = readdlm(testdir);
X_t  = sparsify(test_data[:,1:end-3], V);
S_t  = sparsify(test_data[:,end-2], outdim);
T_t  = sparsify(test_data[:,end-1], outdim);
RP_t = sparsify(test_data[:,end], RPoutdim);


function train(f, data, loss)
    for (x,y) in data
        forw(f, x)
        back(f, y, loss)
        update!(f)
    end
end

function trainloop(net, epochs, lrate, decay, X, Y, X_t, Y_t)
  setp(net; lr=lrate)
  trn = minibatch(X,Y,batchsize)
  tst = minibatch(X_t,Y_t,batchsize)
  for epoch=1:epochs
      train(net, trn, softloss)
      trnerr = test(net, trn, zeroone)
      tsterr = test(net, tst, zeroone)

      println((epoch, lrate, trnerr, tsterr))
      if tsterr > lasterr
          lrate = decay*lrate
          setp(net; lr=lrate)
      end
      lasterr = tsterr
  end
end


function test(f, data, loss)
    sumloss = numloss = 0
    for (x,ygold) in data
        ypred = forw(f, x)
        sumloss += loss(ypred, ygold)
        numloss += 1
    end
    sumloss / numloss
end

# indmax(ypred[:,i])
function predict(f, data)
  P = Int[]
  for i = 1:size(data,2)
    push!(P,indmax(to_host(forw(f,data[:,i]))))
  end
  println(P)
end

@knet function SM(x; dropout=0.5, outdim=20)
  h = wbf(x; out=100, f=:tanh)
  hdrop = drop(h, pdrop=dropout)      ## Prob of dropping
  return wbf(hdrop; out=outdim, f=:soft)
end

function main(args=ARGS)
  s = ArgParseSettings()
    @add_arg_table s begin
        ("--lrate"; arg_type=Float64; default=0.1)
        ("--decay"; arg_type=Float64; default=0.9)
        ("--dropout"; arg_type=Float64; default=0.5)
        ("--seed"; arg_type=Int; default=20160113)
        ("--epochs"; arg_type=Int; default=10)
  end
  isa(args, AbstractString) && (args=split(args))
  o = parse_args(args, s; as_symbols=true); println(o)
  o[:seed] > 0 && setseed(o[:seed]) 

  lrate = o[:lrate]
  decay = o[:decay]
  dropout = o[:dropout]
  epochs = o[:epochs]
  batchsize=100;
  lasterr = 1.0;

  ### Train Source ###
  Snet = compile(:SM, dropout=dropout, outdim=outdim)
  Tnet = compile(:SM, dropout=dropout, outdim=outdim)
  RPnet = compile(:SM, dropout=dropout, outdim=RPoutdim)

  trainloop(Snet, epochs, lrate, decay, X, S, X_t, S_t)
  trainloop(Tnet, epochs, lrate, decay, X, T, X_t, T_t)
  trainloop(RPnet, epochs, lrate, decay, X, RP, X_t, RP_t)

  # Get Predictions
  predict(net, X_t)

  # Save net and parameterize
  JLD.save("ModelA-$lrate-$decay-$dropout-$epochs.jld", "model", clean(net));
  # net = JLD.load("ModelA.jld", "model")
end

main(ARGS)
