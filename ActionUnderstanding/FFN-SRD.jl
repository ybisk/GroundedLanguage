# Feed forward neural network which predicts Source, Target, and Relative Position independantly
# Input File: JSONReader/data/2016-NAACL/STD/*.mat
# Input:  Utterance as sparse Array
# Output:  Source, Target, or Relative Position.  Each prediction is an indep model.

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
# Cut off words from long sentences > 80
longest_sentence = 80
#traindir = "JSONReader/data/2016-NAACL/SRD_Blank/Train.mat"
traindir = "JSONReader/data/2016-NAACL/SRD/Train.mat"
data     = readdlm(traindir)[:,1:83];
data[data.==""]=1
V        = maximum(data[:,4:end])
outdim   = 20
Doutdim  = 9

X  = sparsify(data[:,4:end], V);
S  = sparsify(data[:,1] + 1, outdim);
R  = sparsify(data[:,2] + 1, outdim);
D = sparsify(data[:,3] + 1, Doutdim);

#testdir = "JSONReader/data/2016-NAACL/SRD_Blank/Test.mat"
testdir = "JSONReader/data/2016-NAACL/SRD/Dev.mat"
test_data = readdlm(testdir);
test_data[test_data.==""]=1
if size(test_data,2) < 83
  test_data = hcat(test_data, ones(Int,size(test_data,1),83 - size(test_data,2)))
end
test_data = test_data[:,1:83]
V        = maximum(data[:,4:end])
X_t  = sparsify(test_data[:,4:end], V);
S_t  = sparsify(test_data[:,1] + 1, outdim);
R_t  = sparsify(test_data[:,2] + 1, outdim);
D_t = sparsify(test_data[:,3] + 1, Doutdim);


function train(f, data, loss)
    for (x,y) in data
        forw(f, x)
        back(f, y, loss)
        update!(f)
    end
end

function trainloop(net, epochs, lrate, X, Y, X_t, Y_t)
  batchsize=100;
  lasterr = 1.0;

  setp(net; lr=0.001, adam=true)
  trn = minibatch(X,Y,batchsize)
  tst = minibatch(X_t,Y_t,batchsize)
  for epoch=1:epochs
      train(net, trn, softloss)
      trnerr = test(net, trn, zeroone)
      tsterr = test(net, tst, zeroone)

      println((epoch, lrate, trnerr, tsterr))
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

@knet function SM(x; dropout=0.5, outdim=20, hidden=128)
  h = wbf(x; out=hidden, f=:relu)
  hdrop = drop(h, pdrop=dropout)      ## Prob of dropping
  return wbf(hdrop; out=outdim, f=:soft)
end

function main(args=ARGS)
  s = ArgParseSettings()
    @add_arg_table s begin
        ("--lrate"; arg_type=Float64; default=0.001)
        ("--dropout"; arg_type=Float64; default=0.5)
        ("--seed"; arg_type=Int; default=20160326)
        ("--epochs"; arg_type=Int; default=10)
        ("--hidden"; arg_type=Int; default=128)
        ("--task"; arg_type=Int; default=1)
  end
  isa(args, AbstractString) && (args=split(args))
  o = parse_args(args, s; as_symbols=true); println(o)
  o[:seed] > 0 && setseed(o[:seed])

  lrate = o[:lrate]
  dropout = o[:dropout]
  epochs = o[:epochs]
  hidden = o[:hidden]

  ### Train Source ###
  if o[:task] == 1
    Snet = compile(:SM, dropout=dropout, outdim=outdim, hidden=hidden)
    trainloop(Snet, epochs, lrate, X, S, X_t, S_t)
    predict(Snet, X_t)
  end
  if o[:task] == 2
    Rnet = compile(:SM, dropout=dropout, outdim=outdim, hidden=hidden)
    trainloop(Rnet, epochs, lrate, X, R, X_t, R_t)
    predict(Rnet, X_t)
  end
  if o[:task] == 3
    Dnet = compile(:SM, dropout=dropout, outdim=Doutdim, hidden=hidden)
    trainloop(Dnet, epochs, lrate, X, D, X_t, D_t)
    predict(Dnet, X_t)
  end

  # Save net and parameterize
  #JLD.save("Models/ModelA-$lrate-$dropout-$epochs.jld", "model", clean(Snet));
  # net = JLD.load("ModelA.jld", "model")
end

main(ARGS)
