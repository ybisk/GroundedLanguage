using Knet
using Requests

# turns each row of data into a column of a one-hot sparse matrix
# data should have a one based index for each attribute
function sparsify(data; colmax = maximum(data,1))
    offset = hcat(0,cumsum(colmax,2))
    I = Int[]
    J = Int[]
    V = Float32[]
    for inst=1:size(data,1)
        for attr=1:size(data,2)
            push!(V,1)
            push!(J,inst)
            push!(I, data[inst,attr] + offset[attr])
        end
    end
    sparse(I,J,V)
end


traindir = "BlockWorld/logos/Train.STRP.data"
data = readdlm(traindir);
x  = sparsify(data[:,1:end-3]);
S  = sparsify(data[:,end-2]);
T  = sparsify(data[:,end-1]);
RP = sparsify(data[:,end]);
inputdim = size(x,1)
outputdim = size(S,1)

testdir = "BlockWorld/logos/Dev.STRP.data"
test_data = readdlm(testdir);
x_t  = sparsify(test_data[:,1:end-3], colmax=maximum(data[:,1:end-3],1));
S_t  = sparsify(test_data[:,end-2], colmax=maximum(data[:,end-2]));
T_t  = sparsify(test_data[:,end-1], colmax=maximum(data[:,end-1]));
RP_t = sparsify(test_data[:,end], colmax=maximum(data[:,end]));

function train(f, data, loss)
    for (x,y) in data
        forw(f, x)
        back(f, y, loss)
        update!(f)
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

@knet function SM(x)
  w = par(init=Gaussian(0,0.1), dims=(outputdim,inputdim))
  b = par(init=Constant(0), dims=(outputdim,1))
  return soft(w * x .+ b)
end

batchsize=100;
lrate = 0.1;
decay = 0.9;
lasterr = 1.0;

net = compile(:SM)
setp(net; lr=lrate)

trn = minibatch(x,S,batchsize)
tst = minibatch(x_t,S_t,batchsize)
for epoch=1:100

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
