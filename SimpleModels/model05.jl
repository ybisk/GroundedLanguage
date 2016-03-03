# model05: This is the same as model04, except now we introduce
# multiple names for the same object, i.e. there are synonyms.

# Same hyperparameters, 50 names and 20 blocks, takes about twice as
# many examples to train.

using Knet

# Input w: world (coor) (2,n)
# Input x: name (one-hot id) (n,1)
# Output y: location (coor) (2,1)

@knet function model05(w, x; nblocks=20, nnames=50, init=Gaussian(0,0.1))
    p = par(init=init, dims=(nblocks,nnames))
    z = p * x
    return w * z
end

function train05(; N=2^15, dims=(16,16), nblocks=20, nnames=50, lr=0.0005, init=Constant(0))
    global f = compile(:model05, nblocks=nblocks, nnames=nnames, init=init)
    setp(f, lr=lr)
    loss = 0
    nextn = 1
    ncells = prod(dims)
    global world = zeros(length(dims), nblocks)
    global name2block = rand(1:nblocks, nnames)
    for n=1:N
        locations = randperm(ncells)[1:nblocks]
        for b=1:nblocks
            world[:,b] = [ind2sub(dims, locations[b])...]
        end
        name = rand(1:nnames)
        block = name2block[name]
        global ygold = world[:,block]
        global x = zeros(nnames, 1); x[name] = 1
        global ypred = forw(f, world, x)
        qloss = quadloss(ypred,ygold)
        loss = (n==1 ? qloss : 0.99 * loss + 0.01 * qloss)
        n==nextn && (println((n,loss)); nextn*=2)
        back(f, ygold, quadloss)
        update!(f)
    end
end

train05()
