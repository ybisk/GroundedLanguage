# model04: This is the same as model01, except now we are shuffling
# the vocabulary.  So the solution to y=WPx is not the identity matrix
# but some permutation matrix.  As expected, this has no impact on the
# performance, the model learns the right P with the same
# hyperparameters and same number of examples.

using Knet

# Input w: world (coor) (2,n)
# Input x: block (one-hot id) (n,1)
# Output y: block (coor) (2,1)

@knet function model04(w, x; nblocks=20, init=Gaussian(0,0.1))
    p = par(init=init, dims=(nblocks,nblocks))
    z = p * x
    return w * z
end

function train04(; N=2^14, dims=(16,16), nblocks=20, lr=0.0005, init=Constant(0))
    global f = compile(:model04, nblocks=nblocks, init=init)
    setp(f, lr=lr)
    loss = 0
    nextn = 1
    ncells = prod(dims)
    global world = zeros(length(dims), nblocks)
    global names = randperm(nblocks)
    for n=1:N
        locations = randperm(ncells)[1:nblocks]
        for b=1:nblocks
            world[:,b] = [ind2sub(dims, locations[b])...]
        end
        block = rand(1:nblocks)
        name = names[block]
        global ygold = world[:,block]
        global x = zeros(nblocks, 1); x[name] = 1
        global ypred = forw(f, world, x)
        qloss = quadloss(ypred,ygold)
        loss = (n==1 ? qloss : 0.99 * loss + 0.01 * qloss)
        n==nextn && (println((n,loss)); nextn*=2)
        back(f, ygold, quadloss)
        update!(f)
    end
end

train04()
