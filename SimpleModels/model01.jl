using Knet

# model01: world state is represented by a DxB array: D dimensional
# coordinates, for B blocks.  The goal is given a world and a one-hot
# indicator (name) for a block to return the coordinates of the block.
# The model is y=WPx where W is the world, x is the block indicator, P
# is a permutation matrix and y are the predicted coordinates.

# Input w: world (coor) (2,n)
# Input x: block (one-hot id) (n,1)
# Output y: block (coor) (2,1)

@knet function model01(w, x; nblocks=20, init=Constant(0))
    p = par(init=init, dims=(nblocks,nblocks))
    z = p * x
    return w * z
end


# train01: The coordinates are integers in [1,max] where max is
# specified by the dims argument.  The model is y=WPx and the order of
# names and columns of the world matrix are identical so P=identity
# solves this perfectly.

# Training diverges for lr > 0.0006 starting from P=0.
# best err <0.01 in 8192 examples with lr=0.0005, P=0, no minibatch.

function train01(; model=:model01, nblocks=20, dims=(16,16), init=Constant(0), lr=0.0005, N=2^14)
    global f = compile(model, nblocks=nblocks, init=init)
    setp(f, lr=lr)
    loss = 0
    nextn = 1
    ncells = prod(dims)
    global c = zeros(length(dims), nblocks)
    for n=1:N
        ind = randperm(ncells)[1:nblocks]
        for b=1:nblocks
            c[:,b] = [ind2sub(dims, ind[b])...]
        end
        global x = zeros(nblocks, 1); x[rand(1:nblocks)] = 1
        global ygold = c*x
        global ypred = forw(f, c, x)
        qloss = quadloss(ypred,ygold)
        loss = (n==1 ? qloss : 0.99 * loss + 0.01 * qloss)
        n==nextn && (println((n,loss)); nextn*=2)
        back(f, ygold, quadloss)
        update!(f)
    end
end

train01()
