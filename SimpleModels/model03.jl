# model03: transform coordinates by adding a n+1 dimension and
# normalize so dot products work better.  lr=0.1 works best.  extra
# dimension = 3*maxdim works best.  Get <0.01 err in 2048 examples
# which is between model01 and model02.

using Knet

# Input w: world (coor) (2,n)
# Input x: block (one-hot id) (n,1)
# Output y: block (coor) (2,1)

@knet function model03(w, x; nblocks=20, init=Gaussian(0,0.1))
    p = par(init=init, dims=(nblocks,nblocks))
    z = p * x
    return w * z
end

function train03(; N=2^16, dims=(16,16), nblocks=20, lr=0.1, init=Constant(0), scale=3)
    global f = compile(:model03, nblocks=nblocks, init=init)
    setp(f, lr=lr)
    loss = 0
    nextn = 1
    ncells = prod(dims)
    global c = zeros(length(dims)+1, nblocks)
    for n=1:N
        ind = randperm(ncells)[1:nblocks]
        for b=1:nblocks
            c[1:end-1,b] = [ind2sub(dims, ind[b])...]
        end
        # add an n+1 dimension and normalize
        c[end,:] = scale*maximum(dims)
        c = c./sqrt(sum(c.^2,1))
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

train03()
