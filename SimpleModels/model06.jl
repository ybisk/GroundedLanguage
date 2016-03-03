# model06: This is the additive model.  In models01-05, the two inputs
# w and x get multiplied (y=WPx).  A more basic two input model might
# instead treat these as additive factors (y=f(Aw+Bx)).  Here we will
# try to see if this works.

# So far could not find a way to train...

using Knet

# B: number of blocks, N: number of names, D: number of dimensions
# Input w: world (coor) (D,B)
# Input x: name (one-hot id) (N,1)
# Output y: location (coor) (D,1)
# Here we take name=block and dims=2.

@knet function model06(w, x; nblocks=20, ndims=2, hidden=100, winit=Gaussian(0,0.1), binit=Constant(0))
    pw = par(init=winit, dims=(hidden, nblocks*ndims))
    px = par(init=winit, dims=(hidden, nblocks))
    pb = par(init=binit, dims=(hidden, 0))
    ph = par(init=winit, dims=(ndims, hidden))
    h = relu(pw * w + px * x + pb)
    return ph * h
end

function train06(; N=2^20, dims=(16,16), nblocks=20, lr=0.0005, winit=Gaussian(0,0.1), binit=Constant(0), hidden=100)
    global f = compile(:model06, nblocks=nblocks, ndims=length(dims), hidden=hidden, winit=winit, binit=binit)
    setp(f, lr=lr, adam=true)
    loss = 0
    nextn = 1
    ncells = prod(dims)
    global world = zeros(length(dims), nblocks)
    global x = zeros(nblocks, 1)
    for n=1:N
        locations = randperm(ncells)[1:nblocks]
        for b=1:nblocks
            world[:,b] = [ind2sub(dims, locations[b])...]
        end
        block = rand(1:nblocks)
        global ygold = world[:,block]
        x[:]=0; x[block] = 1
        global w = vec(world)
        global ypred = forw(f, w, x)
        qloss = quadloss(ypred,ygold)
        loss = (n==1 ? qloss : 0.99 * loss + 0.01 * qloss)
        n==nextn && (println((n,loss)); nextn*=2)
        back(f, ygold, quadloss)
        update!(f)
    end
end

train06()
