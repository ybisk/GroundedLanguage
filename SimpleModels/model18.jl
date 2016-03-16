# model18: Given world and block id, return position of block.
# Trying to solve this with a regular single hidden layer MLP.
# Fastest convering net has 2000 hidden units.
# It takes about 10M examples for the error to fall below one block size.

using Knet

@knet function model18(w, x; ndims=2, nblocks=20, winit=Gaussian(0,0.1), hidden=100*nblocks, f1=:relu)
    h1 = wbf2(w, x; out=hidden, f=f1, winit=winit)
    h = wbf(h1; out=hidden, f=f1, winit=winit)
    return wb(h; out=ndims)
end

function train18(; N=2^16, dims=(16,16), ndims=length(dims), nblocks=20, nbatch=128,
                 lr=0.0001, adam=true, winit=Gaussian(0,0.01), o...)
    global f = compile(:model18; ndims=ndims, nblocks=nblocks, winit=winit, o...)
    setp(f, lr=lr, adam=adam)
    qloss = 0
    nextn = 1
    ncells = prod(dims)
    global world = zeros(ndims, nblocks)
    global block = zeros(nblocks)
    global ygold = zeros(ndims)
    global world_ = zeros(length(world), nbatch)
    global block_ = zeros(length(block), nbatch)
    global ygold_ = zeros(length(ygold), nbatch)
    for n=1:N
        for m=1:nbatch
            locations = randperm(ncells)
            for b=1:nblocks
                world[:,b] = [ind2sub(dims, locations[b])...] # place blocks at random locations
            end
            iblock = rand(1:nblocks)
            block[:] = 0; block[iblock] = 1
            ygold[:] = world[:,iblock]
            # fill the minibatch matrices
            world_[:,m] = vec(world)
            block_[:,m] = block
            ygold_[:,m] = ygold
        end
        global ypred_ = forw(f, world_, block_)
        ql = quadloss(ypred_,ygold_); qloss = (n==1 ? ql : 0.99 * qloss + 0.01 * ql)
        n==nextn && (println((n,qloss)); nextn*=2)
        back(f, ygold_, quadloss)
        update!(f)
    end
end

train18()

