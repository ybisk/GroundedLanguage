# model15: Finding the block (id) moved given before and after worlds (coor).

# h=10n .56@8192
# h=20n .95@8192
# h=40n .98@8192

using Knet

@knet function model15(w1, w2; nblocks=20, winit=Gaussian(0,0.05),
                       hidden=20*nblocks)
    h = wbf2(w1, w2; out=hidden, f=:relu, winit=winit)
    return wbf(h; out=nblocks, f=:soft, winit=winit)
end

function train15(; N=2^14, dims=(16,16), nblocks=20, ndims=length(dims), 
                 lr=0.001, adam=true, nbatch=128, winit=Gaussian(0,0.1))
    global f = compile(:model15; nblocks=nblocks, winit=winit)
    setp(f, lr=lr, adam=adam)
    sloss = zloss = 0
    nextn = 1
    ncells = prod(dims)
    global world1 = zeros(ndims, nblocks)
    global world2 = zeros(ndims, nblocks)
    global ygold = zeros(nblocks)
    global world1_ = zeros(length(world1), nbatch)
    global world2_ = zeros(length(world2), nbatch)
    global ygold_ = zeros(length(ygold), nbatch)
    for n=1:N
        for m=1:nbatch
            locations = randperm(ncells)
            for b=1:nblocks
                world1[:,b] = world2[:,b] = [ind2sub(dims, locations[b])...] # fill blocks with random locations
            end
            mblock = rand(1:nblocks)
            world2[:,mblock] = [ind2sub(dims, locations[nblocks+1])...] # move block to an empty location
            ygold[:] = 0; ygold[mblock] = 1
            # fill the minibatch matrices
            world1_[:,m] = vec(world1) 
            world2_[:,m] = vec(world2) 
            ygold_[:,m] = vec(ygold)
        end
        global ypred = forw(f, world1_, world2_)
        sl = softloss(ypred,ygold_); sloss = (n==1 ? sl : 0.99 * sloss + 0.01 * sl)
        zl = zeroone(ypred,ygold_);  zloss = (n==1 ? zl : 0.99 * zloss + 0.01 * zl)
        n==nextn && (println((n,sloss,1-zloss)); nextn*=2)
        back(f, ygold_, softloss)
        update!(f)
    end
end

train15()
