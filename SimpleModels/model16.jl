# model16: Finding a reference block near the target position given before and after states.
# Same as model13 except target position replaced with after state
# Input: before and after world coor (DxN,DxN)
# Output: softmax over block ids (y:Nx1)

# I could not make this simple model work.  Converges to baseline
# random selection:
# (4096,2.995825209809718,0.05087524672196486)

# If we can find the reference block with before + targetpos, but we
# cannot with before + after, maybe the problem is with getting the
# targetpos from before + after.  This may require a multiplicative
# step.

using Knet

@knet function model16(w, x; ndims=2, nblocks=20, winit=Gaussian(0,0.05), hidden=20*nblocks)
    h1 = wbf2(w, x; out=hidden, f=:relu, winit=winit)
    h2 = wbf(h1; out=hidden, f=:relu, winit=winit)
    return wbf(h2; out=nblocks, f=:soft, winit=winit)
end

function train16(; N=2^16, dims=(16,16), ndims=length(dims), nblocks=20, lr=0.001, adam=true, nbatch=128, winit=Gaussian(0,0.1))
    global f = compile(:model16; ndims=ndims, nblocks=nblocks, winit=winit)
    setp(f, lr=lr, adam=adam)
    sloss = zloss = 0
    nextn = 1
    ncells = prod(dims)
    global world1 = zeros(ndims, nblocks)
    global world2 = zeros(ndims, nblocks)
    global target = zeros(ndims)
    global ygold = zeros(nblocks)
    global world1_ = zeros(length(world1), nbatch)
    global world2_ = zeros(length(world2), nbatch)
    global ygold_ = zeros(length(ygold), nbatch)
    for n=1:N
        for m=1:nbatch
            locations = randperm(ncells)
            for b=1:nblocks
                world1[:,b] = world2[:,b] = [ind2sub(dims, locations[b])...] # place blocks at random locations
            end
            target[:] = [ind2sub(dims, locations[nblocks+1])...] # pick target at an empty location
            mblock = rand(1:nblocks)                             # move a random block there
            world2[:,mblock] = target
            d1 = world1 .- target; d2 = sum(d1 .* d1, 1)
            rblock = rand(find(d2 .== minimum(d2))) # pick a reference block close to target, note that this could pick mblock!
            ygold[:] = 0; ygold[rblock] = 1
            # fill the minibatch matrices
            world1_[:,m] = vec(world1)
            world2_[:,m] = vec(world2)
            ygold_[:,m] = ygold
        end
        global ypred = forw(f, world1_, world2_)
        sl = softloss(ypred,ygold_); sloss = (n==1 ? sl : 0.99 * sloss + 0.01 * sl)
        zl = zeroone(ypred,ygold_);  zloss = (n==1 ? zl : 0.99 * zloss + 0.01 * zl)
        n==nextn && (println((n,sloss,1-zloss)); nextn*=2)
        back(f, ygold_, softloss)
        update!(f)
    end
end

train16()
