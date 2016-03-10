# model14: Finding a nearby block and relative direction given a target position.
# Input: world coor (w:DxN), target coor (x:Dx1)
# Output: softmax over block ids x directions (y:R*Nx1)
# Linear input Aw+Bx can calculate signed distances.
# Relative direction can be computed from sign function of distances.
# One hidden layer can calculate absolute distances.
# It needs >=2N hidden units to do so.
# We can flatten w and y to flat vectors.

# Baseline NLL=-log(1/160)=5.075 Acc=1/160=0.00625
# h=20n .25@16K 
# h=40n .47@16K
# h=80n .66@16K 
# h=160n .67@16K .76@32K 
# h=5n+5n .26@16K
# h=10n+10n .30@16K
# h=20n+20n .30@16K
# h=40n+40n .24@16K

# We should be doing weight sharing!

using Knet

@knet function model14(w, x; ndims=2, ndirs=3^ndims-1, nblocks=20, winit=Gaussian(0,0.05),
                       hidden=80*nblocks)
    h = wbf2(w, x; out=hidden, f=:relu, winit=winit)
    return wbf(h; out=nblocks*ndirs, f=:soft, winit=winit)
end

function train14(; N=2^16, dims=(16,16), nblocks=20, ndims=length(dims), ndirs=3^ndims-1,
                 lr=0.001, adam=true, nbatch=128, winit=Gaussian(0,0.05))
    global f = compile(:model14; ndims=ndims, nblocks=nblocks, winit=winit)
    setp(f, lr=lr, adam=adam)
    sloss = zloss = 0
    nextn = 1
    ncells = prod(dims)
    global world = zeros(ndims, nblocks)
    global target = zeros(ndims, 1)
    global ygold = zeros(ndirs, nblocks)
    global world2 = zeros(length(world), nbatch)
    global target2 = zeros(length(target), nbatch)
    global ygold2 = zeros(length(ygold), nbatch)
    for n=1:N
        for m=1:nbatch
            locations = randperm(ncells)
            for b=1:nblocks
                world[:,b] = [ind2sub(dims, locations[b])...] # fill blocks with random locations
            end
            target[:] = [ind2sub(dims, locations[nblocks+1])...] # pick target at an empty location
            d1 = world .- target; d2 = sum(d1 .* d1, 1)
            rblock = rand(find(d2 .== minimum(d2))) # pick one of the closest blocks randomly as reference
            d  = sign(target - world[:,rblock])    # a direction like [-1,0]
            d8 = (d[1]==0 ? 0 : d[1]==1 ? 1 : 2) + 3*(d[2]==0 ? 0 : d[2]==1 ? 1 : 2) # map dir to [1:8]
            @assert (d8 >= 1 && d8 <= 8)
            ygold[:] = 0; ygold[d8,rblock] = 1
            # fill the minibatch matrices
            world2[:,m] = vec(world) 
            target2[:,m] = vec(target)
            ygold2[:,m] = vec(ygold)
        end
        global ypred = forw(f, world2, target2)
        sl = softloss(ypred,ygold2); sloss = (n==1 ? sl : 0.99 * sloss + 0.01 * sl)
        zl = zeroone(ypred,ygold2);  zloss = (n==1 ? zl : 0.99 * zloss + 0.01 * zl)
        n==nextn && (println((n,sloss,1-zloss)); nextn*=2)
        back(f, ygold2, softloss)
        update!(f)
    end
end

train14()
