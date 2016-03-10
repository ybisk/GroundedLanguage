# model13: Finding a nearby block given a position.
# Input: world coor (w:DxN), target coor (x:Dx1)
# Output: softmax over block ids (y:Nx1)
# Linear input Aw+Bx can calculate signed distances.
# We can flatten w to a vector.
# One hidden layer can calculate absolute distances.
# It needs 2N hidden units to do so.
# The softmax probabilities should relate to absolute distances.
# Baseline nll=-log(1/20)=2.995732273553991

# Starts learning in (acc > 0.8) in 8192 iterations with minibatch=128, adam=true, lr=0.001
# More hidden units (20*nblocks) help converge faster. (multiple experiments in parallel?)
# More hidden layers do not seem to help (or I couldn't find the right hyperparameters)
# Converges to ~.85 in ~64K iterations (x minibatch=128).

using Knet

@knet function model13(w, x; ndims=2, nblocks=20, winit=Gaussian(0,0.05), hidden=20*nblocks)
    h = wbf2(w, x; out=hidden, f=:relu, winit=winit)
    return wbf(h; out=nblocks, f=:soft, winit=winit)
end

function train13(; N=2^16, dims=(16,16), nblocks=20, lr=0.001, adam=true, nbatch=128, winit=Gaussian(0,0.05)) # 0.76@8192
    global f = compile(:model13; ndims=length(dims), nblocks=nblocks, winit=winit)
    setp(f, lr=lr, adam=adam)
    sloss = zloss = 0
    nextn = 1
    ncells = prod(dims)
    global world = zeros(length(dims), nblocks)
    global target = zeros(length(dims), 1)
    global ygold = zeros(nblocks,1)
    global world2 = zeros(length(dims)*nblocks, nbatch)
    global target2 = zeros(length(dims), nbatch)
    global ygold2 = zeros(nblocks,nbatch)
    for n=1:N
        for m=1:nbatch
            locations = randperm(ncells)
            for b=1:nblocks
                world[:,b] = [ind2sub(dims, locations[b])...] # fill blocks with random locations
            end
            target[:,1] = [ind2sub(dims, locations[nblocks+1])...] # pick target at an empty location
            d1 = world .- target
            d2 = sum(d1 .* d1, 1)
            d3 = rand(find(d2 .== minimum(d2))) # pick one of the closest blocks randomly
            ygold[:] = 0; ygold[d3] = 1
            world2[:,m] = vec(world) # fill the minibatch matrices
            target2[:,m] = target
            ygold2[:,m] = ygold
        end
        global ypred = forw(f, world2, target2)
        sl = softloss(ypred,ygold2); sloss = (n==1 ? sl : 0.99 * sloss + 0.01 * sl)
        zl = zeroone(ypred,ygold2);  zloss = (n==1 ? zl : 0.99 * zloss + 0.01 * zl)
        n==nextn && (println((n,sloss,1-zloss)); nextn*=2)
        back(f, ygold2, softloss)
        update!(f)
    end
end

#sgd: function train13(; N=2^13, dims=(16,16), nblocks=20, lr=0.02, adam=false, nbatch=128, winit=Gaussian(0,0.05)) # 0.57@8192

train13()
