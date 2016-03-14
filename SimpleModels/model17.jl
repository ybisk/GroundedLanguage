# model17: Finding the target position given before and after states.
# Input: before and after world coor (DxN,DxN)
# Output: target position (Dx1)

# Idea: model15 finds the block id given before+after.  We just need to
# multiply that with after coordinates.
# problem: multiplicative model cannot do minibatching if we don't fix
# the world (model12) and try to keep w2 two dimensional.  To select D
# values from a flattened world vector w2 we need convolution, or we
# need D outputs.  back currently cannot handle D outputs easily.  We
# need to trick back by writing directly dif arrays or figure out a
# convolution solution.

# Idea: just try regular mlp first.

# Baselines: if you pick two random positions on a 16x16 grid,
# expected qloss ~ 42.  If you always pick a random point and the
# center, , the expected qloss ~ 21.. (sqrt(84)~9, sqrt(42)~6.5)

# Hyperparameters: for the tanh model, small winit but nonzero.
# SGD lr=1e-5 works best.  ADAM lr=1e-4 works better.

# Model: the nonlinearity on line 2 (f2):
# tanh: 15.36@8K
# sigm: 16.99@8K convergence slower
# soft1: stuck at 20 (even though this is a modified version of soft that should work in the middle of a model)

# Model: the nonlinearity on line 1 (f1): doesn't make much difference
# relu: 15.36@8K
# sigm: 15.43@8K
# tanh: 15.55@8K

using Knet

@knet function model17(w1, w2; ndims=2, nblocks=20, winit=Gaussian(0,0.1), hidden=20*nblocks, f1=:relu, f2=:tanh)
    h = wbf2(w1, w2; out=hidden, f=f1, winit=winit) # this learns to subtract before and after
    m = wbf(h; out=nblocks, f=f2, winit=winit) # this should learn a target filter, similar to model15, sigmoid better than soft, need to pick out ndims elements (we'll do that outside now), also soft has a problem when used in the middle of the model
    d = arr(init=drepeat(ndims,nblocks))
    dm = d * m
    mw = dm .* w2                                        # this applies the filter to w2 coors
    e = arr(init=eyerepeat(ndims,nblocks))
    return e * mw
end

function train17(; N=2^13, dims=(16,16), ndims=length(dims), nblocks=20, nbatch=128,
                 lr=0.0001, adam=true, winit=Gaussian(0,0.001), o...)
    global f = compile(:model17; ndims=ndims, nblocks=nblocks, winit=winit, o...)
    setp(f, lr=lr, adam=adam)
    qloss = 0
    nextn = 1
    ncells = prod(dims)
    global world1 = zeros(ndims, nblocks)
    global world2 = zeros(ndims, nblocks)
    global ygold = zeros(ndims)
    global world1_ = zeros(length(world1), nbatch)
    global world2_ = zeros(length(world2), nbatch)
    global ygold_ = zeros(length(ygold), nbatch)
    for n=1:N
        for m=1:nbatch
            locations = randperm(ncells)
            for b=1:nblocks
                world1[:,b] = world2[:,b] = [ind2sub(dims, locations[b])...] # place blocks at random locations
            end
            ygold[:] = [ind2sub(dims, locations[nblocks+1])...] # pick target at an empty location
            mblock = rand(1:nblocks)                             # move a random block there
            world2[:,mblock] = ygold
            # fill the minibatch matrices
            world1_[:,m] = vec(world1)
            world2_[:,m] = vec(world2)
            ygold_[:,m] = ygold
        end
        global ypred_ = forw(f, world1_, world2_)
        ql = quadloss(ypred_,ygold_); qloss = (n==1 ? ql : 0.99 * qloss + 0.01 * ql)
        n==nextn && (println((n,qloss)); nextn*=2)
        back(f, ygold_, quadloss)
        update!(f)
    end
end

function eyerepeat(D,N)
    w = zeros(D,D*N)
    for i=1:D*N
        w[mod1(i,D),i] = 1
    end
    return w
end

function drepeat(D,N)
    w = zeros(D*N,N)
    for n=1:N
        for d=1:D
            w[D*(n-1)+d,n] = 1
        end
    end
    return w
end

train17()

# For the tanh model this gives:
# julia> train17(adam=true, lr=0.0001, N=2^20)
# (1,94.43343759674423)
# (2,94.2674811137955)
# (4,93.5496677424914)
# (8,90.93753137621619)
# (16,85.8994963900018)
# (32,76.56159694183638)
# (64,61.404238733021636)
# (128,42.56316744232831)
# (256,26.968469261401854)
# (512,19.565718943081553)
# (1024,16.572647034850924)
# (2048,16.080906803072942)
# (4096,15.874408204069011)
# (8192,15.49930970318984)
# (16384,11.627527984184752)
# (32768,9.997862967152658)
# (65536,5.107752401639812)
# (131072,2.968800589115292)
# (262144,2.0179603028419075)
# (524288,1.5806946078150395)
