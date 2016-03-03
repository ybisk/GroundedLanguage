# model12: Continuing from model11, we'll try to minibatch the world
# as well.  The world will be (D,N,1,B) and the matrix mult will be
# simulated by a convolution with (1,N,1,B).  The one-hot word batches
# will be (N,B).  After they get multiplied by the permutation matrix
# (N,N), the resulting (N,B) needs to be reshaped to (1,N,1,B) for the
# convolution.  We may need a reshape op for Knet?

using Knet

@knet function model12(w, x, r)
    a = par(init=Constant(0), dims=(0,0))
    b = par(init=Constant(0), dims=(0,0))
    c = par(init=Constant(0), dims=(0,0))
    d = par(init=Constant(0), dims=(0,0))
    return w * (a * x + b * r) + (c * x + d * r)
end

function train12(; N=2^15, dims=(16,16), nblocks=20, lr=0.002, adam=false, nbatch=32)
    global f = compile(:model12)
    setp(f, lr=lr, adam=adam)
    loss = 0
    nextn = 1
    ncells = prod(dims)
    ndirs = 9
    global world = zeros(length(dims), nblocks)
    global dirs = [0 0; 1 0; 1 1; 0 1; -1 1; -1 0; -1 -1; 0 -1; 1 -1]'
    global noun = zeros(nblocks, nbatch)
    global prep = zeros(ndirs, nbatch)
    for n=1:N
        locations = randperm(ncells)
        for b=1:nblocks
            world[:,b] = [ind2sub(dims, locations[b])...]
        end
        inoun = rand(1:nblocks, nbatch)
        iprep = rand(1:ndirs, nbatch)
        global ygold = world[:,inoun] + dirs[:,iprep]
        noun[:] = 0; prep[:] = 0
        for b=1:nbatch; noun[inoun[b],b]=prep[iprep[b],b]=1; end
        global ypred = forw(f, world, noun, prep)
        qloss = quadloss(ypred,ygold)
        loss = (n==1 ? qloss : 0.99 * loss + 0.01 * qloss)
        n==nextn && (println((n,loss)); nextn*=2)
        back(f, ygold, quadloss)
        update!(f)
    end
end

train12()
