# model11: Continuing from model09, we'll try minibatching.  It is
# easy to minibatch if the world doesn't change, works in model09 out
# of the box.  But if the world is also different in each instance we
# need to do more work.  First let's try the easy version, world is
# fixed within a minibatch.

# Trains with err<0.01 in 16K batches of 32 = 500K examples.
# Compare to model09 with no minibatch trains in 128K examples.
# model11 takes 5 seconds for 16K batches.
# model09 takes 60 seconds for 128K instances.

using Knet

@knet function model11(w, x, r)
    a = par(init=Constant(0), dims=(0,0))
    b = par(init=Constant(0), dims=(0,0))
    c = par(init=Constant(0), dims=(0,0))
    d = par(init=Constant(0), dims=(0,0))
    return w * (a * x + b * r) + (c * x + d * r)
end

function train11(; N=2^15, dims=(16,16), nblocks=20, lr=0.002, adam=false, nbatch=32)
    global f = compile(:model11)
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

train11()
