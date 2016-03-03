# model08: This adds relative positions on top of model01.  The model
# becomes y=WPx+Ur, where U are the offsets and r are the
# prepositions.  W,x,r are inputs, P and U are learnable parameters.
# W is DxN coordinates, x (object name) and r (relpos) are one-hot, P
# is an NxN permutation matrix (could be NxV if the number of names is
# not equal to the number of objects), U is DxR learnable offsets.

# It learns err<0.01 in 64K instances.  Adam has no advantage over
# lr=0.0005.

using Knet

@knet function model08(w, x, r)
    p = par(init=Constant(0), dims=(0,0))
    u = par(init=Constant(0), dims=(0,0))
    a = p * x
    b = w * a
    c = u * r
    return b + c
end

function train08(; N=2^16, dims=(16,16), nblocks=20, lr=0.0005)
    global f = compile(:model08)
    setp(f, lr=lr)
    loss = 0
    nextn = 1
    ncells = prod(dims)
    ndirs = 9
    global world = zeros(length(dims), nblocks)
    global dirs = [0 0; 1 0; 1 1; 0 1; -1 1; -1 0; -1 -1; 0 -1; 1 -1]'
    global noun = zeros(nblocks, 1)
    global prep = zeros(ndirs, 1)
    for n=1:N
        locations = randperm(ncells)
        for b=1:nblocks
            world[:,b] = [ind2sub(dims, locations[b])...]
        end
        inoun = rand(1:nblocks)
        iprep = rand(1:ndirs)
        global ygold = world[:,inoun] + dirs[:,iprep]
        noun[:] = 0; noun[inoun] = 1
        prep[:] = 0; prep[iprep] = 1
        global ypred = forw(f, world, noun, prep)
        qloss = quadloss(ypred,ygold)
        loss = (n==1 ? qloss : 0.99 * loss + 0.01 * qloss)
        n==nextn && (println((n,loss)); nextn*=2)
        back(f, ygold, quadloss)
        update!(f)
    end
end

train08()
