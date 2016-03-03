# model10: Continuing from model09, maybe we do not need the term
# without W.  After all if we use y=W(Ax+Br), B can learn to offset W
# and the whole expression can generate y=WAx+B'r where B'=WB.  In
# fact W is like a fixed parameter matrix.  The parameters may be a
# bit harder to interpret but worth a shot.

# Unfortunately the error is stuck around 0.40 in 2^20 examples, the
# extra terms seem to be necessary.

using Knet

@knet function model10(w, x, r)
    a = par(init=Constant(0), dims=(0,0))
    b = par(init=Constant(0), dims=(0,0))
    return w * (a * x + b * r)
end

function train10(; N=2^20, dims=(16,16), nblocks=20, lr=0.0002, adam=false)
    global f = compile(:model10)
    setp(f, lr=lr, adam=adam)
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

train10()
