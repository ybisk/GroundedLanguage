# model09: Continuing from model08, what if we don't know which word
# is a noun and which word is a preposition.  Instead of y=WPx+Ur we
# can have: y=WAx+WBr+Cx+Dr or y=W(Ax+Br) + (Cx+Dr).  So if s is a
# representation of the sentence, we'd be better off trying y=WAs+Bs
# to get both nouns and prepositions.

# Takes a bit longer (128K) and smaller lr (0.0002) but it learns with
# err<0.01 in 128K examples.

using Knet

@knet function model09(w, x, r)
    a = par(init=Constant(0), dims=(0,0))
    b = par(init=Constant(0), dims=(0,0))
    c = par(init=Constant(0), dims=(0,0))
    d = par(init=Constant(0), dims=(0,0))
    return w * (a * x + b * r) + (c * x + d * r)
end

function train09(; N=2^17, dims=(16,16), nblocks=20, lr=0.0002, adam=false)
    global f = compile(:model09)
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

train09()
