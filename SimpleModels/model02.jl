# In model01 w'*y was not always maximized for y=w[:,i] because the
# entries of w are not normalized.  There are a few ways to have
# columns of w normalized.  We can keep it 2D and pick entries from
# the unit circle.  We can blow it up to 3D and pick closest positions
# on the surface of a unit sphere.  We can learn how to normalize by
# applying some transformation to w before and after the product with
# Px.  Here we'll try the first one: pick object coordinates uniformly
# on the unit circle.

# When we start from P=identity we stay there because the gradient
# w*(ypred-ygold)=0.

using Knet

# Input: world (coor) (d,n)
# Input: block (one-hot id) (n,1)
# Output: block (coor) (d,1)

@knet function model02(w, x; nblocks=20, init=Gaussian(0,0.1))
    p = par(init=init, dims=(nblocks,nblocks))
    z = p * x
    return w * z
end

# lr=0.1 works best, lr >= 0.2 diverges, lr=0.05 slower.
# Constant(0) works for init, nothing much better.
# err < 0.01 in 1024 examples (no minibatching)

function train02(; N=2^13, dims=(16,16), nblocks=20, lr=0.1, init=Constant(0))
    global f = compile(:model02, nblocks=nblocks, init=init)
    setp(f, lr=lr)
    loss = 0
    nextn = 1
    for n=1:N
        global c = randn(length(dims), nblocks)
        c = c./sqrt(sum(c.^2,1))
        global x = zeros(nblocks, 1); x[rand(1:nblocks)] = 1
        global ygold = c*x
        global ypred = forw(f, c, x)
        qloss = quadloss(ypred,ygold)
        loss = (n==1 ? qloss : 0.99 * loss + 0.01 * qloss)
        n==nextn && (println((n,loss)); nextn*=2)
        back(f, ygold, quadloss)
        update!(f)
    end
end

train02()
