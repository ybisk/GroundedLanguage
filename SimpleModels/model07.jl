# model07: a neural net will learn anything given enough examples /
# hidden units etc.  So the argument is not whether it can learn but
# how efficiently.  Here we'll try an even simpler problem: learning
# to multiply with a regular mlp: input is a 2D vector in [0,1],
# output is the product of the two dimensions.

# Using adam and nbatch=128, I can get this to converge.

# We should probably look at the efficiency papers given at:
# http://neuralnetworksanddeeplearning.com/chap5.html

@knet function model07(x0; hidden=10, layers=2, o...)
    x1 = repeat(x0; o..., frepeat=:wbf, nrepeat=layers, out=hidden, f=:relu)
    return wb(x1; o..., out=1)
end

function train07(; N=2^20, lr=0.001, hidden=1000, layers=1, nbatch=128)
    global f = compile(:model07, hidden=hidden, layers=layers, winit=Gaussian(0,.1), binit=Constant(0))
    setp(f, lr=lr, adam=true)
    loss = 0
    nextn = 1
    x = zeros(2,nbatch)
    for n=1:N
        rand!(x)
        scale!(x,100)
        ygold = prod(x,1)
        ypred = forw(f, x)
        qloss = quadloss(ypred,ygold)
        loss = (n==1 ? qloss : 0.99 * loss + 0.01 * qloss)
        n==nextn && (println((n,loss)); nextn*=2)
        back(f, ygold, quadloss)
        update!(f)
    end
end

train07()
