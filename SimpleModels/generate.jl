# First generate random data in a common format, then worry about
# putting it into various forms.  Yonatan's data coordinates are in
# [-1,1] and block width is 0.1254, i.e. we can fit 16 blocks side by
# side.  We will assume border cells surrounding a 16x16 board, so our
# world is 18x18.  We will place 20 random blocks here and then move
# one randomly to an unoccupied cell.

"A discrete world with given dims (not including borders), a number of blocks and their coordinates, a source and a target position."
type World; dims; nblocks; source; target; state; end

function World(dims, nblocks)
    ncells = prod(dims)
    shuffled_cells = randperm(ncells)            # generate shuffled linear indices
    full_cells = shuffled_cells[1:nblocks]
    empty_cells = shuffled_cells[nblocks+1:end]
    source_cell = rand(full_cells)              # assume dims excludes borders
    target_cell = rand(empty_cells)
    World(dims, nblocks, ind2sub(dims,source_cell), ind2sub(dims,target_cell), map(i->ind2sub(dims,i), full_cells))
end

"returns the state after a block moves from source to target"
function after(w::World)
    [ x == w.source ? w.target : x for x in w.state ]
end

"returns the coordinate representation for a world state"
function coor(state)
    mapreduce(x->Float64[x...], hcat, state)
end

"returns the grid representation for a world state, surrounded by a border"
function grid(state, dims)
    n = length(state)
    # add borders x->x+2
    gdims = map(x->x+2, dims)
    # use 1..n for blocks, n+1 for border and n+2 for empty
    border,empty = n+1,n+2
    g = zeros(gdims..., n+2)
    for i=1:prod(gdims)
        gsub = ind2sub(gdims, i)
        ssub = ntuple(i->gsub[i]-1, length(gsub))
        gobj = findfirst(state, ssub)
        if gobj > 0
            g[gsub...,gobj] = 1
        elseif any([gsub...] .== 1) || any([gsub...] .== [gdims...])
            g[gsub...,border] = 1
        else
            g[gsub...,empty] = 1
        end
    end
    return g
end

"returns the id of an object given its coordinates and world state"
function coor2id(coor, state)
    findfirst(state, coor)
end

"returns the coor of an object given its id and world state"
function id2coor(id, state)
    state[id]
end

"returns a bordered grid with a given coordinate marked"
function coor2grid(coor, dims)
    gdims = map(x->x+2, dims)
    g = zeros(gdims...)
    cdims = map(x->x+1, coor)
    g[cdims...] = 1
    return g
end

:ok
