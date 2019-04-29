#=
  Structs for the approxChol solver
=#


"""
  LLp elements are all in the same column.
  row tells us the row, and val is the entry.
  val is set to zero for some edges that we should remove.
  next gives the next in the column.  It points to itself to terminate.
  reverse is the index into lles of the other copy of this edge,
  since every edge is stored twice as we do not know the order of elimination in advance.
"""
mutable struct LLp{Tind,Tval}
    row::Tind
    val::Tval
    next::LLp{Tind,Tval}
    reverse::LLp{Tind,Tval}

    LLp{Tind,Tval}() where {Tind,Tval} = (x = new(zero(Tind), zero(Tval)); x.next = x; x.reverse = x)
    LLp{Tind,Tval}(row, val, next, rev) where {Tind,Tval} = new(row, val, next, rev)
    LLp{Tind,Tval}(row, val) where {Tind,Tval} = (x = new(row, val); x.next = x; x.reverse = x)
    LLp{Tind,Tval}(row, val, next) where {Tind,Tval} = (x = new(row, val, next); x.reverse = x)
end

"""
  LLmatp is the data structure used to maintain the matrix during elimination.
  It stores the elements in each column in a singly linked list (only next ptrs)
  Each element is an LLp (linked list pointer).
  The head of each column is pointed to by cols.

  We probably can get rid of degs - as it is only used to store initial degrees.
"""
mutable struct LLmatp{Tind,Tval}
    n::Int64
    degs::Array{Tind,1}
    cols::Array{LLp{Tind,Tval},1}
    lles::Array{LLp{Tind,Tval},1}
end

# these are the types we use with a fixed ordering
struct LLord{Tind,Tval}
    row::Tind
    next::Tind
    val::Tval
end

mutable struct LLMatOrd{Tind,Tval}
    n::Int64
    cols::Array{Tind,1}
    lles::Array{LLord{Tind,Tval},1}
end

struct LLcol{Tind,Tval}
      row::Tind
      ptr::Tind
      val::Tval
  end

#=============================================================

LDLinv

=============================================================#

"""
  LDLinv contains the information needed to solve the Laplacian systems.
  It does it by applying Linv, then Dinv, then Linv (transpose).
  But, it is specially constructed for this particular solver.
  It does not explicitly make the matrix triangular.
  Rather, col[i] is the name of the ith col to be eliminated
"""
mutable struct LDLinv{Tind,Tval}
    col::Array{Tind,1}
    colptr::Array{Tind,1}
    rowval::Array{Tind,1}
    fval::Array{Tval,1}
    d::Array{Tval,1}
end

#=============================================================

ApproxCholPQ
the data strcture we use to keep track of degrees

=============================================================#

struct ApproxCholPQElem{Tind}
    prev::Tind
    next::Tind
    key::Tind
end

"""
  An approximate priority queue.
  Items are bundled together into doubly-linked lists with all approximately the same key.
  minlist is the min list we know to be non-empty.
  It should always be a lower bound.
  keyMap maps keys to lists
"""
mutable struct ApproxCholPQ{Tind}
    elems::Array{ApproxCholPQElem{Tind},1} # indexed by node name
    lists::Array{Tind,1}
    minlist::Int
    nitems::Int
    n::Int
end



#=

approxChol Laplacian solver by Daniel A. Spielman, 2017.
This algorithm is an implementation of an approximate edge-by-edge elimination
algorithm inspired by the Approximate Gaussian Elimination algorithm of
Kyng and Sachdeva.

For usage exaples, see http://danspielman.github.io/Laplacians.jl/latest/usingSolvers/index.html

There are two versions of this solver:
one that fixes the order of elimination beforehand,
and one that adapts the order to eliminate verties of low degree.
These use different data structures.
LLOrdMat is for the fixed order, and LLmatp is for the adaptive order.

These coes produce a structure we call LDLinv that is then used in the solve.
The structure of this code is as follows:

The data structures appear in approxCholTypes.jl
We then have the outline:

* constructors for LLmatp and LLMatOrd
* get_ll_col and compress_ll_col : used inside the elimination
* approxChol : the main routine
* LDLsolver, and its forward and backward solve the apply LDLinv
* approxchol_lap: the main solver, which calls approxchol_lap1 on connected
    components.
    This then calls one of approxchol_lapWdeg, approxchol_lapGiven or approxchol_lapGreedy,
    depending on the parameters.

* approxchol_lapChol - for producing a Cholesky factor instead of an LDLinv.
  might be useful if optimized.
* data structures that are used for the adaptive low-degree version to
  choose the next vertex.

=#

"""
    params = ApproxCholParams(order, output)
order can be one of
* :deg (by degree, adaptive),
* :wdeg (by original wted degree, nonadaptive),
* :given
"""
mutable struct ApproxCholParams
    order::Symbol
    stag_test::Integer
end

ApproxCholParams() = ApproxCholParams(:deg, 5)
ApproxCholParams(sym::Symbol) = ApproxCholParams(sym, 5)

LDLinv(a::SparseMatrixCSC{Tval,Tind}) where {Tind,Tval} =
  LDLinv(zeros(Tind,a.n-1), zeros(Tind,a.n),Tind[],Tval[],zeros(Tval,a.n))

LDLinv(a::LLMatOrd{Tind,Tval}) where {Tind,Tval} =
  LDLinv(zeros(Tind,a.n-1), zeros(Tind,a.n),Tind[],Tval[],zeros(Tval,a.n))

LDLinv(a::LLmatp{Tind,Tval}) where {Tind,Tval} =
  LDLinv(zeros(Tind,a.n-1), zeros(Tind,a.n),Tind[],Tval[],zeros(Tval,a.n))


function LLmatp(a::SparseMatrixCSC{Tval,Tind}) where {Tind,Tval}
    n = size(a,1)
    m = nnz(a)

    degs = zeros(Tind,n)

    flips = flipIndex(a)

    cols = Array{LLp{Tind,Tval}}(undef, n)
    llelems = Array{LLp{Tind,Tval}}(undef, m)

    @inbounds for i in 1:n
        degs[i] = a.colptr[i+1] - a.colptr[i]

        ind = a.colptr[i]
        j = a.rowval[ind]
        v = a.nzval[ind]
        llpend = LLp{Tind,Tval}(j,v)
        next = llelems[ind] = llpend
        for ind in (a.colptr[i]+one(Tind)):(a.colptr[i+1]-one(Tind))
            j = a.rowval[ind]
            v = a.nzval[ind]
            next = llelems[ind] = LLp{Tind,Tval}(j,v,next)
        end
        cols[i] = next
    end

    @inbounds for i in 1:n
        for ind in a.colptr[i]:(a.colptr[i+1]-one(Tind))
            llelems[ind].reverse = llelems[flips[ind]]
        end
    end

    return LLmatp{Tind,Tval}(n, degs, cols, llelems)
end

"""
  Print a column in an LLmatp matrix.
  This is here for diagnostics.
"""
function print_ll_col(llmat::LLmatp, i::Int)
    ll = llmat.cols[i]
    println("col $i, row $(ll.row) : $(ll.val)")

    while ll.next != ll
        ll = ll.next
        println("col $i, row $(ll.row) : $(ll.val)")
    end
end



#=============================================================

The approximate factorization

=============================================================#

function get_ll_col(llmat::LLmatp{Tind,Tval},
  i,
  colspace::Vector{LLp{Tind,Tval}}) where {Tind,Tval}


    ll = llmat.cols[i]
    len = 0
    @inbounds while ll.next != ll

        if ll.val > zero(Tval)
            len = len+1
            if (len > length(colspace))
                push!(colspace,ll)
            else
                colspace[len] = ll
            end
        end

        ll = ll.next
    end

    if ll.val > zero(Tval)
        len = len+1
        if (len > length(colspace))
            push!(colspace,ll)
        else
            colspace[len] = ll
        end
    end

    return len
end





function compressCol!(a::LLmatp{Tind,Tval},
  colspace::Vector{LLp{Tind,Tval}},
  len::Int,
  pq::ApproxCholPQ{Tind}) where {Tind,Tval}

    o = Base.Order.ord(isless, x->x.row, false, Base.Order.Forward)

    sort!(colspace, 1, len, QuickSort, o)

    ptr = 0
    currow::Tind = 0

    c = colspace

    @inbounds for i in 1:len

        if c[i].row != currow
            currow = c[i].row
            ptr = ptr+1
            c[ptr] = c[i]

        else
            c[ptr].val = c[ptr].val + c[i].val
            c[i].reverse.val = zero(Tval)

            approxCholPQDec!(pq, currow)
        end
    end


    o = Base.Order.ord(isless, x->x.val, false, Base.Order.Forward)
    sort!(colspace, 1, ptr, QuickSort, o)

    return ptr
end

function compressCol!(
  colspace::Vector{LLcol{Tind,Tval}},
  len::Int
  ) where {Tind,Tval}

    o = Base.Order.ord(isless, x->x.row, false, Base.Order.Forward)

    sort!(colspace, one(len), len, QuickSort, o)

    c = colspace

    ptr = 0
    currow = c[1].row
    curval = c[1].val
    curptr = c[1].ptr

    @inbounds for i in 2:len

        if c[i].row != currow

            ptr = ptr+1
            c[ptr] = LLcol(currow, curptr, curval)  # next is abuse here: reall keep where it came from.

            currow = c[i].row
            curval = c[i].val
            curptr = c[i].ptr

        else

            curval = curval + c[i].val

        end

    end

    # emit the last row

    ptr = ptr+1
    c[ptr] = LLcol(currow, curptr, curval)

    o = Base.Order.ord(isless, x->x.val, false, Base.Order.Forward)
    sort!(colspace, one(ptr), ptr, QuickSort, o)

    return ptr
end


# this one is greedy on the degree - also a big win
function approxChol(a::LLmatp{Tind,Tval}) where {Tind,Tval}
    n = a.n
    ldli = LDLinv(a)
    ldli_row_ptr = one(Tind)

    d = zeros(n)

    pq = ApproxCholPQ(a.degs)

    it = 1

    colspace = Array{LLp{Tind,Tval}}(undef, n)
    cumspace = Array{Tval}(undef, n)
    vals = Array{Tval}(undef, n) # will be able to delete this

    o = Base.Order.ord(isless, identity, false, Base.Order.Forward)

    @inbounds while it < n

        i = approxCholPQPop!(pq)

        ldli.col[it] = i # conversion!
        ldli.colptr[it] = ldli_row_ptr

        it = it + 1

        len = get_ll_col(a, i, colspace)

        len = compressCol!(a, colspace, len, pq)  #3hog

        csum = zero(Tval)
        for ii in 1:len
            vals[ii] = colspace[ii].val
            csum = csum + colspace[ii].val
            cumspace[ii] = csum
        end
        wdeg = csum

        colScale = one(Tval)

        for joffset in 1:(len-1)

            ll = colspace[joffset]
            w = vals[joffset] * colScale
            j = ll.row
            revj = ll.reverse

            f = w/(wdeg)

            vals[joffset] = zero(Tval)

            # kind = Laplacians.blockSample(vals,k=1)[1]
            r = rand() * (csum - cumspace[joffset]) + cumspace[joffset]
            koff = searchsortedfirst(cumspace,r,one(len),len,o)

            k = colspace[koff].row

            approxCholPQInc!(pq, k)

            newEdgeVal = f*(one(Tval)-f)*wdeg

            # fix row k in col j
            revj.row = k   # dense time hog: presumably becaus of cache
            revj.val = newEdgeVal
            revj.reverse = ll

            # fix row j in col k
            khead = a.cols[k]
            a.cols[k] = ll
            ll.next = khead
            ll.reverse = revj
            ll.val = newEdgeVal
            ll.row = j


            colScale = colScale*(one(Tval)-f)
            wdeg = wdeg*(one(Tval)-f)^2

            push!(ldli.rowval,j)
            push!(ldli.fval, f)
            ldli_row_ptr = ldli_row_ptr + one(Tind)

            # push!(ops, IJop(i,j,1-f,f))  # another time suck


        end # for


        ll = colspace[len]
        w = vals[len] * colScale
        j = ll.row
        revj = ll.reverse

        if it < n
            approxCholPQDec!(pq, j)
        end

        revj.val = zero(Tval)

        push!(ldli.rowval,j)
        push!(ldli.fval, one(Tval))
        ldli_row_ptr = ldli_row_ptr + one(Tind)

        d[i] = w

    end

    ldli.colptr[it] = ldli_row_ptr

    ldli.d = d

    return ldli
end



#=============================================================

The routines that do the solve.

=============================================================#

function LDLsolver(ldli::LDLinv, b::Vector)
    y = copy(b)

    forward!(ldli, y)

    @inbounds for i in 1:(length(ldli.d))
        if ldli.d[i] != 0
            y[i] /= ldli.d[i]
        end
    end

    backward!(ldli, y)

    mu = mean(y)
    @inbounds for i in eachindex(y)
        y[i] = y[i] - mu
    end

    return y
end


function forward!(ldli::LDLinv{Tind,Tval}, y::Vector) where {Tind,Tval}

    @inbounds for ii in 1:length(ldli.col)
        i = ldli.col[ii]

        j0 = ldli.colptr[ii]
        j1 = ldli.colptr[ii+1]-one(Tind)

        yi = y[i]

        for jj in j0:(j1-1)
            j = ldli.rowval[jj]
            y[j] += ldli.fval[jj] * yi
            yi *= (one(Tval)-ldli.fval[jj])
        end
        j = ldli.rowval[j1]
        y[j] += yi
        y[i] = yi
    end
end

function backward!(ldli::LDLinv{Tind,Tval}, y::Vector) where {Tind,Tval}
    o = one(Tind)
    @inbounds for ii in length(ldli.col):-1:1
        i = ldli.col[ii]

        j0 = ldli.colptr[ii]
        j1 = ldli.colptr[ii+1]-o

        j = ldli.rowval[j1]
        yi = y[i]
        yi = yi + y[j]

        for jj in (j1-o):-o:j0
            j = ldli.rowval[jj]
            yi = (one(Tval)-ldli.fval[jj])*yi + ldli.fval[jj]*y[j]
        end
        y[i] = yi
    end
end




"""
    solver = approxchol_lap(a); x = solver(b);
    solver = approxchol_lap(a; tol::Real=1e-6, maxits=1000, maxtime=Inf, verbose=false, pcgIts=Int[], params=ApproxCholParams())

A heuristic by Daniel Spielman inspired by the linear system solver in https://arxiv.org/abs/1605.02353 by Rasmus Kyng and Sushant Sachdeva.  Whereas that paper eliminates vertices one at a time, this eliminates edges one at a time.  It is probably possible to analyze it.
The `ApproxCholParams` let you choose one of three orderings to perform the elimination.

* ApproxCholParams(:given) - in the order given.
    This is the fastest for construction the preconditioner, but the slowest solve.
* ApproxCholParams(:deg) - always eliminate the node of lowest degree.
    This is the slowest build, but the fastest solve.
* ApproxCholParams(:wdeg) - go by a perturbed order of wted degree.

For more info, see http://danspielman.github.io/Laplacians.jl/latest/usingSolvers/index.html
"""
function approxchol_lap(a::SparseMatrixCSC{Tv,Ti};
  tol::Real=1e-6,
  maxits=1000,
  maxtime=Inf,
  verbose=false,
  pcgIts=Int[],
  params=ApproxCholParams()) where {Tv,Ti}

    return lapWrapComponents(approxchol_lap1, a,
    verbose=verbose,
    tol=tol,
    maxits=maxits,
    maxtime=maxtime,
    pcgIts=pcgIts,
    params=params)


end

function approxchol_lapGreedy(a::SparseMatrixCSC;
  tol::Real=1e-6,
  maxits=1000,
  maxtime=Inf,
  verbose=false,
  pcgIts=Int[],
  params=ApproxCholParams())

  tol_ =tol
  maxits_ =maxits
  maxtime_ =maxtime
  verbose_ =verbose
  pcgIts_ =pcgIts

  t1 = time()

  la = lap(a) # a hit !?

  llmat = LLmatp(a)
  ldli = approxChol(llmat)
  F(b) = LDLsolver(ldli, b)

  f(b;tol=tol_,maxits=maxits_, maxtime=maxtime_, verbose=verbose_, pcgIts=pcgIts_) = pcg(la, b .- mean(b), F, tol=tol, maxits=maxits, maxtime=maxtime, pcgIts=pcgIts, verbose=verbose, stag_test = params.stag_test)

end


function approxchol_lapGiven(a::SparseMatrixCSC;
  tol::Real=1e-6,
  maxits=1000,
  maxtime=Inf,
  verbose=false,
  pcgIts=Int[],
  params=ApproxCholParams())

  tol_ =tol
  maxits_ =maxits
  maxtime_ =maxtime
  verbose_ =verbose
  pcgIts_ =pcgIts

  t1 = time()

  la = lap(a)

  llmat = LLMatOrd(a)
  ldli = approxChol(llmat)
  F(b) = LDLsolver(ldli, b)

  f(b;tol=tol_,maxits=maxits_, maxtime=maxtime_, verbose=verbose_, pcgIts=pcgIts_) = pcg(la, b .- mean(b), F, tol=tol, maxits=maxits, maxtime=maxtime, pcgIts=pcgIts, verbose=verbose, stag_test = params.stag_test)

end

function approxchol_lapWdeg(a::SparseMatrixCSC;
  tol::Real=1e-6,
  maxits=1000,
  maxtime=Inf,
  verbose=false,
  pcgIts=Int[],
  params=ApproxCholParams())

  tol_ =tol
  maxits_ =maxits
  maxtime_ =maxtime
  verbose_ =verbose
  pcgIts_ =pcgIts

  t1 = time()

  la = lap(a)

  v = vec(sum(a,dims=1))
  v = v .* (1 .+ rand(length(v)))
  p = sortperm(v)

  llmat = LLMatOrd(a,p)
  ldli = approxChol(llmat)

  ip = invperm(p)
  ldlip = LDLinv(p[ldli.col], ldli.colptr, p[ldli.rowval], ldli.fval, ldli.d[ip]);

  F = function(b)
    x = zeros(size(b))
    x = LDLsolver(ldlip, b)
    #x[p] = LDLsolver(ldli, b[p])
    return x
  end

  f(b;tol=tol_,maxits=maxits_, maxtime=maxtime_, verbose=verbose_, pcgIts=pcgIts_) = pcg(la, b .- mean(b), F, tol=tol, maxits=maxits, maxtime=maxtime, pcgIts=pcgIts, verbose=verbose, stag_test = params.stag_test)

end



function approxchol_lap1(a::SparseMatrixCSC{Tv,Ti};
  tol::Real=1e-6,
  maxits=1000,
  maxtime=Inf,
  verbose=false,
  pcgIts=Int[],
  params=ApproxCholParams()) where {Tv,Ti}

    tol_ =tol
    maxits_ =maxits
    maxtime_ =maxtime
    verbose_ =verbose
    pcgIts_ =pcgIts


    if params.order == :deg

      return approxchol_lapGreedy(a,
        verbose=verbose,
        tol=tol,
        maxits=maxits,
        maxtime=maxtime,
        pcgIts=pcgIts,
        params=params)


    elseif params.order == :wdeg

      return approxchol_lapWdeg(a,
        verbose=verbose,
        tol=tol,
        maxits=maxits,
        maxtime=maxtime,
        pcgIts=pcgIts,
        params=params)


    else
      return approxchol_lapGiven(a,
        verbose=verbose,
        tol=tol,
        maxits=maxits,
        maxtime=maxtime,
        pcgIts=pcgIts,
        params=params)


    end

end








#=============================================================

ApproxCholPQ
It only implements pop, increment key, and decrement key.
All nodes with degrees 1 through n appear in their own doubly-linked lists.
Nodes of higher degrees are bundled together.

=============================================================#


function keyMap(x, n)
    return x <= n ? x : n + div(x,n)
end

function ApproxCholPQ(a::Vector{Tind}) where Tind

    n = length(a)
    elems = Array{ApproxCholPQElem{Tind}}(undef, n)
    lists = zeros(Tind, 2*n+1)
    minlist = one(n)

    for i in 1:length(a)
        key = a[i]
        head = lists[key]

        if head > zero(Tind)
            elems[i] = ApproxCholPQElem{Tind}(zero(Tind), head, key)

            elems[head] = ApproxCholPQElem{Tind}(i, elems[head].next, elems[head].key)
        else
            elems[i] = ApproxCholPQElem{Tind}(zero(Tind), zero(Tind), key)

        end

        lists[key] = i
    end

    return ApproxCholPQ(elems, lists, minlist, n, n)
end

function approxCholPQPop!(pq::ApproxCholPQ{Tind}) where Tind
    if pq.nitems == 0
        error("ApproxPQ is empty")
    end
    while pq.lists[pq.minlist] == 0
        pq.minlist = pq.minlist + 1
    end
    i = pq.lists[pq.minlist]
    next = pq.elems[i].next


    pq.lists[pq.minlist] = next
    if next > 0
        pq.elems[next] = ApproxCholPQElem(zero(Tind), pq.elems[next].next, pq.elems[next].key)
    end

    pq.nitems -= 1

    return i
end

function approxCholPQMove!(pq::ApproxCholPQ{Tind}, i, newkey, oldlist, newlist) where Tind

    prev = pq.elems[i].prev
    next = pq.elems[i].next

    # remove i from its old list
    if next > zero(Tind)
        pq.elems[next] = ApproxCholPQElem{Tind}(prev, pq.elems[next].next, pq.elems[next].key)
    end
    if prev > zero(Tind)
        pq.elems[prev] = ApproxCholPQElem{Tind}(pq.elems[prev].prev, next, pq.elems[prev].key)

    else
        pq.lists[oldlist] = next
    end

    # insert i into its new list
    head = pq.lists[newlist]
    if head > 0
        pq.elems[head] = ApproxCholPQElem{Tind}(i, pq.elems[head].next, pq.elems[head].key)
    end
    pq.lists[newlist] = i

    pq.elems[i] = ApproxCholPQElem{Tind}(zero(Tind), head, newkey)

    return nothing
end

"""
    Decrement the key of element i
    This could crash if i exceeds the maxkey
"""
function approxCholPQDec!(pq::ApproxCholPQ{Tind}, i) where Tind

    oldlist = keyMap(pq.elems[i].key, pq.n)
    newlist = keyMap(pq.elems[i].key - one(Tind), pq.n)

    if newlist != oldlist

        approxCholPQMove!(pq, i, pq.elems[i].key - one(Tind), oldlist, newlist)

        if newlist < pq.minlist
            pq.minlist = newlist
        end

    else
        pq.elems[i] = ApproxCholPQElem{Tind}(pq.elems[i].prev, pq.elems[i].next, pq.elems[i].key - one(Tind))
    end


    return nothing
end

"""
    Increment the key of element i
    This could crash if i exceeds the maxkey
"""
function approxCholPQInc!(pq::ApproxCholPQ{Tind}, i) where Tind

    oldlist = keyMap(pq.elems[i].key, pq.n)
    newlist = keyMap(pq.elems[i].key + one(Tind), pq.n)

    if newlist != oldlist

        approxCholPQMove!(pq, i, pq.elems[i].key + one(Tind), oldlist, newlist)

    else
        pq.elems[i] = ApproxCholPQElem{Tind}(pq.elems[i].prev, pq.elems[i].next, pq.elems[i].key + one(Tind))
    end

    return nothing
end
