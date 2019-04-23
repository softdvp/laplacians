mutable struct IJV{Tv,Ti}
    n::Ti
    nnz::Ti
    i::Array{Ti,1}
    j::Array{Ti,1}
    v::Array{Tv,1}
end
import Base.==
==(a::IJV, b::IJV) =
    a.n == b.n &&
    a.nnz == b.nnz &&
    a.i == b.i &&
    a.j == b.j &&
    a.v == b.v

import SparseArrays.nnz
#nnz(a::IJV) = a.nnz

hash(a::IJV) =
    hash((a.n, a.nnz, a.i, a.j, a.v), hash(IJV))

hash(a::IJV, h::UInt) = hash(hash(a), h)

import Base.+
function +(a::IJV, b::IJV)
    @assert a.n == b.n
    IJV(a.n, a.nnz+b.nnz,
        [a.i; b.i],
        [a.j; b.j],
        [a.v; b.v])
end

import Base.*
function *(a::IJV, x::Number)
    ijv = deepcopy(a)
    ijv.v .*= x

    return ijv
end

*(x::Number, a::IJV) = *(a, x)

transpose(ijv::IJV) = IJV(ijv.n, ijv.nnz, ijv.j, ijv.i, ijv.v)
adjoint(ijv::IJV) = IJV(ijv.n, ijv.nnz, ijv.j, ijv.i, adjoint.(ijv.v))

using SparseArrays
using LinearAlgebra
"""
    ijv = IJV(A::SparseMatrixCSC)
Convert a sparse matrix to an IJV.
"""
function IJV(A::SparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
    (ai,aj,av) = findnz(A)
    IJV{Tv,Ti}(A.n, nnz(A), ai, aj, av)
end


import SparseArrays.sparse

sparse(ijv::IJV) = sparse(ijv.i, ijv.j, ijv.v, ijv.n, ijv.n)

compress(ijv::IJV) = IJV(sparse(ijv))




function components(mat::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    n = mat.n

    order = Array{Ti}(undef, n)
    comp = zeros(Ti,n)

    # note that all of this casting is unnecessary.
    # but, some of it speeds up the code
    # I have not figured out the minimal necessary
    c::Ti = 0

    colptr = mat.colptr
    rowval = mat.rowval

    @inbounds for x in 1:n
      if (comp[x] == 0)
        c = c + 1
        comp[x] = c

        if colptr[x+1] > colptr[x]
          ptr::Ti = 1
          orderLen::Ti = 2
          order[ptr] = x

          while ptr < orderLen
            curNode = order[ptr]

            for ind in colptr[curNode]:(colptr[curNode+1]-1)
              nbr = rowval[ind]
              if comp[nbr] == 0
                comp[nbr] = c
                order[orderLen] = nbr
                orderLen += 1
              end # if
            end # for
            ptr += 1
          end # while
        end # if
      end

    end

    return comp
  end # function

  function vecToComps(compvec::Vector{Ti}) where Ti
      nc = maximum(compvec)
      comps = Vector{Vector{Ti}}(undef, nc)

      sizes = zeros(Ti,nc)
      for i in compvec
          sizes[i] += 1
      end

       for i in 1:nc
           comps[i] = zeros(Ti,sizes[i])
       end

      ptrs = zeros(Ti,nc)
       for i in 1:length(compvec)
          c = compvec[i]
          ptrs[c] += 1
          comps[c][ptrs[c]] = i
      end
      return comps
  end

grid2(n::Integer, m::Integer; isotropy=1.0) =
    sparse(grid2_ijv(n, m; isotropy=isotropy))

grid2_ijv(n::Integer, m::Integer; isotropy=1.0) =
    product_graph(isotropy*path_graph_ijv(n), path_graph_ijv(m))

grid2(n::Integer) = grid2(n,n)
grid2_ijv(n::Integer) = grid2_ijv(n,n)

function path_graph_ijv(n::Integer)
    IJV(n, 2*(n-1),
        [collect(1:(n-1)) ; collect(2:n)],
        [collect(2:n); collect(1:(n-1))],
        ones(2*(n-1)))
end

function product_graph(a0::SparseMatrixCSC, a1::SparseMatrixCSC)
    n0 = size(a0)[1]
    n1 = size(a1)[1]gfgff
    a = kron(sparse(I,n0,n0),a1) + kron(a0,sparse(I, n1, n1));

  end # productGraph

  function product_graph(b::IJV{Tva,Tia}, a::IJV{Tvb,Tib}) where {Tva, Tvb, Tia, Tib}
      Ti = promote_type(Tia, Tib)

      n = a.n * b.n

      @assert length(a.i) == a.nnz

      a_edge_from = kron(ones(Ti, a.nnz), a.n*collect(0:(b.n-1)))

            ai = a_edge_from + kron(a.i, ones(Ti, b.n))
      aj = a_edge_from + kron(a.j, ones(Ti, b.n))
      av = kron(a.v, ones(b.n))

      b_edge_from = kron(collect(1:a.n), ones(Ti, b.nnz))
      bi = b_edge_from + kron(ones(Ti, a.n), (b.i .- 1) .* a.n)
      bj = b_edge_from + kron(ones(Ti, a.n), (b.j .- 1) .* a.n)
      bv = kron(ones(a.n), b.v)

      return IJV(n, length(av)+length(bv),
          [ai; bi], [aj; bj], [av; bv])
  end

  function power(a::SparseMatrixCSC, k::Int)
    ap = a^k
    ap = ap - sparse(Diagonal(diag(ap)))
  end

  function diagmat(a::SparseMatrixCSC{Tv, Ti}) where {Tv, Ti}

    return sparse(Diagonal(vec(sum(a,dims=1))))

  end # diagmat

  function lapWrapComponents(solver, a::AbstractArray; tol::Real=1e-6, maxits=Inf, maxtime=Inf, verbose=false, pcgIts=Int[], params...)

    t1 = time()

    if !testZeroDiag(a)
        @warn "The matrix should not have any nonzero diagonal entries."
        a = a - sparse(Diagonal(diag(a)))
    end

    co = components(a)

    if maximum(co) == 1

        s = solver(a; tol=tol, maxits=maxits, maxtime=maxtime, verbose=verbose, pcgIts=pcgIts, params... )
        if verbose
            println("Solver build time: ", round((time() - t1),digits=3), " seconds.")
        end

        # f(b; tol=tol_, maxits=maxits_, maxtime=maxtime_, verbose=verbose_, pcgIts=pcgIts_) =
        return s

    else

        comps = vecToComps(co)

        solvers = []
        for i in 1:length(comps)
            ind = comps[i]

            asub = a[ind,ind]

            if (length(ind) == 1)
                subSolver = nullSolver

            elseif (length(ind) < 50)
                subSolver = lapWrapConnected(chol_sddm,asub)

            else

                subSolver = solver(asub; tol=tol, maxits=maxits, maxtime=maxtime, verbose=verbose, pcgIts=pcgIts, params... );

            end
            push!(solvers, subSolver)
        end

        if verbose
            println("Solver build time: ", round((time() - t1),digits=3), " seconds.")
        end

        return blockSolver(comps,solvers; tol=tol, maxits=maxits, maxtime=maxtime, verbose=verbose, pcgIts=pcgIts)

    end
end


function lapWrapComponents(solver::Function)
    f(a::AbstractArray; tol=1e-6, maxits=Inf, maxtime=Inf, verbose=false, pcgIts=Int[], params...) = lapWrapComponents(solver, a; tol=tol, maxits=maxits, maxtime=maxtime, verbose=verbose, pcgIts=pcgIts, params... )
    return f
end

  function lapWrapSDDM(sddmSolver)

    return lapWrapComponents(lapWrapConnected(sddmSolver))

end

function sddmWrapLap(lapSolver, sddm::AbstractArray; tol::Real=1e-6, maxits=Inf, maxtime=Inf, verbose=false, pcgIts=Int[], params...)

    # Make a new adj matrix, a1, with an extra entry at the end.
    a, d = adj(sddm)
    a1 = extendMatrix(a,d)
    F = lapSolver(a1; tol=tol, maxits=maxits, maxtime=maxtime, verbose=verbose, pcgIts=pcgIts, params...)

    # make a function that solves the extended system, modulo the last entry
    tol_=tol
    maxits_=maxits
    maxtime_=maxtime
    verbose_=verbose
    pcgIts_=pcgIts

    f = function(b; tol=tol_, maxits=maxits_, maxtime=maxtime_, verbose=verbose_, pcgIts=pcgIts_)
        xaug = F([b; -sum(b)], tol=tol, maxits=maxits, maxtime=maxtime, verbose=verbose, pcgIts=pcgIts)
        xaug = xaug .- xaug[end]
        return xaug[1:a.n]
    end

    return f

end

function sddmWrapLap(lapSolver)
    f = function(sddm::AbstractArray; tol::Real=1e-6, maxits=Inf, maxtime=Inf, verbose=false, pcgIts=Int[], params...)
        return sddmWrapLap(lapSolver, sddm;  tol=tol, maxits=maxits, maxtime=maxtime, verbose=verbose, pcgIts=pcgIts, params... )
    end
    return f
end
function testZeroDiag(a)
    n = size(a,1)
    for i in 1:n
        if a[i,i] != 0.0
            return false
        end
    end
    return true
end

lap(a) = sparse(Diagonal(a*ones(size(a)[1]))) - a

function flipIndex(a::SparseMatrixCSC{Tval,Tind}) where {Tval,Tind}

    b = SparseMatrixCSC(a.m, a.n, copy(a.colptr), copy(a.rowval), collect(UnitRange{Tind}(1,nnz(a))) );
    bakMat = copy(b');
    return bakMat.nzval

  end

  function wtedEdgeVertexMat(mat::SparseMatrixCSC)
    (ai,aj,av) = findnz(triu(mat,1))
    m = length(ai)
    n = size(mat)[1]
    v = av.^(1/2)
    return sparse(collect(1:m),ai,v,m,n) - sparse(collect(1:m),aj,v,m,n)
end

function support(a1,a2; tol=1e-5)
    la1 = lap(a1)
    la2 = lap(a2)
    f1 = approxchol_lap(a1,tol=tol)
    f2 = approxchol_lap(a2,tol=tol)
    op12 = SqLinOp(false,1.0,size(a1,1),b->la2*f1(b))
    op21 = SqLinOp(false,1.0,size(a2,1),b->la1*f2(b))

    if isConnected(a1)
      sup12 = abs(eigs(op12;nev=1,which=:LM,tol=tol)[1][1])
    else
      sup12 = Inf
    end
    if isConnected(a2)
      sup21 = abs(eigs(op21;nev=1,which=:LM,tol=tol)[1][1])
    else
      sup21 = Inf
    end


    return sup12, sup21
end

function approxQual(a1,a2; verbose=false, tol=1e-5)
    sup12, sup21 = support(a1, a2, tol=tol)

    if verbose
        println("support12: ", sup12, ", support21: ", sup21)
    end

    return max(sup12-1, sup21-1)
end

#println("grid2=", grid2(5))
