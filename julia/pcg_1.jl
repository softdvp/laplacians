#=

Implementations of cg and pcg.
Need to add conditions for termination when stagnate.
Look at two approaches: Matlab's, and
hypre: https://github.com/LLNL/hypre/blob/master/src/krylov/pcg.c
Started by Dan Spielman.
Contributors:

=#

"""
    x = pcg(mat, b, pre; tol, maxits, maxtime, verbose, pcgIts, stag_test)`

solves a symmetric linear system using preconditioner `pre`.
# Arguments
* `pre` can be a function or a matrix.  If a matrix, a function to solve it is created with cholFact.
* `tol` is set to 1e-6 by default,
* `maxits` defaults to Inf
* `maxtime` defaults to Inf.  It measures seconds.
* `verbose` defaults to false
* `pcgIts` is an array for returning the number of pcgIterations.  Default is length 0, in which case nothing is returned.
* `stag_test=k` stops the code if rho[it] > (1-1/k) rho[it-k].  Set to 0 to deactivate.
"""
#function pcg end


"""
    x = pcgSolver(mat, pre; tol, maxits, maxtime, verbose, pcgIts)

creates a solver for a PSD system using preconditioner `pre`.
The parameters are as described in pcg.
"""
#function pcgSolver end



function pcg(mat, b, pre::Union{AbstractArray,Matrix}; tol::Real=1e-6, maxits=Inf, maxtime=Inf, verbose=false, pcgIts=Int[])
    fact = cholesky(pre)
    F = x->(fact \ x)
    pcg(mat, b, F; tol=tol, maxits=maxits, maxtime=maxtime, verbose=verbose, pcgIts=pcgIts)
end


function pcgSolver(mat, pre; tol::Real=1e-6, maxits=Inf, maxtime=Inf, verbose=false, pcgIts=Int[])
    tol_=tol
    maxits_=maxits
    maxtime_=maxtime
    verbose_=verbose
    pcgIts_=pcgIts

    f(b; tol=tol_, maxits=maxits_, maxtime=maxtime_, verbose=verbose_, pcgIts=pcgIts_) =
        pcg(mat, b, pre, tol=tol, maxits=maxits, maxtime=maxtime, verbose=verbose, pcgIts=pcgIts)
end


function pcg(mat, b::Vector{Tval}, pre::Function;
        tol::Real=1e-6, maxits=Inf, maxtime=Inf, verbose=false, pcgIts=Int[],
        stag_test::Integer=0) where Tval

    local al::Tval

    n = size(mat,2)

    nb = norm(b)

    # If input vector is zero, quit
    if nb == 0
      return zeros(size(b))
    end

    x = zeros(Tval,n)
    bestx = zeros(Tval,n)
    bestnr = 1.0

    r = copy(b)
    z = pre(r)
    p = copy(z)

    rho = dot(r, z)
    best_rho = rho
    stag_count = 0

    t1 = time()

    itcnt = 0
    while itcnt < maxits
        itcnt = itcnt+1

        q = mat*p

        pq = dot(p,q)

        if (pq < eps(Tval) || isinf(pq))
          if verbose
            println("PCG Stopped due to small or large pq")
          end
          break
        end

        al = rho/pq

        # the following line could cause slowdown
        if al*norm(p) < eps(Tval)*norm(x)
          if verbose
            println("PCG: Stopped due to stagnation.")
          end
          break
        end

        axpy2!(al,p,x)
        # x = x + al * p
        #=
        @inbounds @simd for i in 1:n
            x[i] += al*p[i]
        end
        =#
        #axpy

        axpy2!(-al,q,r)
        #r .= r .- al.*q
        #=
        @inbounds @simd for i in 1:n
            r[i] -= al*q[i]
        end
        =#

        nr = norm(r)/nb
        if nr < bestnr
          bestnr = nr
          @inbounds @simd for i in 1:n
            bestx[i] = x[i]
          end
        end
        if nr < tol #Converged?
            break
        end

        # here is the top of the code in numerical templates

        z = pre(r)

        oldrho = rho
        rho = dot(z, r) # this is gamma in hypre.

        if rho < best_rho*(1-1/stag_test)
          best_rho = rho
          stag_count = 0
        else
          if stag_test > 0
            if best_rho > (1-1/stag_test)*rho
              stag_count += 1
              if stag_count > stag_test
                println("PCG Stopped by stagnation test ", stag_test)
                break
              end
            end
          end
        end

        if (rho < eps(Tval) || isinf(rho))
          if verbose
            println("PCG Stopped due to small or large rho")
          end
          break
        end

        # the following would have higher accuracy
        #       rho = sum(r.^2)

        beta = rho/oldrho
        if (beta < eps(Tval) || isinf(beta))
          if verbose
            println("PCG Stopped due to small or large beta")
          end
          break
        end

        bzbeta!(beta,p,z)
        #=
        # p = z + beta*p
        @inbounds @simd for i in 1:n
            p[i] = z[i] + beta*p[i]
        end
        =#

        if (time() - t1) > maxtime
            if verbose
                println("PCG New stopped at maxtime.")
            end
            break
        end

    end

    if verbose
        println("PCG stopped after: ", round((time() - t1),digits=3), " seconds and ", itcnt, " iterations with relative error ", (norm(r)/norm(b)), ".")
    end

    if length(pcgIts) > 0
        pcgIts[1] = itcnt
    end


    return bestx
end



function axpy2!(al,p::Array,x::Array)
  n = length(x)
  @inbounds @simd for i in 1:n
      x[i] = x[i] + al*p[i]
  end
end

# p[i] = z[i] + beta*p[i]
function bzbeta!(beta,p::Array,z::Array)
  n = length(p)
  @inbounds @simd for i in 1:n
      p[i] = z[i] + beta*p[i]
  end
end
