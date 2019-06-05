import Pkg;

#Pkg.add("SparseArrays")
#Pkg.add("RandomV06")
import Base;

import RandomV06
import Random
import SparseArrays

using SparseArrays
using LinearAlgebra

using Statistics

import Base: ==, hash, +, *, transpose
import Laplacians:grid2
import SparseArrays: nnz

V06 = RandomV06.V06
Vcur = RandomV06.Vcur

include("pcg_1.jl")
include("collection.jl")
include("approxChol_1.jl")
include("sparsify.jl")

G = grid2(100)

@show n = size(G,1)
d_ave = nnz(G)/n

Gp = power(G,15)
@show nnz(Gp)/n


Gsparse = sparsify(Gp, ep=1)
println("Average degree of sparsifier: ",nnz(Gsparse)/n)

#println("Approximation quality: ", approxQual(Gp, Gsparse))
#=
=#
