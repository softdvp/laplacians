include("pcg_1.jl")
include("collection.jl")
using SparseArrays

"""
#bzbeta! test
a=[ 1., 2., 3., 4.]
b=[5., 4., 0., 1.]
bzbeta!(2.3, a, b)
println(a)

#axpy2! test
a=[ 1, 2, 3, 4 ]
b=[5, 4, 0, 1]

axpy2!(4, a, b)
println(b)


m=[0 0 0 0; 5 8 0 0; 0 0 3 0; 0 6 0 0]

sp_m=SparseMatrixCSC(m)

ijv=IJV(sp_m)
dump(ijv)


sp_m1=sparse(ijv)

dump(sp_m)
dump(sp_m1)

println(Matrix(sp_m))
println(Matrix(sp_m1))

m2=[1 0 0 0; 6 2 1 0; 2 0 3 0; 0 0 0 0]

sp_m2=SparseMatrixCSC(m2)

ijv2=IJV(sp_m2)
dump(ijv2)


#path_graph_ijv

n=5
IJV(n, 2*(n-1), [collect(1:(n-1)) ; collect(2:n)],
        [collect(2:n); collect(1:(n-1))], ones(2*(n-1)))

"""

A=SparseMatrixCSC{Int};
B=SparseMatrixCSC{Int};
A[1, 1] = 1; A[1, 2] = 4; A[1, 3] = 3; A[2, 1] = 2; A[2, 2] = 1;
B[1, 1] = 5; B[1, 3] = 1; B[2, 1] = 7; B[2,2] = 2; B[3, 1] = 3; B[3, 3] = 1;

println(A)

println(B)
