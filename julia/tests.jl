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

#A(0, 0) = 1; A(0, 1) = 2; A(1, 0) = 3; A(1, 1) = 4;
#B(0, 1) = 5; B(1, 0) = 6; B(1, 1) = 7;

#Test Kronecker product function kron(A, B)

I=[1, 1, 2, 2]
J=[1, 2, 1, 2]
V=[1, 2, 3, 4]

A = sparse(I,J,V,2,2)

I=[1, 2, 2]
J=[2, 1, 2]
V=[5, 6, 7]

B=sparse(I,J,V,2,2)

C=kron(A, B)

println(C)

"""
Out:

0   5   0  10
6   7  12  14
0  15   0  20
18  21  24  28
"""
