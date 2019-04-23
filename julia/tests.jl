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

#Test connections()

m10(1, 0) = 1;	m10(2, 0) = 1;	m10(0, 1) = 1;	m10(5, 1) = 1;	m10(8, 1) = 1;
	m10(0, 2) = 1;	m10(5, 3) = 1;	m10(7, 4) = 1;	m10(1, 5) = 1;	m10(3, 5) = 1;
	m10(8, 5) = 1;	m10(4, 7) = 1;	m10(8, 7) = 1;	m10(1, 8) = 1;	m10(5, 8) = 1;
	m10(7, 8) = 1;

I=[2, 3, 1, 6, 9, 1, 6, 8, 2, 4, 9, 5, 9, 2, 6, 8]
J=[1, 1, 2, 2, 2, 3, 4, 5, 6, 6, 6, 8, 8, 9, 9, 9]
V=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

A = sparse(I, J, V, 10, 10)

#print(components(A))

#Out: [1, 1, 1, 1, 1, 1, 2, 1, 1, 3]

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

#println(Matrix(C))


Out:

0   5   0  10
6   7  12  14
0  15   0  20
18  21  24  28

D=flipIndex(C)

#println(D)

# D = [3, 9, 1, 4, 7, 10, 5, 11, 2, 6, 8, 12]

dg=sum(C, dims=1);

#println("sum(C)) = ", dg)

Out: sum(C)) = [24 48 36 72]

dmatrix=diagmat(C)

#println("diagmat(C)) = ", dmatrix)
#println("power(C, 2) = ")
#display(Matrix(power(C, 2)))

println("\n\nA matrix:")
display(Matrix(A))

println("B matrix:")
display(Matrix(B))

println("\n\nPG = product_graph(IJV(B), IJV(A)):\n");

PG = product_graph(IJV(B), IJV(A))

dump(PG)
Pgs=sparse(PG)
display(Matrix(Pgs))
println()

I=[1, 1, 2, 2]
J=[1, 2, 1, 2]
V=[1, 2, 3, 4]

S1 = sparse(I, J, V)

V=[7, 10, 15, 22]
S2 = sparse(I, J, V)

IJ=IJV(S1)+IJV(S2)
println("IJ=IJV(S1)+IJV(S2) = ")

dump(IJ)
"""

# Test product_graph

#Create two graph matrices of 9 vertices
"""
GrA=[0 1 0 1 0 0 0 0 0; 1 0 1 0 1 0 0 0 0; 0 1 0 0 0 0 0 0 0;
	 1 0 0 0 0 0 1 1 0; 0 1 0 0 0 1 0 0 0; 0 0 0 0 1 0 0 0 1;
	 0 0 0 1 0 0 0 1 0; 0 0 0 1 0 0 0 0 0; 0 0 0 0 0 1 0 0 0]

GrB=[0 0 0 0 1 0 0 0 0; 0 0 1 1 0 0 0 0 0; 0 1 0 0 0 1 0 0 0;
	 0 1 0 0 0 0 0 1 0; 1 0 0 0 0 1 1 0 0; 0 0 0 0 1 0 0 0 1;
	 0 0 0 0 1 0 0 0 0; 0 0 0 1 0 0 0 0 0; 0 0 0 0 0 1 0 0 0]

spGrA=sparse(GrA)
spGrB=sparse(GrB)

ijvA=IJV(spGrA)
ijvB=IJV(spGrB)

#dump(ijvA)
#dump(ijvB)

ijvPg=product_graph(ijvA, ijvB)

dump(ijvPg)

Mx1=Matrix(grid2(5))

#display(Mx1)
LapMx=lap(GrA)

display(LapMx)
"""

V=vec([1, 1, 2, 2, 3])

 R=vecToComps(V)

 display(R)
