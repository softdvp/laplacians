include("pcg_1.jl")
include("collection.jl")
using Laplacians
include("solverInterface.jl")
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
	 0 0 0 1 0 0 0 1 0; 0 0 0 1 0 0 1 0 0; 0 0 0 0 0 1 0 0 0]

GrB=[0 0 0 0 1 0 0 0 0; 0 0 1 1 0 0 0 0 0; 0 1 0 0 0 1 0 0 0;
	 0 1 0 0 0 0 0 1 0; 1 0 0 0 0 1 1 0 0; 0 0 1 0 1 0 0 0 1;
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


V=vec([1, 1, 2, 2, 3])

 R=vecToComps(V)

 display(R)

#Test cholesky

A=[4.0 12.0 -16.0; 12.0 37.0 -43.0; -16.0 -43.0 98.0]
B=[1.0, 2.0, 3.0]

F=cholesky(A)
println("Cholevsky Factorozation=\n")
display(F.L)


X=F  B
println("X=", X)
Bx=A*X
println("Bx=", Bx)

"""

a=[ -0.356543 -0.136045 -1.93844 1.18337 -0.207743;
    -0.67799 1.95279 -0.193003 -1.84183 -0.662046;
     2.61283 1.51118 0.672955 -0.840613 2.01147;
     0.859572 -0.943768 0.375822 -1.57407 -0.858285;
     -0.0863611 -1.47299 1.02716 1.904 -0.42796]

b=transpose([1.064160977905516 -0.3334067812850509  0.7919292830316926 0.01651278833545206 -0.6051230029995152])
a = a * a'
println("a * a' = \n")
display(a)
solvea = wrapInterface(X->cholesky(X,Val(true)), a, maxits=100, verbose=true)
x=solvea(b, verbose=false)
println("\n\nx=\n")
display(x)

println("\n\nb=\n")
display(b)

ax=a*x
println("\n\nax=\n")
display(ax)

ax_b=ax-b
println("\n\nax_b=\n")
display(ax_b)

println("\n\nnorm(ax-b)=", norm(ax_b))

"""
f = wrapInterface(X->cholesky(X,Val(true)))
solvea = f(a, maxits=1000, maxtime = 1)
println(norm(a*solvea(b, verbose=false, maxtime = 10)-b))
"""
