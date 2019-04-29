#include <iostream>
#include <assert.h>
#include <blaze/Math.h>
#include <functional>
#include "collections.h"
#include "approxchol.h"

using blaze::DynamicVector;
using blaze::CompressedMatrix;
using namespace std;

#include "tests.h"

/*void bzbetatest() {
	DynamicVector<double>a{ 1, 2, 3, 4 };
	DynamicVector<double>b{ 5, 4, 0, 1 };
	
	double beta = 2.3;
	bzbeta(beta, a, b);

	cout << "Test of bzbeta function: "<< "\n" << a << "\n";
	//Out: (7.3, 8.6, 6.9, 10.2)
}

void axpy2test() {
	DynamicVector<int>a{ 1, 2, 3, 4 };
	DynamicVector<int>b{ 5, 4, 0, 1 };

	int al = 4;
	axpy2(al, a, b);

	cout << "Test of axpy2 function: " << "\n" << b << "\n";
	//Out: 3, 12, 12, 17
}*/

void IJVtests() {

	CompressedMatrix<int, blaze::columnMajor> z{ {0,0,0,0}, {5,8,0,0}, {0,0,3,0}, {0,6,0,0} };


	cout << "============================================\n";
	for (int i = 0; i < z.rows(); i++) {
		for (int j = 0; j < z.columns(); j++)
			cout << z(i, j) << " ";
		cout << endl;
	}


	cout << z << "\n============================================\n";

	//Convert a BLAZE sparse matrix to an IJV

	// A sample of  matrix was taken from 
	// https://en.wikipedia.org/wiki/Sparse_matrix
	CompressedMatrix<int, blaze::columnMajor> m{ {0,0,0,0}, {5,8,0,0}, {0,0,3,0}, {0,6,0,0} };
	
	SparseMatrixCSC<int> CSCMx(m);

	cout << "Original blaze compressed matrix:\n" << m;
	
	cout << "\nCSC sparse matrix dump:\n";
	cout << "CSCMx.m=" << CSCMx.m << endl;
	cout << "CSCMx.n=" << CSCMx.n << endl;

	cout << "CSCMx.colptr= ";
	for (size_t k = 0; k <= CSCMx.n; ++k)
		cout << CSCMx.colptr[k] << " ";

	cout <<"\n"<< "CSCMx.rowval= ";
	for (size_t k = 0; k < CSCMx.nzval.size(); ++k)
		cout << CSCMx.rowval[k] << " ";

	cout << "\n" << "CSCMx.nzval= ";
	for (size_t k = 0; k < CSCMx.nzval.size(); ++k)
		cout << CSCMx.nzval[k] << " ";

	assert(CSCMx.m == 4 && CSCMx.n == 4 && CSCMx.colptr[0] == 0 && CSCMx.colptr[3] == 4 && 
		CSCMx.rowval[0] == 1 && CSCMx.rowval[3] == 2 && CSCMx.nzval[0] == 5 && CSCMx.nzval[2] == 6);

	/* for column major (default) matrix:
	CSCMx.m=4
	CSCMx.n=4
	CSCMx.colptr= 0 1 3 4 4
	CSCMx.rowval= 1 1 3 2
	CSCMx.nzval= 5 8 6 3
	*/

	cout << endl << "\nTest toCompressedMatrix():\n";
	CompressedMatrix<int, blaze::columnMajor> mxtest;

	mxtest = CSCMx.toCompressedMatrix();
	cout << mxtest;

	assert(m == mxtest);
			
	cout << endl << "Convert from blaze compressed matrix to IJV structure.\n\n";
	IJV<int> ijv0(m);

	dump_ijv(0, ijv0);

	assert(ijv0.n == 4 && ijv0.nnz == 4 && ijv0.i[0] == 1 && ijv0.i[2] == 3 && ijv0.j[0] == 0 && ijv0.j[3] == 2 &&
		ijv0.v[0] == 5 && ijv0.v[2] == 6);
	
	/*Out:
		ijv0.n=4
		ijv0.nnz=4
		ijv0.i= 1 1 3 2
		ijv0.j= 0 1 1 2
		ijv0.v= 5 8 6 3
	*/

	cout << endl << "Convert from Julia SparseMatrixCSC to IJV structure.\n\n";
	IJV<int> ijv10(CSCMx);
	dump_ijv(10, ijv10);

	assert(ijv0 == ijv10);

	/*Out:
		ijv10.n=4
		ijv10.nnz=4
		ijv10.i= 1 1 3 2
		ijv10.j= 0 1 1 2
		ijv10.v= 5 8 6 3
	*/

	cout << "\n";

	cout << "\nTest sparse function.\n\n";
	SparseMatrixCSC<int> CSCMx1;

	Laplacians<int> lapl;

	CSCMx1 = lapl.sparseCSC(ijv0);

	cout << "\nCSC sparse matrix dump:\n";
	cout << "CSCMx1.m=" << CSCMx1.m << endl;
	cout << "CSCMx1.n=" << CSCMx1.n << endl;

	cout << "CSCMx1.colptr= ";
	for (size_t k = 0; k <= CSCMx1.n; ++k)
		cout << CSCMx1.colptr[k] << " ";

	cout << "\n" << "CSCMx1.rowval= ";
	for (size_t k = 0; k < CSCMx1.nzval.size(); ++k)
		cout << CSCMx1.rowval[k] << " ";

	cout << "\n" << "CSCMx1.nzval= ";
	for (size_t k = 0; k < CSCMx1.nzval.size(); ++k)
		cout << CSCMx1.nzval[k] << " ";

	assert(CSCMx == CSCMx1);

	/* for column major (default) matrix:
	 CSCMx1.m=4
	 CSCMx1.n=4
	 CSCMx1.colptr.i= 0 1 3 4 4
	 CSCMx1.rowval= 1 1 3 2
	 CSCMx1.nzval.v= 5 8 6 3
	 */
	
	cout << endl;

	cout << "Test overloaded operators.\n";

	IJV<int> ijv1 = ijv0;

	cout << "\noperator= :\n";

	dump_ijv(1, ijv1);

	bool t = ijv1 == ijv0;
	cout << "\n\noperator== :\n ijv1==ijv: " << (t ? "true" : "false");

	assert(ijv1 == ijv0);

	//Change ijv1
	ijv1.v[0] = 10;

	t = ijv1 == ijv0;
	cout << "\n\noperator== :\n ijv1==ijv: " << (t ? "true" : "false");
	assert(!(ijv1 == ijv0));

	cout << "\n\noperator* :\n ijv * 5";

	ijv1 = ijv0*5;

	cout << "\n" << "ijv1.v= ";
	for (size_t k = 0; k < ijv1.nnz; ++k)
		cout << ijv1.v[k] << " ";

	cout << "\n\noperator* :\n 5 * ijv";

	assert(ijv1.v[0] == 25 && ijv1.v[1] == 40);

	ijv1 = 5 * ijv0;

	cout << "\n" << "ijv1.v= ";
	for (size_t k = 0; k < ijv1.nnz; ++k)
		cout << ijv1.v[k] << " ";

	assert(ijv1.v[0] == 25 && ijv1.v[1] == 40);

	cout << "\n\nTest a constructor:\n";

	IJV<int> ijv2(ijv0.n, ijv0.nnz, ijv0.i, ijv0.j, ijv0.v);

	dump_ijv(2, ijv2);

	assert(ijv2 == ijv0);

	cout << "\n\nTest IJV::ToCompressMatrix():\n";

	CompressedMatrix<int, blaze::columnMajor> newm=ijv0.toCompressedMatrix();
	cout << newm;

	assert(newm == m);

	size_t h1 = lapl.hashijv(ijv0);
	size_t h2 = lapl.hashijv(ijv0, 5);
	cout << "\n\nTest hash(IJV) function:\n";
	cout <<"hash(ijv): " << h1 << endl;
	cout << "hash(ijv, 5): " << h2 << endl;

	assert(h1 != h2);

	cout << "\n\nTest compress(IJV) function:\n";

	IJV<int>ijv3 = lapl.compress(ijv0);

	dump_ijv(3, ijv3);

	assert(ijv0 == ijv3);

	cout << endl << "\nTranspose ijv:\n ";
	IJV<int> ijv4 = lapl.transpose(ijv0);

	dump_ijv(4, ijv4);
	assert(ijv4.i == ijv0.j && ijv4.j == ijv0.i && ijv4.v == ijv0.v);

	cout << endl << ijv4.toCompressedMatrix()<<endl;

//	cout << endl<<"Test a SparseMatrics constructor:\n";
	SparseMatrixCSC<int> CSCMx2(CSCMx.m, CSCMx.n, CSCMx.colptr, CSCMx.rowval, CSCMx.nzval);

	assert(CSCMx2 == CSCMx);

	cout << "\nTest IJV constructor with Dynamic Vectors:\n";

	DynamicVector<int>VI{ 1, 1, 3, 2 }, VJ{ 0, 1, 1, 2 }, VV{ 5, 8, 6, 3 };

	IJV<int> ijv5(4, 4, VI, VJ, VV);
	dump_ijv(5, ijv5);

	assert(ijv5 == ijv0);

}

void CollectionTest() {
	CompressedMatrix<int, blaze::columnMajor> m, m1;

	Laplacians<int> lapl;

	//Test path_graph_ijv
	IJV<int> ijv = lapl.path_graph_ijv(5);

	dump_ijv(0, ijv);
	m1 = ijv.toCompressedMatrix();
	cout << endl << endl << m1 << endl;
	assert(m1(0, 0) == 0 && m1(0, 1) == 1 && m1(1, 0) == 1);

	/* Out:
	0  1  0  0  0
	1  0  1  0  0
	0  1  0  1  0
	0  0  1  0  1
	0  0  0  1  0
	
	*/


	//Test testZeroDiag

	assert(testZeroDiag(m1));

	//Add non zero to a diagonal cell
	m1(2, 2) = 1;

	assert(!testZeroDiag(m1));

	// Test connections()
	CompressedMatrix<int, blaze::columnMajor> m10(10,10);
	m10(1, 0) = 1;	m10(2, 0) = 1;	m10(0, 1) = 1;	m10(5, 1) = 1;	m10(8, 1) = 1;
	m10(0, 2) = 1;	m10(5, 3) = 1;	m10(7, 4) = 1;	m10(1, 5) = 1;	m10(3, 5) = 1;
	m10(8, 5) = 1;	m10(4, 7) = 1;	m10(8, 7) = 1;	m10(1, 8) = 1;	m10(5, 8) = 1;
	m10(7, 8) = 1;

	cout << endl << endl;
	cout << "\n\nCall function componets():\n";

	SparseMatrixCSC<int> sprs(m10);
	vector<size_t> comp = lapl.components(sprs);
	
	for (int i = 0; i < comp.size(); i++)
		cout << comp[i] << " ";

	vector<size_t>comp1 = lapl.components(m10);

	for (int i = 0; i < comp1.size(); i++)
		cout << comp[i] << " ";

	assert(comp == comp1);
	assert(comp[0] == 1 && comp[6] == 2 && comp[9] == 3);

	//Out = 1, 1, 1, 1, 1, 1, 2, 1, 1, 3
	
	cout << endl << endl;
	
	//Test Kronecker product function kron(A, B)

	CompressedMatrix<int, blaze::columnMajor> A(2,2), B(2,2), C;

	//Create two occasional matrices
	//See an example at https://en.wikipedia.org/wiki/Kronecker_product

	A(0, 0) = 1; A(0, 1) = 2; A(1, 0) = 3; A(1, 1) = 4;
	B(0, 1) = 5; B(1, 0) = 6; B(1, 1) = 7;

	C = lapl.kron(A, B);
	cout << "\nC = kron(A, B)=\n" << C;
	assert(C(0, 0) == 0 && C(1, 1) == 7 && C(3, 3) == 28);

/*Out:
	kron(A, B)=

	0   5   0  10
	6   7  12  14
	0  15   0  20
	18  21  24  28
	
	*/

	//Test flipIndex

	cout << endl << endl << "flipIndex(C)=\n";

	vector<size_t> v = flipIndex(C);

	for (int i = 0; i < v.size(); i++) {
		cout << v[i] << " ";
	}
	assert(v[0] == 3 && v[1] == 9 && v[3] == 4);
	//Out="flipIndex(C)=3, 9, 1, 4, 7, 10, 5, 11, 2, 6, 8, 12"

	//Test function diag()

	cout << endl << endl << "diag(C)=\n";

	vector<int> v1 = lapl.diag(C, 1);

	for (int i = 0; i < v1.size(); i++) {
		cout << v1[i] << " ";
	}
	assert(v1[0] == 5 && v1[1] == 12 && v1[2] == 20);
	//Out: diag(C) = 5 12 20

	//Test function Diagonal()
	
	DynamicVector<int> dv=dynvec(v1);
	
	CompressedMatrix<int, blaze::columnMajor>Dg = lapl.Diagonal(dv);

	cout << endl << endl << "Diagonal(v)=\n" << Dg;
	assert(Dg(0, 0) == 5 && Dg(1, 1) == 12 & Dg(2, 2) == 20);

	/* Out:
	5  0  0
	0 12  0
	0  0 20
	*/

	//Test sum()

	/*
	C=	0  5   0  10
	    6  7  12 14
		0  15  0 20
		18 21 24 28
		
		*/

	DynamicVector<int> vec1 = lapl.sum(C);

	cout << "\nsum(C) = \n" << vec1;
	//Out: (24, 48, 36, 72)
	assert(vec1[0] == 24 && vec1[1] == 48 && vec1[2] == 36 && vec1[3] == 72);

	vec1 = lapl.sum(C, false);
	cout << "\nsum(C) = \n" << vec1;
	//Out: (15, 39, 35, 91)
	assert(vec1[0] == 15 && vec1[1] == 39 && vec1[2] == 35 && vec1[3] == 91);

	//Test diagmat()
	CompressedMatrix<int, blaze::columnMajor> DiagMx = lapl.diagmat(C);
	cout << "\ndiagmat(C) = \n" << DiagMx;

	assert(DiagMx(0, 0) == 24 && DiagMx(1, 1) == 48 && DiagMx(2, 2) == 36 && DiagMx(3, 3) == 72);
	
	/*
	Out:
	
	[0, 0]  =  24
	[1, 1]  =  48
	[2, 2]  =  36
	[3, 3]  =  72
  
  */

	//Test pow()
	CompressedMatrix<int, blaze::columnMajor> mx1{ {1,2}, {3,4} };
	CompressedMatrix<int, blaze::columnMajor>powmx = lapl.pow(mx1, 2);

	cout << "\npow(M) = \n" << powmx;

	/* Out:

		7  10
		15  22
	*/

	assert(powmx(0, 0) == 7 && powmx(0, 1) == 10 && powmx(1, 0) == 15 && powmx(1, 1) == 22);

	//Test power():
	CompressedMatrix<int, blaze::columnMajor>powmx1 = lapl.power(C, 2);

	cout << "\npower(C, 2) = \n" << powmx1;

	assert(powmx1(0, 1) == 245 && powmx1(1, 0) == 294);

	/* Out:
		  0  245   300  350
		294     0  420  790
		450   525    0  770
		630  1185  924    0
	
	*/

	//Test kron()  for vectors

	DynamicVector<int>Av{ 1,2,3 }, Bv{ 4, 5, 6 }, Cv;

	Cv = lapl.kron(Av, Bv);
	cout << "\nCv = kron(Av, Bv)=\n" << Cv;

	//Out = (4, 5, 6, 8, 10, 12, 12, 15, 18)
	assert(Cv[0] == 4 && Cv[3] == 8 && Cv[8] == 18);

	cout << "\nTest overloaded IJV operators.\n";

	IJV<int> ijv1(mx1), ijv2(powmx), ijv3;

	ijv3 = ijv1 + ijv2;

	cout << "\noperator+ :\n";

	dump_ijv(1, ijv1);
	cout << endl;
	dump_ijv(2, ijv2);
	cout << endl;
	dump_ijv(3, ijv3);
	cout << endl;
	assert(ijv3.i[0] == 0 && ijv3.i[4] == 0 && ijv3.j[0] == 0 && ijv3.j[4] == 0 && ijv3.v[0] == 1 && ijv3.v[4] == 7);
	
}

void CollectionFunctionTest() {
	
	Laplacians<int> lapl;
	
	//Test product_graph() function
		
	CompressedMatrix<int, blaze::columnMajor>
		GrA{ {0, 1, 0, 1, 0, 0, 0, 0, 0}, {1, 0, 1, 0, 1, 0, 0, 0, 0}, {0, 1, 0, 0, 0, 0, 0, 0, 0},
			{1, 0, 0, 0, 0, 0, 1, 1, 0}, {0, 1, 0, 0, 0, 1, 0, 0, 0}, {0, 0, 0, 0, 1, 0, 0, 0, 1},
			{0, 0, 0, 1, 0, 0, 0, 1, 0}, {0, 0, 0, 1, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 1, 0, 0, 0} };

	//cout << endl << GrA;

	CompressedMatrix<int, blaze::columnMajor>
		GrB{ {0, 0, 0, 0, 1, 0, 0, 0, 0}, {0, 0, 1, 1, 0, 0, 0, 0, 0}, {0, 1, 0, 0, 0, 1, 0, 0, 0},
			 {0, 1, 0, 0, 0, 0, 0, 1, 0}, {1, 0, 0, 0, 0, 1, 1, 0, 0}, {0, 0, 0, 0, 1, 0, 0, 0, 1},
			 { 0, 0, 0, 0, 1, 0, 0, 0, 0}, {0, 0, 0, 1, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 1, 0, 0, 0} };

	//cout << endl << GrB;

	IJV<int>IJVA(GrA), IJVB(GrB);

  	IJV<int> ijv0 = lapl.product_graph(IJVA, IJVB);
	/*cout << "\n\nproduct_graph(B, A)=\n";
	dump_ijv(0, ijv0);*/
	assert(ijv0.i[2] == 22 && ijv0.i[ijv0.i.size() - 3] == 35 && ijv0.j[2] == 18 && ijv0.j[ijv0.j.size() - 3] == 71);

	CompressedMatrix<int, blaze::columnMajor>GridMx;

	//Test grid2()
	GridMx = lapl.grid2(5);

	//cout << GridMx;

	assert(GridMx(0, 1) == 1 && GridMx(0, 5) == 1 && GridMx(1, 0) == 1);

	CompressedMatrix<int, blaze::columnMajor>LapMx = lapl.lap(GrA);
	//cout << LapMx;

	assert(LapMx(0, 0) == 2 && LapMx(1, 1) == 3 && LapMx(3, 0) == -1 && LapMx(0, 1) == -1);

	//Test vecToComps() function

	DynamicVector<size_t> V{ 1, 2, 1, 2, 3, 3, 3 };

	DynamicVector<DynamicVector<size_t>>comp = vecToComps(V);

	//cout << comp;

	assert(comp[0][1] == 2 && comp[1][1] == 3 && comp[2][1]==5);

	CompressedMatrix<int, blaze::columnMajor> Ma{ {1, 2, 3}, {4, 5, 6}, {7, 8, 9} };

	DynamicVector<size_t> Idx1{ 0, 1 }, Idx2{ 1, 2 };

	CompressedMatrix<int, blaze::columnMajor> Midx = lapl.index(Ma, Idx1, Idx2);

	//cout << Midx << endl;

	assert(Midx(0, 0) == 2 && Midx(0, 1) == 3 && Midx(1, 0) == 5 && Midx(1, 1) == 6);

	DynamicVector<size_t> Idx0{ 0 };

	Midx = lapl.index(Ma, Idx1, Idx0);

	//cout << Midx << endl;

	assert(Midx(0, 0) == 1 && Midx(1, 0) == 4);

	Midx = lapl.index(Ma, Idx1);
	//cout << Midx << endl;

	assert(Midx(0, 0) == 1 && Midx(1, 0) == 4);

	DynamicVector<int> vout(10, 0), vin{ 3, 5, 9 };

	DynamicVector<size_t> idx{ 1, 2, 6 };

	lapl.index(vout, idx, vin);

	//cout << vout << endl;

	assert(vout[1] == 3 && vout[2] == 5 && vout[6] == 9);

	//cout << a << endl << b << endl;

	// Cholesky decomposition
	// Test cholesky function

	//Test example from https://en.wikipedia.org/wiki/Cholesky_decomposition

	CompressedMatrix<double, blaze::columnMajor> ChA{ {4.0, 12.0, -16.0}, {12.0, 37.0,	 -43.0}, {-16.0, -43.0, 98.0 } };
	CompressedMatrix<double, blaze::columnMajor> L;

	Laplacians<double> lapld;
	Factorization<double> f;

	try {
		f = cholesky(ChA);
	}
	catch(std::runtime_error ex) {
		cout << ex.what();
	}

	//cout << f.Lower << endl;

	CompressedMatrix<double, blaze::columnMajor> ChA1 = f.Lower * blaze::ctrans(f.Lower);

	assert(ChA1 == ChA);

	//Test chol_subst function

	CompressedMatrix<double, blaze::columnMajor>X{ {1.0,0.0,0.0}, {2.0,0.0,0.0}, {3.,0.,0.} };

	DynamicVector<double> x;

	try {
		x = chol_subst(f.Lower, X);
	}
	catch (std::runtime_error ex) {
		cout << ex.what();
	}
	DynamicVector<double> B = ChA * x;

	//cout <<"X=\n" << x << endl;
	//cout << "B=\n" << B << endl;
		
	assert(abs(B[0] - X(0,0))<1e-6 && abs(B[1] - X(1, 0)) < 1e-6 && abs(B[2] - X(2, 0)) < 1e-6);
	
	//Calculation error

	//Create random matrices A and B
	CompressedMatrix<double, blaze::columnMajor> a{
	{-0.356543, -0.136045, -1.93844, 1.18337, -0.207743},
	{-0.67799, 1.95279, -0.193003, -1.84183, -0.662046},
	{2.61283, 1.51118, 0.672955, -0.840613, 2.01147},
	{0.859572, -0.943768, 0.375822, -1.57407, -0.858285},
	{-0.0863611, -1.47299, 1.02716, 1.904, -0.42796}
	};

	CompressedMatrix<double, blaze::columnMajor> b
	{ {1.064160977905516,0,0,0,0}, 
	{-0.3334067812850509,0,0,0,0},  
	{0.7919292830316926,0,0,0,0}, 
	{0.01651278833545206,0,0,0,0},
	{-0.6051230029995152,0,0,0,0} };

	a = a * blaze::trans(a);
	
	SolverRes<double>SolveA = lapld.wrapInterface(cholesky<double>, a);

	DynamicVector<double> b1(b.rows());

	for (int i = 0; i < b1.size(); i++)
		b1[i] = b(i, 0);

	double l2 = norm(a*SolveA(b) - b1);

	cout << "norm(ax-b)=" << l2;

	assert(abs(l2) < 2e-16);
	
	//CompressedMatrix<double, blaze::columnMajor> x=SolveA
	//auto solvea = lapld.wrapInterface([=](CompressedMatrix<double, blaze::columnMajor> &X) { return lapld.cholesky(X);});

	//solvea = wrapInterface(X->cholesky(X, Val(true)), a, maxits = 100, verbose = true)

	// norm = 7.2165330597487115e-16
}

	