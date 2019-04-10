#include <iostream>
#include <assert.h>
#include <blaze/Math.h>
#include "pcg_1.h"
#include "collections.h"

using blaze::DynamicVector;
using blaze::CompressedMatrix;
using namespace std;

#include "tests.h"

void bzbetatest() {
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
}

void dump_ijv(int ijvn, IJV<int> &ijv ) {
	Laplacians<int> lapl;

	cout << "ijv"<<ijvn<<" matrix dump:\n";

	cout << "\n" << "n= " << ijv.n;
	cout << "\n" << "nnz= " << lapl.nnz(ijv);

	cout << "\ni=";
	for (size_t k = 0; k < ijv.nnz; ++k)
		cout << ijv.i[k] << " ";

	cout << "\n" << "j=";
	for (size_t k = 0; k < ijv.nnz; ++k)
		cout << ijv.j[k] << " ";

	cout << "\n" << "v= ";
	for (size_t k = 0; k < ijv.nnz; ++k)
		cout << ijv.v[k] << " ";
		
}

void IJVtests() {
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

	/* for column major (default) matrix:
	CSCMx.m=4
	CSCMx.n=4
	CSCMx.colptr.i= 0 1 3 4 4
	CSCMx.rowval= 1 1 3 2
	CSCMx.nzval.v= 5 8 6 3
	*/

	cout << endl << "\nTest toCompressedMatrix():\n";
	CompressedMatrix<int, blaze::columnMajor> mxtest;

	mxtest = CSCMx.toCompressedMatrix();
	cout << mxtest;

	assert(m == mxtest);
			
	cout << endl << "Convert from blaze compressed matrix to IJV structure.\n\n";
	IJV<int> ijv0(m);

	dump_ijv(0, ijv0);
	
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

	CSCMx1 = lapl.sparse(ijv0);

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

	assert(CSCMx == CSCMx);

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

	//Change ijv1
	ijv1.v[0] = 10;

	t = ijv1 == ijv0;
	cout << "\n\noperator== :\n ijv1==ijv: " << (t ? "true" : "false");

	cout << "\n\noperator* :\n ijv * 5";

	ijv1 = ijv0*5;

	cout << "\n" << "ijv1.v= ";
	for (size_t k = 0; k < ijv1.nnz; ++k)
		cout << ijv1.v[k] << " ";

	cout << "\n\noperator* :\n 5 * ijv";

	ijv1 = 5 * ijv0;

	cout << "\n" << "ijv1.v= ";
	for (size_t k = 0; k < ijv1.nnz; ++k)
		cout << ijv1.v[k] << " ";

	cout << "Test a constructor";

	IJV<int> ijv2(ijv0.n, ijv0.nnz, ijv0.i, ijv0.j, ijv0.v);

	dump_ijv(2, ijv2);

	cout << "\n" << "nnz= " << lapl.nnz(ijv2);

	cout << "\n\nTest IJV::ToCompressMatrix():\n";

	CompressedMatrix<int, blaze::columnMajor> newm=ijv0.toCompressedMatrix();
	cout << newm;

	assert(newm == m);

	cout << "\n\nTest hash(IJV) function:\n";
	cout <<"hash(ijv): " << lapl.hashijv(ijv0)<<endl;
	cout << "hash(ijv, 5): " << lapl.hashijv(ijv0, 5) << endl;

	cout << "\n\nTest compress(IJV) function:\n";

	IJV<int>ijv3 = lapl.compress(ijv0);

	dump_ijv(3, ijv3);

	assert(ijv0 == ijv3);

	cout << endl << "\nTranspose ijv:\n ";
	IJV<int> ijv4 = lapl.transpose(ijv0);

	dump_ijv(4, ijv4);

	cout << endl << ijv4.toCompressedMatrix()<<endl;

//	cout << endl<<"Test a SparseMatrics constructor:\n";
	SparseMatrixCSC<int> CSCMx2(CSCMx.m, CSCMx.n, CSCMx.colptr, CSCMx.rowval, CSCMx.nzval);

	assert(CSCMx2 == CSCMx);

}

void CollectionTest() {
	CompressedMatrix<int, blaze::columnMajor> m, m1;

	//Test path_graph_ijv
	Laplacians<int> lapl;

	IJV<int> ijv = lapl.path_graph_ijv(5);

	dump_ijv(0, ijv);
	m1 = ijv.toCompressedMatrix();
	cout << endl << endl << m1 << endl;

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

	SparseMatrixCSC<int> sprs(m10);
	vector<size_t> comp = lapl.components(sprs);
	
	for (int i = 0; i < comp.size(); i++)
		cout << comp[i] << " ";

	cout << endl << endl;
	cout << "Call function componets():\n";

	vector<size_t>comp1 = lapl.components(m10);

	for (int i = 0; i < comp1.size(); i++)
		cout << comp[i] << " ";

	assert(comp == comp1);

	//Out = 1, 1, 1, 1, 1, 1, 2, 1, 1, 3
	
	cout << endl << endl;
	
	//Test Kronecker product function kron(A, B)

	CompressedMatrix<int, blaze::columnMajor> A(2,2), B(2,2), C;

	//Create two occasional matrices
	//See an example at https://en.wikipedia.org/wiki/Kronecker_product

	A(0, 0) = 1; A(0, 1) = 2; A(1, 0) = 3; A(1, 1) = 4;
	B(0, 1) = 5; B(1, 0) = 6; B(1, 1) = 7;


	C = lapl.kron(A, B);
	cout << "kron(A, B)=\n" << C;

/*Out:
	kron(A, B)=

	0   5   0  10
	6   7  12  14
	0  15   0  20
	18  21  24  28
	
	*/

	//Test flipIndex

	cout << endl << endl << "flipIndex(C)=\n";

	vector<size_t> v = lapl.flipIndex(C);

	for (int i = 0; i < v.size(); i++) {
		cout << v[i] << " ";
	}

	//Out="flipIndex(C)=3, 9, 1, 4, 7, 10, 5, 11, 2, 6, 8, 12"

	//Test function diag()

	cout << endl << endl << "diag(C)=\n";

	vector<int> v1 = lapl.diag(C, 1);

	for (int i = 0; i < v1.size(); i++) {
		cout << v1[i] << " ";
	}

	//Out: diag(C) = 5 12 20

	//Test function Diagonal()

	CompressedMatrix<int, blaze::columnMajor>Dg = lapl.Diagonal(v1);

	cout << endl << endl << "Diagonal(v)=\n" << Dg;

	/* Out:
	
	5  0 0
	0 12 0
	0 20 0
	
	*/
}