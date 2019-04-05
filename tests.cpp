#include <iostream>
#include <blaze/Math.h>
#include "pcg_1.h"
#include "collections.h"

using blaze::DynamicVector;
using blaze::CompressedMatrix;

#include "tests.h"

void bzbetatest() {
	DynamicVector<double>a{ 1, 2, 3, 4 };
	DynamicVector<double>b{ 5, 4, 0, 1 };
	
	double beta = 2.3;
	bzbeta(beta, a, b);

	std::cout << "Test of bzbeta function: "<< "\n" << a << "\n";
	//Out: (7.3, 8.6, 6.9, 10.2)
}

void axpy2test() {
	DynamicVector<int>a{ 1, 2, 3, 4 };
	DynamicVector<int>b{ 5, 4, 0, 1 };

	int al = 4;
	axpy2(al, a, b);

	std::cout << "Test of axpy2 function: " << "\n" << b << "\n";
	//Out: 3, 12, 12, 17
}

void IJVtests() {
	//Convert a BLAZE sparse matrix to an IJV

	// A sample of  matrix taken from 
	// https://en.wikipedia.org/wiki/Sparse_matrix
	CompressedMatrix<int, blaze::columnMajor> m{ {0,0,0,0}, {5,8,0,0}, {0,0,3,0}, {0,6,0,0} };
	IJV<int> ijv(m), ijv1;

	std::cout << "ijv.i= ";

	for (std::size_t k = 0; k <= ijv.n; ++k)
		std::cout << ijv.i[k] << " ";

	std::cout <<"\n"<< "ijv.j= ";
	for (std::size_t k = 0; k < ijv.nnz; ++k)
		std::cout << ijv.j[k] << " ";

	std::cout << "\n" << "ijv.v= ";
	for (std::size_t k = 0; k < ijv.nnz; ++k)
		std::cout << ijv.v[k] << " ";

	std::cout << "\n" << "nnz= "<<nnz(ijv);

	//Out: jvi.i= 0 0 2 3 4 
	//     for row major matrix:
	//	   ijv.j= 0 1 2 1
	//	   ijv.v= 5 8 3 6	
	//     nnz= 4

	// for column major (default) matrix:
	// ijv.i= 0 1 3 4 4
	// ijv.j= 1 1 3 2
	// ijv.v= 5 8 6 3
	// nnz= 4

	std::cout << "\n";

	CompressedMatrix<int, blaze::columnMajor> newm = sparse(ijv);

	std::cout << "\n Original matrix:\n "<< m <<"\n";
	std::cout << "\n Matrix after convertions:\n " << newm << "\n";

	std::cout << std::endl;

	std::cout << "Test overloaded operators.";

	ijv1 = ijv;

	std::cout << "operator= :\n";

	std::cout << "ijv1.i= ";

	for (std::size_t k = 0; k <= ijv1.n; ++k)
		std::cout << ijv1.i[k] << " ";

	std::cout << "\n" << "ijv1.j= ";
	for (std::size_t k = 0; k < ijv1.nnz; ++k)
		std::cout << ijv1.j[k] << " ";

	std::cout << "\n" << "ijv1.v= ";
	for (std::size_t k = 0; k < ijv1.nnz; ++k)
		std::cout << ijv1.v[k] << " ";

	std::cout << "\n" << "nnz= " << nnz(ijv1);

	bool t = ijv1 == ijv;
	std::cout << "\noperator== :\n ijv1==ijv: " << (t ? "true" : "false");

	//Change ijv1
	ijv1.v[0] = 10;

	t = ijv1 == ijv;
	std::cout << "\noperator== :\n ijv1==ijv: " << (t ? "true" : "false");

	std::cout << "\noperator* :\n ijv*5";

	ijv1 = ijv*5;

	std::cout << "\n" << "ijv1.v= ";
	for (std::size_t k = 0; k < ijv1.nnz; ++k)
		std::cout << ijv1.v[k] << " ";

	std::cout << "\noperator* :\n ijv*5";

	ijv1 = 5 * ijv;

	std::cout << "\n" << "ijv1.v= ";
	for (std::size_t k = 0; k < ijv1.nnz; ++k)
		std::cout << ijv1.v[k] << " ";



}