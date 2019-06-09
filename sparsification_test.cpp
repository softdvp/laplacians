#include <iostream>
#include <blaze/Math.h>
#include "sparsification_test.h"
#include "graphalgs.h"
#include "sparsify.h"

using namespace std;
using namespace laplacians;

using blaze::CompressedMatrix;


void sparsification_test() {

	CompressedMatrix<double, blaze::columnMajor> G = grid2<double>(12);
	
	size_t n = G.rows();
	cout << "Dimension= " << n << endl;
	   
	CompressedMatrix<double, blaze::columnMajor> Gp = power(G, 15);
		
	CompressedMatrix<double, blaze::columnMajor> Gsparse = sparsify(Gp, 0.1);
	
	cout << "Average degree of sparsifier: " << Gsparse.nonZeros() / double(n);
}