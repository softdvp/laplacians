#pragma once

#include <iostream>
#include <blaze/Math.h>
#include <blaze/math/Submatrix.h>
#include <blaze/math/Subvector.h>
#include <vector>

using namespace std;
using blaze::CompressedMatrix;
using blaze::DynamicMatrix;
using blaze::DynamicVector;
using blaze::columnwise;
using blaze::rowwise;

template<typename Tv>
vector<size_t> flipIndex(const CompressedMatrix<Tv, blaze::columnMajor> &A) {
	CompressedMatrix<size_t, blaze::columnMajor>B(A.rows(), A.columns());

	size_t k = 1;

	for (size_t i = 0; i < A.rows(); i++)
		for (auto it = A.cbegin(i); it != A.cend(i); ++it) {
			size_t j = it->index();
			B(i, j) = k;
			++k;
		}

	CompressedMatrix<size_t, blaze::columnMajor> bakMat = ctrans(B);

	vector<size_t> resv;
	for (size_t i = 0; i < B.rows(); i++)
		for (auto it = B.cbegin(i); it != B.cend(i); ++it) {
			size_t v = it->value();

			resv.push_back(v);
		}

	return resv;
}

