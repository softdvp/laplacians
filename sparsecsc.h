#pragma once

#include <blaze/Math.h>
#include <vector>

using namespace std;
using blaze::CompressedMatrix;
using blaze::DynamicMatrix;
using blaze::DynamicVector;

namespace laplacians {

	// Julia sparse matrix class
	template <typename Tv>
	class SparseMatrixCSC {
	public:
		size_t m;
		size_t n;
		vector<size_t> colptr; // vector type for size_t
		vector<size_t> rowval; // vector type for size_t
		DynamicVector<Tv> nzval; // DynamicVector type for Tv type

		SparseMatrixCSC(size_t am, size_t an, vector<size_t> &acolptr, vector<size_t> arowval, DynamicVector<Tv> anzval) :
			m(am), n(an), colptr(acolptr), rowval(arowval), nzval(anzval) {}

		SparseMatrixCSC() {}

		//Convert from blaze::CompressedMatrix
		SparseMatrixCSC(const CompressedMatrix<Tv, blaze::columnMajor> &mat) {
			m = mat.rows();
			n = mat.columns();

			size_t nnz = mat.nonZeros();

			colptr.resize(n + 1);
			colptr[0] = 0;
			colptr[n] = nnz;

			rowval.resize(nnz);
			nzval.resize(nnz);

			size_t k = 0;

			//Fill colptr, rowval and nzval

			std::size_t totalnz = 0;

			for (size_t l = 0UL; l < mat.rows(); ++l) {
				std::size_t rownz = 0;

				for (typename CompressedMatrix<Tv, blaze::columnMajor>::ConstIterator it = mat.cbegin(l); it != mat.cend(l); ++it) {

					nzval[k] = it->value();
					rowval[k] = it->index();
					++k;
					++rownz;
				}
				totalnz += rownz;
				colptr[l + 1] = totalnz;
			}
		}

		// Convert to blaze::CompressedMatrix
		CompressedMatrix<Tv, blaze::columnMajor> toCompressedMatrix() {
			CompressedMatrix<Tv, blaze::columnMajor>res(n, n);

			size_t nnz = nzval.size();
			res.reserve(nnz);

			for (size_t i = 0; i != n; i++) {
				size_t colbegin = colptr[i];
				size_t colend = colptr[i + 1];

				for (size_t row = colbegin; row != colend; row++) {
					size_t rowv = rowval[row];
					Tv v = nzval[row];
					res.append(rowv, i, v);
				}
				res.finalize(i);
			}
			return res;
		}

		const bool operator== (const SparseMatrixCSC<Tv> &b) const {
			return 	m == b.m && n == b.n && colptr == b.colptr && rowval == b.rowval && nzval == b.nzval;
		}
	};
}