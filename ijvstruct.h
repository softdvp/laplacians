#pragma once

#include <blaze/Math.h>
#include <boost/container_hash/hash.hpp>
#include "sparsecsc.h"

using namespace std;

using blaze::CompressedMatrix;
using blaze::DynamicMatrix;
using blaze::DynamicVector;

namespace laplacians {

	template <typename Tv>
	class IJV {
	public:
		std::size_t  n; // dimension
		std::size_t nnz; // number of nonzero elements
		vector<std::size_t> i; // colval
		vector<std::size_t> j; // rowval
		DynamicVector<Tv> v; //nonzero elements

		IJV() :n(0), nnz(0), i(0), j(0), v(0) {}

		const IJV operator+(const IJV &b) const {
			IJV m;

			m.n = n;
			m.nnz = nnz + b.nnz;
			m.i = i;
			m.j = j;
			m.v = v;

			//Append vectors
			m.i.insert(m.i.end(), b.i.begin(), b.i.end());
			m.j.insert(m.j.end(), b.j.begin(), b.j.end());

			m.v.resize(m.nnz);

			for (size_t i = v.size(); i < m.nnz; i++) {
				m.v[i] = b.v[i - v.size()];
			}

			return m;
		}

		const bool operator== (const IJV &b) const
		{
			bool res = n == b.n &&	nnz == b.nnz &&	i == b.i &&	j == b.j &&	v == b.v;
			return res;
		}

		const IJV operator* (const Tv x) const
		{
			IJV m;

			m.n = n;
			m.nnz = nnz;
			m.i = i;
			m.j = j;

			m.v = v * x;

			return m;
		}

		IJV(const IJV &a) {
			n = a.n;
			nnz = a.nnz;
			i = a.i;
			j = a.j;
			v = a.v;
		}

		IJV(const size_t an, const size_t annz,
			const vector<size_t> &ai,
			const vector<size_t> &aj,
			const DynamicVector<Tv> &av) {

			n = an;
			nnz = annz;

			i = ai;
			j = aj;
			v = av;
		}

		IJV(const size_t  an, const size_t annz,
			const DynamicVector<size_t> &ai,
			const DynamicVector<size_t> &aj,
			const DynamicVector<Tv> &av) {
			n = an;
			nnz = annz;

			i.insert(i.begin(), ai.begin(), ai.end());
			j.insert(j.begin(), aj.begin(), aj.end());
			v = av;
		}

		const IJV& operator=(const IJV &a) {
			n = a.n;
			nnz = a.nnz;
			i = a.i;
			j = a.j;
			v = a.v;

			return *this;
		}

		IJV(const CompressedMatrix<Tv, blaze::columnMajor> &mat) {

			n = mat.columns();
			nnz = mat.nonZeros();

			i.resize(nnz);
			j.resize(nnz);
			v.resize(nnz);

			size_t k = 0;

			//Fill i, row and v

			for (size_t l = 0UL; l < mat.rows(); ++l) {
				for (typename CompressedMatrix<Tv, blaze::columnMajor>::ConstIterator it = mat.cbegin(l); it != mat.cend(l); ++it) {

					i[k] = it->index();
					j[k] = l;
					v[k] = it->value();

					++k;
				}
			}
		}

		IJV(const SparseMatrixCSC<Tv> &cscm) {

			n = cscm.n;
			nnz = cscm.nzval.size();

			i.resize(nnz);
			j.resize(nnz);
			v.resize(nnz);

			size_t k = 0;

			for (size_t l = 0; l != n; l++) {
				size_t colbegin = cscm.colptr[l];
				size_t colend = cscm.colptr[l + 1];

				for (size_t row = colbegin; row != colend; row++) {
					i[k] = cscm.rowval[row];
					j[k] = l;
					v[k] = cscm.nzval[row];
					++k;
				}
			}
		}

		// Convert to blaze::CompressedMatrix
		CompressedMatrix<Tv, blaze::columnMajor> toCompressedMatrix() const {
			CompressedMatrix<Tv, blaze::columnMajor>res(n, n);

			res.reserve(nnz);

			for (size_t l = 0; l < nnz; ++l) {
				res(i[l], j[l]) = v[l];
			}

			return res;
		}
	};

	template <typename Tv>
	IJV<Tv> operator* (const Tv x, const IJV<Tv> &ijv) {

		return ijv * x;
	}

	template <typename Tv>
	const std::size_t nnz(const IJV<Tv> &a) {
		return a.nnz;
	}

	template <typename Tv>
	size_t hashijv(const IJV<Tv> &a) {
		size_t seed = boost::hash_range(begin(a.v), end(a.v));
		size_t seed1 = boost::hash_range(begin(a.i), end(a.i));
		size_t seed2 = boost::hash_range(begin(a.j), end(a.j));

		boost::hash_combine(seed, seed1);
		boost::hash_combine(seed, seed2);
		boost::hash_combine(seed, a.n);
		boost::hash_combine(seed, a.nnz);

		return seed;
	}

	//for tests
	template <typename Tv>
	void dump_ijv(int ijvn, IJV<Tv> &ijv) {

		cout << "ijv" << ijvn << " matrix dump:\n";

		cout << "\n" << "n= " << ijv.n;
		cout << "\n" << "nnz= " << nnz(ijv);

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

	//sparse:  convert IJV to SparseMatrixCSC
	template <typename Tv>
	SparseMatrixCSC<Tv> sparseCSC(const IJV<Tv> &ijv) {
		SparseMatrixCSC<Tv>res;
		res.m = ijv.n;
		res.n = res.m;

		size_t nnz = ijv.nnz;

		res.colptr.resize(res.n + 1);
		res.rowval.resize(nnz);
		res.nzval.resize(nnz);

		res.colptr[0] = 0;
		res.colptr[res.n] = nnz;

		size_t k = 0;

		//Fill colptr, rowval and nzval

		std::size_t totalnz = 0, t = 0;

		for (size_t l = 0UL; l < res.n; ++l) {
			std::size_t rownz = 0;

			while (t < nnz && l == ijv.j[t]) {

				res.nzval[k] = ijv.v[t];
				res.rowval[k] = ijv.i[t];
				++k;
				++rownz;
				++t;
			}
			totalnz += rownz;
			res.colptr[l + 1] = totalnz;
		}

		return res;
	}

	template <typename Tv>
	CompressedMatrix<Tv, blaze::columnMajor> sparse(const IJV<Tv> &ijv) {
		return ijv.toCompressedMatrix();
	}

	template <typename Tv>
	size_t hashijv(const IJV<Tv> &a, const unsigned h) {
		size_t seed = hashijv(a);
		boost::hash_combine(seed, h);
		return seed;
	}

	template <typename Tv>
	IJV<Tv> compress(const IJV<Tv> &ijv) {
		return IJV<Tv>(sparseCSC(ijv));
	}

	template <typename Tv>
	IJV<Tv> transpose(const IJV<Tv> &ijv) {
		return IJV<Tv>(ijv.n, ijv.nnz, ijv.j, ijv.i, ijv.v);
	}
}