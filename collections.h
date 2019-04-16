#pragma once
#include <blaze/Math.h>
#include <blaze/math/Submatrix.h>
#include <iostream>
#include <boost/container_hash/hash.hpp>
#include <vector>

using blaze::CompressedMatrix;
using blaze::DynamicVector;
using blaze::columnwise;
using blaze::rowwise;
using std::vector;

// Julia sparse matrix class
template <typename Tv>
class SparseMatrixCSC{
public:
	size_t m;
	size_t n;
	vector<size_t> colptr;
	vector<size_t> rowval;
	vector<Tv> nzval;

	SparseMatrixCSC(size_t am, size_t an, vector<size_t> &acolptr, vector<size_t> arowval, vector<Tv> anzval):
		m(am), n(an), colptr(acolptr), rowval(arowval), nzval(anzval){}

	SparseMatrixCSC() {}

	//Convert from blaze::CompressedMatrix
	SparseMatrixCSC(CompressedMatrix<Tv, blaze::columnMajor> &mat) {
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

template <typename Tv>
class IJV {
public:
	std::size_t  n;
	std::size_t nnz;
	vector<std::size_t> i; //colptr
	vector<std::size_t> j; //rowval
	vector<Tv> v; //nonzero elements

	IJV():n(0), nnz(0), i(0), j(0), v(0){}

	const IJV operator+(const IJV &b) const {
		IJV m;
		m.n = n;
		m.nnz = nnz + b.nnz;

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
				
		for(auto& i:v)
			m.v.push_back(i *x);

		return m;
	}

	IJV(const IJV &a) {
		n = a.n;
		nnz = a.nnz;
		i = a.i;
		j = a.j;
		v = a.v;
	}

	IJV(const size_t  an,	const size_t annz,
	const vector<size_t> &ai,
	const vector<size_t> &aj,
	const vector<Tv> &av) {
		n = an;
		nnz = annz;
		i = ai;
		j = aj;
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

				i[k]= it->index();  
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
	CompressedMatrix<Tv, blaze::columnMajor> toCompressedMatrix() {
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
bool testZeroDiag(const Tv &a) {

	size_t n = a.rows();

	for (size_t i = 0; i < n; i++) {
		if (abs(a(i, i)) > 1E-9)
			return false;
	}

	return true;
}


template <typename Tv>
class Laplacians {
public:

	const std::size_t nnz(const IJV<Tv> &a) {
		return a.nnz;
	}

	//sparse:  convert IJV to SparseMatrixCSC
	SparseMatrixCSC<Tv> sparse(const IJV<Tv> &ijv) {
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

	size_t hashijv(const IJV<Tv> &a, const unsigned h) {
		size_t seed = hashijv(a);
		boost::hash_combine(seed, h);
		return seed;
	}

	IJV<Tv> compress(const IJV<Tv> &ijv) {
		return IJV<Tv>(sparse(ijv));
	}

	IJV<Tv> transpose(const IJV<Tv> &ijv) {
		return IJV<Tv>(ijv.n, ijv.nnz, ijv.j, ijv.i, ijv.v);
	}

	// Retutns vector where comp[i]=component number
	vector<size_t> components(CompressedMatrix<Tv, blaze::columnMajor> &mat) {
		size_t n = mat.rows();

		vector<size_t> order(n);
		vector<size_t> comp(n);

		size_t color = 0;

		for (size_t x = 0UL; x < mat.rows(); ++x) {
			if (!comp[x]) { //not used
				comp[x] = ++color; //insert new color

				if (mat.begin(x) != mat.end(x)) {
					size_t ptr = 0, orderLen = 1;
					order[ptr] = x;

					while (ptr < orderLen) {
						size_t curNode = order[ptr]; // initial curNode=x

						for (auto it = mat.begin(curNode); it != mat.end(curNode); ++it) {
							size_t nbr = it->index();
							if (!comp[nbr]) { //not used
								comp[nbr] = color; // insert current component
								order[orderLen] = nbr;
								++orderLen;
							}
						}
						++ptr;
					}
				}
			}
		}
		return comp;
	}

	vector<size_t> components(const SparseMatrixCSC<Tv> &mat) {
		size_t n = mat.n;

		vector<size_t> order(n);
		vector<size_t> comp(n);

		size_t color = 0;

		for (size_t x = 0; x != n; x++) {
			if (!comp[x]) { //not used
				comp[x] = ++color; //insert new color

				if (mat.colptr[x + 1] > mat.colptr[x]) {
					size_t ptr = 0, orderLen = 1;

					order[ptr] = x;

					while (ptr < orderLen) {
						size_t curNode = order[ptr]; // initial curNode=x

						for (size_t ind = mat.colptr[curNode]; ind != mat.colptr[curNode + 1]; ++ind) { // cycle by rows
							size_t nbr = mat.rowval[ind]; //nbr=row
							if (!comp[nbr]) { //not used
								comp[nbr] = color; // insert current component
								order[orderLen] = nbr;
								++orderLen;
							}
						}
						++ptr;
					}
				}
			}
		}
		return comp;
	}

	IJV<Tv> path_graph_ijv(int n) {
		IJV<Tv> ijv;
		ijv.n = n;
		ijv.nnz = 2 * (n - 1);

		ijv.i.resize(ijv.nnz);
		ijv.j.resize(ijv.nnz);
		ijv.v.resize(ijv.nnz);

		size_t z = 0;

		for (size_t i = 0; i < ijv.n; i++) {
			long long k = i - 1;
			size_t l = i + 1;
			if (k >= 0) {
				ijv.i[z] = i;
				ijv.j[z] = k;
				ijv.v[z] = 1;
				++z;
			}

			if (l < n) {
				ijv.i[z] = i;
				ijv.j[z] = l;
				ijv.v[z] = 1;
				++z;
			}
		}
		return ijv;
	}

	//Kronecker product
	//https://en.wikipedia.org/wiki/Kronecker_product

	CompressedMatrix<Tv, blaze::columnMajor>kron(const CompressedMatrix<Tv,
		blaze::columnMajor> &A, const CompressedMatrix<Tv, blaze::columnMajor> &B) {
		CompressedMatrix<Tv, blaze::columnMajor> Res(A.rows() * B.rows(), A.columns() * B.columns());

		for (size_t i = 0; i < A.rows(); i++)
			for (auto it = A.cbegin(i); it != A.cend(i); ++it){
				size_t j = it->index();
				auto sbm = submatrix(Res, i*B.rows(), j*B.columns(), B.rows(), B.columns());

				sbm = A(i, j) * B;
			}

		return Res;
	}

	//Conjugate transpose
	CompressedMatrix<Tv, blaze::columnMajor>adjoint(const CompressedMatrix<Tv, blaze::columnMajor> &A) {
		CompressedMatrix<Tv, blaze::columnMajor> Res=blaze::ctrans(A);

		return Res;
	}

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

	vector<Tv>diag(const CompressedMatrix<Tv, blaze::columnMajor> &A, size_t diag_n=0) {

		vector<Tv> resv;

		size_t it = A.columns() - diag_n;

		for (size_t i = 0; i < it; ++i) {
			resv.push_back(A(i, diag_n + i));
		}

		return resv;
	}

	CompressedMatrix<Tv, blaze::columnMajor>Diagonal(const DynamicVector<Tv> &V) {
		
		size_t vsize = V.size();

		CompressedMatrix<Tv, blaze::columnMajor>Res(vsize, vsize);

		Res.reserve(vsize);

		for (size_t i = 0; i < vsize; i++) {
			Res.append(i, i, V[i]);
			Res.finalize(i);
		}

		return Res;
	}

	DynamicVector<Tv>dynvec(const vector<Tv> &v) {
		DynamicVector<Tv>res(v.size());

		for (auto i = 0; i < v.size(); ++i) {
			res[i] = v[i];
		}

		return res;
	}

	DynamicVector<Tv>sum(const CompressedMatrix<Tv, blaze::columnMajor> &A, bool row_wise=true) {
		DynamicVector<Tv>res;
		
		if (!row_wise)
			res = blaze::sum<rowwise>(A);
		else
			res = trans(blaze::sum<columnwise>(A));

		return res;
	}

	//Returns the diagonal weighted degree matrix(as a sparse matrix) of a graph
	CompressedMatrix<Tv, blaze::columnMajor>diagmat(const CompressedMatrix<Tv, blaze::columnMajor> &A){
			
		return Diagonal(sum(A));
	}

	CompressedMatrix<Tv, blaze::columnMajor>pow(const CompressedMatrix<Tv, blaze::columnMajor> &A, const int n) {
		CompressedMatrix<Tv, blaze::columnMajor> res;
		
		res = A;
		for (size_t i = 0; i < n-1; i++) {
			res *= A;
		}

		return res;
	}

	CompressedMatrix<Tv, blaze::columnMajor>power(const CompressedMatrix<Tv, blaze::columnMajor> &A, const int k) {
		CompressedMatrix<Tv, blaze::columnMajor>ap;

		ap = pow(A, k);
		ap = ap - Diagonal(dynvec(diag(ap)));

		return ap;
	}
};




