#pragma once
#include <blaze/Math.h>
#include <blaze/math/Submatrix.h>
#include <blaze/math/Subvector.h>
#include <blaze/math/row.h>
#include <blaze/math/column.h>

#include <boost/container_hash/hash.hpp>

#include <iostream>
#include <vector>
#include <chrono>
#include <utility>
#include <functional>

#include "graphalgs.h"
#include "approxchol.h"


using blaze::CompressedMatrix;
using blaze::DynamicMatrix;
using blaze::DynamicVector;
using blaze::columnwise;
using blaze::rowwise;
using std::vector;
using namespace std::chrono;

template<typename Tv>
std::tuple<vector<size_t>, vector<size_t>, DynamicVector<Tv>> findnz(const CompressedMatrix<Tv, blaze::columnMajor> &mat) {

	size_t nnz = mat.nonZeros();

	vector<size_t> i(nnz), j(nnz);
	DynamicVector<Tv>v(nnz);
	
	size_t k = 0;

	//Fill i, row and v

	for (size_t l = 0UL; l < mat.rows(); ++l) {
		for (auto it = mat.cbegin(l); it != mat.cend(l); ++it) {

			i[k] = it->index();
			j[k] = l;
			v[k] = it->value();

			++k;
		}
	}

	return make_tuple(i, j, v);
}

template <typename Tv>
class IJV {
public:
	std::size_t  n;
	std::size_t nnz;
	vector<std::size_t> i; //colptr
	vector<std::size_t> j; //rowval
	DynamicVector<Tv> v; //nonzero elements

	IJV():n(0), nnz(0), i(0), j(0), v(0){}

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
			m.v[i] = b.v[i-v.size()];
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
	CompressedMatrix<Tv, blaze::columnMajor> toCompressedMatrix() const {
		CompressedMatrix<Tv, blaze::columnMajor>res(n, n);

		res.reserve(nnz);

		for (size_t l = 0; l < nnz; ++l) {
			res(i[l], j[l]) = v[l];
		}

		return res;
	}
};

//for tests
void dump_ijv(int ijvn, IJV<int> &ijv);

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
DynamicVector<Tv>dynvec(const vector<Tv> &v) {
	DynamicVector<Tv>res(v.size());

	for (auto i = 0; i < v.size(); ++i) {
		res[i] = v[i];
	}
	return res;
}

const vector<vector<size_t>> vecToComps(vector<size_t> &compvec);

template <typename Tv>
class Factorization {
public:
	CompressedMatrix<Tv, blaze::columnMajor> Lower;
};

//  Solvers for A*x=B where B is a matrix

//Result of wrappers
//Function: pass B matrix, returns x vector

template <typename Tv>
using SolverBMat = function<DynamicVector<Tv>(const CompressedMatrix<Tv, blaze::columnMajor>&)>;

//Result of SolverA functor
//Convert SolverA to a function with 1 paramater B - SolverB
template <typename Tv>
using SubSolverFuncMat = std::function <DynamicVector<Tv>(const CompressedMatrix<Tv, blaze::columnMajor>&, vector<size_t>&, float, double,
	double, bool, ApproxCholParams)>;

template <typename Tv>
class SubSolverMat {
	SubSolverFuncMat<Tv> Solver;
public:
	SubSolverMat(SubSolverFuncMat<Tv> Asolver) : Solver(Asolver) {};

	SubSolverMat(SolverBMat<Tv> solver) {

		Solver = [=](const CompressedMatrix<Tv, blaze::columnMajor> &b, vector<size_t>& pcgIts,
			float tol = 1e-6, double maxits = HUGE_VAL, double maxtime = HUGE_VAL, bool verbose = false,
			ApproxCholParams params = ApproxCholParams()) {

			return solver(b);
		};
	}

	DynamicVector<Tv>operator()(const CompressedMatrix<Tv, blaze::columnMajor> &b, vector<size_t>& pcgIts,
		float tol = 1e-6, double maxits = HUGE_VAL, double maxtime = HUGE_VAL, bool verbose = false,
		ApproxCholParams params = ApproxCholParams()) {

		return Solver(b, pcgIts, tol, maxits, maxtime, verbose, params);
	}

	DynamicVector<Tv>operator()(const CompressedMatrix<Tv, blaze::columnMajor> &b){
		vector<size_t> pcgIts;
		
		return Solver(b, pcgIts, 1e-6F, HUGE_VAL, HUGE_VAL, false, ApproxCholParams());
	}
};

// Function: pass A matrix, return SubSolver
template <typename Tv>
using SolverAFuncMat = std::function<SubSolverMat<Tv>(const CompressedMatrix<Tv, blaze::columnMajor>&, vector<size_t>&, float, double,
	double, bool, ApproxCholParams)>;

template <typename Tv>
class SolverAMat {
	SolverAFuncMat<Tv> Solver;
public:
	SolverAMat(SolverAFuncMat<Tv> solver): Solver(solver){}

	SubSolverMat<Tv> operator()(const CompressedMatrix<Tv, blaze::columnMajor> &a, vector<size_t>& pcgIts,
		float tol = 1e-6, double maxits = HUGE_VAL, double maxtime = HUGE_VAL, bool verbose = false,
		ApproxCholParams params = ApproxCholParams()) {
		
		return Solver(a, pcgIts, tol, maxits, maxtime, verbose, params);
	}

	SubSolverMat<Tv> operator()(const CompressedMatrix<Tv, blaze::columnMajor> &a) {
		vector<size_t> pcgIts;
		return Solver(a, pcgIts, 1e-6F, HUGE_VAL, HUGE_VAL, false, ApproxCholParams());
	}
};

// Function: pass A matrix, return A matrix factorization

template <typename Tv>
using FactorSolver = std::function<Factorization<Tv>(const CompressedMatrix<Tv, blaze::columnMajor>&)>;
 
template <typename Tv>
Factorization<Tv> cholesky(const CompressedMatrix<Tv, blaze::columnMajor> &A) {
	DynamicMatrix<Tv, blaze::columnMajor> A1(A), L;
	Factorization<Tv> F;

	blaze::llh(A1, L);

	F.Lower = L;

	return F;
}

//  Solvers for A*x=b where b is a vector 

//Result of wrappers
//Function: pass B matrix, returns x vector

template <typename Tv>
using SolverB = function<DynamicVector<Tv>(const DynamicVector<Tv>&)>;

//Result of SolverA functor
//Convert SolverA to a function with 1 paramater B - SolverB

template <typename Tv>
using SubSolverFunc = std::function <DynamicVector<Tv>(const DynamicVector<Tv>&, vector<size_t>&, float, double,
	double, bool, ApproxCholParams)>;

template <typename Tv>
class SubSolver {
	SubSolverFunc<Tv> Solver;
public:
	SubSolver(){}

	SubSolver(SubSolverFunc<Tv> Asolver) : Solver(Asolver) {};

	SubSolver(SolverB<Tv> solver) {

		Solver = [=](const DynamicVector<Tv> &b, vector<size_t>& pcgIts,
			float tol = 1e-6, double maxits = HUGE_VAL, double maxtime = HUGE_VAL, bool verbose = false,
			ApproxCholParams params = ApproxCholParams()) {

			return solver(b);
		};
	}

	/*SubSolver &operator=(const SubSolver &s) {
		Solver = s.Solver;

		return *this;
	}*/

	DynamicVector<Tv>operator()(const DynamicVector<Tv> &b, vector<size_t>& pcgIts,
		float tol = 1e-6, double maxits = HUGE_VAL, double maxtime = HUGE_VAL, bool verbose = false,
		ApproxCholParams params = ApproxCholParams()) {
		return Solver(b, pcgIts, tol, maxits, maxtime, verbose, params);
	}

	DynamicVector<Tv>operator()(const DynamicVector<Tv> &b){
		vector<size_t> pcgIts;
		return Solver(b, pcgIts, 1e-6F, HUGE_VAL, HUGE_VAL, false, ApproxCholParams());
	}
};

// Function: pass A matrix, return SubSolver
template <typename Tv>
using SolverAFunc = std::function<SubSolver<Tv>(const CompressedMatrix<Tv, blaze::columnMajor>&, vector<size_t>&, float, double,
	double, bool, ApproxCholParams)>;

template <typename Tv>
class SolverA {
	SolverAFunc<Tv> Solver;
public:
	SolverA(SolverAFunc<Tv> solver): Solver(solver){}

	SubSolver<Tv> operator()(const CompressedMatrix<Tv, blaze::columnMajor> &a, vector<size_t>& pcgIts,
		float tol = 1e-6, double maxits = HUGE_VAL, double maxtime = HUGE_VAL, bool verbose = false,
		ApproxCholParams params = ApproxCholParams()) {

		return Solver(a, pcgIts, tol, maxits, maxtime, verbose, params);
	}

	SubSolver<Tv> operator()(const CompressedMatrix<Tv, blaze::columnMajor> &a) {
		vector<size_t> pcgIts;
		return Solver(a, pcgIts, 1e-6F, HUGE_VAL, HUGE_VAL, false, ApproxCholParams());
	}
};

template <typename Tv>
DynamicVector<Tv> nullSolver(const DynamicVector<Tv>& a, vector<size_t>& pcg, float f, double d,
	double d1, bool b, ApproxCholParams prm) {

	return DynamicVector<Tv>(1, 0);
}

//Cholesky-based Substitution

template <typename Tv>
DynamicVector<Tv> chol_subst(const CompressedMatrix<Tv, blaze::columnMajor> &Lower, const CompressedMatrix<Tv, blaze::columnMajor> &B) {
	DynamicVector<Tv> res(B.rows());
	DynamicMatrix<Tv, blaze::columnMajor> B1 = B, L=Lower;

	potrs(L, B1, 'L');

	res = column(B1, 0);

	return res;
}

template <typename Tv>
DynamicVector<Tv> chol_subst(const CompressedMatrix<Tv, blaze::columnMajor> &Lower, const DynamicVector<Tv> &b) {
	
	DynamicMatrix<Tv, blaze::columnMajor> L = Lower;
	DynamicVector<Tv> b1 = b;

	potrs(L, b1, 'L');

	return b1;
}

template <typename Tv>
class Laplacians {
public:

	const std::size_t nnz(const IJV<Tv> &a) {
		return a.nnz;
	}

	//sparse:  convert IJV to SparseMatrixCSC
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

	CompressedMatrix<Tv, blaze::columnMajor> sparse(const IJV<Tv> &ijv) const {
		return ijv.toCompressedMatrix();
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
		return IJV<Tv>(sparseCSC(ijv));
	}

	IJV<Tv> transpose(const IJV<Tv> &ijv) {
		return IJV<Tv>(ijv.n, ijv.nnz, ijv.j, ijv.i, ijv.v);
	}

	// Returns vector where comp[i]=component number
	vector<size_t> components(const CompressedMatrix<Tv, blaze::columnMajor> &mat) {
		size_t n = mat.rows();

		vector<size_t> order(n, 0);
		vector<size_t> comp(n, 0);

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

		vector<size_t> order(n, 0);
		vector<size_t> comp(n, 0);

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

	IJV<Tv> path_graph_ijv(size_t n) {
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
	//for matrices: https://en.wikipedia.org/wiki/Kronecker_product

	CompressedMatrix<Tv, blaze::columnMajor> kron(const CompressedMatrix<Tv,
		blaze::columnMajor> &A, const CompressedMatrix<Tv, blaze::columnMajor> &B) {
		CompressedMatrix<Tv, blaze::columnMajor> Res(A.rows() * B.rows(), A.columns() * B.columns());

		for (size_t i = 0; i < A.rows(); i++)
			for (auto it = A.cbegin(i); it != A.cend(i); ++it) {
				size_t j = it->index();
				auto sbm = submatrix(Res, i*B.rows(), j*B.columns(), B.rows(), B.columns());

				sbm = A(i, j) * B;
			}

		return Res;
	}

	// for vectors
	DynamicVector<Tv> kron(const DynamicVector<Tv> &A, const DynamicVector<Tv>&B) {
		DynamicVector<Tv> res(A.size()*B.size());

		for (size_t i = 0; i < A.size(); i++) {
			subvector(res, i*B.size(), B.size()) = A[i] * B;
		}

		return res;
	}

	//Conjugate transpose
	CompressedMatrix<Tv, blaze::columnMajor> adjoint(const CompressedMatrix<Tv, blaze::columnMajor> &A) {
		CompressedMatrix<Tv, blaze::columnMajor> Res = blaze::ctrans(A);

		return Res;
	}


	DynamicVector<Tv> diag(const CompressedMatrix<Tv, blaze::columnMajor> &A, size_t diag_n = 0) {

		size_t it = A.columns() - diag_n;
		DynamicVector<Tv> resv(it);

		for (size_t i = 0; i < it; ++i) {
			resv[i] = (A(i, diag_n + i));
		}

		return resv;
	}

	CompressedMatrix<Tv, blaze::columnMajor> Diagonal(const DynamicVector<Tv> &V) {

		size_t vsize = V.size();

		CompressedMatrix<Tv, blaze::columnMajor>Res(vsize, vsize);

		Res.reserve(vsize);

		for (size_t i = 0; i < vsize; i++) {
			Res.append(i, i, V[i]);
			Res.finalize(i);
		}

		return Res;
	}

	DynamicVector<Tv> sum(const CompressedMatrix<Tv, blaze::columnMajor> &A, int wise = 1) {
		DynamicVector<Tv> res;

		if (wise == 1)
			res = blaze::sum<rowwise>(A);
		else
			if (wise == 2)
				res = trans(blaze::sum<columnwise>(A));
			else throw("The value of wise parameter must be 1 or 2.");

		return res;
	}

	//Returns the diagonal weighted degree matrix(as a sparse matrix) of a graph
	CompressedMatrix<Tv, blaze::columnMajor> diagmat(const CompressedMatrix<Tv, blaze::columnMajor> &A) {

		return Diagonal(sum(A));
	}

	CompressedMatrix<Tv, blaze::columnMajor> pow(const CompressedMatrix<Tv, blaze::columnMajor> &A, const int n) {
		CompressedMatrix<Tv, blaze::columnMajor> res;

		res = A;
		for (size_t i = 0; i < n - 1; i++) {
			res *= A;
		}

		return res;
	}

	CompressedMatrix<Tv, blaze::columnMajor> power(const CompressedMatrix<Tv, blaze::columnMajor> &A, const int k) {
		CompressedMatrix<Tv, blaze::columnMajor>ap;

		ap = pow(A, k);
		ap = ap - Diagonal(diag(ap));

		return ap;
	}

	IJV<Tv> product_graph(IJV<Tv> b, IJV<Tv> a) {

		size_t n = a.n *b.n;

		assert(a.i.size() == a.nnz);

		DynamicVector<Tv>bncollect(b.n), ancollect(a.n), annzOnes(a.nnz, 1),
			bnnzOnes(b.nnz, 1), anOnes(a.n, 1), bnOnes(b.n, 1);

		for (size_t i = 0; i < b.n; ++i)
			bncollect[i] = (Tv)i;

		for (size_t i = 0; i < a.n; ++i)
			ancollect[i] = (Tv)i;

		DynamicVector<Tv>ait(a.i.size());

		for (size_t i = 0; i < a.i.size(); ++i) {
			ait[i] = (Tv)a.i[i] + 1;
		}

		DynamicVector<Tv>ajt(a.j.size());

		for (size_t i = 0; i < a.j.size(); ++i) {
			ajt[i] = (Tv)a.j[i] + 1;
		}

		DynamicVector<Tv>bit(b.i.size());

		for (size_t i = 0; i < b.i.size(); ++i) {
			bit[i] = (Tv)(b.i[i] /*- 1*/);
		}

		DynamicVector<Tv>bjt(b.j.size());

		for (size_t i = 0; i < b.j.size(); ++i) {
			bjt[i] = (Tv)(b.j[i] /*- 1*/);
		}

		DynamicVector<Tv> a_edge_from = kron(annzOnes, (Tv)a.n*bncollect);
		DynamicVector<Tv> ai = a_edge_from + kron(ait, bnOnes);
		DynamicVector<Tv> aj = a_edge_from + kron(ajt, bnOnes);
		DynamicVector<Tv> av = kron(a.v, bnOnes);

		DynamicVector<Tv> b_edge_from = kron(ancollect, bnnzOnes);
		DynamicVector<Tv> bi = b_edge_from + kron(anOnes, bit*(Tv)a.n);
		DynamicVector<Tv> bj = b_edge_from + kron(anOnes, bjt*(Tv)a.n);
		DynamicVector<Tv> bv = kron(anOnes, b.v);

		for (size_t i = 0; i < ai.size(); ++i) {
			ai[i] -= 1;
			aj[i] -= 1;
		}

		IJV<Tv>IJVA(n, av.size(), ai, aj, av);
		IJV<Tv>IJVB(n, bv.size(), bi, bj, bv);
		IJV<Tv>IJVRes = IJVA + IJVB;

		return IJVRes;
	}

	IJV<Tv> grid2_ijv(size_t n, size_t m, Tv isotropy = 1) {
		IJV<Tv> isograph = isotropy * path_graph_ijv(n);
		IJV<Tv> pgi = path_graph_ijv(m);
		IJV<Tv> res = product_graph(isograph, pgi);

		return res;
	}

	IJV<Tv> grid2_ijv(size_t n) {
		return grid2_ijv(n, n);
	}

	CompressedMatrix<Tv, blaze::columnMajor>grid2(size_t n, size_t m, Tv isotropy = 1) {
		CompressedMatrix<Tv, blaze::columnMajor>res;
		IJV<Tv> ijv = grid2_ijv(n, m);
		res = sparse(ijv);

		return res;
	}

	CompressedMatrix<Tv, blaze::columnMajor>grid2(size_t n) {
		return grid2(n, n);
	}

	// Create a Laplacian matrix from an adjacency matrix. We might want to do this differently, say by enforcing symmetry

	CompressedMatrix<Tv, blaze::columnMajor> lap(CompressedMatrix<Tv, blaze::columnMajor> A) {
		DynamicVector<Tv> ones(A.rows(), 1);
		CompressedMatrix<Tv, blaze::columnMajor>Dg = Diagonal(A*ones);
		CompressedMatrix<Tv, blaze::columnMajor>Res = Dg - A;

		return Res;
	}
	
	//	lapWrapComponents function

	// vector index of matrix
	CompressedMatrix<Tv> index(const CompressedMatrix <Tv> &A, const vector<size_t> &idx1, const vector<size_t> &idx2) {
		DynamicMatrix<Tv> Res(idx1.size(), idx2.size(), 0);

		for (size_t i = 0; i < idx1.size(); ++i)
			for (size_t j = 0; j < idx2.size(); j++)
				Res(i, j) = A(idx1[i], idx2[j]);

		return Res;
	}

	CompressedMatrix<Tv> index(const CompressedMatrix <Tv> &A, const vector<size_t> &idx) {
		vector<size_t> idx0{ 0 };

		return index(A, idx, idx0);
	}

	//vector index of vector
	DynamicVector<Tv> index(DynamicVector<Tv> vec, vector<size_t> idx) {
		DynamicVector<Tv>res(idx.size());

		for (size_t i = 0; i < idx.size(); i++) {
			res[i] = vec[idx[i]];
		}

		return res;
	}

	//Vec[index]=
	void index(DynamicVector<Tv> &vout, const vector<size_t> &idx, const DynamicVector<Tv> &vin) {

		assert(idx.size() == vin.size());

		for (size_t i = 0; i < idx.size(); ++i) {
			vout[idx[i]] = vin[i];
		}
	}

	//Apply the ith solver on the ith component

	SolverB<Tv> BlockSolver(const vector<vector<size_t>> &comps, vector<SubSolver<Tv>> &solvers, 
		vector<size_t>& pcgIts,	float tol = 1e-6F, double maxits = HUGE_VAL, double maxtime = HUGE_VAL, 
		bool verbose = false) {

		return SolverB<Tv>([=, &pcgIts](const DynamicVector<Tv> &b) {

			vector<size_t> pcgTmp;

			vector<SubSolver<Tv>> l_solvers = solvers;

			if (pcgIts.size()) {
				pcgIts[0] = 0;
				pcgTmp.push_back(0);
			}

			DynamicVector<Tv>x(b.size(), 0);

			for (size_t i = 0; i < comps.size(); ++i) {
				vector<size_t> ind = comps[i];
				DynamicVector<Tv> bi = index(b, ind);
				DynamicVector<Tv> solution = (l_solvers[i])(bi, pcgTmp, tol, maxits, maxtime, verbose);

				index(x, ind, solution);

				if (pcgIts.size())
					pcgIts[0] = pcgIts[0] > pcgTmp[0] ? pcgIts[0] : pcgTmp[0];
			}

			return x;
		});
	}

	SolverBMat<Tv> wrapInterfaceMat(const FactorSolver<Tv> solver, const CompressedMatrix<Tv, blaze::columnMajor> &a, 
		vector<size_t>& pcgIts, float tol = 0,
		double maxits = HUGE_VAL, double maxtime = HUGE_VAL, bool verbose = false,
		ApproxCholParams params = ApproxCholParams())
	{
		auto t1 = high_resolution_clock::now();

		Factorization<Tv> sol = solver(a);

		if (verbose) {
			auto t2 = high_resolution_clock::now();
			auto msec = duration_cast<milliseconds>(t2 - t1).count();
			std::cout << "Solver build time: " << msec << " ms.";
		}

		return SolverBMat<Tv>([=, &pcgIts](const CompressedMatrix<Tv, blaze::columnMajor> &b)->DynamicVector<Tv> {

			if (pcgIts.size())
				pcgIts[0] = 0;

			auto t1 = high_resolution_clock::now();

			DynamicVector<Tv> x = chol_subst(sol.Lower, b);
			 
			if (verbose) {
				auto t2 = high_resolution_clock::now();
				auto msec = duration_cast<milliseconds>(t2 - t1).count();
				std::cout << "Solver build time: " << msec << " ms.";
			}

			return x;
		});
	}
	
	SolverAMat<Tv> wrapInterfaceMat(const FactorSolver<Tv> solver) {
		return SolverAMat<Tv>([=](const CompressedMatrix<Tv, blaze::columnMajor> &a, vector<size_t>& pcgIts, float tol = 1e-6,
			double maxits = HUGE_VAL, double maxtime = HUGE_VAL, bool verbose = false, ApproxCholParams params = ApproxCholParams())
		{
			return wrapInterfaceMat(solver, a, pcgIts, tol, maxits, maxtime, verbose, params);
		});
	}

	SolverB<Tv> wrapInterface(const FactorSolver<Tv> solver, const CompressedMatrix<Tv, blaze::columnMajor> &a,
		vector<size_t>& pcgIts, float tol = 0,
		double maxits = HUGE_VAL, double maxtime = HUGE_VAL, bool verbose = false,
		ApproxCholParams params = ApproxCholParams())
	{
		auto t1 = high_resolution_clock::now();

		Factorization<Tv> sol = solver(a);

		if (verbose) {
			auto t2 = high_resolution_clock::now();
			auto msec = duration_cast<milliseconds>(t2 - t1).count();
			std::cout << "Solver build time: " << msec << " ms.";
		}

		return SolverB<Tv>([=, &pcgIts](const DynamicVector<Tv> &b)->DynamicVector<Tv> {

			if (pcgIts.size())
				pcgIts[0] = 0;

			auto t1 = high_resolution_clock::now();

			DynamicVector<Tv> x = chol_subst(sol.Lower, b);

			if (verbose) {
				auto t2 = high_resolution_clock::now();
				auto msec = duration_cast<milliseconds>(t2 - t1).count();
				std::cout << "Solver build time: " << msec << " ms.";
			}

			return x;
		});
	}

	SolverA<Tv> wrapInterface(const FactorSolver<Tv> solver) {
		return SolverA<Tv>([=](const CompressedMatrix<Tv, blaze::columnMajor> &a, vector<size_t>& pcgIts, float tol = 1e-6,
			double maxits = HUGE_VAL, double maxtime = HUGE_VAL, bool verbose = false, ApproxCholParams params = ApproxCholParams())
		{
			return wrapInterface(solver, a, pcgIts, tol, maxits, maxtime, verbose, params);
		});
	}
	
	//This functions wraps cholfact so that it satsfies our interface.
	SolverAMat<Tv> chol_sddm_mat() {
		return wrapInterfaceMat(cholesky<Tv>);
	}

	SolverA<Tv> chol_sddm() {
		return wrapInterface(cholesky<Tv>);
	}

//			Applies a Laplacian `solver` that satisfies our interface to each connected component of the graph with adjacency matrix `a`.
			

	// Returns true if graph is connected.
	bool isConnected(const CompressedMatrix<Tv, blaze::columnMajor> &mat) {
		if (!mat.rows())
			return false;

		vector<size_t> cm = components(mat);
		return *max_element(cm.begin(), cm.end()) == 1;

	}
	
//	lap function analog. Create a Laplacian matrix from an adjacency matrix.
//	If the input looks like a Laplacian, throw a warning and convert it.
//
	CompressedMatrix<Tv, blaze::columnMajor> forceLap(const CompressedMatrix<Tv, blaze::columnMajor> &a) {
		
		CompressedMatrix<Tv, blaze::columnMajor> af;

		if (blaze::min(a) < 0) {
			 af = blaze::abs(a);
			af = af - Diagonal(diag(af));
		} else 
			if (blaze::sum(blaze::abs(diag(a))) > 0) {
				af = a - Diagonal(diag(a));
			}
			else {
				af = a;
			}
		
		// return diagmat(af) - af;
		// return la(af);
		return Diagonal(sum(af)) - af;
	}

	pair<Tv, size_t> findmax(const CompressedMatrix<Tv, blaze::columnMajor>& A, int wise = 1) {
		size_t index=0;
		Tv maxvalue;

		if (wise == 1) {
			maxvalue = blaze::max(blaze::row(A, 0));

			for (size_t i = 1; i < A.rows(); i++) {

				Tv curvalue = blaze::max(blaze::row(A, i));

				if (curvalue > maxvalue) {
					maxvalue = curvalue;
					index = i;
				}
			}
		} else
			if (wise == 2) {
				maxvalue = blaze::max(blaze::column(A, 0));

				for (size_t i = 1; i < A.columns(); i++) {

					Tv curvalue = blaze::max(blaze::column(A, i));

					if (curvalue > maxvalue) {
						maxvalue = curvalue;
						index = i;
					}
					else throw("The value of wise parameter must be 1 or 2.");
				}
			}

		return pair<Tv, size_t>(maxvalue, index);
	}

	pair<Tv, size_t> findmax(const DynamicVector<Tv>& v) {
		size_t index = 0;
		Tv maxvalue=v[0];

		for (size_t i = 1; i < v.size(); ++i) {
			Tv curvalue = v[i];

			if (curvalue > maxvalue) {
				maxvalue = curvalue;
				index = i;
			}
		}

		return pair<Tv, size_t>(maxvalue, index);
	}
	
	Tv mean(const CompressedMatrix<Tv, blaze::columnMajor>& A) {
		return blaze::sum(A)/(A.rows()*A.columns());
	}

	Tv mean(DynamicVector<Tv> v) {
		return blaze::sum(v) / v.size();
	}

	SolverB<Tv> lapWrapConnected(SolverA<Tv> solver, const CompressedMatrix<Tv, blaze::columnMajor> &a, vector<size_t>& pcgIts,
		float tol = 1e-6, double maxits = HUGE_VAL, double maxtime = HUGE_VAL, bool verbose = false,
		ApproxCholParams params = ApproxCholParams()) {

		CompressedMatrix<Tv, blaze::columnMajor> la = forceLap(a);
		size_t N = la.rows();

		size_t ind = findmax(diag(la)).second;

		vector<size_t> leave;

		// Delete the row with the max value
		for (size_t i = 0; i < N; ++i) {
			if (i != ind)
				leave.push_back(i);
		}
		
		CompressedMatrix<Tv, blaze::columnMajor>lasub = index(la, leave, leave);
		SubSolver<Tv> subSolver = solver(lasub, pcgIts, tol, maxits, maxtime, verbose, params);

		return SolverB<Tv>([=, &pcgIts](const DynamicVector<Tv> &b) {

			SubSolver<Tv> l_subSolver = subSolver;
			DynamicVector<Tv> bs = index(b, leave) - DynamicVector<Tv>(leave.size(), mean(b));

			DynamicVector<Tv> xs = l_subSolver(bs, pcgIts, tol, maxits, maxtime, verbose);

			DynamicVector<Tv> x(b.size(), 0);
			index(x, leave, xs);
			x = x - DynamicVector<Tv>(x.size(), mean(x));

			return x;

		});
	}

	SolverA<Tv> lapWrapConnected(const SolverA<Tv> solver) {
		return SolverA<Tv>([=](const CompressedMatrix<Tv, blaze::columnMajor> &a, vector<size_t>& pcgIts, float tol = 1e-6,
			double maxits = HUGE_VAL, double maxtime = HUGE_VAL, bool verbose = false,
			const ApproxCholParams params = ApproxCholParams())
		{
			return lapWrapConnected(solver, a, pcgIts, tol, maxits, maxtime, verbose, params);
		});
	}

	/*
	function lapWrapConnected(solver::Function)
	    f(a::AbstractArray; tol=1e-6, maxits=Inf, maxtime=Inf, verbose=false, pcgIts=Int[], params...) = lapWrapConnected(solver, a; tol=tol, maxits=maxits, maxtime=maxtime, verbose=verbose, pcgIts=pcgIts, params... )
		return f
	end

	*/

	SolverB<Tv> lapWrapComponents(SolverA<Tv> solver, const CompressedMatrix<Tv, blaze::columnMajor> &a, vector<size_t>& pcgIts,
		float tol = 1e-6, double maxits = HUGE_VAL, double maxtime = HUGE_VAL, bool verbose = false,
		ApproxCholParams params = ApproxCholParams())
	{
		auto t1 = high_resolution_clock::now();

		if (!testZeroDiag(a)) {
//			a = a - Diagonal(diag(a));
		}

		vector<size_t> co = components(a);

		if (*max_element(co.begin(), co.end()) == 1) {

			SubSolver<Tv> s=solver(a, pcgIts, tol, maxits, maxtime, verbose, params);

			if (verbose) {
				auto t2 = high_resolution_clock::now();
				auto msec = duration_cast<milliseconds>(t2 - t1).count();
				std::cout << "Solver build time: " << msec << " ms.";
			}

			return s;
		}
		else {
			vector<vector<size_t>>comps = vecToComps(co);

			vector<SubSolver<Tv>>solvers;

			for (size_t i = 0; i < comps.size(); ++i) {

				vector<size_t> ind = comps[i];

				CompressedMatrix<Tv> asub = index(a, ind, ind);

				SubSolver<Tv> subSolver;

				if (ind.size() == 1) {
					
					subSolver = SubSolver<Tv>(nullSolver<Tv>);
				}
				else
					if (ind.size() < 50) {

						vector<size_t>pcgits;
						subSolver = lapWrapConnected(chol_sddm(), asub, pcgits);
					}
					else {
						subSolver=solver(a, pcgIts, tol, maxits, maxtime, verbose, params);
					}

				solvers.push_back(subSolver);
			}

			if (verbose) {
				auto t2 = high_resolution_clock::now();
				auto msec = duration_cast<milliseconds>(t2 - t1).count();
				std::cout << "Solver build time: " << msec << " ms.";
			}

			return BlockSolver(comps, solvers, pcgIts, tol, maxits, maxtime, verbose);
		}

	}

	SolverA<Tv> lapWrapComponents(const SolverA<Tv> solver) {

		return SolverA<Tv>([=](const CompressedMatrix<Tv, blaze::columnMajor> &a, vector<size_t>& pcgIts, float tol = 1e-6,
			double maxits = HUGE_VAL, double maxtime = HUGE_VAL, bool verbose = false,
			const ApproxCholParams params = ApproxCholParams())
		{
			return lapWrapComponents(solver, a, pcgIts, tol, maxits, maxtime, verbose, params);
		});
	}

	SolverA<Tv> lapWrapSDDM(SolverA<Tv> sddmSolver) {
		return lapWrapComponents(lapWrapConnected(sddmSolver));
	}

	/*
	Create an adjacency matrix and a diagonal vector from an SDD M-matrix.
	That is, from a Laplacian with added diagonal weights
	*/

	pair<CompressedMatrix<Tv, blaze::columnMajor>, DynamicVector<Tv>> adj(const CompressedMatrix<Tv, blaze::columnMajor> &sddm) 
	{
		CompressedMatrix<Tv, blaze::columnMajor> a = Diagonal(diag(sddm)) - sddm;
		DynamicVector<Tv> ones(sddm.rows(), 1);
		DynamicVector<Tv> d = sddm * ones;

		return pair<CompressedMatrix<Tv, blaze::columnMajor>, DynamicVector<Tv>>(a, d);
	}
	 
	// Add a new vertex to a with weights to the other vertices corresponding to diagonal surplus weight.
	
	CompressedMatrix<Tv, blaze::columnMajor> extendMatrix(const CompressedMatrix<Tv, blaze::columnMajor> &a,
		DynamicVector<Tv> d)
	{
		assert(a.rows() == d.size());

		if (blaze::sum(blaze::abs(d)) == 0)
			return a;

		DynamicVector<Tv> dpos(d.size()+1, 0);

		for (size_t i = 0; i < d.size(); i++)
			dpos[i] = d[i] * (d[i] > 0);

		size_t n = d.size();

		CompressedMatrix<Tv, blaze::columnMajor> Res(a.rows() + 1, a.columns() + 1);

		submatrix(Res, 0, 0, a.rows(), a.columns()) = a;
		row(Res, a.rows()) = trans(dpos);
		column(Res, a.columns()) = dpos;

		return Res;
	}

	SolverB<Tv> sddmWrapLap(SolverA<Tv> lapSolver, const CompressedMatrix<Tv, blaze::columnMajor> &sddm, vector<size_t>& pcgIts,
		float tol = 1e-6, double maxits = HUGE_VAL, double maxtime = HUGE_VAL, bool verbose = false,
		ApproxCholParams params = ApproxCholParams())
	{
		/*pair<CompressedMatrix<Tv, blaze::columnMajor>, DynamicVector<Tv>> Adj = adj(sddm);
		CompressedMatrix<Tv, blaze::columnMajor> a = Adj.first;
		DynamicVector<Tv> d = Adj.second;*/

		CompressedMatrix<Tv, blaze::columnMajor> a;
		DynamicVector<Tv> d;

		tie(a, d) = adj(sddm);
		
		CompressedMatrix<Tv, blaze::columnMajor>a1 = extendMatrix(a, d);

		SubSolver<Tv> F = lapSolver(a1, pcgIts, tol, maxits, maxtime, verbose, params);

		return [=, &pcgIts](const DynamicVector<Tv> &b) {
			DynamicVector<Tv> sb(b.size() + 1);
			subvector(sb, 0, b.size()) = b;
			sb[b.size()] = -blaze::sum(b);

			SubSolver<Tv>l_F = F;

			DynamicVector<Tv> xaug = l_F(sb, pcgIts, tol, maxits, maxtime, verbose, params);

			xaug = xaug - DynamicVector<Tv>(xaug.size(), xaug[xaug.size() - 1]);

			return subvector(xaug, 0, a.rows() - 1);

		};
	}

	SolverA<Tv> sddmWrapLap(const SolverA<Tv> solver) {
		
		return SolverA<Tv>([=](const CompressedMatrix<Tv, blaze::columnMajor> &a, vector<size_t>& pcgIts, float tol = 1e-6,
			double maxits = HUGE_VAL, double maxtime = HUGE_VAL, bool verbose = false,
			const ApproxCholParams params = ApproxCholParams())
		{
			return sddmWrapLap(solver, a, pcgIts, tol, maxits, maxtime, verbose, params);
		});
	}

	//Returns the upper triangle of M starting from the kth superdiagonal.
	CompressedMatrix<Tv, blaze::columnMajor> triu(const CompressedMatrix<Tv, blaze::columnMajor> &A, size_t k=0){
		
		CompressedMatrix<Tv, blaze::columnMajor>Res(A.rows(), A.columns());

		for (size_t i = k; i < A.rows(); ++i)
			for (size_t j = i; j < A.columns(); j++)
				Res(i, j) = A(i, j);

		return Res;
	}

	CompressedMatrix<Tv, blaze::columnMajor> wtedEdgeVertexMat(const CompressedMatrix<Tv, blaze::columnMajor> &mat) {
		vector<size_t> ai, aj;
		DynamicVector<Tv> av;

		tie(ai, aj, av) = findnz(triu(mat));

		size_t m = ai.size();
		size_t n = mat.rows();

		DynamicVector<Tv> v = sqrt(av);

		CompressedMatrix<Tv, blaze::columnMajor> Sparse1(m,n), Sparse2(m,n), Res;
		
		for (size_t i = 0; i < m; i++) {
			Sparse1(i, ai[i]) = v[i];
			Sparse2(i, aj[i]) = v[i];
		}

		Res = Sparse1 - Sparse2;

		return Res;
	}
};

/*function wtedEdgeVertexMat(mat::SparseMatrixCSC)
(ai, aj, av) = findnz(triu(mat, 1))
m = length(ai)
n = size(mat)[1]
v = av. ^ (1 / 2)
return sparse(collect(1:m), ai, v, m, n) - sparse(collect(1:m), aj, v, m, n)
end
*/
