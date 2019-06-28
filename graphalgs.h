#pragma once

#include <iostream>
#include <blaze/Math.h>
#include <blaze/math/Submatrix.h>
#include <blaze/math/Subvector.h>
#include <vector>
#include <functional>
#include <random>
#include "sparsecsc.h"
#include "ijvstruct.h"

using namespace std;

using blaze::CompressedMatrix;
using blaze::CompressedVector;
using blaze::DynamicMatrix;
using blaze::DynamicVector;
using blaze::columnwise;
using blaze::rowwise;

namespace laplacians {

	template<typename Tv>
	class Random {
	private:
		random_device rd;
		mt19937 gen;

	public:

		Tv rand0_1() {
			return generate_canonical<Tv, 8*sizeof(Tv)>(gen);
		}

		Tv randn() {
			return (rand0_1() - 0.5) * 6;
		}

		DynamicVector<Tv> randv(size_t sz) {
			DynamicVector<Tv> res(sz);

			for (size_t i = 0; i < sz; i++)
				res[i] = rand0_1();

			return res;
		}

		Random():gen(rd()){}
	};

	template <typename Tv>
	CompressedMatrix<Tv, blaze::columnMajor> ClearDiag(const CompressedMatrix<Tv, blaze::columnMajor>& A) {
		CompressedMatrix<Tv, blaze::columnMajor> res = A;

		for (size_t i = 0; i < A.rows(); i++)
			res(i, i) = 0;

		return res;
	}

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

		vector<size_t> resv;
		for (size_t i = 0; i < B.rows(); i++)
			for (auto it = B.cbegin(i); it != B.cend(i); ++it) {
				size_t v = it->value();

				resv.push_back(v - 1);
			}

		return resv;
	}

	/*template<typename Tv>
	vector<size_t> flipIndex(const SparseMatrixCSC<Tv>& A) {

	}*/

	template <typename Tv>
	DynamicVector<Tv> diag(const CompressedMatrix<Tv, blaze::columnMajor> &A, size_t diag_n = 0) {

		size_t it = A.columns() - diag_n;
		DynamicVector<Tv> resv(it);

		for (size_t i = 0; i < it; ++i) {
			resv[i] = (A(i, diag_n + i));
		}

		return resv;
	}

	template <typename Tv>
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

	//Conjugate transpose
	template <typename Tv>
	CompressedMatrix<Tv, blaze::columnMajor> adjoint(const CompressedMatrix<Tv, blaze::columnMajor> &A) {
		CompressedMatrix<Tv, blaze::columnMajor> Res = blaze::ctrans(A);

		return Res;
	}

	/*
	Create an adjacency matrix and a diagonal vector from an SDD M-matrix.
	That is, from a Laplacian with added diagonal weights
	*/

	template <typename Tv>
	pair<CompressedMatrix<Tv, blaze::columnMajor>, DynamicVector<Tv>> adj(const CompressedMatrix<Tv, blaze::columnMajor> &sddm)
	{
		CompressedMatrix<Tv, blaze::columnMajor> a = Diagonal(diag(sddm)) - sddm;
		DynamicVector<Tv> ones(sddm.rows(), 1);
		DynamicVector<Tv> d = sddm * ones;

		return pair<CompressedMatrix<Tv, blaze::columnMajor>, DynamicVector<Tv>>(a, d);
	}

	// Add a new vertex to a with weights to the other vertices corresponding to diagonal surplus weight.
	template <typename Tv>
	CompressedMatrix<Tv, blaze::columnMajor> extendMatrix(const CompressedMatrix<Tv, blaze::columnMajor> &a,
		DynamicVector<Tv> d)
	{
		assert(a.rows() == d.size());

		if (blaze::sum(blaze::abs(d)) == 0)
			return a;

		DynamicVector<Tv> dpos(d.size() + 1, 0);

		for (size_t i = 0; i < d.size(); i++)
			dpos[i] = d[i] * (d[i] > 0);

		size_t n = d.size();

		CompressedMatrix<Tv, blaze::columnMajor> Res(a.rows() + 1, a.columns() + 1);

		submatrix(Res, 0, 0, a.rows(), a.columns()) = a;
		row(Res, a.rows()) = trans(dpos);
		column(Res, a.columns()) = dpos;

		return Res;
	}

	template <typename Tv>
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

	// vector index of matrix
	template <typename Tv>
	CompressedMatrix<Tv, blaze::columnMajor>index(const CompressedMatrix<Tv, blaze::columnMajor>&A, const vector<size_t> &idx1, const vector<size_t> &idx2) {
		CompressedMatrix<Tv, blaze::columnMajor>Res(idx1.size(), idx2.size(), 0);

		for (size_t i = 0; i < idx1.size(); ++i)
			for (size_t j = 0; j < idx2.size(); j++)
				Res(i, j) = A(idx1[i], idx2[j]);

		return Res;
	}

	// vector index of matrix
	template <typename Tv>
	DynamicVector<Tv> index(const CompressedMatrix<Tv, blaze::columnMajor>& A, const vector<size_t>& idx1, const size_t idx2) {
		DynamicVector<Tv> Res(idx1.size(), 0);

		for (size_t i = 0; i < idx1.size(); ++i)
			Res[i] = A(idx1[i], idx2);

		return Res;
	}

	template <typename Tv>
	DynamicVector<Tv> index(const CompressedMatrix<Tv, blaze::columnMajor>& A, const size_t idx1, const vector<size_t>& idx2) {
		DynamicVector<Tv> Res(idx2.size(), 0);

		for (size_t i = 0; i < idx2.size(); ++i)
			Res[i] = A(idx1, idx2[i]);

		return Res;
	}

	template <typename Tv>
	DynamicVector<Tv> index(const CompressedMatrix<Tv, blaze::columnMajor>&A, const vector<size_t> &idx) {
		
		return index(A, idx, 0);
	}

	//vector index of vector
	template <typename Tv>
	DynamicVector<Tv> index(const DynamicVector<Tv> &vec, const vector<size_t> &idx) {
		DynamicVector<Tv>res(idx.size());

		for (size_t i = 0; i < idx.size(); i++) {
			res[i] = vec[idx[i]];
		}

		return res;
	}

	//Vec[index]=
	template <typename Tv>
	void index(DynamicVector<Tv> &vout, const vector<size_t> &idx, const DynamicVector<Tv> &vin) {

		assert(idx.size() == vin.size());

		for (size_t i = 0; i < idx.size(); ++i) {
			vout[idx[i]] = vin[i];
		}
	}

	template <typename Tv>
	void index(CompressedMatrix<Tv, blaze::columnMajor>& mout, const vector<size_t>& idx, size_t idx2, const DynamicVector<Tv>& vin) {

		assert(idx.size() == vin.size());

		for (size_t i = 0; i < idx.size(); i++)
		{
			mout(idx[i], idx2) = vin[i];
		}
	}

	template <typename Tv>
	DynamicVector<Tv>indexbool(const DynamicVector<Tv> &vect, const vector<bool> &idx) {
		
		assert(vect.size() == idx.size());
		
		vector<Tv> v;

		for (size_t i = 0; i < idx.size(); i++)
			if (idx[i])
				v.push_back(vect[i]);

		DynamicVector<Tv> res(v.size());

		for (size_t i = 0; i < v.size(); i++)
		{
			res[i] = v[i];
		}

		return res;
	}

	inline vector<size_t> indexbool(const vector<size_t>& vect, const vector<bool>& idx) {
		assert(vect.size() == idx.size());

		vector<size_t> v;

		for (size_t i = 0; i < idx.size(); i++)
			if (idx[i])
				v.push_back(vect[i]);

		return v;
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

	inline const vector<vector<size_t>> vecToComps(vector<size_t> &compvec) {

		size_t nc = *max_element(compvec.begin(), compvec.end());

		vector<vector<size_t>> comps(nc);

		vector<size_t> sizes(nc, 0);

		for (size_t i : compvec)
			sizes[i - 1]++;

		for (size_t i = 0; i < nc; i++)
			comps[i].resize(sizes[i]);

		vector<size_t>ptrs(nc, 0);

		for (size_t i = 0; i < compvec.size(); i++)
		{
			size_t c = compvec[i] - 1;

			comps[c][ptrs[c]++] = i;
		}

		return comps;
	}

	template<typename Tv>
	tuple<vector<size_t>, vector<size_t>, DynamicVector<Tv>> findnz(const CompressedMatrix<Tv, blaze::columnMajor> &mat) {

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

	// Returns vector where comp[i]=component number
	template<typename Tv>
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

	template<typename Tv>
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

	//Kronecker product
	//for matrices: https://en.wikipedia.org/wiki/Kronecker_product

	template<typename Tv>
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
	template<typename Tv>
	DynamicVector<Tv> kron(const DynamicVector<Tv> &A, const DynamicVector<Tv>&B) {
		DynamicVector<Tv> res(A.size()*B.size());

		for (size_t i = 0; i < A.size(); i++) {
			subvector(res, i*B.size(), B.size()) = A[i] * B;
		}

		return res;
	}

	//Returns the diagonal weighted degree matrix(as a sparse matrix) of a graph

	template<typename Tv>
	CompressedMatrix<Tv, blaze::columnMajor> diagmat(const CompressedMatrix<Tv, blaze::columnMajor> &A) {

		return Diagonal(sum(A));
	}

	template<typename Tv>
	CompressedMatrix<Tv, blaze::columnMajor> pow(const CompressedMatrix<Tv, blaze::columnMajor> &A, const int n) {
		CompressedMatrix<Tv, blaze::columnMajor> res;

		res = A;
		for (size_t i = 0; i < n - 1; i++) {
			res *= A;
		}

		return res;
	}

	template<typename Tv>
	CompressedMatrix<Tv, blaze::columnMajor> sparse(vector<size_t> I, vector<size_t> J, DynamicVector<Tv> V, size_t m, size_t n) {
		CompressedMatrix<Tv, blaze::columnMajor>res(m, n);

		size_t nnz = I.size();

		res.reserve(nnz);

		for (size_t l = 0; l < nnz; ++l) {
			res(I[l], J[l]) = V[l];
		}

		return res;

	}

	template<typename Tv>
	CompressedMatrix<Tv, blaze::columnMajor> power(const CompressedMatrix<Tv, blaze::columnMajor> &A, const int k) {
		CompressedMatrix<Tv, blaze::columnMajor>ap;
		
		ap = pow(A, k);
		
		ap = ClearDiag(ap);

		return ap;
	}

	// Returns true if graph is connected.
	template<typename Tv>
	bool isConnected(const CompressedMatrix<Tv, blaze::columnMajor> &mat) {
		if (!mat.rows())
			return false;

		vector<size_t> cm = components(mat);
		return *max_element(cm.begin(), cm.end()) == 1;

	}

	template<typename Tv>
	pair<Tv, size_t> findmax(const CompressedMatrix<Tv, blaze::columnMajor>& A, int wise = 1) {
		size_t index = 0;
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
		}
		else
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

	template<typename Tv>
	pair<Tv, size_t> findmax(const DynamicVector<Tv>& v) {
		size_t index = 0;
		Tv maxvalue = v[0];

		for (size_t i = 1; i < v.size(); ++i) {
			Tv curvalue = v[i];

			if (curvalue > maxvalue) {
				maxvalue = curvalue;
				index = i;
			}
		}

		return pair<Tv, size_t>(maxvalue, index);
	}

	template<typename Tv>
	Tv mean(const CompressedMatrix<Tv, blaze::columnMajor>& A) {
		return blaze::sum(A) / (A.rows()*A.columns());
	}

	template<typename Tv>
	Tv mean(DynamicVector<Tv> v) {
		return blaze::sum(v) / v.size();
	}

	//Returns the upper triangle of M starting from the kth superdiagonal.
	template<typename Tv>
	CompressedMatrix<Tv, blaze::columnMajor> triu(const CompressedMatrix<Tv, blaze::columnMajor> &A, size_t k = 0) {

		CompressedMatrix<Tv, blaze::columnMajor>Res(A.rows(), A.columns());

		for (size_t i = k; i < A.rows(); ++i)
			for (size_t j = i; j < A.columns(); j++)
				Res(i, j) = A(i, j);

		return Res;
	}

	//generates vector with values of [start...end)
	inline vector<size_t> collect(size_t start, size_t end) {
		vector<size_t> res(end);

		for (size_t i = 0; i < end; i++)
			res[i] = i;
		
		return res;
	}
	
	template<typename Tv>
	CompressedMatrix<Tv, blaze::columnMajor> wtedEdgeVertexMat(const CompressedMatrix<Tv, blaze::columnMajor> &mat) {

		auto [ai, aj, av] = findnz(triu(mat));

		size_t m = ai.size();
		size_t n = mat.rows();

		DynamicVector<Tv> v = sqrt(av);

		CompressedMatrix<Tv, blaze::columnMajor> Sparse1(m, n), Sparse2(m, n), Res;

		for (size_t i = 0; i < m; i++) {
			Sparse1(i, ai[i]) = v[i];
			Sparse2(i, aj[i]) = v[i];
		}

		Res = Sparse1 - Sparse2;

		return Res;
	}

	inline size_t keyMap(size_t x, size_t n) {
		return x <= n ? x : n + x / n;
	}


	// Create a Laplacian matrix from an adjacency matrix. We might want to do this differently, say by enforcing symmetry

	template <typename Tv>
	CompressedMatrix<Tv, blaze::columnMajor> lap(CompressedMatrix<Tv, blaze::columnMajor> A) {
		DynamicVector<Tv> ones(A.rows(), 1);
		DynamicVector<Tv> d = A * ones;
		CompressedMatrix<Tv, blaze::columnMajor>Dg = Diagonal(d);
		CompressedMatrix<Tv, blaze::columnMajor>Res = Dg - A;

		return Res;
	}

	//	lap function analog. Create a Laplacian matrix from an adjacency matrix.
	//	If the input looks like a Laplacian, throw a warning and convert it.
	//
	template <typename Tv>
	CompressedMatrix<Tv, blaze::columnMajor> forceLap(const CompressedMatrix<Tv, blaze::columnMajor> &a) {

		CompressedMatrix<Tv, blaze::columnMajor> af;

		if (blaze::min(a) < 0) {
			af = blaze::abs(a);
			af = ClearDiag(af);
		}
		else
			if (blaze::sum(blaze::abs(diag(a))) > 0) {
				af = ClearDiag(a);
			}
			else {
				af = a;
			}

		// return diagmat(af) - af;
		// return la(af);
		return Diagonal(sum(af)) - af;
	}

	template <typename Tv>
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

	template <typename Tv>
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

		DynamicVector<Tv> a_edge_from = kron<Tv>(annzOnes, (Tv)a.n*bncollect);
		DynamicVector<Tv> ai = a_edge_from + kron(ait, bnOnes);
		DynamicVector<Tv> aj = a_edge_from + kron(ajt, bnOnes);
		DynamicVector<Tv> av = kron(a.v, bnOnes);

		DynamicVector<Tv> b_edge_from = kron(ancollect, bnnzOnes);
		DynamicVector<Tv> bi = b_edge_from + kron<Tv>(anOnes, bit*(Tv)a.n);
		DynamicVector<Tv> bj = b_edge_from + kron<Tv>(anOnes, bjt*(Tv)a.n);
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

	template <typename Tv>
	IJV<Tv> grid2_ijv(size_t n, size_t m, Tv isotropy = 1) {
		IJV<Tv> isograph = isotropy * path_graph_ijv<Tv>(n);
		IJV<Tv> pgi = path_graph_ijv<Tv>(m);
		IJV<Tv> res = product_graph(isograph, pgi);

		return res;
	}

	template <typename Tv>
	IJV<Tv> grid2_ijv(size_t n) {
		return grid2_ijv<Tv>(n, n);
	}

	template <typename Tv>
	CompressedMatrix<Tv, blaze::columnMajor>grid2(size_t n, size_t m, Tv isotropy = 1) {
		CompressedMatrix<Tv, blaze::columnMajor>res;
		IJV<Tv> ijv = grid2_ijv<Tv>(n, m);
		res = sparse(ijv);

		return res;
	}

	template <typename Tv>
	CompressedMatrix<Tv, blaze::columnMajor>grid2(size_t n) {
		return grid2<Tv>(n, n);
	}
}