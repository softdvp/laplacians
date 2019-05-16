//Structures for the approxChol solver

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

/*
LLp elements are all in the same column.
row tells us the row, and val is the entry.
val is set to zero for some edges that we should remove.
next gives the next in the column.It points to itself to terminate.
reverse is the index into lles of the other copy of this edge,
since every edge is stored twice as we do not know the order of elimination in advance.
*/

template<typename Tv>
class LLp {
public:
	size_t row;
	Tv val;

	LLp* next;
	LLp* reverse;

	LLp() {
		row = 0;
		val = 0;
		next = this;
		reverse = this;
	}

	LLp(size_t Arow, Tv Aval, LLp* Anext, LLp* Areverse):row(Arow), val(Aval), next(Anext), reverse(Areverse) {}

	LLp(size_t Arow, Tv Aval) :row(Arow), val(Aval) {
		next = this;
		reverse = this;
	}

	LLp(size_t Arow, Tv Aval, LLp* Anext) :row(Arow), val(Aval), next(Anext) {
		reverse = this;
	}
};

/*

LLmatp is the data structure used to maintain the matrix during elimination.
It stores the elements in each column in a singly linked list(only next ptrs)
Each element is an LLp(linked list pointer).
The head of each column is pointed to by cols.

We probably can get rid of degs - as it is only used to store initial degrees.

*/

template <typename Tv>
class LLmatp {
public:
	size_t n;
	vector<size_t> degs;
	vector<LLp<Tv>> *cols;
	vector<LLp<Tv>> *llelems;

	LLmatp(const CompressedMatrix<Tv, blaze::columnMajor> &A){
		SparseMatrixCSC<Tv> a(A);
		
		n = A.rows();
		size_t m = A.nonZeros();

		degs.resize(n);
		vector<size_t> flips = flipIndex(A);

		cols.resize(n);
		llelems.resize(m);

		for (size_t i = 0; i < n; i++)
		{
			degs[i] = a.colptr[i + 1] - a.colptr[i];

			size_t ind = a.colptr[i];
			size_t j = a.rowval[ind];
			Tv v = a.nzval[ind];
			LLp<Tv> *llpend = new LLp<Tv>(j, v);
			LLp<Tv> *next = llelems[ind] = llpend;
			
			for (size_t ind = a.colptr[i]; ind < a.colptr[i + 1]; ind++)
			{
				size_t j = a.rowval[ind];
				Tv v = a.nzval[ind];
				next = llelems[ind] = new LLp<Tv>(j, v, next);
			}

			cols[i] = next;
		}

		for (size_t i = 0; i < n; i++)
			for (size_t ind = a.colptr[i]; ind < a.colptr[i + 1]; ind++)
				llelems[ind]->reverse = llelems[flips[ind]];

	}
};

/*

	for ind in (a.colptr[i]+one(Tind)):(a.colptr[i+1]-one(Tind))
			j = a.rowval[ind]
			v = a.nzval[ind]
			next = llelems[ind] = LLp{Tind,Tval}(j,v,next)
		end
		cols[i] = next
	end

	@inbounds for i in 1:n
		for ind in a.colptr[i]:(a.colptr[i+1]-one(Tind))
			llelems[ind].reverse = llelems[flips[ind]]
		end
	end

	return LLmatp{Tind,Tval}(n, degs, cols, llelems)
end

*/

//these are the types we use with a fixed ordering

template<typename Tv>
class LLord {
public:
	size_t row;
	size_t next;
	Tv val;

	LLord(size_t Arow, size_t Anext, Tv Aval):row(Arow), next(Anext), val(Aval){}
};


template<typename Tv>
class LLMatOrd {
public:
	size_t n;
	vector<size_t> cols;
	vector<LLord<Tv>> lles;
};

template<typename Tv>
class LLcol {
public:
	size_t row;
	size_t ptr;
	Tv val;

	LLcol(size_t Arow, size_t Aptr, Tv Aval):row(Arow), ptr(Aptr), val(Aval){}
};

// LDLinv

  
/*  LDLinv contains the information needed to solve the Laplacian systems.
  It does it by applying Linv, then Dinv, then Linv (transpose).
  But, it is specially constructed for this particular solver.
  It does not explicitly make the matrix triangular.
  Rather, col[i] is the name of the ith col to be eliminated
*/

template<typename Tv>
class LDLinv {
public:
	vector<size_t> col;
	vector<size_t> colptr;
	vector<size_t> rowval;
	vector<Tv> fval;
	vector<Tv> d;

	LDLinv(CompressedMatrix<Tv, blaze::columnMajor> A) : col(A.rows() - 1, 0), colptr(A.rows(), 0), d(A.rows(), 0) {}
	LDLinv(LLMatOrd<Tv> A) : col(A.rows() - 1, 0), colptr(A.rows(), 0), d(A.rows(), 0){}
	LDLinv(LLmatp<Tv> A) : col(A.rows() - 1, 0), colptr(A.rows(), 0), d(A.rows(), 0){}
};

/*
ApproxCholPQ

the data structure we use to keep track of degrees
*/


class ApproxCholPQElem {
public:
	size_t prev;
	size_t next;
	size_t key;
	
	ApproxCholPQElem(size_t Aprev, size_t Anext, size_t Akey):prev(Aprev), next(Anext), key(Akey) {}
};

/*
  An approximate priority queue.
  Items are bundled together into doubly-linked lists with all approximately the same key.
  minlist is the min list we know to be non-empty.
  It should always be a lower bound.
  keyMap maps keys to lists
*/


class ApproxCholPQ {
public:
	vector<ApproxCholPQElem> elems; // indexed by node name
	vector<size_t> lists;
	size_t minlist;
	size_t nitems;
	size_t n;
};

/*
	params = ApproxCholParams(order, output)
	order can be one of
	Deg(by degree, adaptive),
	WDeg(by original wted degree, nonadaptive),	
	Given
*/

enum class ApproxCholEnum { Deg, WDeg, Given };

class ApproxCholParams {
public:
	ApproxCholEnum order;
	long stag_test;

	ApproxCholParams() {
		order = ApproxCholEnum::Deg;
		stag_test = 5;
	}

	ApproxCholParams(ApproxCholEnum symb) {
		order = symb;
		stag_test = 5;
	}
};

