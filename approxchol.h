//Structures for the approxChol solver

#pragma once
#include <iostream>
#include <stdint.h>
#include <blaze/Math.h>
#include <blaze/math/Submatrix.h>
#include <blaze/math/Subvector.h>
#include <vector>
#include <chrono>
#include <functional>
#include "graphalgs.h"
#include "solvertypes.h"
#include "pcg_1.h"
#include "lapwrappers.h"

using namespace std;
using blaze::CompressedMatrix;
using blaze::DynamicMatrix;
using blaze::DynamicVector;
using blaze::columnwise;
using blaze::rowwise;

namespace laplacians {

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

		/*LLp() {
			row = 0;
			val = 0;
			next = this;
			reverse = this;
		}*/

		LLp(const size_t Arow, const Tv Aval, const LLp* Anext, const LLp* Areverse) :row(Arow), val(Aval), next(Anext), reverse(Areverse) {}

		LLp(const size_t Arow, const Tv Aval) :row(Arow), val(Aval) {
			next = this;
			reverse = this;
		}

		LLp(const size_t Arow, const Tv Aval, LLp* Anext) :row(Arow), val(Aval), next(Anext) {
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
		vector<LLp<Tv>*> cols;
		vector<LLp<Tv>*> llelems;

		LLmatp(const CompressedMatrix<Tv, blaze::columnMajor> &A) {
			SparseMatrixCSC<Tv> a(A);
			
			n = A.rows();
			size_t m = A.nonZeros();

			degs.resize(n);
			vector<size_t> flips = flipIndex(A);

			cols.resize(n, NULL);
			llelems.resize(m, NULL);

			for (size_t i = 0; i < n; i++)
			{
				degs[i] = a.colptr[i + 1] - a.colptr[i];

				size_t ind = a.colptr[i];
				size_t j = a.rowval[ind];
				Tv v = a.nzval[ind];
				LLp<Tv> *llpend = new LLp<Tv>(j, v);
				LLp<Tv> *next = llelems[ind] = llpend;

				for (size_t ind = a.colptr[i] + 1; ind < a.colptr[i + 1]; ind++)
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

		~LLmatp() {
			cout << "destructor LLmatp";

			for (size_t i = 0; i < llelems.size(); i++)
			{
				delete llelems[i];
			}


		}
	};

	//Print a column in an LLmatp matrix.
	//This is here for diagnostics.

	template <typename Tv>
	void print_ll_col(const LLmatp<Tv> &llmat, const size_t i) {

		LLp<Tv>* ll = llmat.cols[i];

		cout << "col " << i + 1 << " row " << ll->row + 1 << " : " << ll->val << endl;
		cout << "col " << i + 1 << " reverse->row=" << ll->reverse->row + 1 << endl;
		cout << "col " << i + 1 << " next->row=" << ll->next->row + 1 << endl;

		while (ll->next != ll) {
			ll = ll->next;

			cout << "col " << i+1 << " row " << ll->row+1 << " : " << ll->val << endl;
			cout << "col " << i + 1 << " reverse->row=" << ll->reverse->row + 1 << endl;
			cout << "col " << i + 1 << " next->row=" << ll->next->row + 1 << endl;

		}
	}

	//these are the types we use with a fixed ordering

	template<typename Tv>
	class LLord {
	public:
		size_t row;
		size_t next;
		Tv val;

		LLord(const size_t Arow, const size_t Anext, const Tv Aval) :row(Arow), next(Anext), val(Aval) {}
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

		LLcol(const size_t Arow, const size_t Aptr, const Tv Aval) :row(Arow), ptr(Aptr), val(Aval) {}
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

		LDLinv(const CompressedMatrix<Tv, blaze::columnMajor> &A) : col(A.columns() - 1, 0), colptr(A.columns(), 0), 
			d(A.columns(), 0) /* rowval and fval are empty*/	{}

		LDLinv(const LLMatOrd<Tv> &A) : col(A.n - 1, 0), colptr(A.n, 0), d(A.n, 0) {}

		LDLinv(const LLmatp<Tv> &A) : col(A.n - 1, 0), colptr(A.n, 0), d(A.n, 0) {}

		void debug() const {
			cout << "\ncol=";
			for (size_t i = 0; i < col.size(); i++)
				cout << col[i]+1 << " ";
			cout << endl;

			cout << "\ncolptr=";
			for (size_t i = 0; i < colptr.size(); i++)
				cout << colptr[i] + 1 << " ";
			cout << endl;

			cout << "\nrowval=";
			for (size_t i = 0; i < rowval.size(); i++)
				cout << rowval[i] + 1 << " ";
			cout << endl;

			cout << "\nfval=";
			for (size_t i = 0; i < fval.size(); i++)
				cout << fval[i] << " ";
			cout << endl;

			cout.precision(8);
			cout << "\nd=";
			for (size_t i = 0; i < d.size(); i++)
				cout << d[i] << " ";
			cout << endl;
		}
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

		ApproxCholPQElem() {};
		ApproxCholPQElem(const size_t Aprev, const size_t Anext, const size_t Akey) :prev(Aprev), next(Anext), key(Akey) {}
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

		ApproxCholPQ(const vector<size_t> &a) {

			n = a.size();
			elems.resize(n);
			lists = vector<size_t>(2 * n + 1, SIZE_MAX);
			minlist = 0;

			for (size_t i = 0; i < n; i++)
			{
				size_t key = a[i]-1;
				size_t head = lists[key];

				if (head != SIZE_MAX)
				{
					elems[i] = ApproxCholPQElem(SIZE_MAX, head, key);
					elems[head] = ApproxCholPQElem(i, elems[head].next, elems[head].key);
				}
				else
					elems[i] = ApproxCholPQElem(SIZE_MAX, SIZE_MAX, key);

				lists[key] = i;

			}
			
			// Assigned with a default constructor
			nitems = n;
		}

		void move(const size_t i, const size_t newkey, const size_t oldlist, const size_t newlist);
		void inc(const size_t i);
		void dec(const size_t i);
		size_t pop();

		void debug() {
		/*	cout << endl << "elems" << endl;

			for (size_t i = 0; i < elems.size(); i++)
			{
				cout << "prev= " << elems[i].prev+1 << " next= " << elems[i].next+1 << " key= " << elems[i].key+1 << endl;
			}
			
			cout << endl << "lists" << endl;*/
			for (size_t i = 0; i < lists.size(); i++)
			{
				cout << lists[i]+1 << " ";
			}
			
			cout << endl << endl;
			/*
			cout << "minlist= " << minlist+1 << " nitems= " << nitems << " n= " << n;
			cout << endl;*/
		}
	};

	//The approximate factorization

	template<typename Tv>
	size_t get_ll_col(const LLmatp<Tv>& llmat, size_t i, vector<LLp<Tv>*>& colspace) {

		LLp<Tv>* ll = llmat.cols[i-1];
		size_t len = 0;

		while (ll->next != ll) {
			//if (ll->val > 1e-6) {
			if (ll->val > 0) {
				++len;

				if (len > colspace.size())
					colspace.push_back(ll);
				else colspace[len - 1] = ll;
			}

			ll = ll->next;
		}

		//if (ll->val > 1e-6) {
		if (ll->val > 0) {
			++len;

			if (len > colspace.size())
				colspace.push_back(ll);
			else colspace[len - 1] = ll;
		}

		return len;
	}

	template<typename Tv>
	void debugLLp(const vector<LLp<Tv>*> &colspace, size_t len) {

		for (size_t i = 0; i < len; i++)
		{
			cout << "row= " << colspace[i]->row+1 << " ";
		}
		cout << endl;

		for (size_t i = 0; i < len; i++)
		{
			cout << "val= " << colspace[i]->val << " ";
		}
		cout << endl;


		for (size_t i = 0; i < len; i++)
		{
			cout << "reverse->row= " << colspace[i]->reverse->row + 1 << " ";
		}
		cout << endl;

		for (size_t i = 0; i < len; i++)
		{
			cout << "next->row= " << colspace[i]->next->row + 1 << " ";
		}
		cout << endl;
	}

	template<typename Tv>
	size_t compressCol(vector<LLp<Tv>*> &colspace, size_t len, ApproxCholPQ &pq) {

		sort(colspace.begin(), colspace.begin()+len, [](auto x1, auto x2) { return x1->row < x2->row; });

		size_t ptr = 0;
		size_t currow = SIZE_MAX;

		vector<LLp<Tv>*> &c = colspace;

		for (size_t i = 0; i < len; i++)
		{
			if (c[i]->row != currow) {

				currow = c[i]->row;
				c[ptr] = c[i];
				ptr++;
			}
			else
			{
				c[ptr-1]->val += c[i]->val;
				c[i]->reverse->val = 0;

				pq.dec(currow);
			}
		}

		sort(colspace.begin(), colspace.begin() + ptr, [](auto x1, auto x2) { return x1->val < x2->val; });

		return ptr;
	}

	template<typename Tv>
	size_t compressCol(vector<LLcol<Tv>> &colspace, size_t len)
	{
		sort(colspace.begin(), colspace.begin()+1, [](auto x1, auto x2) {return x1.row < x2.row; });

		vector<LLcol<Tv>> &c = colspace;
		size_t ptr = 0;
		size_t currow = c[0].row;
		Tv curval = c[0].val;
		size_t curptr = c[0].ptr;

		for (size_t i = 1; i < len; i++)
		{
			if (c[i].row != currow) {

				c[ptr] = LLcol<Tv>(currow, curptr, curval); //next is abuse here: really keep where it came from.
				++ptr;

				currow = c[i].row;
				curval = c[i].val;
				curptr = c[i].ptr;
			}
			else
				curval = curval + c[i].val;
		}

		// emit the last row

		c[ptr] = LLcol<Tv>(currow, curptr, curval);
		ptr++;

		sort(colspace.begin(), colspace.begin() + ptr, [](auto x1, auto x2) { return x1.val < x2.val; });

		return ptr;
	}

	// this one is greedy on the degree - also a big win
	template<typename Tv>
	LDLinv<Tv> approxChol(LLmatp<Tv> &a) {
		
		size_t n = a.n;
		LDLinv ldli(a);
		size_t ldli_row_ptr = 0;

		vector<Tv> d(n, 0);
		ApproxCholPQ pq(a.degs);

		size_t it = 0;

		vector<LLp<Tv>*> colspace(n);
		vector<Tv> cumspace(n);

		vector<Tv> vals(n);
		size_t len=0;

		while (it + 1 < n) {

				
			size_t i = pq.pop();


			ldli.col[it] = i-1;
			ldli.colptr[it] = ldli_row_ptr;
			
			it++;

			len = get_ll_col(a, i, colspace);//changed

			len = compressCol(colspace, len, pq);

			Tv csum = 0;

			for (size_t ii = 0; ii < len; ii++)
			{
				vals[ii] = colspace[ii]->val;
				csum += colspace[ii]->val;
				cumspace[ii] = csum;
			}


			Tv wdeg = csum;
			Tv colScale = 1;

			Random<double> rnd;

			for (size_t joffset = 0; joffset + 1 < len; joffset++)
			{
				LLp<Tv>* ll = colspace[joffset];
				
				
				Tv w = vals[joffset] * colScale;
				size_t j = ll->row;


				LLp<Tv>* revj = ll->reverse;

				Tv f = w / wdeg;

				vals[joffset] = 0;

				Tv rand0_1 = rnd.rand0_1();
				Tv r = rand0_1 * (csum - cumspace[joffset]) + cumspace[joffset];
				//Tv r = 0.2 * (csum - cumspace[joffset]) + cumspace[joffset];

				auto firstit = find_if(cumspace.begin(), cumspace.begin()+len, [=](auto x) { return x > r; });

				size_t koff;

				if (firstit != cumspace.end())
					koff = firstit - cumspace.begin();
				else koff = 0;

				size_t k = colspace[koff]->row;

				
				pq.inc(k);

				Tv newEdgeVal = f * (1 - f)*wdeg;

				//fix row k in col j
				revj->row = k;	//dense time hog: presumably because of cache
				revj->val = newEdgeVal;
				revj->reverse = ll;

				//fix row j in col k

				LLp<Tv>* khead = a.cols[k];
				a.cols[k] = ll;
				ll->next = khead;
				ll->reverse = revj;
				ll->val = newEdgeVal; 
				ll->row = j;

				colScale = colScale * (1 - f);
				wdeg = wdeg * std::pow(1 - f, 2);

				ldli.rowval.push_back(j);
				ldli.fval.push_back(f);
				ldli_row_ptr++;

			}
			
	
			LLp<Tv>* ll = colspace[len-1];
			Tv w = vals[len-1] * colScale;
			size_t j = ll->row;
			LLp<Tv>* revj = ll->reverse;

			if (it + 1 < n) {
				pq.dec(j);
			}

			revj->val = 0;

			ldli.rowval.push_back(j);
			ldli.fval.push_back(1);
			ldli_row_ptr++;

			d[i-1] = w;

		}

		ldli.colptr[it] = ldli_row_ptr;
		ldli.d = d;

		return ldli;

	}

	//The routines that do the solve.
	template<typename Tv>
	void forward(const LDLinv<Tv> &ldli, DynamicVector<Tv> &y) {

		for (size_t ii = 0; ii < ldli.col.size(); ii++)
		{
			size_t i = ldli.col[ii];

			size_t j0 = ldli.colptr[ii];
			size_t j1 = ldli.colptr[ii + 1] - 1;

			Tv yi = y[i];
			
			//cout << "\nii=" << ii+1 <<"======="<< endl;
			//cout << "\ni= " << i+1 << " j0= " << j0+1 << " j1= " << j1+1 << endl;
			
			for (size_t jj = j0; jj < j1; jj++)
			{
				size_t j = ldli.rowval[jj];
				y[j] += ldli.fval[jj] * yi;
				yi *= (1 - ldli.fval[jj]);
			}

			size_t j = ldli.rowval[j1];
			y[j] += yi;
			y[i] = yi;
		}
	}

	template<typename Tv>
	void backward(const LDLinv<Tv> &ldli, DynamicVector<Tv> &y) {
		
		size_t sz = ldli.col.size();

		for (size_t ii = 0; ii < sz; ++ii)
		{
			
			size_t iib = sz - ii - 1;
			size_t i = ldli.col[iib];

			size_t j0 = ldli.colptr[iib];
			size_t j1 = ldli.colptr[iib + 1] - 1;

			size_t j = ldli.rowval[j1];
			Tv yi = y[i] + y[j];

			

			for (size_t jj = 0; jj < j1-j0; jj++)
			{
				size_t jjb = j1 - jj - 1;
				
				size_t j = ldli.rowval[jjb];
				yi = (1 - ldli.fval[jjb])*yi + ldli.fval[jjb] * y[j];
				
			}

			y[i] = yi;

		}
	}

	template<typename Tv>
	DynamicVector<Tv> LDLsolver(const LDLinv<Tv> &ldli,  const DynamicVector<Tv> &b) {

		DynamicVector<Tv> y = b;
		
		
		forward(ldli, y);
		

		for (size_t i = 0; i < ldli.d.size(); i++)
			//if (abs(ldli.d[i]) > 1e-6)
			if (ldli.d[i]!=0)
				y[i] /= ldli.d[i];

		backward(ldli, y);

		Tv mu = mean(y);

		for (size_t i = 0; i < y.size(); i++)
			y[i] = y[i]-mu;

		
		//cout << "\n\ny=" << y;
		return y;
	}

	template<typename Tv>
	SubSolver<Tv> approxchol_lapGreedy(const CompressedMatrix<Tv, blaze::columnMajor> &a, vector<size_t>& pcgIts, float tol = 1e-6,
		double maxits = 1000, double maxtime = HUGE_VAL, bool verbose = false,
		ApproxCholParams params = ApproxCholParams())
	{
		//auto t1 = high_resolution_clock::now();
		
		CompressedMatrix<Tv, blaze::columnMajor> la=lap(a);
		LLmatp<Tv> llmat(a);
		LDLinv <Tv> ldli = approxChol(llmat);

		SolverB<Tv> F = [=](const DynamicVector<Tv>& b) {
			return LDLsolver(ldli, b);
		};

		return SubSolver<Tv>([=](const DynamicVector<Tv>& b, vector<size_t>& pcgIts) {
			//cout << "b=" << b;

			Tv mn = mean(b);
			DynamicVector<Tv> b1(b.size());
			
			for (size_t i = 0; i < b.size(); i++)
				b1[i]= b[i] - mn;

			//cout << "b1=" << b1;

			DynamicVector<Tv> res = pcg(la, b1, F, pcgIts, tol, maxits, maxtime, verbose, params.stag_test);
			return res;
		});

	}

	template<typename Tv>
	SubSolver<Tv> approxchol_lap1(const CompressedMatrix<Tv, blaze::columnMajor> &a, vector<size_t>& pcgIts, float tol = 1e-6,
		double maxits = 1000, double maxtime = HUGE_VAL, bool verbose = false,
		ApproxCholParams params = ApproxCholParams())
	{
		//if (params.order == ApproxCholEnum::Deg) //not implemented yet
			return approxchol_lapGreedy(a, pcgIts, tol, maxits, maxtime, verbose, params);
	}

	template<typename Tv>
	SubSolver<Tv> approxchol_lap(const CompressedMatrix<Tv, blaze::columnMajor> &a, vector<size_t>& pcgIts, float tol = 1e-6,
		double maxits = 1000, double maxtime = HUGE_VAL, bool verbose = false,
		ApproxCholParams params = ApproxCholParams())
	{

		return lapWrapComponents(SolverA<Tv>(approxchol_lap1<Tv>), a, pcgIts, tol, maxits, maxtime, false);
	}

	/* ApproxCholPQ
	It only implements pop, increment key, and decrement key.
	All nodes with degrees 1 through n appear in their own doubly - linked lists.
	Nodes of higher degrees are bundled together.B
	*/

	inline void ApproxCholPQ::move(size_t i, size_t newkey, size_t oldlist, size_t newlist) {

		size_t prev = elems[i].prev;
		size_t next = elems[i].next;

		// remove i from its old list

		if (next != SIZE_MAX)
			elems[next] = ApproxCholPQElem(prev, elems[next].next, elems[next].key);

		if (prev != SIZE_MAX) 
			elems[prev] = ApproxCholPQElem(elems[prev].prev, next, elems[prev].key);
		else
			lists[oldlist] = next;

		// insert i into its new list
		size_t head = lists[newlist];

		if (head != SIZE_MAX) 
			elems[head] = ApproxCholPQElem(i, elems[head].next, elems[head].key);
		
		lists[newlist] = i;
		elems[i] = ApproxCholPQElem(SIZE_MAX, head, newkey);
	}

	//Increment the key of element i
	//This could crash if i exceeds the maxkey

	inline void ApproxCholPQ::inc(size_t i) {

		size_t oldlist = keyMap(elems[i].key, n);
		size_t newlist = keyMap(elems[i].key + 1, n);

		if (newlist != oldlist)
			move(i, elems[i].key + 1, oldlist, newlist);
		else
			elems[i] = ApproxCholPQElem(elems[i].prev, elems[i].next, elems[i].key + 1);

	}

	inline size_t ApproxCholPQ::pop() {
		assert(nitems != 0);

		while (lists[minlist] == SIZE_MAX)
			minlist++;

		size_t i = lists[minlist];
		size_t next = elems[i].next;

		lists[minlist] = next;

		if (next != SIZE_MAX)
			elems[next] = ApproxCholPQElem(SIZE_MAX, elems[next].next, elems[next].key);

		nitems--;

		return i+1;
	}

	// Decrement the key of element i
	// This could crash if i exceeds the maxkey

	inline void ApproxCholPQ::dec(size_t i) {

		size_t oldlist = keyMap(elems[i].key, n);
		size_t newlist = keyMap(elems[i].key - 1, n);

		if (newlist != oldlist) {
			move(i, elems[i].key - 1, oldlist, newlist);

			if (newlist < minlist)
				minlist = newlist;
		}
		else
			elems[i] = ApproxCholPQElem(elems[i].prev, elems[i].next, elems[i].key - 1);
	}
}
