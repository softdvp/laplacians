#pragma once
#include <blaze/Math.h>
#include <iostream>
using blaze::DynamicVector;
using blaze::CompressedMatrix;

template <typename Tv>
class IJV {
public:
	std::size_t  n;
	std::size_t nnz;
	DynamicVector<std::size_t> i; //colptr
	DynamicVector<std::size_t> j; //rowval
	DynamicVector<Tv> v; //nonzero elements

	IJV(const IJV &a) {
		n = a.n;
		nnz = a.nnz;
		i = a.i;
		j = a.j;
		v = a.v;
	}

	IJV(const std::size_t  an,	const std::size_t annz,
	const DynamicVector<std::size_t> ai,
	const DynamicVector<std::size_t> aj,
	const DynamicVector<Tv> av) {
		n = an;
		nnz = annz;
		i = ai;
		j = aj;
		v = av;
	}

	IJV& operator=(const IJV &a) {
		n = a.n;
		nnz = a.nnz;
		i = a.i;
		j = a.j;
		v = a.v;

		return *this;
	}
	
	IJV(CompressedMatrix<Tv, blaze::columnMajor> &mat) {

		n = mat.rows();
		nnz = mat.nonZeros();

		i.resize(n + 1);
		i[0] = 0;
		i[n] = nnz;

		j.resize(nnz);
		v.resize(nnz);

		std::size_t k = 0; 

		//Fill i, j and v
		
		std::size_t totalnz = 0;

		for (size_t l = 0UL; l < mat.rows(); ++l) {
			std::size_t rownz = 0;

			for (typename CompressedMatrix<Tv, blaze::columnMajor>::Iterator it = mat.begin(l); it != mat.end(l); ++it) {
				
				 v[k]= it->value();  
				 j[k]= it->index();  
				 ++k;
				 ++rownz;
			}
			totalnz += rownz;
			i[l+1] = totalnz;
		}
	}
	/*== (a::IJV, b::IJV) =
		a.n == b.n &&
		a.nnz == b.nnz &&
		a.i == b.i &&
		a.j == b.j &&
		a.v == b.v*/

	/*function *(a::IJV, x::Number)
		ijv = deepcopy(a)
		ijv.v .* = x

		return ijv
		end
		*/

};

template <typename Tv>
std::size_t nnz(const IJV<Tv> &a) {
	return a.nnz;
}
