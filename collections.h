#pragma once
#include <blaze/Math.h>
#include <iostream>
#include <boost/container_hash/hash.hpp>

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

	IJV():n(0), nnz(0), i(0), j(0), v(0){}

	const IJV operator+(const IJV b) const {
		IJV m;
		m.n = n;
		m.nnz = nnz + b.nnz;

		return m;
	}
		
	const bool operator== (const IJV b) const
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

	const IJV& operator=(const IJV &a) {
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

		//Fill i, row and v
		
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
};

template <typename Tv>
const std::size_t nnz(const IJV<Tv> &a) {
	return a.nnz;
}

 template <typename Tv>
CompressedMatrix<Tv, blaze::columnMajor> sparse(const IJV<Tv> &ijv) {
	CompressedMatrix<Tv, blaze::columnMajor>res(ijv.n, ijv.n);
	res.reserve(ijv.nnz);       
	for (size_t i = 0; i != ijv.n; i++) {
		size_t colbegin = ijv.i[i];
		size_t colend = ijv.i[i + 1];

		for (size_t row = colbegin; row != colend; row++) {
			size_t rowv = ijv.j[row]; 
			Tv v = ijv.v[row];
			res.append(rowv, i, v); 
		}
		res.finalize(i);      
	}
	return res;
}

template <typename Tv>
IJV<Tv> operator* (const Tv x, const IJV<Tv> ijv) {

	return ijv*x;
}

/*template <typename Tv>
size_t hash(IJV<Tv> a) {

	return
}*/
/*hash(a::IJV) =
hash((a.n, a.nnz, a.i, a.j, a.v), hash(IJV))*/