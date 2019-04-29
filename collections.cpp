#include <iostream>
#include "collections.h"
#include "approxchol.h"

using std::cout;

void dump_ijv(int ijvn, IJV<int> &ijv) {
	Laplacians<int> lapl;

	cout << "ijv" << ijvn << " matrix dump:\n";

	cout << "\n" << "n= " << ijv.n;
	cout << "\n" << "nnz= " << lapl.nnz(ijv);

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

const DynamicVector<DynamicVector<size_t>>vecToComps(DynamicVector<size_t> &compvec) {
		
		size_t nc = blaze::max(compvec);

		DynamicVector<DynamicVector<size_t>> comps(nc);

		DynamicVector<size_t> sizes(nc, 0);

		for(size_t i:compvec) 
			sizes[i-1]++;
		
		for(size_t i = 0; i < nc; i++)
			comps[i].resize(sizes[i]);

		DynamicVector<size_t>ptrs(nc, 0);

		for (size_t i = 0; i < compvec.size(); i++)
		{
			size_t c = compvec[i]-1;
			
			comps[c][ptrs[c]++] = i;
		}
		
		return comps;
	}

