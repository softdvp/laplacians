#pragma once

#include <cmath>
#include <blaze/Math.h>
#include <vector>
#include <algorithm>
#include "solvertypes.h"
#include "approxchol.h"
#include "graphalgs.h"
#include "crandom.h"

using namespace std;
using blaze::CompressedMatrix;
using blaze::DynamicMatrix;
using blaze::DynamicVector;

/*
	Just implements Spielman-Srivastava
		as = sparsify(a; ep=0.5)

	Apply Spielman-Srivastava sparsification: sampling by effective resistances.
	`ep` should be less than 1.
*/


namespace laplacians {
	template<typename Tv>
	CompressedMatrix<Tv, blaze::columnMajor> sparsify(const CompressedMatrix<Tv, blaze::columnMajor>& a,
		float ep = 0.3F, float matrixConcConst = 4.0F, float JLfac = 4.0F) {

		Random<double> rnd;

		vector<size_t> pcgIts;
		
		SolverB<Tv> f = approxchol_lap(a, pcgIts, 1e-2F);

		size_t n = a.rows();
		size_t k = (size_t)round(JLfac * log(n)); //number of dims for JL

		CompressedMatrix<Tv, blaze::columnMajor> U = wtedEdgeVertexMat(a);

		size_t m = U.rows();

		DynamicMatrix<double, blaze::columnMajor> R(m, k);

		for (size_t i = 0; i < m; i++)
			for (size_t j = 0; j < k; j++)
				R(i, j) = rnd.randn();

		CompressedMatrix<Tv, blaze::columnMajor> UR = adjoint(U) * R;

		CompressedMatrix<Tv, blaze::columnMajor> V(n, k, 0);

		for (size_t i = 0; i < k; i++) {
			DynamicVector<Tv> x, b;

			vector<size_t> idx = collect(0, UR.rows());
			b = index(UR, idx, i);
			
			x = f(b);

			idx = collect(0, V.rows());
			index(V, idx, i, x);
		}

		auto [ai, aj, av] = findnz(triu(a));

		DynamicVector<Tv> prs(av.size());
		for (size_t h = 0; h < av.size(); h++)
		{
			size_t i = ai[h];
			size_t j = aj[h];

			DynamicVector<Tv>vi, vj, vr;

			vector<size_t> idx = collect(0, V.columns());
			vi = index(V, i, idx);
			vj = index(V, j, idx);
			Tv nr = std::pow(norm(vi - vj), 2)/ k;
			Tv tmp = av[h] * nr * matrixConcConst * log(n) / std::pow(ep, 2);
			prs[h] = (1 < tmp) ? 1 : tmp;
		}

		vector<bool>ind(prs.size());
		DynamicVector<double> rndvec = rnd.randv(prs.size());

		/*DynamicVector<double> rndvec(prs.size());

		for (size_t i = 0; i < rndvec.size(); i++)
			rndvec[i] = crand01();*/
		
		for (size_t i = 0; i < prs.size(); i++)
			ind[i] = rndvec[i] < prs[i];

		vector<size_t> ai_ind = indexbool(ai, ind);
		vector<size_t> aj_ind = indexbool(aj, ind);
		DynamicVector<Tv> av_ind = indexbool(av, ind);
		DynamicVector<Tv> prs_ind = indexbool(prs, ind);

		DynamicVector<Tv> divs = av_ind / prs_ind;
		CompressedMatrix<Tv, blaze::columnMajor> as = sparse(ai_ind, aj_ind, divs, n, n);

		as = as + adjoint(as);

		return as;
	}
}