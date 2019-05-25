#pragma once

#include <blaze/Math.h>
#include <vector>
#include "solvertypes.h"
#include "approxchol.h"
#include "graphalgs.h"

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

template<typename Tv>
CompressedMatrix<Tv, blaze::columnMajor> sparsify(const CompressedMatrix<Tv, blaze::columnMajor>& a,
	float ep=0.3F, float matrixConcConst=4.0F, float JLfac=4.0F) {

	vector<size_t> pcgIts;

	SolverB<Tv> f = approxchol_lap(a, PcgIts, 1e-2);

	size_t n = a.rows();

	size_t k = roundl(JLfac*log(n));

	CompressedMatrix<Tv, blaze::columnMajor> U = wtedEdgeVertexMat(a);

	size_t m = U.rows();
	double R = randn(m, k);
	CompressedMatrix<Tv, blaze::columnMajor> UR = ctrans(U)*R;
	

}


/*
function sparsify(a; ep=0.3, matrixConcConst=4.0, JLfac=4.0)

  f = approxchol_lap(a,tol=1e-2);

  n = size(a,1)
  k = round(Int, JLfac*log(n)) # number of dims for JL

  U = wtedEdgeVertexMat(a)
  m = size(U,1)
  R = randn(Float64, m,k)
  UR = U'*R;

  V = zeros(n,k)
  for i in 1:k
	V[:,i] = f(UR[:,i])
  end

  (ai,aj,av) = findnz(triu(a))
  prs = zeros(size(av))
  for h in 1:length(av)
	  i = ai[h]
	  j = aj[h]
	  prs[h] = min(1,av[h]* (norm(V[i,:]-V[j,:])^2/k) * matrixConcConst *log(n)/ep^2)
  end

  ind = rand(Float64,size(prs)) .< prs

  as = sparse(ai[ind],aj[ind],av[ind]./prs[ind],n,n)
  as = as + as'

  return as

end

*/

