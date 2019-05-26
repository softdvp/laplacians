/* 
	Implementations of cg and pcg.
	Look at two approaches: Matlab's, and
	hypre: https://github.com/LLNL/hypre/blob/master/src/krylov/pcg.c
	Started by Dan Spielman.
*/

/*
	x = pcg(mat, b, pre; tol, maxits, maxtime, verbose, pcgIts, stag_test)

	solves a symmetric linear system using preconditioner `pre`.
	# Arguments
	* `pre` can be a function or a matrix.  If a matrix, a function to solve it is created with cholFact.
	* `tol` is set to 1e-6 by default,
	* `maxits` defaults to MAX_VAL
	* `maxtime` defaults to MAX_VAL.  It measures seconds.
	* `verbose` defaults to false
	* `pcgIts` is an array for returning the number of pcgIterations.  Default is length 0, in which case nothing is returned.
	* `stag_test=k` stops the code if rho[it] > (1-1/k) rho[it-k].  Set to 0 to deactivate.
*/

#pragma once
#include <functional>
#include <chrono>
#include <blaze/Math.h>
#include "graphalgs.h"
#include "solvertypes.h"

using namespace std;
using blaze::CompressedMatrix;
using blaze::DynamicVector;
using namespace std::chrono;

namespace laplacians {

	const double EPS = 2.220446049250313e-16;

	template<typename Tv>
	void axpy2(const Tv al, const DynamicVector<Tv> &p, DynamicVector<Tv> &x) {
		x += al * p;
	}

	template <typename Tv>
	void bzbeta(const Tv beta, DynamicVector<Tv> &p, const DynamicVector<Tv> &z) {
		p = z + beta * p;
	}

	template<typename Tv>
	DynamicVector<Tv> pcg(const CompressedMatrix<Tv, blaze::columnMajor>& mat, const DynamicVector<Tv> &b, SolverB<Tv> pre,
		vector<size_t>& pcgIts, float tol = 1e-6, double maxits = HUGE_VAL, double maxtime = HUGE_VAL, bool verbose = false,
		size_t stag_test = 0) {

		Tv al;

		size_t n = mat.rows();

		Tv nb = blaze::norm(b);

		// If input vector is zero, quit

		if (abs(nb) < 1e-6)
			return DynamicVector<Tv>(b.size(), 0);

		DynamicVector<Tv> x(n, 0), bestx(n, 0), r, z, p;

		double bestnr = 1.0;

		r = b;
		z = pre(b);
		p = z;

		Tv rho = r * blaze::trans(z);
		Tv best_rho = rho;
		size_t stag_count = 0;

		auto t1 = high_resolution_clock::now();

		size_t itcnt = 0;

		while (itcnt++ < maxits)
		{
			DynamicVector<Tv> q = mat * p;

			Tv pq = p * blaze::trans(q);

			if (pq < EPS || pq >= HUGE_VAL) {
				if (verbose)
					cout << endl << "PCG Stopped due to small or large pq";

				break;
			}

			al = rho / q;

			// the following line could cause slowdown

			if (al* norm(p) < EPS*norm(x)) {
				if (verbose)
					cout << endl << "PCG: Stopped due to stagnation." << endl;

				break;
			}

			axpy2(al, p, x);
			axpy2(-al, q, r);

			Tv nr = norm(r) / nb;

			if (nr < bestnr) {
				bestnr = nr;
				bestx = x;
			}

			if (nr < tol)
				break;

			z = pre(r);

			Tv oldrho = rho;
			rho = z * blaze::trans(r); //this is gamma in hypre.

			if (stag_test != 0) // If stag_test=0 skip this check
				if (rho < best_rho*(1 - (Tv)(1 / stag_test))) {
					best_rho = rho;
					stag_count = 0;

				}
				else {
					if (stag_test > 0)
						if (best_rho > (1 - 1 / stag_test) * rho) {
							stag_count++;

							if (stag_count > stag_test)
							{
								if (verbose)
									cout << endl << "PCG Stopped by stagnation test.\n";

								break;
							}
						}
				}

			if (rho < EPS || rho >= HUGE_VAL) {
				if (verbose)
					cout << endl << "PCG Stopped due to small or large rho.\n";

				break;
			}

			Tv beta = rho / oldrho;

			if (beta < EPS || beta >= HUGE_VAL) {
				if (verbose)
					cout << endl << "PCG Stopped due to small or large beta.\n";

				break;
			}

			bzbeta(beta, p, z);

			auto t2 = high_resolution_clock::now();
			auto sec = duration_cast<milliseconds>(t2 - t1).count() * 1000;

			if (sec > maxtime)
			{
				if (verbose)
					cout << endl << "PCG stopped at maxtime.";

				break;
			}
		}

		auto t2 = high_resolution_clock::now();
		auto sec = duration_cast<milliseconds>(t2 - t1).count() * 1000;

		if (verbose) {
			cout << endl << "PCG stopped after: " << sec << " seconds and " << itcnt <<
				" iterations with relative error " << (norm(r) / norm(b)) << ".";
		}

		if (pcgIts.size())
			pcgIts[0] = itcnt;

		return bestx;
	}

	template<typename Tv>
	DynamicVector<Tv> pcg(const CompressedMatrix<Tv, blaze::columnMajor>& mat, const DynamicVector<Tv> &b, const CompressedMatrix<Tv, blaze::columnMajor>& pre,
		vector<size_t>& pcgIts, float tol = 1e-6, double maxits = HUGE_VAL, double maxtime = HUGE_VAL, bool verbose = false,
		size_t stag_test = 0)
	{
		Factorization<Tv> fact = cholesky(pre);

		SolverB<Tv> F = [=](const DynamicVector<Tv> &b) {
			DynamicVector<Tv> x = chol_subst(fact.Lower, b);
			return x;
		};

		return pcg(mat, b, F, pcgIts, tol, maxits, maxtime, verbose);
	}

	template<typename Tv>
	SubSolver<Tv> pcgSolver(const CompressedMatrix<Tv, blaze::columnMajor>& mat, SolverB<Tv> pre, vector<size_t>& pcgIts,
		float tol = 1e-6, double maxits = HUGE_VAL, double maxtime = HUGE_VAL, bool verbose = false) {

		return [=, &mat](const DynamicVector<Tv> &b, vector<size_t>& pcgIts) {
			return pcg(mat, b, pre, pcgIts, tol, maxits, maxtime, verbose);
		};
	}

	template<typename Tv>
	SubSolver<Tv> pcgSolver(const CompressedMatrix<Tv, blaze::columnMajor>& mat, const CompressedMatrix<Tv, blaze::columnMajor>& pre, vector<size_t>& pcgIts,
		float tol = 1e-6, double maxits = HUGE_VAL, double maxtime = HUGE_VAL, bool verbose = false) {

		return [=, &mat](const DynamicVector<Tv> &b, vector<size_t>& pcgIts) {
			return pcg(mat, b, pre, pcgIts, tol, maxits, maxtime, verbose);
		};
	}
}