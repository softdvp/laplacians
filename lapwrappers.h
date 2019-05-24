#pragma once

#include <blaze/Math.h>
#include <vector>
#include <functional>
#include <chrono>
#include "graphalgs.h"
#include "approxchol.h"

using namespace std;
using blaze::CompressedMatrix;
using blaze::DynamicMatrix;
using blaze::DynamicVector;
using std::vector;
using namespace std::chrono;

template <typename Tv>
DynamicVector<Tv> nullSolver(const DynamicVector<Tv>& a, vector<size_t>& pcg, float f, double d,
	double d1, bool b, ApproxCholParams prm) {

	return DynamicVector<Tv>(1, 0);
}

//	lapWrapComponents function

//Apply the ith solver on the ith component
template <typename Tv>
SolverB<Tv> BlockSolver(const vector<vector<size_t>> &comps, vector<SubSolver<Tv>> &solvers,
	vector<size_t>& pcgIts, float tol = 1e-6F, double maxits = HUGE_VAL, double maxtime = HUGE_VAL,
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

template <typename Tv>
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

template <typename Tv>
SolverAMat<Tv> wrapInterfaceMat(const FactorSolver<Tv> solver) {
	return SolverAMat<Tv>([=](const CompressedMatrix<Tv, blaze::columnMajor> &a, vector<size_t>& pcgIts, float tol = 1e-6,
		double maxits = HUGE_VAL, double maxtime = HUGE_VAL, bool verbose = false, ApproxCholParams params = ApproxCholParams())
	{
		return wrapInterfaceMat(solver, a, pcgIts, tol, maxits, maxtime, verbose, params);
	});
}

template <typename Tv>
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

template <typename Tv>
SolverA<Tv> wrapInterface(const FactorSolver<Tv> solver) {
	return SolverA<Tv>([=](const CompressedMatrix<Tv, blaze::columnMajor> &a, vector<size_t>& pcgIts, float tol = 1e-6,
		double maxits = HUGE_VAL, double maxtime = HUGE_VAL, bool verbose = false, ApproxCholParams params = ApproxCholParams())
	{
		return wrapInterface(solver, a, pcgIts, tol, maxits, maxtime, verbose, params);
	});
}

//This functions wraps cholfact so that it satisfies our interface.
template <typename Tv>
SolverAMat<Tv> chol_sddm_mat() {
	return wrapInterfaceMat<Tv>(cholesky<Tv>);
}

template <typename Tv>
SolverA<Tv> chol_sddm() {
	return wrapInterface<Tv>(cholesky<Tv>);
}

//			Applies a Laplacian `solver` that satisfies our interface to each connected component of the graph with adjacency matrix `a`.
template <typename Tv>
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

	CompressedMatrix<Tv, blaze::columnMajor>lasub = index<Tv>(la, leave, leave);
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

template <typename Tv>
SolverA<Tv> lapWrapConnected(const SolverA<Tv> solver) {
	return SolverA<Tv>([=](const CompressedMatrix<Tv, blaze::columnMajor> &a, vector<size_t>& pcgIts, float tol = 1e-6,
		double maxits = HUGE_VAL, double maxtime = HUGE_VAL, bool verbose = false,
		const ApproxCholParams params = ApproxCholParams())
	{
		return lapWrapConnected(solver, a, pcgIts, tol, maxits, maxtime, verbose, params);
	});
}

template <typename Tv>
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

		SubSolver<Tv> s = solver(a, pcgIts, tol, maxits, maxtime, verbose, params);

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

			CompressedMatrix<Tv> asub = index<Tv>(a, ind, ind);

			SubSolver<Tv> subSolver;

			if (ind.size() == 1) {

				subSolver = SubSolver<Tv>(nullSolver<Tv>);
			}
			else
				if (ind.size() < 50) {

					vector<size_t>pcgits;
					subSolver = lapWrapConnected<Tv>(chol_sddm<Tv>(), asub, pcgits);
				}
				else {
					subSolver = solver(a, pcgIts, tol, maxits, maxtime, verbose, params);
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

template <typename Tv>
SolverA<Tv> lapWrapComponents(const SolverA<Tv> solver) {

	return SolverA<Tv>([=](const CompressedMatrix<Tv, blaze::columnMajor> &a, vector<size_t>& pcgIts, float tol = 1e-6,
		double maxits = HUGE_VAL, double maxtime = HUGE_VAL, bool verbose = false,
		const ApproxCholParams params = ApproxCholParams())
	{
		return lapWrapComponents(solver, a, pcgIts, tol, maxits, maxtime, verbose, params);
	});
}

template <typename Tv>
SolverA<Tv> lapWrapSDDM(SolverA<Tv> sddmSolver) {
	return lapWrapComponents(lapWrapConnected(sddmSolver));
}

template <typename Tv>
SolverB<Tv> sddmWrapLap(SolverA<Tv> lapSolver, const CompressedMatrix<Tv, blaze::columnMajor> &sddm, vector<size_t>& pcgIts,
	float tol = 1e-6, double maxits = HUGE_VAL, double maxtime = HUGE_VAL, bool verbose = false,
	ApproxCholParams params = ApproxCholParams())
{
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

template <typename Tv>
SolverA<Tv> sddmWrapLap(const SolverA<Tv> solver) {

	return SolverA<Tv>([=](const CompressedMatrix<Tv, blaze::columnMajor> &a, vector<size_t>& pcgIts, float tol = 1e-6,
		double maxits = HUGE_VAL, double maxtime = HUGE_VAL, bool verbose = false,
		const ApproxCholParams params = ApproxCholParams())
	{
		return sddmWrapLap(solver, a, pcgIts, tol, maxits, maxtime, verbose, params);
	});
}





