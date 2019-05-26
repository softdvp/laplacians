#pragma once

template <typename Tv>
class Factorization {
public:
	CompressedMatrix<Tv, blaze::columnMajor> Lower;
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

//  Types of solvers for A*x=B where B is a matrix

template <typename Tv>
using FactorSolver = function<Factorization<Tv>(const CompressedMatrix<Tv, blaze::columnMajor>&)>;


//Result of wrappers
//Function: pass B matrix, returns x vector

template <typename Tv>
using SolverBMat = function<DynamicVector<Tv>(const CompressedMatrix<Tv, blaze::columnMajor>&)>;

//Result of SolverA functor
//Convert SolverA to a function with 1 paramater B - SolverB
template <typename Tv>
using SubSolverFuncMat = function <DynamicVector<Tv>(const CompressedMatrix<Tv, blaze::columnMajor>&, vector<size_t>&)>;

template <typename Tv>
class SubSolverMat {
	SubSolverFuncMat<Tv> Solver;
public:
	SubSolverMat(SubSolverFuncMat<Tv> Asolver) : Solver(Asolver) {};

	SubSolverMat(SolverBMat<Tv> solver) {

		Solver = [=](const CompressedMatrix<Tv, blaze::columnMajor> &b, vector<size_t>& pcgIts) {

			return solver(b);
		};
	}

	DynamicVector<Tv>operator()(const CompressedMatrix<Tv, blaze::columnMajor> &b, vector<size_t>& pcgIts) {

		return Solver(b, pcgIts);
	}

	DynamicVector<Tv>operator()(const CompressedMatrix<Tv, blaze::columnMajor> &b) {
		vector<size_t> pcgIts;

		return Solver(b, pcgIts);
	}
};

// Function: pass A matrix, return SubSolver
template <typename Tv>
using SolverAFuncMat = std::function<SubSolverMat<Tv>(const CompressedMatrix<Tv, blaze::columnMajor>&, vector<size_t>&)>;

template <typename Tv>
class SolverAMat {
	SolverAFuncMat<Tv> Solver;
public:
	SolverAMat(SolverAFuncMat<Tv> solver) : Solver(solver) {}

	SubSolverMat<Tv> operator()(const CompressedMatrix<Tv, blaze::columnMajor> &a, vector<size_t>& pcgIts) {

		return Solver(a, pcgIts);
	}

	SubSolverMat<Tv> operator()(const CompressedMatrix<Tv, blaze::columnMajor> &a) {
		vector<size_t> pcgIts;
		return Solver(a, pcgIts);
	}
};

//  Solvers for A*x=b where b is a vector 

//Result of wrappers
//Function: pass B matrix, returns x vector

template <typename Tv>
using SolverB = function<DynamicVector<Tv>(const DynamicVector<Tv>&)>;

//Result of SolverA functor
//Convert SolverA to a function with 1 paramater B - SolverB

template <typename Tv>
using SubSolverFunc = std::function <DynamicVector<Tv>(const DynamicVector<Tv>&, vector<size_t>&)>;

template <typename Tv>
class SubSolver {
	SubSolverFunc<Tv> Solver;
public:
	SubSolver() {}

	SubSolver(SubSolverFunc<Tv> Asolver) : Solver(Asolver) {};

	SubSolver(SolverB<Tv> solver) {

		Solver = [=](const DynamicVector<Tv> &b, vector<size_t>& pcgIts) {

			return solver(b);
		};
	}

	/*SubSolver &operator=(const SubSolver &s) {
		Solver = s.Solver;

		return *this;
	}*/

	DynamicVector<Tv>operator()(const DynamicVector<Tv> &b, vector<size_t>& pcgIts) {
		return Solver(b, pcgIts);
	}

	DynamicVector<Tv>operator()(const DynamicVector<Tv> &b) {
		vector<size_t> pcgIts;
		return Solver(b, pcgIts);
	}
};

// Function: pass A matrix, return SubSolver
template <typename Tv>
using SolverAFunc = function<SubSolver<Tv>(const CompressedMatrix<Tv, blaze::columnMajor>&, vector<size_t>&, float, double,
	double, bool, ApproxCholParams)>;

template <typename Tv>
class SolverA {
	SolverAFunc<Tv> Solver;
public:
	SolverA(SolverAFunc<Tv> solver) : Solver(solver) {}

	SubSolver<Tv> operator()(const CompressedMatrix<Tv, blaze::columnMajor> &a, vector<size_t>& pcgIts,
		float tol = 1e-6, double maxits = HUGE_VAL, double maxtime = HUGE_VAL, bool verbose = false,
		ApproxCholParams params = ApproxCholParams()) {

		return Solver(a, pcgIts, tol, maxits, maxtime, verbose, params);
	}

	SubSolver<Tv> operator()(const CompressedMatrix<Tv, blaze::columnMajor> &a) {
		vector<size_t> pcgIts;
		return Solver(a, pcgIts, 1e-6F, HUGE_VAL, HUGE_VAL, false, ApproxCholParams());
	}
};

