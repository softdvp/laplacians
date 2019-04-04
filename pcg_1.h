#pragma once
#include <blaze/Math.h>

using blaze::DynamicVector;

template <typename El, typename Vec>
void axpy2(const El al, const Vec &p, Vec &x) {
	x = x + al * p;
}

template <typename El, typename Vec>
void bzbeta(const El beta, Vec &p, const Vec &z) {
	p = z + beta * p;

	return;
}
