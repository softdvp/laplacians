#include "crandom.h"

const unsigned long randmax = 32767;
unsigned long long next = 1;

long long crand(void)
{
	next = (unsigned int)((next * 1103515245 + 12345) / 65536) % randmax;

	return next;
}

double crand01(void) {
	return (double)crand() / randmax;
}

double crandn(void) {
	return (double)crand() / randmax * 5.2 - 2.6;
}
