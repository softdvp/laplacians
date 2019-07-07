#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>

#include <iostream>
#include "tests.h"
#include "sparsification_test.h"

int main()
{
	pcg_tests();

	IJVtests();

	CollectionTest();

	CollectionFunctionTest();

	sparsification_test();

	_CrtDumpMemoryLeaks();

}

