// Laplacians.cpp : This file contains the 'main' function. Program execution begins and ends there.
//


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

}

