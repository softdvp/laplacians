#include <iostream>
#include <stdint.h>
#include <vector>
#include "approxchol.h"

/* ApproxCholPQ
It only implements pop, increment key, and decrement key.
All nodes with degrees 1 through n appear in their own doubly - linked lists.
Nodes of higher degrees are bundled together.
*/


size_t keyMap(size_t x, size_t n) {
	return x <= n ? n : n + x / n;
}

void ApproxCholPQ::move(size_t i, size_t newkey, size_t oldlist, size_t newlist) {

	size_t prev = elems[i].prev;
	size_t next = elems[i].next;

	// remove i from its old list

	if (next != SIZE_MAX)
	{
		ApproxCholPQElem newpqel(prev, elems[next].next, elems[next].key);
		elems[next] = newpqel;
	}

	if (prev != SIZE_MAX) {
		ApproxCholPQElem newpqel(elems[prev].prev, next, elems[prev].key);
		elems[prev] = newpqel;
	}
	else
		lists[oldlist] = next;

	// insert i into its new list
	size_t head = lists[newlist];

	if (head != SIZE_MAX) {
		ApproxCholPQElem newpqel(i, elems[head].next, elems[head].key);
		elems[head] = newpqel;
	}

	lists[newlist] = i;
	elems[i] = ApproxCholPQElem(SIZE_MAX, head, newkey);
}

//Increment the key of element i
//This could crash if i exceeds the maxkey

void ApproxCholPQ::inc(size_t i) {

	size_t oldlist = keyMap(elems[i].key, n);
	size_t newlist = keyMap(elems[i].key + 1, n);

	if (newlist != oldlist)
		move(i, elems[i].key + 1, oldlist, newlist);
	else
		elems[i] = ApproxCholPQElem(elems[i].prev, elems[i].next, elems[i].key + 1);

}

size_t ApproxCholPQ::pop() {
	assert(nitems != 0);

	while (lists[minlist] == SIZE_MAX)
		minlist++;

	size_t i = lists[minlist];
	size_t next = elems[i].next;

	lists[minlist] = next;

	if (next != SIZE_MAX)
		elems[next] = ApproxCholPQElem(SIZE_MAX, elems[next].next, elems[next].key);

	nitems--;

	return i;
}

// Decrement the key of element i
// This could crash if i exceeds the maxkey

void ApproxCholPQ::dec(size_t i) {

	size_t oldlist = keyMap(elems[i].key, n);
	size_t newlist = keyMap(elems[i].key - 1, n);

	if (newlist != oldlist) {
		move(i, elems[i].key - 1, oldlist, newlist);

		if (newlist < minlist)
			minlist = newlist;
	}
	else
	{
		ApproxCholPQElem newpqel(elems[i].prev, elems[i].next, elems[i].key - 1);
		elems[i] = newpqel;
	}
}
