/*
To generate extension module with both sparse and dense methods in the
same binary.

*/

#include "svm.h"

#define _DENSE_REP
#include "svm.cpp"
//#undef _DENSE_REP
//#include "svm.cpp"

