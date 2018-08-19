#include <stdlib.h>
#include <numpy/arrayobject.h>
#include "svm.h"

/*
	Some utility funcions for converting NumPy arrays to LIBSVM structures.

*/


struct SVMNode* denseToLIBSVM(double *x, npy_intp* dims)
{
	struct SVMNode *node;
	npy_intp len_row = dims[1];
    double *tx = x;
	int i;

	node = malloc(dims[0] * sizeof(struct SVMNode));

}

void setProblem(struct SVMProblem *prob, char *X, char *Y, npy_intp *dims)
{
	problem->l = (int) dims[0]; // Number of samples
	problem->y = (double *) Y;


}
