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
	npy_intp len_col = dims[0];
    double *tx = x;
	int i;

	printf("Row: %d, Col: %d\n", (int) len_row, (int) len_col);

	node = (struct SVMNode*) malloc(dims[0] * sizeof(struct SVMNode));

	if(node == NULL) return NULL;
	for(i = 0; i < dims[0]; ++i)
	{
		node[i].values = tx;
		node[i].dim = (int) len_row;
		node[i].ind = i;

		tx += len_row;
	}


	return node;

}


// Filling struct SVMParameter
void setParameter(struct SVMParameter *param, double gamma, double C, double e)
{
    param->C = C;
    param->gamma = gamma;
    param->e = e;
}


void setProblem(struct SVMProblem *prob, char *X, char *Y, npy_intp *dims)
{
	prob->l = (int) dims[0]; // Number of samples
	prob->y = (double *) Y;
	prob->x = denseToLIBSVM((double *) X, dims);

}

