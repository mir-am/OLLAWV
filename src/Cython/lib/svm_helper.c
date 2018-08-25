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


/*
    Create and return an instance of SVMModel
*/
struct SVMModel *setModel(struct SVMParameter *param, int nrClass, char *SV, npy_intp *dimSV,
                          char *support, npy_intp supportDim, npy_intp *stridesSV, char* svCoef,
                          char* bias, char* nSV)
{


}


/*
    Convert from LIBSVM sparse representation and copy into NumPy array
*/
void copySvCoef(char *data, struct SVMModel *model)
{
    int i , len = model->numClass - 1;
    double *temp = (double *) data;

    for(i = 0; i < len; ++i)
    {
        memcpy(temp, model->svCoef[i], sizeof(double) * model->numSV);
        temp += model->numSV;
    }
}


/*
    Copy Bias from model struct
*/
void copyIntercept(char *data, struct SVMModel *model, npy_intp *dims)
{

    /* intercept = -rho */

    npy_intp i , n = dims[0];
    double t, *ddata = (double *) data;

    for(i = 0; i < n; ++i)
    {
        t = model->bias[i];

        *ddata = (t != 0) ? t : 0;
        ++ddata;
    }

}


/*
    Copy indices of SVs from model struct
*/
void copySupport(char *data, struct SVMModel *model)
{
    memcpy(data, model->svIndices, (model->numSV) * sizeof(int));
}


/*
    Copy SVs from sparse structures
*/
void copySV(char *data, struct SVMModel *model, npy_intp *dims)
{
    int i , n = model->numSV;
    double *tdata = (double *) data;
    int dim = model->SV[0].dim;

    for(i = 0; i < n; ++i)
    {
        memcpy(tdata, model->SV[i].values, dim * sizeof(double));
        tdata += dim;
    }
}


/*
    Copy number SVs for each class from model struct
*/
void copySVClass(char *data, struct SVMModel *model)
{
    memcpy(data, model->svClass, model->numClass * sizeof(int));
}


/*
    Get number of support vectors in a model
*/
npy_intp getNumSV(struct SVMModel *model)
{
    return (npy_intp) model->numSV;
}


/*
    Get number of classes in a model
*/
npy_intp getNumClass(struct SVMModel *model)
{
    return (npy_intp) model->numClass;
}
