#include <stdlib.h>
#include <numpy/arrayobject.h>

#include "svm.h"


#define Malloc(type,n) (type *)malloc((n) * sizeof(type))


/*
	Some utility funcions for converting NumPy arrays to LIBSVM structures.

*/


struct SVMNode* denseToLIBSVM(double *x, npy_intp* dims)
{
	struct SVMNode *node;

	npy_intp len_row = dims[1];
    npy_intp len_col = dims[0];

    double *tx = x;
	int i, j;

	//printf("Row: %d, Col: %d\n", (int) len_row, (int) len_col);

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
void setParameter(struct SVMParameter *param, int kernelType, double gamma,
                  double C, double e)
{
    param->kernelType = kernelType;
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
                          char *support, npy_intp* supportDim, npy_intp *stridesSV, char* svCoef,
                          char* bias, char* nSV)
{
    struct SVMModel *model;
    double *dsvCoef = (double *) svCoef;

    int i, m;

    m = nrClass * (nrClass -1) / 2;

    if((model = Malloc(struct SVMModel, 1)) == NULL)
        goto modelError;

    if((model->svClass = Malloc(int, nrClass)) == NULL)
        goto nsvError;

    if((model->label = Malloc(int, nrClass)) == NULL)
        goto labelError;

    if((model->svCoef = Malloc(double *, (nrClass - 1))) == NULL)
        goto svCoefError;

    if((model->bias = Malloc(double, m)) == NULL)
        goto biasError;

    model->numClass = nrClass;
    model->param = *param;
    model->numSV = (int) supportDim[0];

    model->SV = denseToLIBSVM((double *) SV, dimSV);

    memcpy(model->svClass, nSV, model->numClass * sizeof(int));

    for(i = 0; i < model->numClass; ++i)
        model->label[i] = i;

    for(i = 0; i < model->numClass - 1; ++i)
        model->svCoef[i] = dsvCoef + i * model->numSV;

    for(i = 0; i < m; ++i)
        (model->bias)[i] = ((double *) bias)[i];

    model->freeSV = 0;

    //printf("Model created...\n");

    return model;

    // Handling errors
    biasError:
        free(model->svCoef);

    svCoefError:
        free(model->label);

    labelError:
        free(model->svClass);

    nsvError:
        free(model);

    modelError:
        return NULL;

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
    Predict using SVM model
*/
int copyPredict(char *predict, struct SVMModel *model, npy_intp *predictDims,
                char *decValues)
{
    double *t = (double *) decValues;
    struct SVMNode *predictNodes;
    npy_intp i;

    predictNodes = denseToLIBSVM((double *) predict, predictDims);

    if(predictNodes == NULL)
        return -1;

    for(i = 0; i < predictDims[0]; ++i)
    {
        *t = SVMPredict(model, &predictNodes[i]);

        ++t;
    }

    free(predictNodes);
    //printf("Predicted....\n");

    return 0;

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


/*
    Some free rotines
    Like SVMFreeModel but does not free svCoef
*/
int freeModel(struct SVMModel *model)
{
    if(model == NULL) return -1;

    free(model->SV);

    // Do not free mode->svIndices, sice it was not created in setModel
    free(model->svCoef);
    free(model->bias);
    free(model->label);
    free(model->svClass);
    free(model);

    return 0;

}

