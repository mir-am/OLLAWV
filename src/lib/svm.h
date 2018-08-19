#ifndef SVM_H
#define SVM_H

#include<string>

struct SVMNode
{
    int index;
    double value;
};

struct SVMDenseNode
{
	int dim;
	int ind; // Precomputed kernel.
	double* values;

};


struct SVMProblem
{
    int l; // Number of training samples
    double *y; // Class labels
	SVMNode *x;

    //SVMNode **x; // Points to array pointers
    //SVMNode *xSpace; // Points to the elements (index, value)
};


struct SVMParameter
{
    double C; // Penalty parameter
    double gamma; // Parameter of RBF function
    // a value between 0 and 1. For calculating stopping criteria
    double e;
};


struct SVMModel
{
    SVMParameter param;
    int numClass;
    int *label; // Label of each class

    int numSV; // total SupportVectors
    SVMNode **SV;
    double **svCoef; // coefficients for SVs in decision function
    int *svIndices;
    int *svClass; // Number of SVs for each class

    double *bias;
};


struct decisionFunction
{
    double *alpha;
    double bias;
    double obj; // Objective value
};


template <typename T>
inline void swapVar(T& x, T& y);

double kernelRBF(const SVMNode *x, const SVMNode *y, const double& gamma);

void groupClasses(const SVMProblem& prob, int& numClass, int** label_ret,
                          int** strat_ret, int** count_ret, int* perm);

decisionFunction trainOneSVM(const SVMProblem& prob, const SVMParameter& param);

SVMModel* trainSVM(const SVMProblem& prob, const SVMParameter& param);

void SVMSolver(const SVMProblem& prob, const SVMParameter& para,
               decisionFunction& solution);

// Multiclass classification, a vote strategy would be used.
double computeVotes(const SVMModel* model, const SVMNode* x, double* decValues);

double SVMPredict(const SVMModel* model, const SVMNode* x);

void predict(std::string testFile, const SVMModel* model);

void crossValidation(const SVMProblem& prob, SVMParameter& param, int numFolds);

#endif // SVM_H
