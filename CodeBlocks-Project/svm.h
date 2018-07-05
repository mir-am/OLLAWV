#ifndef SVM_H
#define SVM_H

struct SVMNode
{
    int index;
    double value;
};


struct SVMProblem
{
    int l; // Number of training samples
    double *y; // Class labels
    SVMNode **x; // Points to array pointers
    SVMNode *xSpace; // Points to the elements (index, value)
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
    int *svIndices;
    int *svClass; // Number of SVs for each class

    double *bias;
};


struct decisionFunction
{
    double *alpha;
    double bias;
};


template <typename T>
inline void swapVar(T& x, T& y);

void groupClasses(const SVMProblem& prob, int& numClass, int** label_ret,
                          int** strat_ret, int** count_ret, int* perm);

SVMModel trainSVM(const SVMProblem& prob, const SVMParameter& param);

class SVM
{
    public:
        SVM();

    protected:

    private:
};

#endif // SVM_H
