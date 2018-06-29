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
    SVMNode **x; //
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

};


class SVM
{
    public:
        SVM();

    protected:

    private:
};

#endif // SVM_H
