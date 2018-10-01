#include "svm.h"
#include <cstddef>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <math.h>
#include <cstring>
#include "misc.h"
//#include "FileReader.h"


#define Malloc(type,n) (type *)malloc((n) * sizeof(type))

template <typename T>
static inline void swapVar(T& x, T& y)
{
    T temp = x;
    x = y;
    y = temp;
}

template <typename S, typename T>
static inline void cloneVar(T*& dst, S* src, int n)
{
    dst = new T[n];
    memcpy((void *) dst, (void *) src, sizeof(T) * n);
}

typedef float Qfloat;
typedef signed char schar;

// Dense representation
#ifdef _DENSE_REP

	#ifdef PREFIX
		#undef PREFIX
	#endif

	#ifdef NAMESPACE
		#undef NAMESPACE
	#endif

	#define PREFIX(name) SVM##name
	#define NAMESPACE svm
	namespace svm {

// Sparse representation
#else

	#ifdef PREFIX
		#undef PREFIX
	#endif

	#ifdef NAMESPACE
		#undef NAMESPACE
	#endif

	#define PREFIX(name) SVMSparse##name
	#define NAMESPACE svm_csr
	namespace svm_csr {

#endif


// Kernel cache
class Cache
{

    public:

        Cache(int l, long int size);
        ~Cache();

        // Request data
        int getData(const int index, Qfloat **data, int len);
        void swapIndex(int i, int j);

    private:

        int l;
        long int size;

        // Doubly circular linked list
        struct head_t
        {
            head_t *prev, *next;
            Qfloat *data;
            int len;
        };

        head_t *head;
        head_t lru_head;
        void lru_delete(head_t *h);
        void lru_insert(head_t *h);

};


Cache::Cache(int l_, long int size_)
    :l(l_), size(size_)
{
    head = (head_t *) calloc(l, sizeof(head_t)); // initialized to zero.
    size /= sizeof(Qfloat);
    size -= l * sizeof(head_t) / sizeof(Qfloat);
    size = std::max(size, 2 * (long int) l);
    lru_head.next = lru_head.prev = &lru_head;

}


Cache::~Cache()
{
    for(head_t *h = lru_head.next; h != &lru_head; h = h->next)
        free(h->data);

    free(head);
}


void Cache::lru_delete(head_t *h)
{
    // Delete from current location
    h->prev->next = h->next;
    h->next->prev = h->prev;

}


void Cache::lru_insert(head_t *h)
{
    // Insert to last posistion
    h->next = &lru_head;
    h->prev = lru_head.prev;
    h->prev->next = h;
    h->next->prev = h;
}


int Cache::getData(const int index, Qfloat **data, int len)
{
    head_t *h = &head[index];
    if(h->len) lru_delete(h);
    int more = len - h->len;

    if(more > 0)
    {
        // free old space
        while(size < more)
        {
            head_t *old = lru_head.next;
            lru_delete(old);
            free(old->data);
            size += old->len;
            old->data = 0;
            old->len = 0;
        }

        // Allocate new space
        h->data = (Qfloat *) realloc(h->data, sizeof(Qfloat) * len);
        size -= more;
        swapVar(h->len, len);

    }

    lru_insert(h);
    *data = h->data;
    return len;

}


void Cache::swapIndex(int i, int j)
{
    if(i == j) return;

    if(head[i].len) lru_delete(&head[i]);
    if(head[j].len) lru_delete(&head[j]);

    swapVar(head[i].data, head[j].data);
    swapVar(head[i].len, head[j].len);

    if(head[i].len) lru_insert(&head[i]);
    if(head[j].len) lru_insert(&head[j]);

    if(i > j) swapVar(i, j);
    for(head_t *h = lru_head.next; h != &lru_head; h = h->next)
    {
        if(h->len > i)
        {
            if(h->len > j)
                swapVar(h->data[i], h->data[j]);
            else
            {
                lru_delete(h);
                free(h->data);
                size += h->len;
                h->data = 0;
                h->len = 0;
            }
        }

    }


}


class QMatrix
{
    public:

        // Getting one column from Q matrix
        virtual Qfloat* get_Q(int column, int len) const = 0;
        virtual double* get_QD() const = 0;
        virtual void swap_index(int i, int j) const = 0;
        virtual ~QMatrix() {}

};


class Kernel: public QMatrix
{

    public:

#ifdef _DENSE_REP

        Kernel(int l, PREFIX(Node) *x, const SVMParameter& param);

#else

        Kernel(int l, PREFIX(Node) * const *x, const SVMParameter& param);

#endif // _DENSE_REP

        virtual ~Kernel();

        static double k_function(const PREFIX(Node) *x, const PREFIX(Node) *y,
                                 const SVMParameter& param);

        virtual Qfloat* get_Q(int column, int len) const = 0;
        virtual double* get_QD() const = 0;

        void swap_index(int i, int j);

    protected:

        double (Kernel::*kernel_function)(int i, int j) const;

    private:

#ifdef _DENSE_REP

        PREFIX(Node) *x;

#else

        const PREFIX(Node) **x;

#endif // _DENSE_REP

        double *x_square;

        // SVM parameters
        int kernelType;
        const double gamma;

        static double dot(const PREFIX(Node) *px, const PREFIX(Node) *py);

#ifdef _DENSE_REP

        static double dot(const PREFIX(Node) &px, const PREFIX(Node) &py);

#endif

        double kernelLinear(int i, int j) const
        {
            return dot(x[i], x[j]);
        }

        double kernelRBF(int i, int j) const
        {
            return exp(-gamma * (x_square[i] + x_square[j] - 2 * dot(x[i], x[j])));
        }


};


#ifdef _DENSE_REP

        Kernel::Kernel(int l, PREFIX(Node) *x_, const SVMParameter& param)

#else

        Kernel::Kernel(int l, PREFIX(node) * const *x_, const SVMParameter& param)

#endif // _DENSE_REP
:kernelType(param.kernelType), gamma(param.gamma)
{
    switch(kernelType)
    {
        case LINEAR:

            kernel_function = &Kernel::kernelLinear;
            break;

        case RBF:

            kernel_function = &Kernel::kernelRBF;
            break;

    }

    cloneVar(x, x_, l);

    if(kernelType == RBF)
    {
        x_square = new double[l];

        for(int i = 0; i < l; ++i)
            x_square[i] = dot(x[i], x[i]);
    }
    else

        x_square = 0;

}

Kernel::~Kernel()
{
    delete[] x;
    delete[] x_square;
}


#ifdef _DENSE_REP

double Kernel::dot(const PREFIX(Node) *px, const PREFIX(Node) *py)
{
    double sum = 0;
    int dim = std::min(py->dim, px->dim);

    for(int i = 0; i < dim; ++i)
        sum += (px->values)[i] * (py->values)[i];

    return sum;

}

double Kernel::dot(const PREFIX(Node) &px, const PREFIX(Node) &py)
{
    double sum = 0;
    int dim = std::min(py.dim, px.dim);

    for(int i = 0; i < dim; ++i)
        sum += px.values[i] * py.values[i];

    return sum;

}

#else

double Kernel::dot(const PREFIX(node) *px, const PREFIX(node) *py)
{
    double sum = 0;

    while(px->index != -1 && py->index != -1)
    {
        if(px->index == py->index)
        {
            sum += px->value * py->value;
            ++px;
            ++py;
        }
        else
        {
            if(px->index > py->index)
                ++py;
            else
                ++px;
        }
    }

    return sum;

}

#endif // _DENSE_REP

double Kernel::k_function(const PREFIX(Node) *x, const PREFIX(Node) *y,
                          const SVMParameter& param)
{

    switch(param.kernelType)
    {
        case LINEAR:

            return dot(x, y);

        case RBF:
        {
            double sum = 0;

#ifdef _DENSE_REP

            int dim = std::min(x->dim, y->dim), i;

            for(i = 0; i < dim; ++i)
            {
                double d = x->values[i] - y->values[i];
                sum += d * d;
            }

            for( ; i < x->dim; ++i)
                sum += x->values[i] * x->values[i];

            for( ; i < y->dim; ++i)
                sum += y->values[i] * y->values[i];

#else

        while(x->index != -1 && y->index != -1)
        {
            if(x->index == y->index)
            {
                double d = x->value - y->value;
                sum += d * d;
                ++x;
                ++y;
            }
            else
            {
                if(x->index > y->index)
                {
                    sum += y->value * y->value;
                    ++y;
                }
                else
                {
                    sum += x->value * x->value;
                    ++x;
                }
            }
        }

    while(x->index != -1)
    {
        sum += x->value * x->value;
        ++x;
    }

    while(y->index != -1)
    {
        sum += y->value * y->value;
        ++y;
    }

#endif // _DENSE_REP

            return exp(-param.gamma * sum);

        }

    }
}


class SVC_Q: public Kernel
{

    private:

        schar *y;
        Cache *cache;
        double *QD;

    public:

        SVC_Q(const PREFIX(problem) &prob, const SVMParameter &param, const schar *y_)
        :Kernel(prob.l, prob.x, param)
        {
            cloneVar(y, y_, prob.l);
            cache = new Cache(prob.l, (long int)(param.cacheSize * (1 << 20)));
            QD = new double[prob.l];
            for(int i = 0; i < prob.l; ++i)
                QD[i] = (this->*kernel_function)(i, i);
        }

};


struct decisionFunction
{
    double *alpha;
    double bias;
    double obj; // Objective value
};


static double kernelRBF(const PREFIX(Node) *x, const PREFIX(Node) *y, const double& gamma)
{
    double sum = 0;

#ifdef _DENSE_REP

    int dim = std::min(x->dim, y->dim), i;

    for(i = 0; i < dim; ++i)
    {
        double d = x->values[i] - y->values[i];
        sum += d * d;
    }

    for( ; i < x->dim; ++i)
        sum += x->values[i] * x->values[i];

    for( ; i < y->dim; ++i)
        sum += y->values[i] * y->values[i];

#else

    while(x->index != -1 && y->index != -1)
    {
        if(x->index == y->index)
        {
            double d = x->value - y->value;
            sum += d * d;
            ++x;
            ++y;
        }
        else
        {
            if(x->index > y->index)
            {
                sum += y->value * y->value;
                ++y;
            }
            else
            {
                sum += x->value * x->value;
                ++x;
            }
        }
    }

    while(x->index != -1)
    {
        sum += x->value * x->value;
        ++x;
    }

    while(y->index != -1)
    {
        sum += y->value * y->value;
        ++y;
    }

#endif // _DENSE_REP

    return exp(-gamma * sum);
}


static void SVMSolver(const PREFIX(Problem)& prob, const SVMParameter& para,
               decisionFunction& solution)
{

    // M value is scaled value of C.
    double M = para.e * para.C;

    // Output vector
    std::vector<double> outputVec(prob.l, 0);
    int t = 0; // Iterations
    // Learn rate
    double learnRate;
    double B;

    // Initialize hinge loss error and worst violator index
    unsigned int idxWV = 0;
    double yo = prob.y[idxWV] * outputVec[idxWV];
    double hingeLoss;

    // Indexes
    std::vector<size_t> nonSVIdx(prob.l);
    std::iota(nonSVIdx.begin(), nonSVIdx.end(), 0);

    while(yo < M)
    {
        ++t;
        learnRate = 2 / sqrt(t);

        //std::cout << "Worst violator index: " << idxWV << " | Output: " << yo << std::endl;

        // Remove worst violator from index set
        nonSVIdx.erase(std::remove(nonSVIdx.begin(), nonSVIdx.end(), idxWV),
                        nonSVIdx.end());

        // Calculate
        hingeLoss = learnRate * para.C * prob.y[idxWV];

        // Calculate bias term
        B = hingeLoss / prob.l;

        // Update worst violator's alpha value
        solution.alpha[idxWV] += hingeLoss;
        solution.bias += B;

        if (nonSVIdx.size() != 0)
        {

#ifdef _DENSE_REP
            outputVec[nonSVIdx[0]] += ((hingeLoss * kernelRBF(&prob.x[nonSVIdx[0]],
                                      &prob.x[idxWV], para.gamma)) + B);

#else

            outputVec[nonSVIdx[0]] += ((hingeLoss * kernelRBF(prob.x[nonSVIdx[0]],
                                      prob.x[idxWV], para.gamma)) + B);

#endif // _DENSE_REP

            // Suppose that first element of nonSVIdx vector is worst violator sample
            unsigned int newIdxWV = nonSVIdx[0];
            yo = prob.y[newIdxWV] * outputVec[newIdxWV];

            for(size_t idx = 1; idx < nonSVIdx.size(); ++idx)
            {

#ifdef _DENSE_REP
                outputVec[nonSVIdx[idx]] += ((hingeLoss * kernelRBF(&prob.x[nonSVIdx[idx]],
                                            &prob.x[idxWV], para.gamma)) + B);

#else

                outputVec[nonSVIdx[idx]] += ((hingeLoss * kernelRBF(&prob.x[nonSVIdx[idx]],
                                            &prob.x[idxWV], para.gamma)) + B);

#endif // _DENSE_REP


                //std::cout << outputVec[nonSVIdx[idx]] << std::endl;

                // Find worst violator
                if((prob.y[nonSVIdx[idx]] * outputVec[nonSVIdx[idx]]) < yo)
                {
                    newIdxWV = nonSVIdx[idx];
                    yo = prob.y[nonSVIdx[idx]] * outputVec[nonSVIdx[idx]];
                }
            }

            // Found new worst violator sample
            idxWV = newIdxWV;

            //std::cout << "Worst violator idx: " << idxWV << std::endl;
            //std::cout << "M: " << M << "yo: " << yo << std::endl;
        }
        else
        {
            break;
        }

    }

    solution.obj = yo;

    //std::cout << "Iterations: " << t << std::endl;

}


static void groupClasses(const PREFIX(Problem) *prob, int* numClass, int** label_ret,
                          int** start_ret, int** count_ret, int* perm)
{
    int l = prob->l;
    int maxNumClass = 16;
    int nrClass = 0;

    int* label = Malloc(int, maxNumClass);
    int* countLables = Malloc(int, maxNumClass);
    int* dataLabel = Malloc(int, l);

    int i, j, thisLabel, thisCount;

    // Count number of samples of each class.
    // Also it stores class labels.
    for(i = 0; i < l; ++i)
    {
        thisLabel = (int) prob->y[i];

        for(j = 0; j < nrClass; ++j)
        {
            if(thisLabel == label[j])
            {
                ++countLables[j];
                break;
            }
        }

        //dataLabel[i] = j;

//        std::cout << "a-label: " << dataLabel[i] << " r-label: " << thisLabel
//         << std::endl;

        if(j == nrClass)
        {
            // If number of classes is more than 2
            // you need to re allocate memory here later.
            if(nrClass == maxNumClass)
            {
                maxNumClass *= 2;
                label = (int *) realloc(label, maxNumClass * sizeof(int));
                countLables = (int *) realloc(countLables, maxNumClass * sizeof(int));
            }


            label[nrClass] = thisLabel;
            countLables[nrClass] = 1;
            ++nrClass;
        }
    }

     // FOR DEBUGGING Purpose
//    for(j = 0; j < nrClass; ++j)
//    {
//        std::cout << "Label: " << label[j] << " | Count: " << countLables[j] << std::endl;
//    }




//    // For binary classification, we need to swap labels
//    if(nrClass == 2 && label[0] == -1 && label[1] == 1)
//    {
//        swapVar(label[0], label[1]);
//        swapVar(countLables[0], countLables[1]);
//
//        for(int i = 0; i < l; ++i)
//        {
//            if(dataLabel[i] == 0)
//                dataLabel[i] = 1;
//            else
//                dataLabel[i] = 0;
//        }
//    }

    // Sort labels by straight insertion
    for(j = 1; j < nrClass; ++j)
    {
        i = j - 1;
        thisLabel = label[j];
        thisCount = countLables[j];

        while(i >=0 && label[i] > thisLabel)
        {
            label[i + 1] = label[i];
            countLables[i + 1] = countLables[i];
            i--;
        }

        label[i + 1] = thisLabel;
        countLables[i + 1] = thisCount;

    }

    for(i = 0; i < l; ++i)
    {
        j = 0;
        thisLabel = (int) prob->y[i];

        while(thisLabel != label[j])
        {
            j++;
        }
        dataLabel[i] = j;

    }


    int* start = Malloc(int, nrClass);
    start[0] = 0;

    for(i = 1; i < nrClass; ++i)
        start[i] = start[i-1] + countLables[i-1];

    // Store indices of samples in perm for grouping classes.
    for(i = 0; i < l; ++i)
    {
        perm[start[dataLabel[i]]] = i;

//        std::cout << "Org place: " << i << " Label: " << dataLabel[i]
//        << " perm" <<"["<< start[dataLabel[i]] << "]: " << i << std::endl;

        ++start[dataLabel[i]];
    }

    // For Debugging
//    for(i = 0; i < l; ++i)
//        std::cout << "Label of perm[" << i << "]: " << prob->y[perm[i]] << std::endl;

    // Reset
    start[0] = 0;
    for(i = 1; i < nrClass; ++i)
        start[i] = start[i-1] + countLables[i-1];

    *numClass = nrClass;
    *label_ret = label;
    *start_ret = start;
    *count_ret = countLables;
    free(dataLabel);

    //std::cout << "Classes are grouped!" << std::endl;

}


static decisionFunction trainOneSVM(const PREFIX(Problem) *prob, const SVMParameter* param, int* status)
{

    decisionFunction solutionInfo;
    solutionInfo.alpha = Malloc(double, prob->l);

    // Initialize the solution
    for(int i = 0; i < prob->l; ++i)
        solutionInfo.alpha[i] = 0;

    solutionInfo.bias = 0.0;

    SVMSolver(*prob, *param, solutionInfo);

    //std::cout << "obj = " << solutionInfo.obj << " Bias = " << solutionInfo.bias
     //    << std::endl;

    // Count number of support vectors
    int nSV = 0;
    for(int i = 0; i < prob->l; ++i)
    {
        if(fabs(solutionInfo.alpha[i]) > 0)
            ++nSV;
    }

    //std::cout << "num. of SVs: " << nSV << std::endl;

    return solutionInfo;
}

} // End of namespace

PREFIX(Model) *PREFIX(Train)(const PREFIX(Problem) *prob, const SVMParameter *param, int *status)
{

    std::cout << "Training started... | C: " << param->C << " Gamma: " << param->gamma << " Kernel: " << param->kernelType << std::endl;

    // Print elements - FOR DEBUGGING PURPOSE
    //printData(prob->x, prob->l);

    // Classification
    PREFIX(Model)* model = Malloc(PREFIX(Model), 1);
    model->param = *param;
    model->freeSV = 0;

    int numSamples = prob->l;
    int numClass;
    int* label = NULL;
    int* start = NULL;
    int* count = NULL;
    int* perm = Malloc(int, numSamples);

    NAMESPACE::groupClasses(prob, &numClass, &label, &start, &count, perm);

#ifdef _DENSE_REP

    PREFIX(Node) *x = Malloc(PREFIX(Node), numSamples);

#else

    // Allocate space for samples with respect to perm
    PREFIX(Node) **x = Malloc(PREFIX(Node), numSamples);

#endif // __DENSE_REP

    for(int i = 0; i < numSamples; ++i)
        x[i] = prob->x[perm[i]];

    // Train k*(k-1)/2 models
    bool* nonZero = Malloc(bool, numSamples);

    for(int i = 0; i < numSamples; ++i)
        nonZero[i] = false;

    // Allocate space for each model's parameters such as weights and bias
    NAMESPACE::decisionFunction* f = Malloc(NAMESPACE::decisionFunction, numClass*(numClass-1)/2);

    int p = 0;
    for(int i = 0; i < numClass; ++i)
    {
        for(int j = i + 1; j < numClass; ++j)
        {

            PREFIX(Problem) subProb; // A sub problem for i-th and j-th class

            // start points of i-th and j-th classes
            int si = start[i], sj = start[j];
            // Number of samples in i-th and j-th class
            int ci = count[i], cj = count[j];

            // For debugging
            //std::cout << "si: " << si << " sj: " << sj << std::endl;
            //std::cout << "ci: " << ci << " cj: " << cj << std::endl;

            subProb.l = ci + cj;

#ifdef _DENSE_REP

            subProb.x = Malloc(SVMNode, subProb.l);
#else
            subProb.x = Malloc(SVMNode *, subProb.l);
#endif // _DENSE_REP

            subProb.y = Malloc(double, subProb.l);

            // select all the samples of j-th class
            for(int k = 0; k < ci;++k)
            {
                subProb.x[k] = x[si+k];
                subProb.y[k] = +1;
            }
            for(int k = 0; k < cj;++k)
            {
                subProb.x[ci+k] = x[sj+k];
                subProb.y[ci+k] = -1;
            }

            f[p] = NAMESPACE::trainOneSVM(&subProb, param, status);

            // Count number of support vectors of each class
            for(int k = 0; k < ci; ++k)
            {
                if(!nonZero[si + k] && fabs(f[p].alpha[k]) > 0)
                    nonZero[si + k] = true;
            }

            for(int k = 0; k < cj; ++k)
            {
                if(!nonZero[sj + k] && fabs(f[p].alpha[ci + k]) > 0)
                    nonZero[sj + k] = true;

            }

            // Free memory!
            free(subProb.x);
            free(subProb.y);
            ++p;
        }
    }

    // Build model
    model->numClass = numClass;

    // Copy the labels
    model->label = Malloc(int, numClass);
    for(int i = 0; i < numClass; ++i)
        model->label[i] = label[i];

    model->bias = Malloc(double, numClass * (numClass - 1) / 2);
    for(int i = 0; i < numClass * (numClass - 1) / 2; ++i)
         model->bias[i] = f[i].bias;

    int totalSV = 0;
    int* nzCount = Malloc(int, numClass);
    model->svClass = Malloc(int, numClass);

    for(int i = 0; i < numClass; ++i)
    {
        int nSV = 0;
        for(int j = 0; j < count[i]; ++j)
        {
            if(nonZero[start[i] + j])
            {
                ++nSV;
                ++totalSV;
            }
        }

        model->svClass[i] = nSV;
        nzCount[i] = nSV;

        // For debugging
        //std::cout << "Label: " << model->label[i] << " SVs: " <<
        //model->svClass[i] << std::endl;
    }


    //std::cout << "Total nSV: " << totalSV << std::endl;

    model->numSV = totalSV;

    model->svIndices = Malloc(int, totalSV);

#ifdef _DENSE_REP

    model->SV = Malloc(SVMNode, totalSV);
#else
    model->SV = Malloc(SVMNode *, totalSV);
#endif // _DENSE_REP


    p = 0;
    for(int i = 0; i < numSamples; ++i)
    {
        if(nonZero[i])
        {
            model->SV[p] = x[i];
            model->svIndices[p++] = perm[i] + 1;
        }
    }

    int* nzStart = Malloc(int, numClass);
    nzStart[0] = 0;

    for(int i = 1; i < numClass; ++i)
        nzStart[i] = nzStart[i - 1] + nzCount[i - 1];

    model->svCoef = Malloc(double *, numClass - 1);
    for(int i = 0; i < numClass - 1; ++i)
        model->svCoef[i] = Malloc(double, totalSV);

    p = 0;
    for(int i = 0; i < numClass; ++i)
    {
        for(int j = i + 1; j < numClass; ++j)
        {
            // classifier (i,j): coefficients with
            // i are in sv_coef[j-1][nz_start[i]...],
            // j are in sv_coef[i][nz_start[j]...]

            int si = start[i];
            int sj = start[j];
            int ci = count[i];
            int cj = count[j];

            int q = nzStart[i];
            for(int k = 0; k < ci; ++k)
            {
                if(nonZero[si + k])
                    model->svCoef[j - 1][q++] = f[p].alpha[k];
            }

            q = nzStart[j];
            for(int k = 0; k < cj; ++k)
            {
                if(nonZero[sj + k])
                    model->svCoef[i][q++] = f[p].alpha[ci + k];
            }

            ++p;
        }
    }

    // Free memory
    free(label);
    free(count);
    free(perm);
    free(start);
    free(x);
    free(nonZero);

    // Delete decision functions
    for(int i = 0; i < numClass * (numClass - 1) / 2 ; ++i)
        free(f[i].alpha);

    free(f);
    free(nzCount);
    free(nzStart);

    *status = 1;
    //std::cout << "Training Finished!!" << std::endl;

    return model;
}


const char *PREFIX(CheckParameter)(const SVMParameter *param)
{

    if(param->gamma < 0)
        return "gamma < 0. gamma is a positive parameter.";

    if(param->e <= 0 || param->e >= 1)
        return "Optimizer stopping criteria should be in the interval(0, 1).";

    if(param->C <= 0)
        return "C <= 0. C penalty parameter should be positive.";

    return NULL;
}


void PREFIX(FreeModelContent)(PREFIX(Model) *model_ptr)
{
    if(model_ptr->freeSV && model_ptr->numSV > 0  && model_ptr->SV != NULL)
    {
#ifdef _DENSE_REP

        for(int i = 0; i < model_ptr->numSV; ++i)
            free(model_ptr->SV[i].values);
#else
        free((void *) (model_ptr->SV[0]));

#endif // _DENSE_REP

        if(model_ptr->svCoef)
        {
            for(int i = 0; i < model_ptr->numClass - 1; ++i)
                free(model_ptr->svCoef[i]);
        }

        free(model_ptr->SV);
        model_ptr->SV = NULL;

        free(model_ptr->svCoef);
        model_ptr->svCoef = NULL;

        free(model_ptr->svIndices);
        model_ptr->svIndices = NULL;

        free(model_ptr->bias);
        model_ptr->bias = NULL;

        free(model_ptr->label);
        model_ptr->label = NULL;

        free(model_ptr->svClass);
        model_ptr->svClass = NULL;

    }
}


void PREFIX(FreeModel)(PREFIX(Model)** model_ptr_ptr)
{
    if(model_ptr_ptr != NULL && *model_ptr_ptr != NULL)
    {
        PREFIX(FreeModelContent)(*model_ptr_ptr);
        free(*model_ptr_ptr);

        *model_ptr_ptr = NULL;

        //std::cout << "The model successfully destroyed." << std::endl;

    }

}


double PREFIX(computeVotes) (const PREFIX(Model) *model, const PREFIX(Node) *x, double *decValues)
{
    int numClasses = model->numClass;
    int numSamples = model->numSV;

    // Kernel values
    double* kValue = Malloc(double, numSamples);

    for(int i = 0; i < numSamples; ++i)

#ifdef _DENSE_REP

        kValue[i] = NAMESPACE::kernelRBF(x, model->SV + i, model->param.gamma);
#else

        kValue[i] = NAMESPACE::kernelRBF(x, model->SV[i], model->param.gamma);

#endif // _DENSE_REP


    int* start = Malloc(int, numClasses);
    start[0] = 0;
    for(int i = 1; i < numClasses; ++i)
        start[i] = start[i - 1] + model->svClass[i - 1];

    // Initialize votes
    int* vote = Malloc(int, numClasses);
    for(int i = 0; i < numClasses; ++i)
        vote[i] = 0;

    int p = 0;
    for(int i = 0; i < numClasses; ++i)
        for(int j = i + 1; j < numClasses; ++j)
        {
            double sum = 0;
            int si = start[i];
            int sj = start[j];
            int ci = model->svClass[i];
            int cj = model->svClass[j];

            double* coef1 = model->svCoef[j - 1];
            double* coef2 = model->svCoef[i];

            for(int k = 0; k < ci; ++k)
                sum += coef1[si + k] * kValue[si + k];

            for(int k = 0; k < cj; ++k)
                sum += coef2[sj + k] * kValue[sj + k];

            // Bias
            sum -= model->bias[p];
            decValues[p] = sum;

            if(decValues[p] > 0)
                ++vote[i];
            else
                ++vote[j];

            // For debugging purpose
            //std::cout << "Vote " << vote[i] << " |Vote " << vote[j] << std::endl;

            p++;
        }

    int voteMaxIdx = 0;
    for(int i = 1; i < numClasses; ++i)
        if(vote[i] > vote[voteMaxIdx])
            voteMaxIdx = i;

//    std::cout << "VoteMax: " << voteMaxIdx << std::endl;
//    std::cout << "Predicted label: " << model->label[voteMaxIdx] << std::endl;

    free(kValue);
    free(start);
    free(vote);

    return model->label[voteMaxIdx];
}


double PREFIX(Predict)(const PREFIX(Model) *model, const PREFIX(Node) *x)
{
    int numClass = model->numClass;
    double* decValues;

    decValues = Malloc(double, numClass * (numClass - 1) / 2);

    double predResult = PREFIX(computeVotes)(model, x, decValues);

    free(decValues);

    return predResult;
}


//void predict(std::string testFile, const SVMModel* model)
//{
//    int correct = 0;
//    int total = 0;
//    int numClass = model->numClass;
//    // Allocating 64 SVM nodes. Suppose that features are not more than 64.
//    unsigned int maxNumAttr = 64;
//
//    // Read test datafile
//    std::ifstream testDataFile(testFile.c_str());
//
//    if(testDataFile.is_open())
//    {
//        std::cout << "Successfully read test datafile!" << std::endl;
//        std::string line;
//
//        SVMNode* x = new SVMNode[maxNumAttr];
//
//        // Reading each test sample
//        while(std::getline(testDataFile, line))
//        {
//            int i = 0;
//            double targetLabel, predictLabel;
//
//            std::vector<std::string> testSample = splitString(line);
//
//            // Reallocating memory if num. of features are more than 64
//            if(testSample.size() - 1 >= maxNumAttr - 1)
//            {
//
//                // Delete the previous allocated mem. blocks to avoid mem. leak
//                delete[] x;
//
//                maxNumAttr *= 2;
//                x = new SVMNode[maxNumAttr];
//
//            }
//
//            targetLabel = atof(testSample[0].c_str());
//
//            for(unsigned int j = 1; j < testSample.size(); ++j)
//            {
//                std::vector<std::string> node = splitString(testSample[j], ':');
//
//                x[i].index = atoi(node[0].c_str());
//                x[i].value = atof(node[1].c_str());
//
//                ++i;
//            }
//
//            x[i].index = -1;
//
//            predictLabel = SVMPredict(model, x);
//
//            if(predictLabel == targetLabel)
//                ++correct;
//
//            ++total;
//
//        }
//
//        std::cout << "Accuracy: " << (double) correct / total * 100 <<
//         " % (" << correct << "/" << total << ")" << " Classification" << std::endl;
//
//        testDataFile.close();
//    }
//    else
//    {
//        std::cout << "Failed to open test file. " << testFile << std::endl;
//    }
//
//}


//void crossValidation(const SVMProblem& prob, SVMParameter& param, int numFolds)
//{
//    int totalCorrect = 0;
//    double *target = new double[prob.l];
//    int* foldStart;
//    int numSamples = prob.l;
//    int* perm = new int[numSamples];
//    int numClasses;
//
//    int* start = NULL;
//    int* label = NULL;
//    int* count = NULL;
//
//    groupClasses(prob, numClasses, &label, &start, &count, perm);
//
//    // Random shuffle
//    foldStart = new int[numFolds + 1];
//    int* foldCount = new int[numFolds];
//    int* index = new int[numSamples];
//
//    for(int i = 0; i < numSamples; ++i)
//        index[i] = perm[i];
//
//    for(int c = 0; c < numClasses; ++c)
//        for(int i = 0; i < count[c]; ++i)
//        {
//            int j = i + rand() % (count[c] - i);
//            swapVar(index[start[c] + j], index[start[c] + i]);
//        }
//
//    for(int i = 0; i < numFolds; ++i)
//    {
//        foldCount[i] = 0;
//        for(int c = 0; c < numClasses; ++c)
//            foldCount[i] += (i + 1) * count[c] / numFolds - i * count[c] / numFolds;
//
//        std::cout << "Fold " << i << ": " << foldCount[i] << std::endl;
//    }
//
//    foldStart[0] = 0;
//    for(int i = 1; i <= numFolds; ++i)
//    {
//        foldStart[i] = foldStart[i - 1] + foldCount[i - 1];
//        std::cout << "startFold " << i << ": " << foldCount[i] << std::endl;
//    }
//
//    for(int c = 0; c < numClasses; ++c)
//        for(int i = 0; i < numFolds; ++i)
//        {
//            int begin = start[c] + i * count[c] / numFolds;
//            int end = start[c] + (i + 1) * count[c] / numFolds;
//
//            //std::cout << "C:" << c << " Fold " << i << " " << "Begin: "
//             //<< begin << " End: " << end << std::endl;
//
//             for(int j = begin; j < end; ++j)
//             {
//                 perm[foldStart[i]] = index[j];
//                 ++foldStart[i];
//             }
//
//        }
//
//    foldStart[0] = 0;
//    for(int i = 0; i < numFolds; ++i)
//        foldStart[i] = foldStart[i - 1] + foldCount[i - 1];
//
//    for(int i = 0; i < numFolds; ++i)
//    {
//        int begin = foldStart[i];
//        int end = foldStart[i + 1];
//        int k = 0;
//
//        SVMProblem subProb;
//        subProb.l = numSamples - (end - begin);
//        subProb.x = new SVMNode*[subProb.l];
//        subProb.y = new double[subProb.l];
//
//        for(int j = 0 ;j < begin; ++j)
//        {
//            subProb.x[k] = prob.x[perm[j]];
//            subProb.y[k] = prob.y[perm[j]];
//            ++k;
//        }
//        for(int j = end;j < numSamples; ++j)
//        {
//            subProb.x[k] = prob.x[perm[j]];
//            subProb.y[k] = prob.y[perm[j]];
//            ++k;
//        }
//
//        SVMModel* subModel = trainSVM(subProb, param);
//
//        for(int j = begin; j < end; ++j)
//        {
//            target[perm[j]] = SVMPredict(subModel, prob.x[j]);
//        }
//
//
//        delete[] subProb.x;
//        delete[] subProb.y;
//
//    }
//
//    // Compute accuracy
//    for(int i = 0; i < numSamples; ++i)
//    {
//        if(target[i] == prob.y[i])
//            ++totalCorrect;
//    }
//
//    std::cout << "Total correct: " << totalCorrect << std::endl;
//    std::cout << std::fixed << "Cross Validation Accuracy: " << 100.0 * totalCorrect / numSamples << std::endl;
//
//    delete[] start;
//    delete[] label;
//    delete[] count;
//    delete[] index;
//    delete[] foldCount;
//    delete[] foldStart;
//    delete[] perm;
//    delete[] target;
//
//}

