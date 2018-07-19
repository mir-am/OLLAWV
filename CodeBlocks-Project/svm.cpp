#include "svm.h"
#include <cstddef>
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <math.h>


template <typename T>
inline void swapVar(T& x, T& y)
{
    T temp = x;
    x = y;
    y = temp;
}


double kernelRBF(const SVMNode *x, const SVMNode *y, const double& gamma)
{
    double sum = 0;

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

    return exp(-gamma * sum);
}

void groupClasses(const SVMProblem& prob, int& numClass, int** label_ret,
                          int** start_ret, int** count_ret, int* perm)
{
    int l = prob.l;
    int maxNumClass = 2; // Binary classification
    numClass = 0;

    int* label = new int[maxNumClass];
    int* countLables = new int[maxNumClass];
    int* dataLabel = new int[l];

    for(int i = 0; i < l; ++i)
    {
        int thisLabel = (int) prob.y[i];
        int j;

        for(j = 0; j < numClass; ++j)
        {
            if(thisLabel == label[j])
            {
                ++countLables[j];
                break;
            }
        }

        dataLabel[i] = j;

//        std::cout << "a-label: " << dataLabel[i] << " r-label: " << thisLabel
//         << std::endl;

        if(j == numClass)
        {
            // If number of classes is more than 2
            // you need to re allocate memory here later.

            label[numClass] = thisLabel;
            countLables[numClass] = 1;
            ++numClass;
        }
    }

    //std::cout << "0: " << countLables[0] << "| 1:" << countLables[1] << std::endl;

    // For binary classification, we need to swap labels
    if(numClass == 2 && label[0] == -1 && label[1] == 1)
    {
        swapVar(label[0], label[1]);
        swapVar(countLables[0], countLables[1]);

        for(int i = 0; i < l; ++i)
        {
            if(dataLabel[i] == 0)
                dataLabel[i] = 1;
            else
                dataLabel[i] = 0;
        }
    }

    int* start = new int[numClass];
    start[0] = 0;

    for(int i = 1; i < numClass; ++i)
        start[i] = start[i-1] + countLables[i-1];

    for(int i = 0; i < l; ++i)
    {
        perm[start[dataLabel[i]]] = i;

//        std::cout << "Org place: " << i << " Label: " << dataLabel[i]
//        << " perm" <<"["<< start[dataLabel[i]] << "]:" << i << std::endl;

        ++start[dataLabel[i]];
    }

    // Reset
    start[0] = 0;
    for(int i = 1; i < numClass; ++i)
        start[i] = start[i-1] + countLables[i-1];


    *label_ret = label;
    *start_ret = start;
    *count_ret = countLables;
    delete[] dataLabel;

}


decisionFunction trainOneSVM(const SVMProblem& prob, const SVMParameter& param)
{

    decisionFunction solutionInfo;
    solutionInfo.alpha = new double[prob.l];

    // Initialize the solution
    for(int i = 0; i < prob.l; ++i)
        solutionInfo.alpha[i] = 0;

    solutionInfo.bias = 0.0;

    SVMSolver(prob, param, solutionInfo);

    std::cout << "obj = " << solutionInfo.obj << " Bias = " << solutionInfo.bias
         << std::endl;

    // Count number of support vectors
    int nSV = 0;
    for(int i = 0; i < prob.l; ++i)
    {
        if(fabs(solutionInfo.alpha[i]) > 0)
            ++nSV;
    }

    std::cout << "num. of SVs: " << nSV << std::endl;

    return solutionInfo;
}


SVMModel* trainSVM(const SVMProblem& prob, const SVMParameter& param)
{
    // Classification
    SVMModel* model = new SVMModel;
    model->param = param;

    int numSamples = prob.l;
    int numClass;
    int* label = NULL;
    int* start = NULL;
    int* count = NULL;
    int* perm = new int[numSamples];

    groupClasses(prob, numClass, &label, &start, &count, perm);

    // Allocate space for samples with respect to perm
    SVMNode** x = new SVMNode*[numSamples];

    for(int i = 0; i < numSamples; ++i)
        x[i] = prob.x[perm[i]];

    // Train k*(k-1)/2 models
    bool* nonZero = new bool[numSamples];

    for(int i = 0; i < numSamples; ++i)
        nonZero[i] = false;

    // Allocate space for each model's parameters such as weights and bias
    decisionFunction* f = new decisionFunction[numClass*(numClass-1)/2];

    int p = 0;
    for(int i = 0; i < numClass; ++i)
    {
        for(int j = i + 1; j < numClass; ++j)
        {
            SVMProblem subProb; // A sub problem for i-th and j-th class

            // start points of i-th and j-th classes
            int si = start[i], sj = start[j];
            // Number of samples in i-th and j-th class
            int ci = count[i], cj = count[j];

            // For debugging
            //std::cout << "si: " << si << " sj: " << sj << std::endl;
            //std::cout << "ci: " << ci << " cj: " << cj << std::endl;

            subProb.l = ci + cj;
            subProb.x = new SVMNode*[subProb.l];
            subProb.y = new double[subProb.l];

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

            f[p] = trainOneSVM(subProb, param);

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
            delete[] subProb.x;
            delete[] subProb.y;
            ++p;
        }
    }

    // Build model
    model->numClass = numClass;
    model->label = label;

    model->bias = new double[numClass * (numClass - 1) / 2];
    for(int i = 0; i < numClass * (numClass - 1) / 2; ++i)
         model->bias[i] = f[i].bias;

    int totalSV = 0;
    int* nzCount = new int[numClass];
    model->svClass = new int[numClass];

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


    std::cout << "Total nSV: " << totalSV << std::endl;

    model->numSV = totalSV;
    model->SV = new SVMNode*[totalSV];
    model->svIndices = new int[totalSV];

    p = 0;
    for(int i = 0; i < numSamples; ++i)
    {
        if(nonZero[i])
        {
            model->SV[p] = x[i];
            model->svIndices[p++] = perm[i] + 1;
        }
    }

    int* nzStart = new int[numClass];
    nzStart[0] = 0;

    for(int i = 1; i < numClass; ++i)
        nzStart[i] = nzStart[i - 1] + nzCount[i - 1];

    model->svCoef = new double*[numClass - 1];
    for(int i = 0; i < numClass - 1; ++i)
        model->svCoef[i] = new double[totalSV];

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
    delete[] label;
    delete[] count;
    delete[] perm;
    delete[] start;
    delete[] x;
    delete[] nonZero;

    // Delete decision functions
    for(int i = 0; i < numClass * (numClass - 1) / 2 ; ++i)
        delete[] f[i].alpha;

    delete[] f;
    delete[] nzCount;
    delete[] nzStart;

    return model;
}


void SVMSolver(const SVMProblem& prob, const SVMParameter& para,
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
            outputVec[nonSVIdx[0]] += ((hingeLoss * kernelRBF(prob.x[nonSVIdx[0]],
                                      prob.x[idxWV], para.gamma)) + B);

            // Suppose that first element of nonSVIdx vector is worst violator sample
            unsigned int newIdxWV = nonSVIdx[0];
            yo = prob.y[newIdxWV] * outputVec[newIdxWV];

            for(size_t idx = 1; idx < nonSVIdx.size(); ++idx)
            {
                outputVec[nonSVIdx[idx]] += ((hingeLoss * kernelRBF(prob.x[nonSVIdx[idx]],
                                            prob.x[idxWV], para.gamma)) + B);

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

    std::cout << "Iterations: " << t << std::endl;

}

