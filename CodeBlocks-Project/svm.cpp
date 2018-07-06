#include "svm.h"
#include <cstddef>
#include <iostream>


template <typename T>
inline void swapVar(T& x, T& y)
{
    T temp = x;
    x = y;
    y = temp;
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

    for(int i = 0; i < prob.l; ++i)
        solutionInfo.alpha[i] = 0;


}


SVMModel trainSVM(const SVMProblem& prob, const SVMParameter& param)
{
    // Classification
    SVMModel model;
    model.param = param;

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
        for(int j = i + 1; j < numClass; ++j)
        {
            SVMProblem subProb; // A sub problem for i-th and j-th class
            //subProb

            // start points of i-th and j-th classes
            int si = start[i], sj = start[j];
            // Number of samples in i-th and j-th class
            int ci = count[i], cj = count[j];

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

            //f[p] =

        }

}


