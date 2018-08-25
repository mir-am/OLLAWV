"""
Created on Mon Aug 20 10:57:25 2018

@author: Mir, A

Wrting a wrapper function for methods that implemented in C/C++

"""

import numpy as np
cimport numpy as np
cimport cy_wrap
from libc.stdlib cimport free

np.import_array()

# Wrapper function
def fit(np.ndarray[np.float64_t, ndim=2, mode='c'] X,
        np.ndarray[np.float64_t, ndim=1, mode='c'] Y,
        double C=1, double gamma=0.1, double tol=0.1):
    
    """
    Train the SVM model with OLLAWV using low-level mothods
    
    Parameters:
        
        X: Training samples, dtype=float64, size=[n_samples, n_features]
        Y: Target vector, dtype=float64, size=[n_samples]
        C: Penalty parameter, float64
        gamma: Parameter for RBF function, float64
        tol: Stopping criteria, float64
        
    Returns:
        support: Index of support vectors, (n_support,)
        support_vectors: support vectors, (n_support, features)
        n_class_SV: Number of support vectors for each class
        svCoef: coefficient of support vectors in decision function
        intercept: intercept in decision function
        
    """
    
    cdef SVMParameter param
    cdef SVMProblem prob
    cdef SVMModel *model
    cdef const char* errorMsg
    cdef np.npy_intp SV_len
    cdef np.npy_intp nr
    
    setProblem(&prob, X.data, Y.data, X.shape)
    
    if prob.x == NULL:
        
        raise MemoryError("Ran out of memory.")
        
    setParameter(&param, gamma, C, tol)
    
    errorMsg = SVMCheckParameter(&param)
    
    if errorMsg:
        
        raise ValueError(errorMsg.decode('utf-8'))

    cdef int fit_status = 0
    with nogil:
        
        model = SVMTrain(&prob, &param, &fit_status)
        
    # Copy the data returned by SVMTrain
    numSV = getNumSV(model)
    numClasses = getNumClass(model)
    
    #print("NumSV: %d, Classes: %d" % (numSV, numClasses))
    
    cdef np.ndarray[np.float64_t, ndim=2, mode='c'] svCoef
    svCoef = np.empty((numClasses - 1, numSV), dtype=np.float64)
    copySvCoef(svCoef.data, model)
    
    #print(svCoef.shape[0], svCoef.shape[1])
    
    # Copy the intercept (Bias)
    cdef np.ndarray[np.float64_t, ndim=1, mode='c'] intercept
    intercept = np.empty(int(numClasses * (numClasses - 1) / 2), dtype=np.float64)
    copyIntercept(intercept.data, model, intercept.shape)
    
    #print(intercept)
    
    # Indices of SVs
    cdef np.ndarray[np.int32_t, ndim=1, mode='c'] support
    support = np.empty(numSV, dtype=np.int32)
    copySupport(support.data, model)
    
    #print(support.shape[0], support.shape[1])
    
    # Copy model.SV
    # Samples that are SVs.
    cdef np.ndarray[np.float64_t, ndim=2, mode='c'] supportVectors
    supportVectors = np.empty((numSV, X.shape[1]), dtype=np.float64)
    copySV(supportVectors.data, model, supportVectors.shape)
    
    #print(supportVectors.shape[0], supportVectors.shape[1])
    
    # Number of SVs for each class
    cdef np.ndarray[np.int32_t, ndim=1, mode='c'] n_class_SV
    n_class_SV = np.empty(numClasses, dtype=np.int32)
    copySVClass(n_class_SV.data, model)
    
    #print(n_class_SV)
    
    SVMFreeModel(&model)
    free(prob.x)
    
    return (support, supportVectors, n_class_SV, svCoef, intercept, fit_status)
    
    
def predict(np.ndarray[np.float64_t, ndim=2, mode='c'] X,
            np.ndarray[np.int32_t, ndim=1, mode='c'] support,
            np.ndarray[np.float64_t, ndim=2, mode='c'] SV,
            np.ndarray[np.int32_t, ndim=1, mode='c'] nSV,
            np.ndarray[np.float64_t, ndim=2, mode='c'] sv_coef,
            np.ndarray[np.float64_t, ndim=1, mode='c'] intercept,
            double gamma=0.1):
    
    """
    Predicts target values of X given a model. (low-level methods)
    
    Parameters:
        X: Test samples, dtype=float, shape=[n_samples, n_features]
        support: Index of support vectors, (n_support,)
        SV: support vectors, (n_support, features)
        nSV: Number of support vectors for each class
        svCoef: coefficient of support vectors in decision function
        intercept: intercept in decision function
        gamma: Parameter of RBF function, float, default=0.1
        
    Returns:
        dec_values: predicted values
    """
    
    cdef np.ndarray[np.float64_t, ndim=1, mode='c'] dev_values
    cdef SVMParameter param
    cdef SVMModel *model
    cdef int rv
    
    # Training time-only parameters
    cdef double C = .0
    cdef double tol = .1
    
    setParameter(&param, gamma, C, tol)
    
    model = setModel(&param, <int> nSV.shape[0], SV.data, SV.shape, support.data,
                     support.shape, sv_coef.strides, sv_coef.data, intercept.data,
                     nSV.data)
    


