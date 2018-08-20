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

#    cdef int fit_status = 0
#    with nogil:
#        
#        model = SVMTrain(&prob, &param, &fit_status)

