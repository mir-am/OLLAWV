"""
Created on Sun Aug 19 17:02:10 2018
@author: Mir, A.

Cython header file

"""

cimport numpy as np

# Include

cdef extern from "../lib/svm.h":
    
    cdef struct SVMDenseNode
    cdef struct SVMModel
    
    cdef struct SVMParameter:
        
        double gamma # For RBF function
        double C # Penlaty parameter
        double e # A value between 0 and 1. Stopping criteria
        
    cdef struct SVMProblem:
        
        int l
        double *y
        SVMNode *x
        



