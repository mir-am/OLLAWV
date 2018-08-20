"""
Created on Sun Aug 19 17:02:10 2018
@author: Mir, A.

Cython header file

"""

cimport numpy as np

# Include

cdef extern from "lib/svm.h":
    
    cdef struct SVMNode
    cdef struct SVMModel
    
    cdef struct SVMParameter:
        
        double gamma # For RBF function
        double C # Penlaty parameter
        double e # A value between 0 and 1. Stopping criteria
        
    cdef struct SVMProblem:
        
        int l
        double *y
        SVMNode *x
        
    char* SVMCheckParameter(SVMParameter*);
        
    SVMModel* SVMTrain(SVMProblem*, SVMParameter* , int*) nogil
        

cdef extern from "lib/svm_helper.c":
    
    # This file contains utility functions
    
    void setParameter(SVMParameter*, double, double, double)
    
    void setProblem(SVMProblem* , char*, char*, np.npy_intp *)
    
    
    
    

