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
        
        int kernelType
        double gamma # For RBF function
        double C # Penlaty parameter
        double e # A value between 0 and 1. Stopping criteria
        
    cdef struct SVMProblem:
        
        int l
        double *y
        SVMNode *x
        
    char* SVMCheckParameter(SVMParameter*);
        
    SVMModel* SVMTrain(SVMProblem*, SVMParameter* , int*) nogil
    void SVMFreeModel(SVMModel **model_ptr_ptr)
        

cdef extern from "lib/svm_helper.c":
    
    # This file contains utility functions
    
    void setParameter(SVMParameter*, double, double, double)
    
    void setProblem(SVMProblem* , char*, char*, np.npy_intp *)
    
    SVMModel *setModel(SVMParameter *, int, char *, np.npy_intp *, char *,
                       np.npy_intp *, np.npy_intp *, char *, char *, char *)
    

    int copyPredict(char *, SVMModel *, np.npy_intp *, char *) nogil    
    void copySvCoef(char *, SVMModel *)
    void copyIntercept(char *, SVMModel *, np.npy_intp *)
    void copySupport(char *, SVMModel *)
    void copySV(char *, SVMModel *, np.npy_intp *)
    void copySVClass(char *, SVMModel *)
    
    np.npy_intp getNumSV(SVMModel *)
    np.npy_intp getNumClass(SVMModel *)
    
    int freeModel(SVMModel *)
    