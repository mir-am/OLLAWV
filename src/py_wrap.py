#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 15:46:58 2018

@author: Mir, A.

Python's wrapper for SVM classifier which is implemented in C++.

"""

class SVM:
    
    """
    Implementaion of Support Vector Machines using OLLAWV algorithm
    """
    
    def __init__(self, C=1.0, gamma=1.0, tol=0.1):
        
        """
        Parameters:
            C: float, (default=1.0)
               Penalty parameter
               
            gamma: float, (default=1.0)
                   Kernel coefficient for RBF function
                       
            tol: float, (default=0.1)
                 Tolerance for stopping criterion
        """
        
        
        self.C = C
        self.gamma = gamma
        self.tol = tol
    
    def fit(self):
        
        pass
    
    def predict(self):
        
        pass

