#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 15:46:58 2018

@author: Mir, A.

Python's wrapper for SVM classifier which is implemented in C++.

"""

from dataproc import read_data
from sklearn.model_selection import train_test_split
import cy_wrap

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
    
    def fit(self, X_train, y_train):
        
        cy_wrap.fit(X_train, y_train.astype('float64'), self.C, self.gamma,
                    self.tol)
    
    def predict(self):
        
        pass


if __name__ == '__main__':
    
    train_data, lables, file_name = read_data('../dataset/pima-indian.csv')
    
    X_t, X_te, y_tr, y_te = train_test_split(train_data, lables, test_size=0.3,\
                                                    random_state=42)
    
    model = SVM(1, 1)
    model.fit(X_t, y_tr)

