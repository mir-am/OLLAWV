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
import time

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
        
        # Parameters
        self.C = C
        self.gamma = gamma
        self.tol = tol
        
        # Model
        self.support_ = self.support_vectors_ = self.n_support_ = self.dual_coef = \
        self.intercept = self.fit_status_ = None
        
    
    def fit(self, X_train, y_train):
        
        """
        Given training set, it creates a SVM model
        
        Parameters:
            X_train: Training samples, (n_samples, n_features)
            y_train: Target values, (n_samples, )
        """
        
        self.support_, self.support_vectors_, self.n_support_, \
        self.dual_coef_, self.intercept_, self.fit_status_ = cy_wrap.fit(
                X_train, y_train.astype('float64'), self.C, self.gamma, self.tol)
    
    def predict(self, X_test):
        
        """
        Predicits lables of test samples
        
        Parameters:
            X_test: test samples, (n_samples, n_features)
        
        Returns:
            y_pred: array, (n_samples,)
        
        """
        
        return cy_wrap.predict(X_test, self.support_, self.support_vectors_,
                        self.n_support_, self.dual_coef_, self.intercept_, self.gamma)
        


if __name__ == '__main__':
    
    train_data, lables, file_name = read_data('../dataset/pima-indian.csv')
    
    X_t, X_te, y_tr, y_te = train_test_split(train_data, lables, test_size=0.3,\
                                                    random_state=42)
    
    start_t = time.time()
    
    model = SVM(1, 2)
    model.fit(X_t, y_tr)
    pred = model.predict(X_te)
    
    print(pred)

    print("Finished in %.3f ms" % ((time.time() - start_t) * 1000))

