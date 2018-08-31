#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 15:46:58 2018

@author: Mir, A.

Python's wrapper for SVM classifier which is implemented in C++.

"""

from dataproc import read_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import cy_wrap
import time


class SVM:
    
    """
    Implementaion of Support Vector Machines using OLLAWV algorithm
    """
    
    def __init__(self, kernel='RBF', C=1.0, gamma=1.0, tol=0.1):
        
        """
        Parameters:
            
            kernel: Kernel function. Currenty, only RBF is supported.
            
            C: float, (default=1.0)
               Penalty parameter
               
            gamma: float, (default=1.0)
                   Kernel coefficient for RBF function
                       
            tol: float, (default=0.1)
                 Tolerance for stopping criterion
        """
        
        # Parameters
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.tol = tol
        
        self.cl_name = "SVM"
        
        # Model
        self.support_ = self.support_vectors_ = self.n_support_ = self.dual_coef_ = \
        self.intercept = self.fit_status_ = self.classes_ = None
        
    def set_parameter(self, C, gamma):
        
        """
        It changes the parametes for SVM classifier.
        DO NOT USE THIS METHOD AFTER INSTANTIATION OF SVM CLASS!
        THIS METHOD CREATED ONLY FOR Validator CLASS.
        Input:
            C: Penalty parameter
            gamma: RBF function parameter
        """

        self.C = C
        self.gamma = gamma
        
        
    def _validate_targets(self, labels):
    
        """
        Validates labels for training and testing classifier
        """
        
        self.classes_, y = np.unique(labels, return_inverse=True)
        
        return np.asarray(y, dtype=np.float64, order='C')
        
    
    def fit(self, X_train, y_train):
        
        """
        Given training set, it creates a SVM model
        
        Parameters:
            X_train: Training samples, (n_samples, n_features)
            y_train: Target values, (n_samples, )
        """
        
        y = self._validate_targets(y_train)
        
        print("First row: ", X_train[0, :])
        print("Last row: ", X_train[-1, :])
        
        self.support_, self.support_vectors_, self.n_support_, \
        self.dual_coef_, self.intercept_, self.fit_status_ = cy_wrap.fit(
                X_train, y, self.C, self.gamma, self.tol)
    
    def predict(self, X_test):
        
        """
        Predicits lables of test samples
        
        Parameters:
            X_test: test samples, (n_samples, n_features)
        
        Returns:
            y_pred: array, (n_samples,)
        
        """
        
        y = cy_wrap.predict(X_test, self.support_, self.support_vectors_,
                        self.n_support_, self.dual_coef_, self.intercept_, self.gamma)
        
        return self.classes_.take(np.asarray(y, dtype=np.intp))
        #return np.asarray(y, dtype=np.intp)


if __name__ == '__main__':
    
    train_data, lables, file_name = read_data('../dataset/checkerboard.csv')
    
    X_t, X_te, y_tr, y_te = train_test_split(train_data, lables, test_size=0.2,\
                                                    random_state=42)
    
    start_t = time.time()
    
    model = SVM('RBF', 2.0, 0.5)
    model.fit(X_t, y_tr)
    pred = model.predict(X_te)
    
    print(pred)
    
    print("Accuracy: %.2f" % (accuracy_score(y_te, pred) * 100))
    
    print("Finished in %.3f ms" % ((time.time() - start_t) * 1000))

