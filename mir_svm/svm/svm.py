#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 15:46:58 2018

@author: Mir, A.

Python's wrapper for SVM classifier which is implemented in C++.

To solve SVM's primal problems, Online learning algorithm using worst violators
(OLLAWV) is implemented.

"""

from dataproc import read_data
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils import column_or_1d
from sklearn.metrics import accuracy_score
import numpy as np
import scipy.sparse as sp
import cy_wrap
import time


class SVM(BaseEstimator, ClassifierMixin):
    
    """
    Implementaion of Support Vector Machines using OLLAWV algorithm
    """
    
    def __init__(self, kernel='rbf', C=1.0, gamma=1.0, tol=0.1):
        
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
        #self.support_ = self.support_vectors_ = self.n_support_ = self.dual_coef_ = \
        #self.intercept = self.fit_status_ = self.classes_ = None
        
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
        
        
    def _validate_targets(self, y):
    
        """
        Validates labels for training and testing classifier
        """
        y_ = column_or_1d(y, warn=True)
        check_classification_targets(y)
        self.classes_, y = np.unique(y_, return_inverse=True)
        
        return np.asarray(y, dtype=np.float64, order='C')
    
    def _validate_for_predict(self, X):
        
        """
        Checks that the classifier is already trained and also test samples are
        valid
        """
        
        check_is_fitted(self, ['support_'])
        X = check_array(X, dtype=np.float64, order="C")
        
        n_samples, n_features = X.shape
        
        if n_features != self.shape_fit_[1]:
            
            raise ValueError("X.shape[1] = %d should be equal to %d," 
                             "the number of features of training samples" % 
                             (n_features, self.shape_fit_[1]))
        
        return X
        
    def fit(self, X, y):
        
        """
        Given training set, it creates a SVM model
        
        Parameters:
            X_train: Training samples, (n_samples, n_features)
            y_train: Target values, (n_samples, )
        """
        
        y = self._validate_targets(y)
        X, y = check_X_y(X, y, dtype=np.float64,
                                     order='C')
        
        
        self.support_, self.support_vectors_, self.n_support_, \
        self.dual_coef_, self.intercept_, self.fit_status_ = cy_wrap.fit(
                X, y, self.C, self.gamma, self.tol)
        
        self.shape_fit_ = X.shape
        self.X_ = X
        self.y_ = y
        
        return self
    
    def predict(self, X_test):
        
        """
        Predicits lables of test samples
        
        Parameters:
            X_test: test samples, (n_samples, n_features)
        
        Returns:
            y_pred: array, (n_samples,)
        
        """
        
        X_test = self._validate_for_predict(X_test)
        
        y = cy_wrap.predict(X_test, self.support_, self.support_vectors_,
                        self.n_support_, self.dual_coef_, self.intercept_, self.gamma)
        
        return self.classes_.take(np.asarray(y, dtype=np.intp))
        #return np.asarray(y, dtype=np.intp)


if __name__ == '__main__':
    
    train_data, lables, file_name = read_data('../../dataset/pima-indian.csv')
    
#    X_t, X_te, y_tr, y_te = train_test_split(train_data, lables, test_size=0.2,\
#                                                    random_state=42)
    
    param = {'C': [float(2**i) for i in range(-5, 6)],
             'gamma': [float(2**i) for i in range(-10, 3)]}
    
    start_t = time.time()
    
    model = SVM()
    
    #scores = cross_val_score(model, train_data, lables, cv=5)
    
    result = GridSearchCV(model, param, cv=5, n_jobs=4, refit=False, verbose=1)
    result.fit(train_data, lables)
    
    print(result.best_score_ * 100)
    print(result.best_params_)
    print(result.cv_results_['std_test_score'][result.best_index_] * 100)
    
    #print("Accuracy: %.2f" % (scores.mean() * 100))
    
    #model.fit(X_t, y_tr)
    
    #check_estimator(SVM)
    
    
    #print("Training Finished in %.3f ms" % ((time.time() - start_t) * 1000))
    
    #t_test = time.time()
    
    #pred = model.predict(X_te)
    
    #print("Prediction Finished in %.3f ms" % ((time.time() - t_test) * 1000))

    #pred_train = model.predict(X_t)
    
    #print("Targets: \n", y_te)
    #print("Predictions: \n", pred)
    
    #print("Test Accuracy: %.2f" % (accuracy_score(y_te, pred) * 100))
    #print("Training Accuracy: %.2f" % (accuracy_score(y_tr, pred_train) * 100))
    
    print("Finished in %.3f ms" % ((time.time() - start_t) * 1000))

