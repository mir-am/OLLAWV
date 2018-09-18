#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 17:44:18 2018

@author: Mir, A.
"""

# A comparison between Scikit-learn implementation and Mir's

from dataproc import read_data
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from svm import SVM
import numpy as np
import time

train_data, lables, file_name = read_data('../dataset/checkerboard.csv')

sk_model = SVC()
mir_model = SVM()

param = {'C': [float(2**i) for i in range(-10, 6)],
          'gamma': [float(2**i) for i in range(-10, 5)]}
    
mir_start_t = time.time()

mir_result = GridSearchCV(mir_model, param, cv=10, n_jobs=-1, refit=False, verbose=1)
mir_result.fit(train_data, lables)

mir_end = time.time()



sk_start_t = time.time()

sk_result = GridSearchCV(sk_model, param, cv=10, n_jobs=-1, refit=False, verbose=1)
sk_result.fit(train_data, lables)

sk_end = time.time()

print("Mir's Acc: ", mir_result.best_score_ * 100)
print(mir_result.best_params_)
print("Mir's Std: ", mir_result.cv_results_['std_test_score'][mir_result.best_index_] * 100)
print("Finished in %.3f ms" % ((mir_end - mir_start_t) * 1000))

print("Sk's Acc: ", sk_result.best_score_ * 100)
print(sk_result.best_params_)
print("Sk's Std: ", sk_result.cv_results_['std_test_score'][sk_result.best_index_] * 100)

print("Finished in %.3f ms" % ((sk_end - sk_start_t) * 1000))
