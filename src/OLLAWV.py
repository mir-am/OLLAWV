#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of Online Learning Algorithm using Worst Violators
Version: 0.1.0-alpha - 2018-06-11
Developer: Mir, A. (mir-am@hotmail.com)
License: GNU General Public License v3.0
"""

from dataproc import read_data
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold
import numpy as np
import time


def eval_metrics(y_true, y_pred):

    """
        Input:
            
            y_true: True label of samples
            y_pred: Prediction of classifier for test samples
    
        output: Elements of confusion matrix and Evalaution metrics such as
        accuracy, precision, recall and F1 score
    
    """        
    
    # Elements of confusion matrix
    tp, tn, fp, fn = 0, 0, 0, 0
    
    for i in range(y_true.shape[0]):
        
        # True positive 
        if y_true[i] == 1 and y_pred[i] == 1:
            
            tp = tp + 1
        
        # True negative 
        elif y_true[i] == -1 and y_pred[i] == -1:
            
            tn = tn + 1
        
        # False positive
        elif y_true[i] == -1 and y_pred[i] == 1:
            
            fp = fp + 1
        
        # False negative
        elif y_true[i] == 1 and y_pred[i] == -1:
            
            fn = fn + 1
            
    # Compute total positives and negatives
    positives = tp + fp
    negatives = tn + fn

    # Initialize
    accuracy = 0
    # Positive class
    recall_p = 0
    precision_p = 0
    f1_p = 0
    # Negative class
    recall_n = 0
    precision_n = 0
    f1_n = 0
    
    try:
        
        accuracy = (tp + tn) / (positives + negatives)
        # Positive class
        recall_p = tp / (tp + fn)
        precision_p = tp / (tp + fp)
        f1_p = (2 * recall_p * precision_p) / (precision_p + recall_p)
        
        # Negative class
        recall_n = tn / (tn + fp)
        precision_n = tn / (tn + fn)
        f1_n = (2 * recall_n * precision_n) / (precision_n + recall_n)
        
    except ZeroDivisionError:
        
        pass # Continue if division by zero occured


    return tp, tn, fp, fn, accuracy * 100 , recall_p * 100, precision_p * 100, f1_p * 100, \
           recall_n * 100, precision_n * 100, f1_n * 100

class OLLAWV:

    def __init__(self, C, gamma):

        """
        Input:
            C: Penalty parameter
            gamma: gamma parameter for RBF function
        """

        # SVM hyper-parameters
        self.c = C
        self.y = gamma

        # Model parameters
        self.alpha = None
        self.bias = None

        self.X_train = None

    def fit(self, X_train, y_train):

        """
        Input:
            X_train: Training samples
            y_train: Labels
        """
        
        self.X_train = X_train
        
        M = 1 # Scaled value of C

        # Initialize OLLAWV model parameters
        self.alpha = np.zeros((self.X_train.shape[0]))
        self.bias = 0
        #error_idx_vec = np.zeros((X_train.shape[0]), dtype=int)

        # Initialize the output vector and iteration counter
        output_vec = np.zeros((self.X_train.shape[0]))
        t = 0

        # Initialize hinge loss error and worst-violator index
        wv = 0
        yo = y_train[wv] * output_vec[wv]

        # Indexes
        non_sv_idx = list(range(0, self.X_train.shape[0])) 

        while (yo < M): #and len(non_sv_idx) > 1:

            t = t + 1
            learn_rate = 2 / np.sqrt(t)
            non_sv_idx.remove(wv) # Save index of worst violator

            # Calculate hingeloss update
            loss = learn_rate * self.c * y_train[wv]
            # Calculate bias update
            B = loss / X_train.shape[0]
            
            # Update worst violator's alpha value
            self.alpha[wv] = self.alpha[wv] + loss
            # Update bias term
            self.bias = self.bias + B
            
            if len(non_sv_idx) != 0:

                output_vec[non_sv_idx[0]] = output_vec[non_sv_idx[0]] + loss * rbf_kernel(non_sv_idx[0], \
                              X_train[wv], self.y) + B
                          
                wv = non_sv_idx[0]
                yo = y_train[wv] * output_vec[wv]
    
                # Update output vector         
                for idx in non_sv_idx[1:]:
    
                    output_vec[idx] = output_vec[idx] + loss * rbf_kernel(X_train[idx], \
                              X_train[wv], self.y) + B
                              
                    # Find the worst violator
                    if y_train[idx] * output_vec[idx] < yo:
                        
                        wv = idx
                        yo = y_train[idx] * output_vec[idx]
                        
                        
            else:
                
                break

            #print(yo)
    
    def predict(self, X_test):

        pre_labels = np.zeros((X_test.shape[0]), dtype=np.int)

        for i in range(X_test.shape[0]):

            svm_output = 0

            for j in range(self.X_train.shape[0]):

                svm_output += self.alpha[j] * rbf_kernel(X_test[i, :], \
                                        self.X_train[j, :], self.y)

            svm_output +=  self.bias
            
            pre_labels[i] = np.sign(svm_output)

        return pre_labels

        
def rbf_kernel(x, y, gamma):

    """
    Input:
        x, y: samples in input space
        gamma: RBF kernel's parameter
    """

    return np.exp(-1 * gamma * np.power(np.linalg.norm(x - y), 2))
     
# cross validation - SVM - Non-linear
def cv_svm_nl(data_train, data_labels, k, c, rbf_u):
    
    # Instance of SVM
    svm = OLLAWV(c, rbf_u)
     
    # Number of folds for cross validation
    k_fold = KFold(k)
    
    # Store result after each run
    mean_accuracy = []
    
    k_time = 1
    
    # Train and test LSTSVM K times
    for train_index, test_index in k_fold.split(data_train):
        
        # Extract data based on index created by k_fold
        X_train = np.take(data_train, train_index, axis=0) 
        X_test = np.take(data_train, test_index, axis=0)
        
        X_train_label = np.take(data_labels, train_index, axis=0)
        X_test_label = np.take(data_labels, test_index, axis=0)
        
        # Train SVM classifier
        svm.fit(X_train, X_train_label)
        
        # Predict
        output = svm.predict(X_test)
        
        accuracy_test = accuracy_score(X_test_label, output)
                              
        mean_accuracy.append(accuracy_test * 100)
        
        print("K_fold %d finished..." % k_time)
        
        k_time = k_time + 1
        
    # m_a=0, m_r_p=1, m_p_p=2, m_f1_p=3, k=4, c=5, w=6, b=7 m_r_n=8, m_p_n=9, m_f1_n=10, 
    # tp=11, tn=12, fp=13, fn=14, rbf=15 iter=16    
    return np.mean(mean_accuracy)
        
        
# Test
c = 4 ** -1
y = 4 ** 1

train_data, lables, filename = read_data('../dataset/checkerboard.csv')
X_t, X_te, y_tr, y_te = train_test_split(train_data, lables, test_size=0.2,\
                                                    random_state=42)

start_t = time.time()

svm_1 = OLLAWV(c, y)
svm_1.fit(X_t, y_tr)
result = svm_1.predict(X_te)

cv_test = cv_svm_nl(train_data, lables, 5, c, y)

a = svm_1.alpha
b = svm_1.bias

print("Finished in %.2f sec" % ((time.time() - start_t)))
print("CV acc: %.2f " % cv_test)
print('percent of support vectors: %.2f' % ((np.count_nonzero(a) / a.shape[0])*100))
print("Accuracy: %.2f" % (accuracy_score(y_te, result) * 100))
