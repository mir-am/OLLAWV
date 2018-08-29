#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LightTwinSVM Program - Simple and Fast
Version: 0.2.0-alpha - 2018-05-30
Developer: Mir, A. (mir-am@hotmail.com)
License: GNU General Public License v3.0

Module: eval_classifier.py
In this module, methods are defined for evluating TwinSVM perfomance such as cross validation
train/test split, grid search and generating the detailed result.
"""


from svm import SVM
from misc import progress_bar_gs, time_fmt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from itertools import product
from datetime import datetime
import numpy as np
import pandas as pd
import os
# import time

#class Parameter:
#    
#    """
#    Parameters of a classfier
#    """
#    
#    def __init__(self, p_1='', p_2='', p_3='', p_4='', p_5=''):
#        
#        self.P1 = p_1  # C1 penalty parameter
#        self.P2 = p_2  # C2 parameter
#        self.P3 = p_3  # C3 parameter
#        self.P4 = p_4  # k parameter or C4 parameter
#        self.P5 = p_5  # Gamma parameter
#
#        
#    def get_parameters(self):
#        
#        return tuple([self.__dict__[para] for para in sorted(self.__dict__.keys())  \
#                      if self.__dict__[para] != ''])
#    
#    def get_dict_para(self):
#        
#        """
#        Returns list of parameters and their corresponding value in dictionary
#        """
#        
#        para_names = {'C1':self.P1, 'C2':self.P2, 'C3':self.P3, 'k': self.P4, \
#                           'u': self.P5}
#        
#        # Dictionary comprehension
#        return {k:v for k,v in para_names.items() if para_names[k] != ''}

    
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


class Validator:

    """
    It applies a test method such as cross validation on a classifier like SVM
    """

    def __init__(self, X_train, y_train, validator_type, obj_svm):

        """
        It constructs and returns a validator 
        Input:
            X_train: Samples in dataset (2-d NumPy array)
            y_train: Labels of samples in dataset (1-d NumPy array)
            validator_type: Type of test methodology and its parameter.(Tuple - ('CV', 5))
            obj_tsvm:  Instance of SVM classifier. (SVM class)

        """

        self.train_data = X_train
        self.labels_data = y_train
        self.validator = validator_type
        self.obj_SVM = obj_svm

#    def cv_validator(self, elem_obj):
#
#        """
#        It applies cross validation on instance of TSVM classifier
#        output:
#            Evaluation metrics such as accuracy, precision, recall and F1 score
#            for each class.
#        """
#        
#        # Set parameters of TSVM classifer
#        self.obj_TSVM._set_parameter(elem_obj)
#        
#        # K-Fold Cross validation, divide data into K subsets
#        k_fold = KFold(self.validator[1])    
#
#        # Store result after each run
#        mean_accuracy = []
#        # Postive class
#        mean_recall_p, mean_precision_p, mean_f1_p = [], [], []
#        # Negative class
#        mean_recall_n, mean_precision_n, mean_f1_n = [], [], []
#        
#        # Count elements of confusion matrix
#        tp, tn, fp, fn = 0, 0, 0, 0
#        
#        # Train and test TSVM classifier K times
#        for train_index, test_index in k_fold.split(self.train_data):
#
#            # Extract data based on index created by k_fold
#            X_train = np.take(self.train_data, train_index, axis=0) 
#            X_test = np.take(self.train_data, test_index, axis=0)
#
#            y_train = np.take(self.labels_data, train_index, axis=0)
#            y_test = np.take(self.labels_data, test_index, axis=0)
#
#            # fit - create two non-parallel hyperplanes
#            self.obj_TSVM.fit(X_train, y_train)
#
#            # Predict
#            output = self.obj_TSVM.predict(X_test)
#
#            accuracy_test = eval_metrics(y_test, output)
#
#            mean_accuracy.append(accuracy_test[4])
#            # Positive cass
#            mean_recall_p.append(accuracy_test[5])
#            mean_precision_p.append(accuracy_test[6])
#            mean_f1_p.append(accuracy_test[7])
#            # Negative class    
#            mean_recall_n.append(accuracy_test[8])
#            mean_precision_n.append(accuracy_test[9])
#            mean_f1_n.append(accuracy_test[10])
#
#            # Count
#            tp = tp + accuracy_test[0]
#            tn = tn + accuracy_test[1]
#            fp = fp + accuracy_test[2]
#            fn = fn + accuracy_test[3]
#
#        # m_a=0, m_r_p=1, m_p_p=2, m_f1_p=3, k=4, c1=5, c2=6, gamma=7,
#        # m_r_n=8, m_p_n=9, m_f1_n=10, tp=11, tn=12, fp=13, fn=14, iter=15    
#        return np.mean(mean_accuracy), np.std(mean_accuracy), [np.mean(mean_accuracy), \
#               np.std(mean_accuracy), np.mean(mean_recall_p), np.std(mean_recall_p), \
#               np.mean(mean_precision_p), np.std(mean_precision_p), np.mean(mean_f1_p), \
#               np.std(mean_f1_p), np.mean(mean_recall_n), np.std(mean_recall_n), \
#               np.mean(mean_precision_n), np.std(mean_precision_n), np.mean(mean_f1_n), \
#               np.std(mean_f1_n), tp, tn, fp, fn, *elem_obj.get_parameters()]


    def split_tt_validator(self, elem_obj):
        
        """
        It trains TwinSVM classifier on random training set and tests the classifier
        on test set.
        output:
            Evaluation metrics such as accuracy, precision, recall and F1 score
            for each class.
        """

        # Set parameters of TSVM classifer
        self.obj_SVM._set_parameter(elem_obj)

        X_train, X_test, y_train, y_test = train_test_split(self.train_data, \
                                           self.labels_data, test_size=self.validator[1] / 100, \
                                           random_state=42)

        # fit - create two non-parallel hyperplanes
        self.obj_SVM.fit(X_train, y_train)

        output = self.obj_SVM.predict(X_test)

        tp, tn, fp, fn, accuracy, recall_p, precision_p, f1_p, recall_n, precision_n, \
        f1_n = eval_metrics(y_test, output)

       # m_a=0, m_r_p=1, m_p_p=2, m_f1_p=3, k=4, c1=5, c2=6, gamma=7,
       # m_r_n=8, m_p_n=9, m_f1_n=10, tp=11, tn=12, fp=13, fn=14,   
        return accuracy, 0.0, [accuracy, recall_p, precision_p, f1_p, recall_n, \
               precision_n, f1_n, tp, tn, fp, fn, *elem_obj.get_parameters()]

    def cv_validator(self, c=2**0, gamma=2**0):

        """
        It applies cross validation on instance of multiclass SVM classifier
        """

        # Set parameters of multiclass TSVM classifer
        self.obj_SVM.set_parameter(c, gamma)

        # K-Fold Cross validation, divide data into K subsets
        k_fold = KFold(self.validator[1])    

        # Store result after each run
        mean_accuracy = []
        
        # Evaluation metrics
        mean_recall, mean_precision, mean_f1 = [], [], []
        
        # Train and test multiclass TSVM classifier K times
        for train_index, test_index in k_fold.split(self.train_data):

            # Extract data based on index created by k_fold
            X_train = np.take(self.train_data, train_index, axis=0) 
            X_test = np.take(self.train_data, test_index, axis=0)

            y_train = np.take(self.labels_data, train_index, axis=0)
            y_test = np.take(self.labels_data, test_index, axis=0)

            # fit - creates K-binary TSVM classifier
            self.obj_SVM.fit(X_train, y_train)

            # Predict
            output = self.obj_SVM.predict(X_test)

            mean_accuracy.append(accuracy_score(y_test, output) * 100)
            mean_recall.append(recall_score(y_test, output, average='micro') * 100)
            mean_precision.append(precision_score(y_test, output, average='micro') * 100)
            mean_f1.append(f1_score(y_test, output, average='micro') * 100)

        return np.mean(mean_accuracy), np.std(mean_accuracy), [np.mean(mean_accuracy), \
               np.std(mean_accuracy), np.mean(mean_recall), np.std(mean_recall), \
               np.mean(mean_precision), np.std(mean_precision), np.mean(mean_f1), \
               np.std(mean_f1), c, gamma if self.obj_SVM.kernel == 'RBF' else '']

    def choose_validator(self):

        """
        It returns choosen validator method.
        """

        if self.validator[0] == 'CV':

            return self.cv_validator

        elif self.validator[0] == 't_t_split':

            return self.split_tt_validator



def search_space(kernel_type, c_l_bound, c_u_bound, rbf_lbound, rbf_ubound,
                 step=1):

    """
    It generates combination of search elements for grid search
    Input:
        cl_obj: classifier object
        c_l_bound, c_u_bound: Range of C penalty parameter for grid search(e.g 2^-5 to 2^+5)
        rbf_lbound, rbf_ubound: Range of gamma parameter
    Output:
        return search elements for grid search (List)
    """

    c_range = [2 ** i for i in np.arange(c_l_bound, c_u_bound + 1, step,
                                         dtype=np.float)]
            
    if kernel_type == 'linear':
        
        raise NotImplementedError("Linear kernel is NOT supported yet.")
        
    else:
        
        return list(product(*[c_range, [2 ** i for i in np.arange(rbf_lbound,
                rbf_ubound + 1, step, dtype=np.float)]]))
        
            
#def search_space(dict_paramter, step=1):
#    
#    """
#    It generates combination of search elements for grid search
#        Input: A dictionary that contains lower and bound of each parameter
#               example, dict_para = {'C':(-5, 5), 'k':(2, 8), 'u':(-2, 2)}
#               (0, 0) -> this parameter is disabled.
#    """
#    
#    p_range = lambda l_bound, u_bound: [2 ** i for i in np.arange(l_bound, u_bound + 1, \
#                                        step, dtype=np.float)]
#            
#    k_range = lambda l_bound, u_bound: np.arange(l_bound, u_bound + 1)
#    
#    para_list = []
#    
#    # Dictionary keys should be sorted. C parameter should be first.
#    for para in sorted(dict_paramter.keys()):
#        
#        if dict_paramter[para] != None:
#        
#            if 'C' in para:
#                
#                para_list.append(p_range(dict_paramter[para][0], \
#                                         dict_paramter[para][1]))
#            
#            elif 'k' in para:
#                
#                para_list.append(k_range(dict_paramter[para][0], \
#                                         dict_paramter[para][1]))
#            
#            elif 'u' in para:
#                
#                para_list.append(p_range(dict_paramter[para][0], \
#                                         dict_paramter[para][1]))
#                
#        else:
#            
#            para_list.append([''])
#        
#    return build_search_space(list(product(*para_list)))    
    
#def build_search_space(search_elements):
#
#    search_elem = []
#    
#    for elem in search_elements:
#        
#        cl_par = Parameter()
#        cl_par.P1, cl_par.P2, cl_par.P3, cl_par.P4, cl_par.P5 = elem
#        
#        search_elem.append(cl_par)
#        
#    return search_elem
    
def grid_search(search_space, func_validator):

    """
        It applies grid search which finds C and gamma paramters for obtaining
        best classification accuracy.
    
        Input:
           search_space: search_elements (List)
           func_validator: Validator function
            
        output:
            returns classification result (List)
    
    """

    # Store 
    result_list = []
    
    # Tracking Max accuracy and corresponding optimal parameters 
    max_acc, max_acc_std = 0, 0
    #optimal_para = None

    # Total number of search elements
    search_total = len(search_space)

	# Dispaly headers and progress bar
#    print("TSVM-%s    Dataset: %s    Total Search Elements: %d" % (kernel_type, \
#          file_name, search_total))
    
    progress_bar_gs(0, search_total, '0:00:00', (0.0, 0.0), (0.0, 0.0), prefix='', \
                    suffix='')

    start_time = datetime.now()

    run = 1   

    # Ehaustive Grid search for finding optimal parameters
    for element_obj in search_space:

        try:

            #start_time = time.time()

            # Save result after each run
            acc, acc_std, result = func_validator(*element_obj)

            #end = time.time()

            result_list.append(result)

            # Save best accuracy
            if acc > max_acc:
                
                max_acc = acc
                max_acc_std = acc_std  
                #optimal_para = element_obj
                
            
            elapsed_time = datetime.now() - start_time
            progress_bar_gs(run, search_total, time_fmt(elapsed_time.seconds), \
                            (acc, acc_std), (max_acc, max_acc_std), prefix='', suffix='') 
            

            run = run + 1

        # Some parameters cause errors such as Singular matrix        
        except np.linalg.LinAlgError:
        
            run = run + 1
            

    return result_list  #, (max_acc, max_acc_std, optimal_para)


def save_result(file_name, validator_obj, gs_result, output_path):

    """
        It saves detailed result in spreadsheet file(Excel).

        Input:
            file_name: Name of spreadsheet file
            col_names: Column names for spreadsheet file
            gs_result: result produced by grid search
            para_names: Names of parameters 
            output_path: Path to store the spreadsheet file.

        output:
            returns path of spreadsheet file

    """

    column_names = {'CV': ['accuracy', 'acc_std', 'micro_recall', 'm_rec_std', 'micro_precision', \
                                         'm_prec_std', 'mirco_f1', 'm_f1_std', 'C', 'gamma'],
                    't_t_split': ['accuracy', 'recall_p', 'precision_p', 'f1_p', 'recall_n', 'precision_n', \
                                  'f1_n', 'tp', 'tn', 'fp', 'fn', 'C', 'gamma']}

    # (Name of validator, validator's attribute) - ('CV', 5-folds)
    validator_type, validator_attr = validator_obj.validator

    name_classifier = validator_obj.obj_SVM.cl_name           

    output_file = os.path.join(output_path, "%s_%s_%s_%s_%s.xlsx") % (name_classifier, validator_obj.obj_SVM.kernel, \
                  "%d-F-CV" % validator_attr if validator_type == 'CV' else 'Tr%d-Te%d' % \
                  (100 - validator_attr, validator_attr), file_name, datetime.now().strftime('%Y-%m-%d %H-%M'))

    excel_file = pd.ExcelWriter(output_file, engine='xlsxwriter')
    
    result_frame = pd.DataFrame(gs_result, columns=column_names[validator_type]) 

    result_frame.to_excel(excel_file, sheet_name='Sheet1', index=False)

    excel_file.save()

    return os.path.abspath(output_file)  


def initializer(user_input_obj):

    """
    It gets user input and passes function and classes arguments to run the program
    Input:
        user_input_obj: User input (UserInput class)
    """

    cl_obj = SVM(user_input_obj.kernel_type)
        
    validate = Validator(user_input_obj.X_train, user_input_obj.y_train, \
                         user_input_obj.test_method_tuple, cl_obj)

    search_elements = search_space(user_input_obj.kernel_type,
                                   user_input_obj.lower_b_c,
                                   user_input_obj.upper_b_c,
                                   user_input_obj.lower_b_u,
                                   user_input_obj.upper_b_u)

    # Dispaly headers
    print("%s-%s    Dataset: %s    Total Search Elements: %d" % (cl_obj.cl_name, user_input_obj.kernel_type, \
          user_input_obj.filename, len(search_elements)))

    result = grid_search(search_elements, validate.choose_validator())
    
    # Extract names of parameters
#    names_param = [name for name in sorted(user_input_obj.dict_parameters.keys()) \
#                   if user_input_obj.dict_parameters[name] != None]


    save_result(user_input_obj.filename, validate, result, user_input_obj.result_path)
    
    # Returns (max_acc, max_acc_std, optimal_parameters)
    #return result[1][0], result[1][1], result[1][2]
