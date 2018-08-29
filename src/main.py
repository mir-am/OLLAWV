#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 13:54:15 2018

@author: Mir, A.
"""

from dataproc import read_data
from eval_classifier import initializer


class UserInput:

    """
    This class stores user inputs
    """

    def __init__(self, data_tuple, result_path, kernel, test_m_tuple,
                 l_b_c, u_b_c, l_b_u, u_b_u):

        # Initializing all the inputs
        self.X_train, self.y_train = data_tuple[0], data_tuple[1]
        self.filename = data_tuple[2]
        self.result_path = result_path
        self.kernel_type = kernel
        self.test_method_tuple = test_m_tuple
       
        # Parameters
        self.lower_b_c, self.upper_b_c = l_b_c, u_b_c
        # Lower and upper bounds of gamma parameter
        self.lower_b_u, self.upper_b_u = l_b_u, u_b_u
        
        #self.dict_parameters = dict_para


if __name__ == '__main__':

    
    dataset = read_data('../dataset/checkerboard.csv')
    
    user_in_obj = UserInput(dataset, './result', 'RBF', ('CV', 5), -5, 5, -5, 5)
    
    initializer(user_in_obj)
    