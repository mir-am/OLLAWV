#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 11:26:12 2018

@author: Mir, A.

Building extension module

"""

from distutils.core import setup, Extension
from Cython.Build import cythonize
from os.path import join
import numpy as np

# join('lib', 'svm.h')
# 

#setup(name='cy_wrap',
#      version='0.1',
#      author='Mir, A.',
#      description="A cython extension for SVM classifier",
#      ext_modules=cythonize(Extension(
#              'cy_wrap',
#              language='c++',
#              sources=[join('Cython', 'cy_wrap.pyx'),
#                       join('Cython', 'lib', 'svm_template.cpp'),
#                       join('Cython', 'lib', 'svm.cpp'),
#                       join('Cython', 'lib', 'svm_helper.c')],
#              
#              )),
#              include_dirs=[join('Cython', 'lib'), np.get_include()])

def configuration(parent_package='', top_path=None):
    
    from numpy.distutils.misc_util import Configuration
    
    config = Configuration('SVM', parent_package, top_path)
    
    # Extension module of SVM
    config.add_library('libsvm', )
      
