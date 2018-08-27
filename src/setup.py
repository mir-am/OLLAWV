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


#def configuration(parent_package='', top_path=None):
#    
#    from numpy.distutils.misc_util import Configuration
#    from Cython.Build import cythonize
#    
#    config = Configuration('SVM', parent_package, top_path)
#    
#    # Extension module of SVM
#    config.add_library('svm_ollawv', 
#                       sources=[join('lib', 'svm_template.cpp')],
#                       depends=[join('lib', 'svm.cpp'),
#                                join('lib', 'svm.h')],
#                       extra_link_args=['-lstdc++']
#                       )
#    
#    svm_sources = [join('cy_wrap.pyx')]
#    svm_depends = [join('lib', 'svm_helper.c'),
#                   join('lib', 'svm_template.cpp'),
#                   join('lib', 'svm.cpp'),
#                   join('lib', 'svm.h')]
#    
#    config.add_extension('cy_wrap',
#                         sources=svm_sources,
#                         include_dirs=[np.get_include(),
#                                       'lib'],
#                          libraries=['svm_ollawv'],
#                          depends=svm_depends)
#    
#    #config.ext_modules[-1] = cythonize(config.ext_modules[-1])
#    
#    return config


if __name__ == '__main__':
    
    #from numpy.distutils.core import setup
    
    #setup(**configuration(top_path='').todict())
      
    setup(name='cy_wrap',
      version='0.1',
      author='Mir, A.',
      description="A cython extension for SVM classifier",
      ext_modules=cythonize(Extension(
              'cy_wrap',
              language='c++',
              sources=[join('Cython', 'cy_wrap.pyx'),
                       join('Cython', 'lib', 'svm.cpp'),
                       join('Cython', 'lib', 'svm_template.cpp'),
                       join('Cython', 'lib', 'svm_helper.c')],
             extra_compile_args=['-std=c++11']
              
              )),
              include_dirs=[np.get_include(), join('Cython', 'lib')])

