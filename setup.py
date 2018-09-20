from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
from os.path import join
import numpy as np

# Build C/C++ extension for SVM sub-package
svm_path = join('mir_svm', 'svm', 'Cython')

svm_ext = Extension(
              'cy_wrap',
              language='c++',
              sources=[join(svm_path, 'cy_wrap.pyx'),
                       join(svm_path, 'lib', 'svm_template.cpp'),
                       join(svm_path, 'lib', 'misc.cpp'),
                       join(svm_path, 'lib', 'svm_helper.c')],
             extra_compile_args=['-std=c++11'],
             include_dirs=[np.get_include(), join(svm_path, 'lib')]
             )


setup(
name="Mir's SVM API",
version='0.1-dev',
author='Mir, A.',
description=("A simple SVM API."),
packages=find_packages(),
license='GNU General Public License v3.0',
ext_modules=[cythonize(svm_ext)]
)

