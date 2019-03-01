##
# @file   setup.py
# @author Yibo Lin
# @date   Jun 2018
#

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

import os 
import sys

def add_prefix(filename):
    return os.path.join('${CMAKE_CURRENT_SOURCE_DIR}/src', filename)

modules = []

modules.extend([
    CppExtension('greedy_legalize_cpp', 
        [
            add_prefix('greedy_legalize.cpp'),
            add_prefix('legalize_bin_cpu.cpp'), 
            add_prefix('bin_assignment_cpu.cpp'), 
            add_prefix('merge_bin_cpu.cpp'), 
            add_prefix('greedy_legalize_cpu.cpp') 
            ], 
        extra_compile_args={
            #'cxx': ['-g', '-O0'], 
            'cxx': ['-O2'], 
            }
        )
    ])

setup(
        name='greedy_legalize',
        ext_modules=modules,
        cmdclass={
            'build_ext': BuildExtension
            })
