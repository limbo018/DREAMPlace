##
# @file   setup.py
# @author Yibo Lin
# @date   Jun 2018
#

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

import os 
import sys

setup(
        name='greedy_legalize',
        ext_modules=[
            CppExtension('greedy_legalize_cpp', 
                [
                    'greedy_legalize.cpp',
                    'legalize_bin_cpu.cpp', 
                    'bin_assignment_cpu.cpp', 
                    'merge_bin_cpu.cpp', 
                    'greedy_legalize_cpu.cpp' 
                    ], 
                extra_compile_args={
                    #'cxx': ['-g', '-O0'], 
                    'cxx': ['-O2'], 
                    }
                ),
            ],
        cmdclass={
            'build_ext': BuildExtension
            })
