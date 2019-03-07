##
# @file   setup.py
# @author Yibo Lin
# @date   Jun 2018
#

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

import os 
import sys

utility_dir = os.environ['UTILITY_DIR']
ops_dir = os.environ['OPS_DIR']

include_dirs = [os.path.abspath(ops_dir)]
lib_dirs = [utility_dir]
libs = ['utility'] 

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
                include_dirs=include_dirs, 
                library_dirs=lib_dirs,
                libraries=libs,
                extra_compile_args={
                    #'cxx': ['-g', '-O0'], 
                    'cxx': ['-O2'], 
                    }
                ),
            ],
        cmdclass={
            'build_ext': BuildExtension
            })
