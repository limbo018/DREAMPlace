##
# @file   setup.py
# @author Yibo Lin
# @date   Jun 2018
#

from setuptools import setup
import torch 
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

import os 
import sys

utility_dir = os.environ['UTILITY_DIR']
ops_dir = os.environ['OPS_DIR']

include_dirs = [os.path.abspath(ops_dir)]
lib_dirs = [utility_dir]
libs = ['utility'] 

tokens = str(torch.__version__).split('.')
torch_major_version = "-DTORCH_MAJOR_VERSION=%d" % (int(tokens[0]))
torch_minor_version = "-DTORCH_MINOR_VERSION=%d" % (int(tokens[1]))

setup(
        name='greedy_legalize',
        ext_modules=[
            CppExtension('greedy_legalize_cpp', 
                [
                    'src/greedy_legalize.cpp',
                    'src/legalize_bin_cpu.cpp', 
                    'src/bin_assignment_cpu.cpp', 
                    'src/merge_bin_cpu.cpp', 
                    'src/greedy_legalize_cpu.cpp' 
                    ], 
                include_dirs=include_dirs, 
                library_dirs=lib_dirs,
                libraries=libs,
                extra_compile_args={
                    #'cxx': ['-g', '-O0'], 
                    'cxx': ['-O2', torch_major_version, torch_minor_version], 
                    }
                ),
            ],
        cmdclass={
            'build_ext': BuildExtension
            })
