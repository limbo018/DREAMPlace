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

cuda_flags = os.environ['CUDAFLAGS']
print("cuda_flags = %s" % (cuda_flags))

include_dirs = [os.path.abspath(ops_dir)]
lib_dirs = [utility_dir]
libs = ['utility'] 

tokens = str(torch.__version__).split('.')
torch_major_version = "-DTORCH_MAJOR_VERSION=%d" % (int(tokens[0]))
torch_minor_version = "-DTORCH_MINOR_VERSION=%d" % (int(tokens[1]))

setup(
        name='density_overflow',
        ext_modules=[
            CppExtension('density_overflow_cpp', 
                [
                    'density_overflow.cpp'
                    ], 
                include_dirs=include_dirs, 
                library_dirs=lib_dirs,
                libraries=libs,
                extra_compile_args={
                    'cxx' : [torch_major_version, torch_minor_version]
                    }
                ),
            CUDAExtension('density_overflow_cuda_thread_map', 
                [
                    'density_overflow_cuda_thread_map.cpp',
                    'density_overflow_cuda_kernel.cu', 
                    'density_overflow_cuda_thread_map_kernel.cu'
                    ], 
                include_dirs=include_dirs, 
                library_dirs=lib_dirs,
                libraries=['cusparse', 'culibos'] + libs,
                extra_compile_args={
                    'cxx': ['-O2', torch_major_version, torch_minor_version], 
                    'nvcc': [cuda_flags]
                    }
                ),
            CUDAExtension('density_overflow_cuda_by_node', 
                [
                    'density_overflow_cuda_by_node.cpp',
                    'density_overflow_cuda_by_node_kernel.cu'
                    ], 
                include_dirs=include_dirs, 
                library_dirs=lib_dirs,
                libraries=['cusparse', 'culibos'] + libs,
                extra_compile_args={
                    'cxx': ['-O2', torch_major_version, torch_minor_version], 
                    'nvcc': [cuda_flags]
                    }
                ),
            ],
        cmdclass={
            'build_ext': BuildExtension
            })
