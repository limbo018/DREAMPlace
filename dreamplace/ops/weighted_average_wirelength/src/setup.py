##
# @file   setup.py
# @author Yibo Lin
# @date   Jun 2018
#

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

import os 
import sys

ops_dir = os.environ['OPS_DIR']

cuda_flags = os.environ['CUDAFLAGS']
print("cuda_flags = %s" % (cuda_flags))

include_dirs = [os.path.abspath(ops_dir)]

setup(
        name='weighted_average_wirelength',
        ext_modules=[
            CppExtension('weighted_average_wirelength_cpp', 
                [
                    'weighted_average_wirelength.cpp'
                    ]),
            CUDAExtension('weighted_average_wirelength_cuda', 
                [
                    'weighted_average_wirelength_cuda.cpp',
                    'weighted_average_wirelength_cuda_kernel.cu'
                    ], 
                include_dirs=include_dirs, 
                ),
            CUDAExtension('weighted_average_wirelength_cuda_atomic', 
                [
                    'weighted_average_wirelength_cuda_atomic.cpp',
                    'weighted_average_wirelength_cuda_atomic_kernel.cu'
                    ],
                include_dirs=include_dirs, 
                extra_compile_args={
                    'cxx': ['-O2'], 
                    'nvcc': [cuda_flags]
                    }),
            CUDAExtension('weighted_average_wirelength_cuda_sparse', 
                [
                    'weighted_average_wirelength_cuda_sparse.cpp',
                    'weighted_average_wirelength_cuda_sparse_kernel.cu'
                    ],
                include_dirs=include_dirs, 
                extra_compile_args={
                    'cxx': ['-O2'], 
                    'nvcc': [cuda_flags]
                    }),
            ],
        cmdclass={
            'build_ext': BuildExtension
            })
