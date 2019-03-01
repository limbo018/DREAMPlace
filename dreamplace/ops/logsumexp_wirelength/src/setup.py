##
# @file   setup.py
# @author Yibo Lin
# @date   Jun 2018
#

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

import os 
import sys

cuda_flags = os.environ['CUDAFLAGS']
print("cuda_flags = %s" % (cuda_flags))

setup(
        name='logsumexp_wirelength',
        ext_modules=[
            CppExtension('logsumexp_wirelength_cpp', 
                [
                    'logsumexp_wirelength.cpp'
                    ]),
            CUDAExtension('logsumexp_wirelength_cuda', 
                [
                    'logsumexp_wirelength_cuda.cpp',
                    'logsumexp_wirelength_cuda_kernel.cu'
                    ]),
            CUDAExtension('logsumexp_wirelength_cuda_atomic', 
                [
                    'logsumexp_wirelength_cuda_atomic.cpp',
                    'logsumexp_wirelength_cuda_atomic_kernel.cu'
                    ],
                extra_compile_args={
                    'cxx': ['-O2'], 
                    'nvcc': [cuda_flags]
                    }),
            ],
        cmdclass={
            'build_ext': BuildExtension
            })
