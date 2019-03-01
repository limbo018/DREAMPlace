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
        name='hpwl',
        ext_modules=[
            CppExtension('hpwl_cpp', 
                [
                    'hpwl.cpp'
                    ]),
            CUDAExtension('hpwl_cuda', 
                [
                    'hpwl_cuda.cpp',
                    'hpwl_cuda_kernel.cu'
                    ]),
            CUDAExtension('hpwl_cuda_atomic', 
                [
                    'hpwl_cuda_atomic.cpp',
                    'hpwl_cuda_atomic_kernel.cu'
                    ],
                extra_compile_args={
                    'cxx': ['-O2'], 
                    'nvcc': [cuda_flags]
                    }),
            ],
        cmdclass={
            'build_ext': BuildExtension
            })
