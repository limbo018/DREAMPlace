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
        name='density_potential',
        ext_modules=[
            CppExtension('density_potential_cpp', 
                [
                    'density_potential.cpp'
                    ]),
            CUDAExtension('density_potential_cuda', 
                [
                    'density_potential_cuda.cpp',
                    'density_potential_cuda_kernel.cu',
                    'density_overflow_cuda_kernel.cu',
                    ], 
                include_dirs=include_dirs, 
                libraries=['cusparse', 'culibos'],
                extra_compile_args={
                    'cxx': ['-O2'], 
                    'nvcc': [cuda_flags]
                    }
                ),
            ],
        cmdclass={
            'build_ext': BuildExtension
            })
