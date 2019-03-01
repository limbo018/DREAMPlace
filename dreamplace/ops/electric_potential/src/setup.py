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
        name='electric_potential',
        ext_modules=[
            CppExtension('electric_potential_cpp', 
                [
                    'electric_density_map.cpp', 
                    'electric_force.cpp'
                    ]),
            CUDAExtension('electric_potential_cuda', 
                [
                    'electric_density_map_cuda.cpp',
                    'electric_density_map_cuda_kernel.cu',
                    'electric_force_cuda.cpp', 
                    'electric_force_cuda_kernel.cu',
                    ], 
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
