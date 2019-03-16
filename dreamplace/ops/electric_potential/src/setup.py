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
        name='electric_potential',
        ext_modules=[
            CppExtension('electric_potential_cpp', 
                [
                    'electric_density_map.cpp', 
                    'electric_force.cpp'
                    ], 
                include_dirs=include_dirs, 
                library_dirs=lib_dirs,
                libraries=libs,
                extra_compile_args={
                    'cxx' : [torch_major_version, torch_minor_version]
                    }),
            CUDAExtension('electric_potential_cuda', 
                [
                    'electric_density_map_cuda.cpp',
                    'electric_density_map_cuda_kernel.cu',
                    'electric_force_cuda.cpp', 
                    'electric_force_cuda_kernel.cu',
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
