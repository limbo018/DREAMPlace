##
# @file   setup.py
# @author Yibo Lin
# @date   Jun 2018
#

import os 
from setuptools import setup
import torch 
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

utility_dir = os.environ['UTILITY_DIR']
ops_dir = os.environ['OPS_DIR']

include_dirs = [os.path.abspath(ops_dir)]
lib_dirs = [utility_dir]
libs = ['utility'] 

tokens = str(torch.__version__).split('.')
torch_major_version = "-DTORCH_MAJOR_VERSION=%d" % (int(tokens[0]))
torch_minor_version = "-DTORCH_MINOR_VERSION=%d" % (int(tokens[1]))

setup(
        name='move_boundary',
        ext_modules=[
            CppExtension('move_boundary_cpp', 
                [
                    'src/move_boundary.cpp'
                    ], 
                include_dirs=include_dirs, 
                library_dirs=lib_dirs,
                libraries=libs,
                extra_compile_args={
                    'cxx' : [torch_major_version, torch_minor_version]
                    }),
            CUDAExtension('move_boundary_cuda', 
                [
                    'src/move_boundary_cuda.cpp',
                    'src/move_boundary_cuda_kernel.cu'
                    ], 
                include_dirs=include_dirs, 
                library_dirs=lib_dirs,
                libraries=['cusparse', 'culibos'] + libs, 
                extra_compile_args={
                    'cxx': ['-O2', torch_major_version, torch_minor_version], 
                    'nvcc': []
                    }
                ),
            ],
        cmdclass={
            'build_ext': BuildExtension
            })
