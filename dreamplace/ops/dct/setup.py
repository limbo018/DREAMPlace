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
        name='dct',
        ext_modules=[
            CppExtension('dct_cpp', 
                [
                    'src/dct.cpp',
                    'src/dst.cpp',
                    'src/dxt.cpp', 
                    'src/dct_2N.cpp'
                    ], 
                include_dirs=include_dirs, 
                library_dirs=lib_dirs,
                libraries=libs,
                extra_compile_args={
                    'cxx' : [torch_major_version, torch_minor_version]
                    }),
            CUDAExtension('dct_cuda', 
                [
                    'src/dct_cuda.cpp',
                    'src/dct_cuda_kernel.cu',
                    'src/dst_cuda.cpp',
                    'src/dst_cuda_kernel.cu',
                    'src/dxt_cuda.cpp', 
                    'src/dct_2N_cuda.cpp'
                    ], 
                include_dirs=include_dirs, 
                library_dirs=lib_dirs,
                libraries=libs,
                extra_compile_args={
                    'cxx' : [torch_major_version, torch_minor_version], 
                    'nvcc': []
                    }),
            CppExtension('dct_lee_cpp', 
                [
                    'src/dct_lee.cpp'
                    ], 
                include_dirs=include_dirs, 
                library_dirs=lib_dirs,
                libraries=libs,
                extra_compile_args={
                    'cxx' : [torch_major_version, torch_minor_version]
                    }),
            CUDAExtension('dct_lee_cuda', 
                [
                    'src/dct_lee_cuda.cpp',
                    'src/dct_lee_cuda_kernel.cu', 
                    'src/dct_cuda_kernel.cu',
                    'src/dst_cuda_kernel.cu'
                    ], 
                include_dirs=include_dirs, 
                library_dirs=lib_dirs,
                libraries=libs,
                extra_compile_args={
                    'cxx' : [torch_major_version, torch_minor_version], 
                    'nvcc': []
                    }),
            ],
        cmdclass={
            'build_ext': BuildExtension
            })
