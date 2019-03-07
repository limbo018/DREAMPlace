##
# @file   setup.py
# @author Yibo Lin
# @date   Jun 2018
#

import os 
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

ops_dir = os.environ['OPS_DIR']

include_dirs = [os.path.abspath(ops_dir)]

setup(
        name='move_boundary',
        ext_modules=[
            CppExtension('move_boundary_cpp', 
                [
                    'move_boundary.cpp'
                    ]),
            CUDAExtension('move_boundary_cuda', 
                [
                    'move_boundary_cuda.cpp',
                    'move_boundary_cuda_kernel.cu'
                    ], 
                include_dirs=include_dirs, 
                libraries=['cusparse', 'culibos']
                ),
            ],
        cmdclass={
            'build_ext': BuildExtension
            })
