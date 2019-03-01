##
# @file   setup.py
# @author Yibo Lin
# @date   Jun 2018
#

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

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
                libraries=['cusparse', 'culibos']
                ),
            ],
        cmdclass={
            'build_ext': BuildExtension
            })
