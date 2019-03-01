##
# @file   setup.py
# @author Yibo Lin
# @date   Jun 2018
#

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

import os 
import sys

cuda_arch = '${CMAKE_CUDA_FLAGS}'
print("cuda_arch = %s" % (cuda_arch))

def add_prefix(filename):
    return os.path.join('${CMAKE_CURRENT_SOURCE_DIR}/src', filename)

modules = []

modules.append(CppExtension('logsumexp_wirelength_cpp', 
                [
                    add_prefix('logsumexp_wirelength.cpp')
                    ])
                )

if not "${CUDA_FOUND}" or "${CUDA_FOUND}".upper() == 'TRUE': 
    modules.append(CUDAExtension('logsumexp_wirelength_cuda', 
                    [
                        add_prefix('logsumexp_wirelength_cuda.cpp'),
                        add_prefix('logsumexp_wirelength_cuda_kernel.cu')
                        ])
    modules.append(CUDAExtension('logsumexp_wirelength_cuda_atomic', 
                    [
                        add_prefix('logsumexp_wirelength_cuda_atomic.cpp'),
                        add_prefix('logsumexp_wirelength_cuda_atomic_kernel.cu')
                        ],
                    extra_compile_args={
                        'cxx': ['-O2'], 
                        'nvcc': [cuda_arch]
                        })
                    )

setup(
        name='logsumexp_wirelength',
        ext_modules=modules,
        cmdclass={
            'build_ext': BuildExtension
            })
