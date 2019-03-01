##
# @file   setup.py
# @author Yibo Lin
# @date   Jun 2018
#

import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

def add_prefix(filename):
    return os.path.join('${CMAKE_CURRENT_SOURCE_DIR}/src', filename)

modules = []

modules.extend([
    CppExtension('rmst_wl_cpp', 
        [
            add_prefix('rmst_wl.cpp')
            ], 
        include_dirs=['${FLUTE_INCLUDE_DIRS}'], 
        library_dirs=['${FLUTE_LINK_DIRS}'], 
        libraries=['flute']
        ),
    ])

setup(
        name='rmst_wl',
        ext_modules=modules,
        cmdclass={
            'build_ext': BuildExtension
            })
