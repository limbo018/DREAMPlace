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
flute_dir = os.environ['FLUTE_DIR']

tokens = str(torch.__version__).split('.')
torch_major_version = "-DTORCH_MAJOR_VERSION=%d" % (int(tokens[0]))
torch_minor_version = "-DTORCH_MINOR_VERSION=%d" % (int(tokens[1]))

setup(
        name='rmst_wl',
        ext_modules=[
            CppExtension('rmst_wl_cpp', 
                [
                    'rmst_wl.cpp'
                    ], 
                include_dirs=[os.path.abspath("%s/include" % (flute_dir)), ops_dir], 
                library_dirs=[os.path.abspath("%s/lib" % (flute_dir)), utility_dir], 
                libraries=['flute', 'utility'], 
                extra_compile_args={
                    'cxx' : [torch_major_version, torch_minor_version]
                    }
                ),
            ],
        cmdclass={
            'build_ext': BuildExtension
            })
