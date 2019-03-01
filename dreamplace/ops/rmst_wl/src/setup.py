##
# @file   setup.py
# @author Yibo Lin
# @date   Jun 2018
#

import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

flute_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))), "lib/flute")

setup(
        name='rmst_wl',
        ext_modules=[
            CppExtension('rmst_wl_cpp', 
                [
                    'rmst_wl.cpp'
                    ], 
                include_dirs=[os.path.abspath("%s/include" % (flute_dir))], 
                library_dirs=[os.path.abspath("%s/lib" % (flute_dir))], 
                libraries=['flute']
                ),
            ],
        cmdclass={
            'build_ext': BuildExtension
            })
