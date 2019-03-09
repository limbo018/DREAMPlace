##
# @file   setup.py
# @author Yibo Lin
# @date   Jan 2019
#

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

import os 
import sys
import pkgconfig

boost_dir = os.environ['BOOST_DIR']
limbo_dir = os.environ['LIMBO_DIR']
utility_dir = os.environ['UTILITY_DIR']
ops_dir = os.environ['OPS_DIR']

include_dirs = [os.path.join(os.path.abspath(boost_dir), 'include'), os.path.join(os.path.abspath(limbo_dir), 'include'), os.path.abspath(ops_dir)]
lib_dirs = [os.path.join(os.path.abspath(boost_dir), 'lib'), os.path.join(os.path.abspath(limbo_dir), 'lib'), utility_dir]
libs = ['gdsparser', 'boost_iostreams', 'z', 'utility'] 

if pkgconfig.exists('cairo'):
    print("found cairo and enable")
    include_dirs.append(pkgconfig.cflags('cairo')[2:])
    libs.append(pkgconfig.libs('cairo')[2:])
    cairo_compile_args = '-DDRAWPLACE=1'
else:
    print("not found cairo and disable")
    cairo_compile_args = '-DDRAWPLACE=0'

setup(
        name='draw_place',
        ext_modules=[
            CppExtension('draw_place_cpp', 
                [
                    'draw_place.cpp', 
                    ], 
                include_dirs=include_dirs, 
                library_dirs=lib_dirs,
                libraries=libs,
                extra_compile_args={
                    'cxx': ['-fvisibility=hidden', '-D_GLIBCXX_USE_CXX11_ABI=0', cairo_compile_args], 
                    }
                ),
            ],
        cmdclass={
            'build_ext': BuildExtension
            })
