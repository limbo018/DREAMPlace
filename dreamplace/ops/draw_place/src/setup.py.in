##
# @file   setup.py
# @author Yibo Lin
# @date   Jan 2019
#

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

import os 
import sys

limbo_dir = "${LIMBO_DIR}"
ops_dir = "${OPS_DIR}"

include_dirs = [os.path.join(os.path.abspath(limbo_dir), 'include'), ops_dir, '${Boost_INCLUDE_DIRS}', '${ZLIB_INCLUDE_DIRS}']
lib_dirs = [os.path.join(os.path.abspath(limbo_dir), 'lib'), '${Boost_LIBRARY_DIRS}', os.path.dirname('${ZLIB_LIBRARIES}'), '${UTILITY_LIBRARY_DIRS}']
libs = ['gdsparser', 'boost_iostreams', 'z', 'utility'] 

if "${CAIRO_FOUND}".upper() == 'TRUE': 
    print("found Cairo and enable")
    include_dirs.append('${CAIRO_INCLUDE_DIRS}')
    lib_dirs.append(os.path.dirname('${CAIRO_LIBRARIES}'))
    libs.append('cairo')
    cairo_compile_args = '-DDRAWPLACE=1'
else:
    print("not found Cairo and disable")
    cairo_compile_args = '-DDRAWPLACE=0'

def add_prefix(filename):
    return os.path.join('${CMAKE_CURRENT_SOURCE_DIR}/src', filename)

setup(
        name='draw_place',
        ext_modules=[
            CppExtension('draw_place_cpp', 
                [
                    add_prefix('draw_place.cpp'), 
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
