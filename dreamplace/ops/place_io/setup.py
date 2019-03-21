##
# @file   setup.py
# @author Yibo Lin
# @date   Jun 2018
#

import os
import sys 
from setuptools import setup
import torch 
from torch.utils.cpp_extension import BuildExtension, CppExtension

# I removed boost dependency by removing timers 
boost_dir = os.environ['BOOST_DIR']
limbo_dir = os.environ['LIMBO_DIR']
utility_dir = os.environ['UTILITY_DIR']
ops_dir = os.environ['OPS_DIR']

include_dirs = [os.path.join(os.path.abspath(boost_dir), 'include'), os.path.join(os.path.abspath(limbo_dir), 'include'), os.path.abspath(ops_dir)]
lib_dirs = [os.path.join(os.path.abspath(boost_dir), 'lib'), os.path.join(os.path.abspath(limbo_dir), 'lib'), utility_dir]
libs = ['lefparseradapt', 'defparseradapt', 'verilogparser', 'gdsparser', 'bookshelfparser', 'programoptions', 
                    'boost_system', 'boost_timer', 'boost_chrono', 'boost_iostreams', 'z', 'utility'] 

tokens = str(torch.__version__).split('.')
torch_major_version = "-DTORCH_MAJOR_VERSION=%d" % (int(tokens[0]))
torch_minor_version = "-DTORCH_MINOR_VERSION=%d" % (int(tokens[1]))

setup(
        name='place_io',
        ext_modules=[
            CppExtension('place_io_cpp', 
                [
                    'src/place_io.cpp',  
                    'src/BenchMetrics.cpp',  
                    'src/BinMap.cpp',  
                    'src/Enums.cpp',  
                    'src/Net.cpp',  
                    'src/Node.cpp',  
                    'src/Params.cpp',  
                    'src/PlaceDB.cpp',  
                    'src/DefWriter.cpp',
                    'src/BookshelfWriter.cpp'
                    ],
                include_dirs=include_dirs, 
                library_dirs=lib_dirs,
                libraries=libs,
                extra_compile_args={
                    'cxx': ['-fvisibility=hidden', '-D_GLIBCXX_USE_CXX11_ABI=0', torch_major_version, torch_minor_version], 
                    }
                ),
            ],
        cmdclass={
            'build_ext': BuildExtension
            })
