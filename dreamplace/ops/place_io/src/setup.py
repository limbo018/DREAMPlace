##
# @file   setup.py
# @author Yibo Lin
# @date   Jun 2018
#

import os
import sys 
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

# I removed boost dependency by removing timers 
boost_dir = os.environ['BOOST_DIR']
limbo_dir = os.environ['LIMBO_DIR']

setup(
        name='place_io',
        ext_modules=[
            CppExtension('place_io_cpp', 
                [
                    'place_io.cpp',  
                    'BenchMetrics.cpp',  
                    'BinMap.cpp',  
                    'Enums.cpp',  
                    'Msg.cpp',  
                    'Net.cpp',  
                    'Node.cpp',  
                    'Params.cpp',  
                    'PlaceDB.cpp',  
                    'DefWriter.cpp',
                    'BookshelfWriter.cpp'
                    ],
                #include_dirs=[os.path.join(os.path.abspath(limbo_dir), 'include'), os.path.join(os.path.abspath(boost_dir), 'include')], 
                #library_dirs=[os.path.join(os.path.abspath(limbo_dir), 'lib'), os.path.join(os.path.abspath(boost_dir), 'lib')],
                include_dirs=[os.path.join(os.path.abspath(limbo_dir), 'include')], 
                library_dirs=[os.path.join(os.path.abspath(limbo_dir), 'lib')],
                libraries=['lefparseradapt', 'defparseradapt', 'verilogparser', 'gdsparser', 'bookshelfparser', 'programoptions', 
                    'boost_system', 'boost_timer', 'boost_chrono', 'boost_iostreams', 'z'],
                extra_compile_args={
                    'cxx': ['-fvisibility=hidden', '-D_GLIBCXX_USE_CXX11_ABI=0'], 
                    }
                ),
            ],
        cmdclass={
            'build_ext': BuildExtension
            })
