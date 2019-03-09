##
# @file   __init__.py
# @author Yibo Lin
# @date   Jun 2018
#

import os 
import sys
# this is a bad practice for importing, but I want to make it generic to python2 and python3 
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import dct 
import dct_lee 
import discrete_spectral_transform
sys.path.pop()
