##
# @file   draw_place_unitest.py
# @author Yibo Lin
# @date   Mar 2019
#

import os 
import sys 
import time 
import numpy as np
import unittest
import gzip 
import pdb 

import torch
from torch.autograd import Function, Variable
if sys.version_info[0] < 3: 
    import cPickle as pickle
else:
    import _pickle as pickle
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dreamplace.ops.draw_place import draw_place
sys.path.pop()

class DrawPlaceOpTest(unittest.TestCase):
    def test_drawPlaceRandom(self):
        dtype = np.float32
        np.random.seed(100)
        xx = np.array([1.0, 2.0]).astype(dtype)
        yy = np.array([3.0, 1.5]).astype(dtype)
        node_size_x = np.array([0.5, 1.0]).astype(dtype)
        node_size_y = np.array([1.0, 1.0]).astype(dtype)
        pin2node_map = np.array([0, 0, 1, 1]).astype(np.int32)
        pin_offset_x = np.array([0.1, 0.3, 0.2, 0.6]).astype(dtype)
        pin_offset_y = np.array([0.1, 0.8, 0.2, 0.6]).astype(dtype)
        num_nodes = len(xx)
        
        xl = 1.0 
        yl = 1.0 
        xh = 5.0
        yh = 5.0
        bin_size_x = 2.0
        bin_size_y = 2.0
        site_width = 1.0
        row_height = 2.0
        num_bins_x = int(np.ceil((xh-xl)/bin_size_x))
        num_bins_y = int(np.ceil((yh-yl)/bin_size_y))
        num_movable_nodes = len(xx)
        num_terminals = 0 
        num_filler_nodes = 0

        # test cpu 
        custom = draw_place.DrawPlaceFunction.forward(
                    torch.from_numpy(np.concatenate([xx, yy])), 
                    torch.from_numpy(node_size_x), torch.from_numpy(node_size_y), 
                    torch.from_numpy(pin_offset_x), torch.from_numpy(pin_offset_y), 
                    torch.from_numpy(pin2node_map), 
                    xl, yl, xh, yh, 
                    site_width, row_height, 
                    bin_size_x, bin_size_y, 
                    num_movable_nodes, 
                    num_filler_nodes, 
                    "test.gds" # png, jpg, eps, pdf, gds 
                    )
        print(custom)

if __name__ == '__main__':
    unittest.main()
