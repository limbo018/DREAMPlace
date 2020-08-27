##
# @file   move_boundary_unitest.py
# @author Yibo Lin
# @date   Mar 2019
#

import os 
import sys
import numpy as np
import unittest

import torch
from torch.autograd import Function, Variable

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dreamplace.ops.move_boundary import move_boundary
sys.path.pop()

class MoveBoundaryOpTest(unittest.TestCase):
    def test_densityOverflowRandom(self):
        dtype = np.float32
        xx = np.array([1.0, 4.6]).astype(dtype)
        yy = np.array([0.5, 4.1]).astype(dtype)
        node_size_x = np.array([0.5, 1.0]).astype(dtype)
        node_size_y = np.array([1.0, 1.0]).astype(dtype)
        #xx = np.array([2.0]).astype(dtype)
        #yy = np.array([1.5]).astype(dtype)
        #node_size_x = np.array([1.0]).astype(dtype)
        #node_size_y = np.array([1.0]).astype(dtype)
        num_nodes = len(xx)
        scale_factor = 1.0
        
        xl = 1.0 
        yl = 1.0 
        xh = 5.0
        yh = 5.0
        num_movable_nodes = len(xx)
        num_terminals = 0 
        num_filler_nodes = 0

        # test cpu 
        custom = move_boundary.MoveBoundary(
                    torch.from_numpy(node_size_x), torch.from_numpy(node_size_y), 
                    xl=xl, yl=yl, xh=xh, yh=yh, 
                    num_movable_nodes=num_movable_nodes, 
                    num_filler_nodes=num_filler_nodes)

        pos = Variable(torch.from_numpy(np.concatenate([xx, yy])))
        result = custom.forward(pos)
        print("custom_result = ", result)

        #result.retain_grad()
        #result.sum().backward()
        #print("custom_result.grad = ", result.grad)

        # test cuda 
        if torch.cuda.device_count(): 
            custom_cuda = move_boundary.MoveBoundary(
                        torch.from_numpy(node_size_x).cuda(), torch.from_numpy(node_size_y).cuda(), 
                        xl=xl, yl=yl, xh=xh, yh=yh, 
                        num_movable_nodes=num_movable_nodes, 
                        num_filler_nodes=num_filler_nodes)

            pos = Variable(torch.from_numpy(np.concatenate([xx, yy]))).cuda()
            result_cuda = custom_cuda.forward(pos)
            print("custom_result = ", result_cuda.data.cpu())


            np.testing.assert_allclose(result, result_cuda.data.cpu())

if __name__ == '__main__':
    unittest.main()
