##
# @file   density_overflow_unitest.py
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
from dreamplace.ops.density_overflow import density_overflow
sys.path.pop()

"""
return bin xl
"""
def bin_xl(id_x, xl, bin_size_x):
    return xl+id_x*bin_size_x

"""
return bin xh
"""
def bin_xh(id_x, xl, xh, bin_size_x):
    return min(bin_xl(id_x, xl, bin_size_x)+bin_size_x, xh)

"""
return bin yl
"""
def bin_yl(id_y, yl, bin_size_y):
    return yl+id_y*bin_size_y

"""
return bin yh
"""
def bin_yh(id_y, yl, yh, bin_size_y):
    return min(bin_yl(id_y, yl, bin_size_y)+bin_size_y, yh)

class DensityOverflowOpTest(unittest.TestCase):
    def test_densityOverflowRandom(self):
        dtype = np.float32
        xx = np.array([1.0, 2.0]).astype(dtype)
        yy = np.array([3.0, 1.5]).astype(dtype)
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
        bin_size_x = 2.0
        bin_size_y = 2.0
        target_density = 0.1
        num_bins_x = int(np.ceil((xh-xl)/bin_size_x))
        num_bins_y = int(np.ceil((yh-yl)/bin_size_y))
        num_movable_nodes = len(xx) - 1
        num_terminals = 1 
        num_filler_nodes = 0

        bin_center_x = np.zeros(num_bins_x, dtype=dtype)
        for id_x in range(num_bins_x): 
            bin_center_x[id_x] = (bin_xl(id_x, xl, bin_size_x)+bin_xh(id_x, xl, xh, bin_size_x))/2

        bin_center_y = np.zeros(num_bins_y, dtype=dtype)
        for id_y in range(num_bins_y): 
            bin_center_y[id_y] = (bin_yl(id_y, yl, bin_size_y)+bin_yh(id_y, yl, yh, bin_size_y))/2

        # test cpu 
        custom = density_overflow.DensityOverflow(
                    torch.from_numpy(node_size_x), torch.from_numpy(node_size_y), 
                    torch.from_numpy(bin_center_x), torch.from_numpy(bin_center_y), 
                    target_density=target_density, 
                    xl=xl, yl=yl, xh=xh, yh=yh, 
                    bin_size_x=bin_size_x, bin_size_y=bin_size_y, 
                    num_movable_nodes=num_movable_nodes, 
                    num_terminals=num_terminals, 
                    num_filler_nodes=num_filler_nodes)

        pos = Variable(torch.from_numpy(np.concatenate([xx, yy])))
        result, max_density = custom.forward(pos)
        print("custom_result = ", result)
        print("custom_max_density = ", max_density)

        # test cuda 
        if torch.cuda.device_count(): 
            custom_cuda = density_overflow.DensityOverflow(
                        torch.from_numpy(node_size_x).cuda(), torch.from_numpy(node_size_y).cuda(), 
                        torch.from_numpy(bin_center_x).cuda(), torch.from_numpy(bin_center_y).cuda(), 
                        target_density=target_density, 
                        xl=xl, yl=yl, xh=xh, yh=yh, 
                        bin_size_x=bin_size_x, bin_size_y=bin_size_y, 
                        num_movable_nodes=num_movable_nodes, 
                        num_terminals=num_terminals, 
                        num_filler_nodes=num_filler_nodes, 
                        )

            pos = Variable(torch.from_numpy(np.concatenate([xx, yy]))).cuda()
            result_cuda, max_density_cuda = custom_cuda.forward(pos)
            print("custom_result = ", result_cuda.data.cpu())
            print("custom_max_density_cuda = ", max_density_cuda.data.cpu())

            np.testing.assert_allclose(result, result_cuda.data.cpu())
            np.testing.assert_allclose(max_density, max_density_cuda.data.cpu())

def eval_runtime(design):
    with gzip.open(design, "rb") as f:
        node_size_x, node_size_y, bin_center_x, bin_center_y, target_density, xl, yl, xh, yh, bin_size_x, bin_size_y, num_movable_nodes, num_terminals, num_filler_nodes = pickle.load(f)

    pos_var = Variable(torch.empty(len(node_size_x)*2, dtype=torch.float64).uniform_(xl, xh), requires_grad=True).cuda()
    custom_cuda = density_overflow.DensityOverflow(
                torch.from_numpy(node_size_x).cuda(), torch.from_numpy(node_size_y).cuda(), 
                torch.from_numpy(bin_center_x).cuda(), torch.from_numpy(bin_center_y).cuda(), 
                target_density=target_density, 
                xl=xl, yl=yl, xh=xh, yh=yh, 
                bin_size_x=bin_size_x, bin_size_y=bin_size_y, 
                num_movable_nodes=num_movable_nodes, 
                num_terminals=num_terminals, 
                num_filler_nodes=num_filler_nodes, 
                )

    torch.cuda.synchronize()
    iters = 10 
    tt = time.time()
    for i in range(iters): 
        result = custom_cuda.forward(pos_var)
    torch.cuda.synchronize()
    print("custom_cuda takes %.3f ms" % ((time.time()-tt)/iters*1000))

if __name__ == '__main__':

    if len(sys.argv) < 2: 
        unittest.main()
    else:
        design = sys.argv[1]
        eval_runtime(design)
