##
# @file   density_potential_unitest.py
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
from dreamplace.ops.density_potential import density_potential
sys.path.pop()
import inspect
import pdb 

class DensityPotentialOpTest(unittest.TestCase):
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
        
        xl = 0.0 
        yl = 0.0 
        xh = 5.0
        yh = 5.0
        bin_size_x = 1.0
        bin_size_y = 1.0
        target_density = 0.1
        num_bins_x = int(np.ceil((xh-xl)/bin_size_x))
        num_bins_y = int(np.ceil((yh-yl)/bin_size_y))
        
        ax = (4 / (node_size_x + 2*bin_size_x) / (node_size_x + 4*bin_size_x)).astype(dtype)
        bx = (2 / bin_size_x / (node_size_x + 4*bin_size_x)).astype(dtype)
        ay = (4 / (node_size_y + 2*bin_size_y) / (node_size_y + 4*bin_size_y)).astype(dtype)
        by = (2 / bin_size_y / (node_size_y + 4*bin_size_y)).astype(dtype)

        #cx = np.zeros(num_nodes)
        #cy = np.zeros(num_nodes)
        #for i in range(num_nodes):
        #    sum_potential = 0.0
        #    count = 0 
        #    for dist in np.arange(-(node_size_x[i]/2+2*bin_size_x)+node_size_x[i]/2, node_size_x[i]/2+2*bin_size_x, bin_size_x):
        #        if np.absolute(dist) < node_size_x[i]/2+bin_size_x:
        #            print("dist1 = %g, add %g" % (dist, 1 - ax[i]*dist*dist))
        #            sum_potential += 1 - ax[i]*dist*dist 
        #        else:
        #            print("dist2 = %g, add %g" % (dist, bx[i]*(np.absolute(dist)-node_size_x[i]/2-2*bin_size_x)*(np.absolute(dist)-node_size_x[i]/2-2*bin_size_x)))
        #            print("dddd = %g" % (np.absolute(dist)-node_size_x[i]/2-2*bin_size_x))
        #            sum_potential += bx[i]*(np.absolute(dist)-node_size_x[i]/2-2*bin_size_x)*(np.absolute(dist)-node_size_x[i]/2-2*bin_size_x)
        #        count += 1
        #        #if count > num_bins_x:
        #        #    break 
        #    print("sum_potential = ", sum_potential)
        #    cx[i] = node_size_x[i]/sum_potential
        #for i in range(num_nodes):
        #    sum_potential = 0.0
        #    count = 0 
        #    for dist in np.arange(-(node_size_y[i]/2+2*bin_size_y)+node_size_y[i]/2, node_size_y[i]/2+2*bin_size_y, bin_size_y):
        #        if np.absolute(dist) < node_size_y[i]/2+bin_size_y:
        #            sum_potential += 1 - ay[i]*dist*dist 
        #        else:
        #            sum_potential += by[i]*(np.absolute(dist)-node_size_y[i]/2-2*bin_size_y)*(np.absolute(dist)-node_size_y[i]/2-2*bin_size_y)
        #        count += 1
        #        #if count > num_bins_x:
        #        #    break 
        #    cy[i] = node_size_y[i]/sum_potential
        #print("cx = ", cx)
        #print("cy = ", cy)

        # bell shape overlap function 
        def npfx1(dist):
            # ax will be broadcast from num_nodes*1 to num_nodes*num_bins_x
            return 1.0-ax.reshape([num_nodes, 1])*np.square(dist)
        def npfx2(dist):
            # bx will be broadcast from num_nodes*1 to num_nodes*num_bins_x
            return bx.reshape([num_nodes, 1])*np.square(dist-node_size_x/2-2*bin_size_x).reshape([num_nodes, 1])
        def npfy1(dist):
            # ay will be broadcast from num_nodes*1 to num_nodes*num_bins_y
            return 1.0-ay.reshape([num_nodes, 1])*np.square(dist)
        def npfy2(dist):
            # by will be broadcast from num_nodes*1 to num_nodes*num_bins_y
            return by.reshape([num_nodes, 1])*np.square(dist-node_size_y/2-2*bin_size_y).reshape([num_nodes, 1])
        # should not use integral, but sum; basically sample 5 distances, -2wb, -wb, 0, wb, 2wb; the sum does not change much when shifting cells 
        integral_potential_x = npfx1(0) + 2*npfx1(bin_size_x) + 2*npfx2(2*bin_size_x)
        print("integral_potential_x = ", integral_potential_x)
        cx = (node_size_x.reshape([num_nodes, 1]) / integral_potential_x).reshape([num_nodes, 1])
        # should not use integral, but sum; basically sample 5 distances, -2wb, -wb, 0, wb, 2wb; the sum does not change much when shifting cells 
        integral_potential_y = npfy1(0) + 2*npfy1(bin_size_y) + 2*npfy2(2*bin_size_y)
        cy = (node_size_y.reshape([num_nodes, 1]) / integral_potential_y).reshape([num_nodes, 1])

        """
        return bin xl
        """
        def bin_xl(id_x):
            return xl+id_x*bin_size_x

        """
        return bin xh
        """
        def bin_xh(id_x):
            return min(bin_xl(id_x)+bin_size_x, xh)

        """
        return bin yl
        """
        def bin_yl(id_y):
            return yl+id_y*bin_size_y

        """
        return bin yh
        """
        def bin_yh(id_y):
            return min(bin_yl(id_y)+bin_size_y, yh)

        bin_center_x = np.zeros(num_bins_x, dtype=dtype)
        for id_x in range(num_bins_x): 
            bin_center_x[id_x] = (bin_xl(id_x)+bin_xh(id_x))/2*scale_factor

        bin_center_y = np.zeros(num_bins_y, dtype=dtype)
        for id_y in range(num_bins_y): 
            bin_center_y[id_y] = (bin_yl(id_y)+bin_yh(id_y))/2*scale_factor

        print("target_area = ", target_density*bin_size_x*bin_size_y)

        sigma = 0.25 
        delta = 2.0

        # test cpu 
        custom = density_potential.DensityPotential(
                    torch.tensor(node_size_x, requires_grad=False), torch.tensor(node_size_y, requires_grad=False), 
                    torch.tensor(ax, requires_grad=False), torch.tensor(bx, requires_grad=False), torch.tensor(cx, requires_grad=False), 
                    torch.tensor(ay, requires_grad=False), torch.tensor(by, requires_grad=False), torch.tensor(cy, requires_grad=False), 
                    torch.tensor(bin_center_x, requires_grad=False), torch.tensor(bin_center_y, requires_grad=False), 
                    target_density=torch.tensor(target_density, requires_grad=False), 
                    xl=torch.tensor(xl, requires_grad=False), yl=torch.tensor(yl, requires_grad=False), xh=torch.tensor(xh, requires_grad=False), yh=torch.tensor(yh, requires_grad=False), 
                    bin_size_x=torch.tensor(bin_size_x, requires_grad=False), bin_size_y=torch.tensor(bin_size_y, requires_grad=False), 
                    num_movable_nodes=torch.tensor(num_nodes, requires_grad=False), 
                    num_terminals=0, 
                    num_filler_nodes=0, 
                    padding=torch.tensor(0, dtype=torch.int32, requires_grad=False), 
                    sigma=sigma, delta=delta)

        pos = Variable(torch.from_numpy(np.concatenate([xx, yy])), requires_grad=True)
        result = custom.forward(pos)
        print("custom_result = ", result)
        result.backward()
        grad = pos.grad.clone()
        print("custom_grad = ", grad)

        # test cuda 
        if torch.cuda.device_count(): 
            custom_cuda = density_potential.DensityPotential(
                        torch.tensor(node_size_x, requires_grad=False).cuda(), torch.tensor(node_size_y, requires_grad=False).cuda(), 
                        torch.tensor(ax, requires_grad=False).cuda(), torch.tensor(bx, requires_grad=False).cuda(), torch.tensor(cx, requires_grad=False).cuda(), 
                        torch.tensor(ay, requires_grad=False).cuda(), torch.tensor(by, requires_grad=False).cuda(), torch.tensor(cy, requires_grad=False).cuda(), 
                        torch.tensor(bin_center_x, requires_grad=False).cuda(), torch.tensor(bin_center_y, requires_grad=False).cuda(), 
                        target_density=torch.tensor(target_density, requires_grad=False).cuda(), 
                        xl=torch.tensor(xl, requires_grad=False).cuda(), yl=torch.tensor(yl, requires_grad=False).cuda(), xh=torch.tensor(xh, requires_grad=False).cuda(), yh=torch.tensor(yh, requires_grad=False).cuda(), 
                        bin_size_x=torch.tensor(bin_size_x, requires_grad=False).cuda(), bin_size_y=torch.tensor(bin_size_y, requires_grad=False).cuda(), 
                        num_movable_nodes=torch.tensor(num_nodes, requires_grad=False).cuda(), 
                        num_terminals=0, 
                        num_filler_nodes=0, 
                        padding=torch.tensor(0, dtype=torch.int32, requires_grad=False).cuda(), 
                        sigma=sigma, delta=delta)

            pos = Variable(torch.from_numpy(np.concatenate([xx, yy])).cuda(), requires_grad=True)
            #pos.grad.zero_()
            result_cuda = custom_cuda.forward(pos)
            print("custom_result_cuda = ", result_cuda.data.cpu())
            result_cuda.backward()
            grad_cuda = pos.grad.clone()
            print("custom_grad_cuda = ", grad_cuda.data.cpu())

            np.testing.assert_allclose(result.detach().numpy(), result_cuda.data.cpu().detach().numpy())

if __name__ == '__main__':
    unittest.main()
