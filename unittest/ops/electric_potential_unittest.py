##
# @file   electric_potential_unitest.py
# @author Yibo Lin
# @date   Mar 2019
#

import time
import numpy as np
import unittest
import logging

import torch
from torch.autograd import Function, Variable
import os
import sys
import gzip

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
from dreamplace.ops.dct import dct
from dreamplace.ops.dct import discrete_spectral_transform
from dreamplace.ops.electric_potential import electric_potential
sys.path.pop()

if sys.version_info[0] < 3:
    import cPickle as pickle
else:
    import _pickle as pickle
import inspect
import pdb
from scipy import fftpack

import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


class ElectricPotentialOpTest(unittest.TestCase):
    def test_densityOverflowRandom(self):
        dtype = np.float64
        xx = np.array([
            1000, 11148, 11148, 11148, 11148, 11148, 11124, 11148, 11148,
            11137, 11126, 11148, 11130, 11148, 11148, 11148, 11148, 11148,
            11148, 0, 11148, 11148, 11150, 11134, 11148, 11148, 11148, 10550,
            11148, 11148, 11144, 11148, 11148, 11148, 11148, 11140, 11120,
            11154, 11148, 11133, 11148, 11148, 11134, 11125, 11148, 11148,
            11148, 11155, 11127, 11148, 11148, 11148, 11148, 11131, 11148,
            11148, 11148, 11148, 11136, 11148, 11146, 11148, 11135, 11148,
            11125, 11150, 11148, 11139, 11148, 11148, 11130, 11148, 11128,
            11148, 11138, 11148, 11148, 11148, 11130, 11148, 11132, 11148,
            11148, 11090
        ]).astype(dtype)
        yy = np.array([
            1000, 11178, 11178, 11190, 11400, 11178, 11172, 11178, 11178,
            11418, 11418, 11178, 11418, 11178, 11178, 11178, 11178, 11178,
            11178, 11414, 11178, 11178, 11172, 11418, 11406, 11184, 11178,
            10398, 11178, 11178, 11172, 11178, 11178, 11178, 11178, 11418,
            11418, 11172, 11178, 11418, 11178, 11178, 11172, 11418, 11178,
            11178, 11178, 11418, 11418, 11178, 11178, 11178, 11178, 11418,
            11178, 11178, 11394, 11178, 11418, 11178, 11418, 11178, 11418,
            11178, 11418, 11418, 11178, 11172, 11178, 11178, 11418, 11178,
            11418, 11178, 11418, 11412, 11178, 11178, 11172, 11178, 11418,
            11178, 11178, 11414
        ]).astype(dtype)
        node_size_x = np.array([
            6, 3, 3, 3, 3, 3, 5, 3, 3, 1, 1, 3, 1, 3, 3, 3, 3, 3, 3, 16728, 3,
            3, 5, 1, 3, 3, 3, 740, 3, 3, 5, 3, 3, 3, 3, 5, 5, 5, 3, 1, 3, 3, 5,
            1, 3, 3, 3, 5, 1, 3, 3, 3, 3, 1, 3, 3, 3, 3, 5, 3, 5, 3, 1, 3, 5,
            5, 3, 5, 3, 3, 5, 3, 1, 3, 1, 3, 3, 3, 5, 3, 1, 3, 3, 67
        ]).astype(dtype)
        node_size_y = np.array([
            6, 240, 240, 6, 6, 240, 6, 240, 240, 6, 6, 240, 6, 240, 240, 240,
            240, 240, 240, 10, 240, 6, 6, 6, 6, 6, 240, 780, 240, 240, 6, 240,
            240, 240, 240, 6, 6, 6, 240, 6, 240, 240, 6, 6, 240, 240, 240, 6,
            6, 240, 240, 240, 240, 6, 240, 240, 6, 240, 6, 240, 6, 240, 6, 240,
            6, 6, 240, 6, 240, 240, 6, 240, 6, 240, 6, 6, 240, 240, 6, 240, 6,
            240, 240, 10
        ]).astype(dtype)
        #xx = np.array([2.0]).astype(dtype)
        #yy = np.array([1.5]).astype(dtype)
        #node_size_x = np.array([1.0]).astype(dtype)
        #node_size_y = np.array([1.0]).astype(dtype)
        num_nodes = len(xx)
        num_movable_nodes = 1
        num_terminals = len(xx) - num_movable_nodes

        scale_factor = 1.0

        xl = 0.0
        yl = 6.0
        xh = 16728.0
        yh = 11430.0
        target_density = 0.7
        num_bins_x = 1024
        num_bins_y = 1024
        bin_size_x = (xh - xl) / num_bins_x
        bin_size_y = (yh - yl) / num_bins_y
        """
        return bin xl
        """
        def bin_xl(id_x):
            return xl + id_x * bin_size_x

        """
        return bin xh
        """

        def bin_xh(id_x):
            return min(bin_xl(id_x) + bin_size_x, xh)

        """
        return bin yl
        """

        def bin_yl(id_y):
            return yl + id_y * bin_size_y

        """
        return bin yh
        """

        def bin_yh(id_y):
            return min(bin_yl(id_y) + bin_size_y, yh)

        bin_center_x = np.zeros(num_bins_x, dtype=dtype)
        for id_x in range(num_bins_x):
            bin_center_x[id_x] = (bin_xl(id_x) +
                                  bin_xh(id_x)) / 2 * scale_factor

        bin_center_y = np.zeros(num_bins_y, dtype=dtype)
        for id_y in range(num_bins_y):
            bin_center_y[id_y] = (bin_yl(id_y) +
                                  bin_yh(id_y)) / 2 * scale_factor

        print("target_area = ", target_density * bin_size_x * bin_size_y)

        if dtype == np.float64:
            dtype = torch.float64
        elif dtype == np.float32:
            dtype = torch.float32
        movable_size_x = node_size_x[:num_movable_nodes]
        _, sorted_node_map = torch.sort(
            torch.tensor(movable_size_x, requires_grad=False, dtype=dtype))
        sorted_node_map = sorted_node_map.to(torch.int32).contiguous()
        # test cpu
        custom = electric_potential.ElectricPotential(
            torch.tensor(node_size_x, requires_grad=False, dtype=dtype),
            torch.tensor(node_size_y, requires_grad=False, dtype=dtype),
            torch.tensor(bin_center_x, requires_grad=False, dtype=dtype),
            torch.tensor(bin_center_y, requires_grad=False, dtype=dtype),
            target_density=torch.tensor(target_density,
                                        requires_grad=False,
                                        dtype=dtype),
            xl=xl,
            yl=yl,
            xh=xh,
            yh=yh,
            bin_size_x=bin_size_x,
            bin_size_y=bin_size_y,
            num_movable_nodes=num_movable_nodes,
            num_terminals=num_terminals,
            num_filler_nodes=0,
            padding=0,
            sorted_node_map=sorted_node_map,
            movable_macro_mask=None,
            deterministic_flag=True)

        pos = Variable(torch.from_numpy(np.concatenate([xx, yy])),
                       requires_grad=True)
        result = custom.forward(pos)
        print("custom_result = ", result)
        print(result.type())
        result.backward()
        grad = pos.grad.clone()
        print("custom_grad = ", grad)

        # test cuda
        if torch.cuda.device_count():
            custom_cuda = electric_potential.ElectricPotential(
                torch.tensor(node_size_x, requires_grad=False,
                             dtype=dtype).cuda(),
                torch.tensor(node_size_y, requires_grad=False,
                             dtype=dtype).cuda(),
                torch.tensor(bin_center_x, requires_grad=False,
                             dtype=dtype).cuda(),
                torch.tensor(bin_center_y, requires_grad=False,
                             dtype=dtype).cuda(),
                target_density=torch.tensor(target_density,
                                            requires_grad=False,
                                            dtype=dtype).cuda(),
                xl=xl,
                yl=yl,
                xh=xh,
                yh=yh,
                bin_size_x=bin_size_x,
                bin_size_y=bin_size_y,
                num_movable_nodes=num_movable_nodes,
                num_terminals=num_terminals,
                num_filler_nodes=0,
                padding=0,
                sorted_node_map=sorted_node_map.cuda(),
                movable_macro_mask=None,
                deterministic_flag=False)

            pos = Variable(torch.from_numpy(np.concatenate([xx, yy])).cuda(),
                           requires_grad=True)
            #pos.grad.zero_()
            result_cuda = custom_cuda.forward(pos)
            print("custom_result_cuda = ", result_cuda.data.cpu())
            print(result_cuda.type())
            result_cuda.backward()
            grad_cuda = pos.grad.clone()
            print("custom_grad_cuda = ", grad_cuda.data.cpu())

            np.testing.assert_allclose(result.detach().numpy(),
                                       result_cuda.data.cpu().detach().numpy())
            np.testing.assert_allclose(grad.detach().numpy(),
                                       grad_cuda.data.cpu().detach().numpy())


def plot(plot_count, density_map, padding, name):
    """
    density map contour and heat map
    """
    density_map = density_map[padding:density_map.shape[0] - padding,
                              padding:density_map.shape[1] - padding]
    print("max density = %g" % (np.amax(density_map)))
    print("mean density = %g" % (np.mean(density_map)))

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    x = np.arange(density_map.shape[0])
    y = np.arange(density_map.shape[1])

    x, y = np.meshgrid(x, y)
    # looks like x and y should be swapped
    ax.plot_surface(y, x, density_map, alpha=0.8)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('density')

    # plt.tight_layout()
    plt.savefig(name + ".3d.png")
    plt.close()

    # plt.clf()

    #fig, ax = plt.subplots()

    # ax.pcolor(density_map)

    # Loop over data dimensions and create text annotations.
    # for i in range(density_map.shape[0]):
    # for j in range(density_map.shape[1]):
    # text = ax.text(j, i, density_map[i, j],
    # ha="center", va="center", color="w")
    # fig.tight_layout()
    #plt.savefig(name+".2d.%d.png" % (plot_count))
    # plt.close()


def eval_runtime(design):
    # e.g., adaptec1_density.pklz
    with gzip.open(design, "rb") as f:
        node_size_x, node_size_y, bin_center_x, bin_center_y, target_density, xl, yl, xh, yh, bin_size_x, bin_size_y, num_movable_nodes, num_terminals, num_filler_nodes = pickle.load(
            f)

    dtype = torch.float64
    num_threads = 10
    torch.set_num_threads(num_threads)
    print("num_threads = %d" % (torch.get_num_threads()))
    movable_size_x = node_size_x[:num_movable_nodes]
    _, sorted_node_map = torch.sort(
        torch.tensor(movable_size_x, requires_grad=False, dtype=dtype).cuda())
    sorted_node_map = sorted_node_map.to(torch.int32).contiguous()

    pos_var = Variable(torch.empty(len(node_size_x) * 2,
                                   dtype=dtype).uniform_(xl, xh),
                       requires_grad=True)
    custom = electric_potential.ElectricPotential(
        torch.tensor(node_size_x, requires_grad=False, dtype=dtype).cpu(),
        torch.tensor(node_size_y, requires_grad=False, dtype=dtype).cpu(),
        torch.tensor(bin_center_x, requires_grad=False, dtype=dtype).cpu(),
        torch.tensor(bin_center_y, requires_grad=False, dtype=dtype).cpu(),
        target_density=torch.tensor(target_density,
                                    requires_grad=False,
                                    dtype=dtype).cpu(),
        xl=xl,
        yl=yl,
        xh=xh,
        yh=yh,
        bin_size_x=bin_size_x,
        bin_size_y=bin_size_y,
        num_movable_nodes=num_movable_nodes,
        num_terminals=num_terminals,
        num_filler_nodes=num_filler_nodes,
        padding=0,
        sorted_node_map=sorted_node_map.cpu())

    custom_cuda = electric_potential.ElectricPotential(
        torch.tensor(node_size_x, requires_grad=False, dtype=dtype).cuda(),
        torch.tensor(node_size_y, requires_grad=False, dtype=dtype).cuda(),
        torch.tensor(bin_center_x, requires_grad=False, dtype=dtype).cuda(),
        torch.tensor(bin_center_y, requires_grad=False, dtype=dtype).cuda(),
        target_density=torch.tensor(target_density,
                                    requires_grad=False,
                                    dtype=dtype).cuda(),
        xl=xl,
        yl=yl,
        xh=xh,
        yh=yh,
        bin_size_x=bin_size_x,
        bin_size_y=bin_size_y,
        num_movable_nodes=num_movable_nodes,
        num_terminals=num_terminals,
        num_filler_nodes=num_filler_nodes,
        padding=0,
        sorted_node_map=sorted_node_map)

    torch.cuda.synchronize()
    iters = 100
    tbackward = 0
    tt = time.time()
    for i in range(iters):
        result = custom.forward(pos_var)
        ttb = time.time()
        result.backward()
        tbackward += time.time() - ttb
    torch.cuda.synchronize()
    print("custom takes %.3f ms, backward %.3f ms" %
          ((time.time() - tt) / iters * 1000, (tbackward / iters * 1000)))

    pos_var = pos_var.cuda()
    tt = time.time()
    for i in range(iters):
        result = custom_cuda.forward(pos_var)
        result.backward()
    torch.cuda.synchronize()
    print("custom_cuda takes %.3f ms" % ((time.time() - tt) / iters * 1000))


if __name__ == '__main__':
    logging.root.name = 'DREAMPlace'
    logging.basicConfig(level=logging.DEBUG,
                        format='[%(levelname)-7s] %(name)s - %(message)s',
                        stream=sys.stdout)
    if len(sys.argv) < 2:
        unittest.main()
    else:
        design = sys.argv[1]
        eval_runtime(design)
