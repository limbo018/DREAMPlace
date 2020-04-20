##
# @file   density_map.py
# @author Yibo Lin
# @date   Jun 2018
# @brief  Compute density map
#

import math
import torch
from torch import nn
from torch.autograd import Function

import dreamplace.ops.density_map.density_map_cpp as density_map_cpp
import dreamplace.configure as configure
if configure.compile_configurations["CUDA_FOUND"] == "TRUE":
    import dreamplace.ops.density_map.density_map_cuda as density_map_cuda

import numpy as np
import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import pdb


class DensityMapFunction(Function):
    """
    @brief compute density map.
    """
    @staticmethod
    def forward(pos, node_size_x, node_size_y, bin_center_x, bin_center_y,
                initial_density_map, xl, yl, xh, yh, bin_size_x, bin_size_y,
                num_movable_nodes, num_filler_nodes):
        if pos.is_cuda:
            func = density_map_cuda.forward
        else:
            func = density_map_cpp.forward
        output = func(pos.view(pos.numel()), node_size_x, node_size_y,
                      bin_center_x, bin_center_y, initial_density_map, xl, yl,
                      xh, yh, bin_size_x, bin_size_y, num_movable_nodes,
                      num_filler_nodes)
        return output


class DensityMap(object):
    """
    @brief Compute density map for both movable and fixed cells.
    The density map for fixed cells is pre-computed. 
    Each call will only compute the density map for movable cells. 
    """
    def __init__(self, node_size_x, node_size_y, bin_center_x, bin_center_y,
                 xl, yl, xh, yh, bin_size_x, bin_size_y, num_movable_nodes,
                 num_terminals, num_filler_nodes):
        """
        @brief initialization 
        @param node_size_x cell width array consisting of movable cells, fixed cells, and filler cells in order  
        @param node_size_y cell height array consisting of movable cells, fixed cells, and filler cells in order   
        @param bin_center_x bin center x locations 
        @param bin_center_y bin center y locations 
        @param xl left boundary 
        @param yl bottom boundary 
        @param xh right boundary 
        @param yh top boundary 
        @param bin_size_x bin width 
        @param bin_size_y bin height 
        @param num_movable_nodes number of movable cells 
        @param num_terminals number of fixed cells 
        @param num_filler_nodes number of filler cells 
        """
        super(DensityMap, self).__init__()
        self.node_size_x = node_size_x
        self.node_size_y = node_size_y
        self.bin_center_x = bin_center_x
        self.bin_center_y = bin_center_y
        self.xl = xl
        self.yl = yl
        self.xh = xh
        self.yh = yh
        self.bin_size_x = bin_size_x
        self.bin_size_y = bin_size_y
        self.num_movable_nodes = num_movable_nodes
        self.num_terminals = num_terminals
        self.num_filler_nodes = num_filler_nodes
        self.initial_density_map = None

    def forward(self, pos):
        """
        @brief API 
        @param pos cell locations. The array consists of x locations of movable cells, fixed cells, and filler cells, then y locations of them 
        """
        if self.initial_density_map is None:
            if pos.is_cuda:
                func = density_map_cuda.fixed_density_map
            else:
                func = density_map_cpp.fixed_density_map
            self.initial_density_map = func(
                pos, self.node_size_x, self.node_size_y, self.bin_center_x,
                self.bin_center_y, self.xl, self.yl, self.xh, self.yh,
                self.bin_size_x, self.bin_size_y, self.num_movable_nodes,
                self.num_terminals)
            #plot(self.initial_density_map.clone().div(self.bin_size_x*self.bin_size_y).cpu().numpy(), 'initial_density_map')

        density_map = DensityMapFunction.forward(
            pos=pos,
            node_size_x=self.node_size_x,
            node_size_y=self.node_size_y,
            bin_center_x=self.bin_center_x,
            bin_center_y=self.bin_center_y,
            initial_density_map=self.initial_density_map,
            xl=self.xl,
            yl=self.yl,
            xh=self.xh,
            yh=self.yh,
            bin_size_x=self.bin_size_x,
            bin_size_y=self.bin_size_y,
            num_movable_nodes=self.num_movable_nodes,
            num_filler_nodes=self.num_filler_nodes)

        return density_map


def plot(density_map, name):
    """
    @brief density map contour and heat map 
    """
    print(np.amax(density_map))
    print(np.mean(density_map))
    fig = plt.figure(figsize=(4, 3))
    ax = fig.gca(projection='3d')

    x = np.arange(density_map.shape[0])
    y = np.arange(density_map.shape[1])

    x, y = np.meshgrid(x, y)
    ax.plot_surface(x, y, density_map, alpha=0.8)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('density')

    #plt.tight_layout()
    plt.savefig(name + ".3d.png")

    plt.clf()

    fig, ax = plt.subplots()

    ax.pcolor(density_map)

    # Loop over data dimensions and create text annotations.
    #for i in range(density_map.shape[0]):
    #    for j in range(density_map.shape[1]):
    #        text = ax.text(j, i, density_map[i, j],
    #                ha="center", va="center", color="w")
    fig.tight_layout()
    plt.savefig(name + ".2d.png")
