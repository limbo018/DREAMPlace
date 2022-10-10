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
    @brief compute density map given a range of nodes.
    """
    @staticmethod
    def forward(pos, node_size_x, node_size_y, 
                initial_density_map, xl, yl, xh, yh, 
                num_bins_x, num_bins_y, 
                range_begin, range_end, 
                deterministic_flag):
        if pos.is_cuda:
            func = density_map_cuda.forward
        else:
            func = density_map_cpp.forward
        output = func(pos.view(pos.numel()), node_size_x, node_size_y,
                      initial_density_map, xl, yl, xh, yh, 
                      num_bins_x, num_bins_y, 
                      range_begin, range_end, 
                      deterministic_flag)
        return output


class DensityMap(object):
    """
    @brief Compute density map given a range of cells.
    """
    def __init__(self, node_size_x, node_size_y, 
                 xl, yl, xh, yh, num_bins_x, num_bins_y, 
                 range_list, 
                 deterministic_flag, 
                 initial_density_map=None):
        """
        @brief initialization 
        @param node_size_x cell width array consisting of movable cells, fixed cells, and filler cells in order  
        @param node_size_y cell height array consisting of movable cells, fixed cells, and filler cells in order   
        @param xl left boundary 
        @param yl bottom boundary 
        @param xh right boundary 
        @param yh top boundary 
        @param num_bins_x number of bins in x direction 
        @param num_bins_y number of bins in y direction  
        @param num_movable_nodes number of movable cells 
        @param range_list array of [begin, end) index range 
        @param deterministic_flag whether ensure run-to-run determinism 
        @param initial_density_map initial density map 
        """
        super(DensityMap, self).__init__()
        self.node_size_x = node_size_x
        self.node_size_y = node_size_y
        self.xl = xl
        self.yl = yl
        self.xh = xh
        self.yh = yh
        self.num_bins_x = num_bins_x
        self.num_bins_y = num_bins_y
        self.range_list = range_list
        self.deterministic_flag = deterministic_flag
        self.initial_density_map = initial_density_map

    def forward(self, pos):
        """
        @brief API 
        @param pos cell locations. The array consists of x locations of movable cells, fixed cells, and filler cells, then y locations of them 
        """
        if self.initial_density_map is None:
            self.initial_density_map = torch.zeros(self.num_bins_x, self.num_bins_y, dtype=pos.dtype, device=pos.device)
            #plot(self.initial_density_map.clone().div(self.bin_size_x*self.bin_size_y).cpu().numpy(), 'initial_density_map')

        density_map = self.initial_density_map
        for index_range in self.range_list: 
            if index_range[0] < index_range[1]: 
                density_map = DensityMapFunction.forward(
                    pos=pos,
                    node_size_x=self.node_size_x,
                    node_size_y=self.node_size_y,
                    initial_density_map=density_map,
                    xl=self.xl,
                    yl=self.yl,
                    xh=self.xh,
                    yh=self.yh,
                    num_bins_x=self.num_bins_x,
                    num_bins_y=self.num_bins_y,
                    range_begin=index_range[0],
                    range_end=index_range[1], 
                    deterministic_flag=self.deterministic_flag)

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
