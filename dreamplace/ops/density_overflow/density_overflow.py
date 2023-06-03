##
# @file   density_overflow.py
# @author Yibo Lin
# @date   Jun 2018
# @brief  Compute density overflow
#

import math
import torch
from torch import nn
from torch.autograd import Function

import dreamplace.ops.density_map.density_map as density_map

import pdb


class DensityOverflow(object):
    """
    @brief Compute density overflow for both movable and fixed cells.
    The density map for fixed cells is pre-computed. 
    Each call will only compute the density map for movable cells. 
    """
    def __init__(self, node_size_x, node_size_y, 
                 xl, yl, xh, yh, 
                 num_bins_x, num_bins_y, 
                 num_movable_nodes, num_terminals, num_filler_nodes,
                 target_density, deterministic_flag):
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
        @param num_terminals number of fixed cells 
        @param num_filler_nodes number of filler cells 
        @param target_density target density 
        @param deterministic_flag whether ensure run-to-run determinism 
        """
        self.node_size_x = node_size_x
        self.node_size_y = node_size_y
        self.xl = xl
        self.yl = yl
        self.xh = xh
        self.yh = yh
        self.num_bins_x = num_bins_x
        self.num_bins_y = num_bins_y
        self.initial_range_list = [[num_movable_nodes, num_movable_nodes+num_terminals]]
        self.range_list = [[0, num_movable_nodes], [node_size_x.numel() - num_filler_nodes, node_size_x.numel()]]
        self.density_map_op = None 
        self.target_density = target_density
        self.deterministic_flag = deterministic_flag

    def forward(self, pos):
        """
        @brief API 
        @param pos cell locations. The array consists of x locations of movable cells, fixed cells, and filler cells, then y locations of them 
        """
        if self.density_map_op is None:
            initial_density_map_op = density_map.DensityMap(
                    node_size_x=self.node_size_x,
                    node_size_y=self.node_size_y,
                    xl=self.xl,
                    yl=self.yl,
                    xh=self.xh,
                    yh=self.yh,
                    num_bins_x=self.num_bins_x,
                    num_bins_y=self.num_bins_y,
                    range_list=self.initial_range_list,
                    deterministic_flag=self.deterministic_flag,
                    initial_density_map=None
                    )
            initial_density_map = initial_density_map_op.forward(pos)
            self.density_map_op = density_map.DensityMap(
                    node_size_x=self.node_size_x,
                    node_size_y=self.node_size_y,
                    xl=self.xl,
                    yl=self.yl,
                    xh=self.xh,
                    yh=self.yh,
                    num_bins_x=self.num_bins_x,
                    num_bins_y=self.num_bins_y,
                    range_list=self.range_list,
                    deterministic_flag=self.deterministic_flag,
                    initial_density_map=initial_density_map
                    )
        total_density_map = self.density_map_op.forward(pos)

        bin_area = (self.xh - self.xl) * (self.yh - self.yl) / (self.num_bins_x * self.num_bins_y)
        density_cost = (total_density_map - self.target_density * bin_area).clamp_(min=0.0).sum()

        return density_cost, total_density_map.max() / bin_area
