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

from dreamplace.ops.density_map.density_map import DensityMap as DensityMap

import pdb


class DensityOverflow(DensityMap):
    """
    @brief Compute density overflow for both movable and fixed cells.
    The density map for fixed cells is pre-computed. 
    Each call will only compute the density map for movable cells. 
    """
    def __init__(self, node_size_x, node_size_y, bin_center_x, bin_center_y,
                 target_density, xl, yl, xh, yh, bin_size_x, bin_size_y,
                 num_movable_nodes, num_terminals, num_filler_nodes):
        """
        @brief initialization 
        @param node_size_x cell width array consisting of movable cells, fixed cells, and filler cells in order  
        @param node_size_y cell height array consisting of movable cells, fixed cells, and filler cells in order   
        @param bin_center_x bin center x locations 
        @param bin_center_y bin center y locations 
        @param target_density target density 
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
        super(DensityOverflow,
              self).__init__(node_size_x=node_size_x,
                             node_size_y=node_size_y,
                             bin_center_x=bin_center_x,
                             bin_center_y=bin_center_y,
                             xl=xl,
                             yl=yl,
                             xh=xh,
                             yh=yh,
                             bin_size_x=bin_size_x,
                             bin_size_y=bin_size_y,
                             num_movable_nodes=num_movable_nodes,
                             num_terminals=num_terminals,
                             num_filler_nodes=num_filler_nodes)
        self.target_density = target_density

    def forward(self, pos):
        """
        @brief API 
        @param pos cell locations. The array consists of x locations of movable cells, fixed cells, and filler cells, then y locations of them 
        """
        density_map = super(DensityOverflow, self).forward(pos)

        bin_area = self.bin_size_x * self.bin_size_y
        density_cost = (density_map -
                        self.target_density * bin_area).clamp_(min=0.0).sum()

        return density_cost, density_map.max() / bin_area
