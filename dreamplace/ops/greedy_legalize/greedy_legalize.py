##
# @file   greedy_legalize.py
# @author Yibo Lin
# @date   Jun 2018
#

import math 
import torch
from torch import nn
from torch.autograd import Function

import dreamplace.ops.greedy_legalize.greedy_legalize_cpp as greedy_legalize_cpp

class GreedyLegalizeFunction(Function):
    """ Legalize cells with greedy approach 
    """
    @staticmethod
    def forward(
          pos,
          node_size_x,
          node_size_y,
          xl, 
          yl, 
          xh, 
          yh, 
          site_width, 
          row_height, 
          num_bins_x, 
          num_bins_y, 
          num_movable_nodes, 
          num_filler_nodes
          ):
        if pos.is_cuda:
            output = greedy_legalize_cpp.forward(
                    pos.view(pos.numel()).cpu(), 
                    node_size_x.cpu(),
                    node_size_y.cpu(),
                    xl, 
                    yl, 
                    xh, 
                    yh, 
                    site_width, 
                    row_height, 
                    num_bins_x, 
                    num_bins_y, 
                    num_movable_nodes, 
                    num_filler_nodes
                    )
        else:
            output = greedy_legalize_cpp.forward(
                    pos.view(pos.numel()), 
                    node_size_x,
                    node_size_y,
                    xl, 
                    yl, 
                    xh, 
                    yh, 
                    site_width, 
                    row_height, 
                    num_bins_x, 
                    num_bins_y, 
                    num_movable_nodes, 
                    num_filler_nodes
                    )
        return output

class GreedyLegalize(Function):
    """ Legalize cells with greedy approach 
    """
    def __init__(self, node_size_x, node_size_y, xl, yl, xh, yh, site_width, row_height, num_bins_x, num_bins_y, num_movable_nodes, num_filler_nodes):
        super(GreedyLegalize, self).__init__()
        self.node_size_x = node_size_x
        self.node_size_y = node_size_y
        self.xl = xl 
        self.yl = yl
        self.xh = xh 
        self.yh = yh 
        self.site_width = site_width 
        self.row_height = row_height 
        self.num_bins_x = num_bins_x 
        self.num_bins_y = num_bins_y
        self.num_movable_nodes = num_movable_nodes
        self.num_filler_nodes = num_filler_nodes
    def forward(self, pos): 
        return GreedyLegalizeFunction.forward(
                pos,
                node_size_x=self.node_size_x,
                node_size_y=self.node_size_y,
                xl=self.xl, 
                yl=self.yl, 
                xh=self.xh, 
                yh=self.yh, 
                site_width=self.site_width, 
                row_height=self.row_height, 
                num_bins_x=self.num_bins_x, 
                num_bins_y=self.num_bins_y,
                num_movable_nodes=self.num_movable_nodes, 
                num_filler_nodes=self.num_filler_nodes, 
                )
