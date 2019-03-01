##
# @file   move_boundary.py
# @author Yibo Lin
# @date   Jun 2018
#

import math 
import torch
from torch import nn
from torch.autograd import Function

import move_boundary_cpp
import move_boundary_cuda

class MoveBoundaryFunction(Function):
    """ Bound cells into layout boundary, perform in-place update 
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
          num_movable_nodes, 
          num_filler_nodes
          ):
        if pos.is_cuda:
            output = move_boundary_cuda.forward(
                    pos.view(pos.numel()), 
                    node_size_x,
                    node_size_y,
                    xl, 
                    yl, 
                    xh, 
                    yh, 
                    num_movable_nodes, 
                    num_filler_nodes
                    )
        else:
            output = move_boundary_cpp.forward(
                    pos.view(pos.numel()), 
                    node_size_x,
                    node_size_y,
                    xl, 
                    yl, 
                    xh, 
                    yh, 
                    num_movable_nodes, 
                    num_filler_nodes
                    )
        return output

class MoveBoundary(Function):
    """ Bound cells into layout boundary, perform in-place update 
    """
    def __init__(self, node_size_x, node_size_y, xl, yl, xh, yh, num_movable_nodes, num_filler_nodes):
        super(MoveBoundary, self).__init__()
        self.node_size_x = node_size_x
        self.node_size_y = node_size_y
        self.xl = xl 
        self.yl = yl
        self.xh = xh 
        self.yh = yh 
        self.num_movable_nodes = num_movable_nodes
        self.num_filler_nodes = num_filler_nodes
    def forward(self, pos): 
        return MoveBoundaryFunction.forward(
                pos,
                node_size_x=self.node_size_x,
                node_size_y=self.node_size_y,
                xl=self.xl, 
                yl=self.yl, 
                xh=self.xh, 
                yh=self.yh, 
                num_movable_nodes=self.num_movable_nodes, 
                num_filler_nodes=self.num_filler_nodes, 
                )
