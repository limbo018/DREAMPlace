##
# @file   legality_check.py
# @author Yibo Lin
# @date   Jan 2020
#

import math 
import torch
from torch import nn
from torch.autograd import Function

import dreamplace.ops.legality_check.legality_check_cpp as legality_check_cpp

class LegalityCheckFunction(Function):
    """ Check legality including, 
    1. out of boundary 
    2. row and site alignment 
    3. overlap 
    4. fence region 
    """
    @staticmethod
    def forward(
          pos,
          node_size_x,
          node_size_y,
          flat_region_boxes, 
          flat_region_boxes_start, 
          node2fence_region_map, 
          xl, 
          yl, 
          xh, 
          yh, 
          site_width, 
          row_height, 
          num_terminals, 
          num_movable_nodes
          ):
        if pos.is_cuda:
            output = greedy_legalize_cpp.forward(
                    pos.view(pos.numel()).cpu(), 
                    node_size_x.cpu(),
                    node_size_y.cpu(),
                    flat_region_boxes.cpu(), 
                    flat_region_boxes_start.cpu(), 
                    node2fence_region_map.cpu(), 
                    site_width, 
                    row_height, 
                    xl, 
                    yl, 
                    xh, 
                    yh, 
                    num_terminals, 
                    num_movable_nodes
                    ).cuda()
        else:
            output = greedy_legalize_cpp.forward(
                    pos.view(pos.numel()), 
                    node_size_x,
                    node_size_y,
                    flat_region_boxes, 
                    flat_region_boxes_start, 
                    node2fence_region_map, 
                    xl, 
                    yl, 
                    xh, 
                    yh, 
                    site_width, 
                    row_height, 
                    num_terminals, 
                    num_movable_nodes
                    )
        return output

class LegalityCheck(object):
    """ Legalize cells with greedy approach 
    """
    def __init__(self, node_size_x, node_size_y, 
            flat_region_boxes, flat_region_boxes_start, node2fence_region_map, 
            xl, yl, xh, yh, site_width, row_height, 
            num_terminals, 
            num_movable_nodes
            ):
        super(LegalityCheck, self).__init__()
        self.node_size_x = node_size_x
        self.node_size_y = node_size_y
        self.flat_region_boxes = flat_region_boxes 
        self.flat_region_boxes_start = flat_region_boxes_start 
        self.node2fence_region_map = node2fence_region_map
        self.xl = xl 
        self.yl = yl
        self.xh = xh 
        self.yh = yh 
        self.site_width = site_width 
        self.row_height = row_height 
        self.num_terminals = num_terminals 
        self.num_movable_nodes = num_movable_nodes

    def __call__(self, pos):
        return self.forward(pos)

    def forward(self, pos): 
        """ 
        @param pos current roughly legal position
        """
        return LegalityCheckFunction.forward(
                pos,
                node_size_x=self.node_size_x,
                node_size_y=self.node_size_y,
                flat_region_boxes=self.flat_region_boxes, 
                flat_region_boxes_start=self.flat_region_boxes_start, 
                node2fence_region_map=self.node2fence_region_map, 
                xl=self.xl, 
                yl=self.yl, 
                xh=self.xh, 
                yh=self.yh, 
                site_width=self.site_width, 
                row_height=self.row_height, 
                num_terminals=self.num_terminals, 
                num_movable_nodes=self.num_movable_nodes
                )
