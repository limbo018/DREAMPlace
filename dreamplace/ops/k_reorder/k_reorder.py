##
# @file   k_reorder.py
# @author Yibo Lin
# @date   Jan 2019
# @brief  detailed placement using local reordering
#

import math 
import torch
from torch import nn
from torch.autograd import Function

import dreamplace.ops.k_reorder.k_reorder_cpp as k_reorder_cpp
try: 
    import dreamplace.ops.k_reorder.k_reorder_cuda as k_reorder_cuda
except:
    pass 

import pdb 

class KReorderFunction(Function):
    """ Detailed placement with k-reorder 
    """
    @staticmethod
    def forward(
          pos,
          node_size_x,
          node_size_y,
          flat_region_boxes, 
          flat_region_boxes_start, 
          node2fence_region_map, 
          flat_net2pin_map, 
          flat_net2pin_start_map, 
          pin2net_map, 
          flat_node2pin_map, 
          flat_node2pin_start_map, 
          pin2node_map, 
          pin_offset_x, 
          pin_offset_y, 
          net_mask, 
          xl, 
          yl, 
          xh, 
          yh, 
          site_width, 
          row_height, 
          num_bins_x, 
          num_bins_y, 
          num_movable_nodes, 
          num_terminal_NIs, 
          num_filler_nodes, 
          K, 
          max_iters, 
          num_threads
          ):
        if pos.is_cuda:
            output = k_reorder_cuda.k_reorder(
                    pos.view(pos.numel()), 
                    node_size_x,
                    node_size_y,
                    flat_region_boxes, 
                    flat_region_boxes_start, 
                    node2fence_region_map, 
                    flat_net2pin_map, 
                    flat_net2pin_start_map, 
                    pin2net_map, 
                    flat_node2pin_map, 
                    flat_node2pin_start_map, 
                    pin2node_map, 
                    pin_offset_x, 
                    pin_offset_y, 
                    net_mask, 
                    xl, 
                    yl, 
                    xh, 
                    yh, 
                    site_width, 
                    row_height, 
                    num_bins_x, 
                    num_bins_y, 
                    num_movable_nodes, 
                    num_terminal_NIs, 
                    num_filler_nodes, 
                    K, 
                    max_iters, 
                    num_threads
                    )
        else:
            output = k_reorder_cpp.k_reorder(
                    pos.view(pos.numel()), 
                    node_size_x,
                    node_size_y,
                    flat_region_boxes, 
                    flat_region_boxes_start, 
                    node2fence_region_map, 
                    flat_net2pin_map, 
                    flat_net2pin_start_map, 
                    pin2net_map, 
                    flat_node2pin_map, 
                    flat_node2pin_start_map, 
                    pin2node_map, 
                    pin_offset_x, 
                    pin_offset_y, 
                    net_mask, 
                    xl, 
                    yl, 
                    xh, 
                    yh, 
                    site_width, 
                    row_height, 
                    num_bins_x, 
                    num_bins_y, 
                    num_movable_nodes, 
                    num_terminal_NIs, 
                    num_filler_nodes, 
                    K, 
                    max_iters, 
                    num_threads
                    )
        return output

class KReorder(object):
    """ Detailed placement with k-reorder
    """
    def __init__(self, 
            node_size_x, node_size_y, 
            flat_region_boxes, flat_region_boxes_start, node2fence_region_map, 
            flat_net2pin_map, flat_net2pin_start_map, pin2net_map, 
            flat_node2pin_map, flat_node2pin_start_map, pin2node_map, 
            pin_offset_x, pin_offset_y, 
            net_mask, 
            xl, yl, xh, yh, 
            site_width, row_height, 
            num_bins_x, num_bins_y, 
            num_movable_nodes, num_terminal_NIs, num_filler_nodes, 
            K, 
            max_iters=10, 
            num_threads=8):
        super(KReorder, self).__init__()
        self.node_size_x = node_size_x
        self.node_size_y = node_size_y
        self.flat_region_boxes = flat_region_boxes 
        self.flat_region_boxes_start = flat_region_boxes_start 
        self.node2fence_region_map = node2fence_region_map
        self.flat_net2pin_map = flat_net2pin_map 
        self.flat_net2pin_start_map = flat_net2pin_start_map 
        self.pin2net_map = pin2net_map 
        self.flat_node2pin_map = flat_node2pin_map 
        self.flat_node2pin_start_map = flat_node2pin_start_map 
        self.pin2node_map = pin2node_map 
        self.pin_offset_x = pin_offset_x 
        self.pin_offset_y = pin_offset_y 
        self.net_mask = net_mask
        self.xl = xl 
        self.yl = yl
        self.xh = xh 
        self.yh = yh 
        self.site_width = site_width 
        self.row_height = row_height 
        self.num_bins_x = num_bins_x 
        self.num_bins_y = num_bins_y
        self.num_movable_nodes = num_movable_nodes
        self.num_terminal_NIs = num_terminal_NIs 
        self.num_filler_nodes = num_filler_nodes
        self.K = K
        self.max_iters = max_iters
        self.num_threads = num_threads
    def __call__(self, pos): 
        return KReorderFunction.forward(
                pos,
                node_size_x=self.node_size_x,
                node_size_y=self.node_size_y,
                flat_region_boxes=self.flat_region_boxes, 
                flat_region_boxes_start=self.flat_region_boxes_start, 
                node2fence_region_map=self.node2fence_region_map, 
                flat_net2pin_map=self.flat_net2pin_map, 
                flat_net2pin_start_map=self.flat_net2pin_start_map, 
                pin2net_map=self.pin2net_map, 
                flat_node2pin_map=self.flat_node2pin_map, 
                flat_node2pin_start_map=self.flat_node2pin_start_map, 
                pin2node_map=self.pin2node_map, 
                pin_offset_x=self.pin_offset_x, 
                pin_offset_y=self.pin_offset_y, 
                net_mask=self.net_mask, 
                xl=self.xl, 
                yl=self.yl, 
                xh=self.xh, 
                yh=self.yh, 
                site_width=self.site_width, 
                row_height=self.row_height, 
                num_bins_x=self.num_bins_x, 
                num_bins_y=self.num_bins_y,
                num_movable_nodes=self.num_movable_nodes, 
                num_terminal_NIs=self.num_terminal_NIs, 
                num_filler_nodes=self.num_filler_nodes, 
                K=self.K, 
                max_iters=self.max_iters, 
                num_threads=self.num_threads
                )
