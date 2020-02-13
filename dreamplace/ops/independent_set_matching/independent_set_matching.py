##
# @file   independent_set_matching.py
# @author Yibo Lin
# @date   Jan 2019
# @brief  detailed placement using independent set matching
#

import math 
import torch
from torch import nn
from torch.autograd import Function

import dreamplace.ops.independent_set_matching.independent_set_matching_cpp as independent_set_matching_cpp
import dreamplace.ops.independent_set_matching.independent_set_matching_sequential_cpp as independent_set_matching_sequential_cpp
import dreamplace.configure as configure
if configure.compile_configurations["CUDA_FOUND"] == "TRUE": 
    import dreamplace.ops.independent_set_matching.independent_set_matching_cuda as independent_set_matching_cuda

import pdb 

class IndependentSetMatchingFunction(Function):
    """ Detailed placement with independent set matching
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
          batch_size, 
          set_size, 
          max_iters, 
          algorithm, 
          num_threads
          ):
        if pos.is_cuda:
            output = independent_set_matching_cuda.independent_set_matching(
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
                    batch_size, 
                    set_size, 
                    max_iters,
                    num_threads
                    )
        elif algorithm == 'sequential':
            output = independent_set_matching_sequential_cpp.independent_set_matching(
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
                    batch_size, 
                    set_size, 
                    max_iters
                    )
        else:
            output = independent_set_matching_cpp.independent_set_matching(
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
                    batch_size, 
                    set_size, 
                    max_iters, 
                    num_threads
                    )
        return output

class IndependentSetMatching(object):
    """ Detailed placement with independent set matching
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
            batch_size, 
            set_size, 
            max_iters, 
            algorithm="concurrent", 
            num_threads=8
            ):
        super(IndependentSetMatching, self).__init__()
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
        self.batch_size = batch_size
        self.set_size = set_size 
        self.max_iters = max_iters
        self.algorithm = algorithm
        self.num_threads = num_threads 
    def __call__(self, pos): 
        return IndependentSetMatchingFunction.forward(
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
                batch_size=self.batch_size, 
                set_size=self.set_size, 
                max_iters=self.max_iters, 
                algorithm=self.algorithm, 
                num_threads=self.num_threads
                )
