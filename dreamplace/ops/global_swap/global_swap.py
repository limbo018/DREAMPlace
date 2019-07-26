##
# @file   global_swap.py
# @author Yibo Lin
# @date   Jan 2019
# @brief  detailed placement using global swap 
#

import math 
import torch
from torch import nn
from torch.autograd import Function

import dreamplace.ops.global_swap.global_swap_cpp as global_swap_cpp
import dreamplace.ops.global_swap.global_swap_concurrent_cpp as global_swap_concurrent_cpp
try:
    import dreamplace.ops.global_swap.global_swap_cuda as global_swap_cuda
except:
    pass 

import pdb 

class GlobalSwapFunction(Function):
    """ Detailed placement with global swap 
    """
    @staticmethod
    def forward(
          pos,
          node_size_x,
          node_size_y,
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
          num_filler_nodes, 
          batch_size, 
          max_iters, 
          algorithm, 
          num_threads
          ):
        if pos.is_cuda:
            output = global_swap_cuda.global_swap(
                    pos.view(pos.numel()), 
                    node_size_x,
                    node_size_y,
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
                    num_filler_nodes, 
                    batch_size, 
                    max_iters
                    )
        else:
            if algorithm == 'concurrent': 
                output = global_swap_concurrent_cpp.global_swap(
                        pos.view(pos.numel()), 
                        node_size_x,
                        node_size_y,
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
                        num_filler_nodes, 
                        batch_size, 
                        max_iters, 
                        num_threads
                        )
            else:
                output = global_swap_cpp.global_swap(
                        pos.view(pos.numel()), 
                        node_size_x,
                        node_size_y,
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
                        num_filler_nodes, 
                        max_iters
                        )
        return output

class GlobalSwap(Function):
    """ Detailed placement with global swap
    """
    def __init__(self, 
            node_size_x, node_size_y, 
            flat_net2pin_map, flat_net2pin_start_map, pin2net_map, 
            flat_node2pin_map, flat_node2pin_start_map, pin2node_map, 
            pin_offset_x, pin_offset_y, 
            net_mask, 
            xl, yl, xh, yh, 
            site_width, row_height, 
            num_bins_x, num_bins_y, 
            num_movable_nodes, num_filler_nodes, 
            batch_size=32, 
            max_iters=10, 
            algorithm='concurrent', 
            num_threads=8):
        super(GlobalSwap, self).__init__()
        self.node_size_x = node_size_x
        self.node_size_y = node_size_y
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
        self.num_filler_nodes = num_filler_nodes
        self.batch_size = batch_size
        self.max_iters = max_iters
        self.algorithm = algorithm 
        self.num_threads = num_threads 
    def forward(self, pos): 
        return GlobalSwapFunction.forward(
                pos,
                node_size_x=self.node_size_x,
                node_size_y=self.node_size_y,
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
                num_filler_nodes=self.num_filler_nodes, 
                batch_size=self.batch_size, 
                max_iters=self.max_iters, 
                algorithm=self.algorithm, 
                num_threads=self.num_threads
                )
