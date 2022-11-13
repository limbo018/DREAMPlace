##
# @file   hpwl.py
# @author Yibo Lin
# @date   Jun 2018
#

import torch
from torch.autograd import Function
from torch import nn
import numpy as np
import pdb

import dreamplace.ops.dump_boxes.dump_boxes_cpp as dump_boxes_cpp


class DumpBoxes(nn.Module):
    """ 
    @brief Dump box data 
    """
    def __init__(self,
            node_size_x, node_size_y, 
            flat_netpin,
            netpin_start, net_weights, net_mask, 
            xl, yl, xh, yh, 
            num_bins_x, num_bins_y, 
            num_movable_nodes, num_terminals):
        super(DumpBoxes, self).__init__()
        self.node_size_x = node_size_x.cpu()
        self.node_size_y = node_size_y.cpu()
        self.flat_netpin = flat_netpin.cpu()
        self.netpin_start = netpin_start.cpu()
        self.net_weights = net_weights.cpu()
        self.net_mask = net_mask.cpu()
        self.xl = xl 
        self.yl = yl 
        self.xh = xh 
        self.yh = yh 
        self.num_bins_x = num_bins_x
        self.num_bins_y = num_bins_y
        self.num_movable_nodes = num_movable_nodes
        self.num_terminals = num_terminals

    def forward(self, pos, pin_pos):
        dump_boxes_cpp.forward(pos.cpu(), 
                self.node_size_x, self.node_size_y, 
                pin_pos.cpu(), self.flat_netpin,
                self.netpin_start, self.net_weights, self.net_mask, 
                self.xl, self.yl, self.xh, self.yh, 
                self.num_bins_x, self.num_bins_y, 
                self.num_movable_nodes, self.num_terminals) 
