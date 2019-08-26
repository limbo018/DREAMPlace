##
# @file   pin_pos.py
# @author Yibo Lin
# @date   Jun 2018
# @brief  Compute density overflow 
#

import math 
import torch
from torch import nn
from torch.autograd import Function

import dreamplace.ops.pin_pos.pin_pos_cpp as pin_pos_cpp

import pdb 

class PinPosFunction(Function):
    """
    @brief Given cell locations, compute pin locations.
    """
    @staticmethod
    def forward(
            ctx, 
            pos,
            pin_offset_x, 
            pin_offset_y, 
            pin2node_map, 
            flat_node2pin_map, 
            flat_node2pin_start_map, 
            num_physical_nodes, 
            num_threads
          ):
        ctx.pos = pos .view(pos.numel())
        if pos.is_cuda:
            assert 0, "CUDA version NOT implemented"
        else:
            output = pin_pos_cpp.forward(
                    ctx.pos, 
                    pin_offset_x, 
                    pin_offset_y, 
                    pin2node_map, 
                    flat_node2pin_map, 
                    flat_node2pin_start_map, 
                    num_threads
                    )
        ctx.pin_offset_x = pin_offset_x
        ctx.pin_offset_y = pin_offset_y
        ctx.pin2node_map = pin2node_map
        ctx.flat_node2pin_map = flat_node2pin_map 
        ctx.flat_node2pin_start_map = flat_node2pin_start_map
        ctx.num_physical_nodes = num_physical_nodes
        ctx.num_threads = num_threads
        return output

    @staticmethod
    def backward(ctx, grad_pin_pos): 
        # grad_pin_pos is not contiguous
        return pin_pos_cpp.backward(
                grad_pin_pos.contiguous(), 
                ctx.pos,  
                ctx.pin_offset_x, 
                ctx.pin_offset_y, 
                ctx.pin2node_map, 
                ctx.flat_node2pin_map, 
                ctx.flat_node2pin_start_map, 
                ctx.num_physical_nodes, 
                ctx.num_threads
                ), None, None, None, None, None, None, None

class PinPos(nn.Module):
    """
    @brief Given cell locations, compute pin locations.
    Different from torch.index_add which computes x[index[i]] += t[i], 
    the forward function compute x[i] += t[index[i]]
    """
    def __init__(self, pin_offset_x, pin_offset_y, pin2node_map, flat_node2pin_map, flat_node2pin_start_map, num_physical_nodes, num_threads=8):
        """
        @brief initialization 
        @param pin_offset pin offset in x or y direction, only computes one direction 
        @param num_threads number of threads 
        """
        super(PinPos, self).__init__()
        self.pin_offset_x = pin_offset_x
        self.pin_offset_y = pin_offset_y
        self.pin2node_map = pin2node_map.long()
        self.flat_node2pin_map = flat_node2pin_map
        self.flat_node2pin_start_map = flat_node2pin_start_map
        self.num_physical_nodes = num_physical_nodes
        self.num_threads = num_threads
    def forward(self, pos): 
        """
        @brief API 
        @param pos cell locations. The array consists of x locations of movable cells, fixed cells, and filler cells, then y locations of them 
        """
        num_nodes = pos.numel()/2;
        if pos.is_cuda: 
            pin_x = self.pin_offset_x.add(torch.index_select(pos[0:self.num_physical_nodes], dim=0, index=self.pin2node_map))
            pin_y = self.pin_offset_y.add(torch.index_select(pos[num_nodes:num_nodes+self.num_physical_nodes], dim=0, index=self.pin2node_map))
            return torch.cat([pin_x, pin_y], dim=0)
        else:
            return PinPosFunction.apply(
                    pos,
                    self.pin_offset_x, 
                    self.pin_offset_y, 
                    self.pin2node_map, 
                    self.flat_node2pin_map, 
                    self.flat_node2pin_start_map, 
                    self.num_physical_nodes, 
                    self.num_threads
                    )
