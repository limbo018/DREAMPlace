import torch
from torch.autograd import Function
from torch import nn
import numpy as np
import pdb

import dreamplace.ops.pin_weight_sum.pws_cpp as pws_cpp
import dreamplace.configure as configure
if configure.compile_configurations["CUDA_FOUND"] == "TRUE":
    import dreamplace.ops.pin_weight_sum.pws_cuda as pws_cuda

class PinWeightSumFunction(Function):
    """accumulate pin weights of a node.
    @param net_weights weight of nets
    @param flat_nodepin flat nodepin map, length of #pins 
    @param nodepin_start starting index in nodepin map for each net, length of #nodes+1,
            the last entry is #pins  
    @param pin2net_map pin2net map, second set of options
    @param num_nodes the total number of nodes including fillers
    """
    @staticmethod
    def forward(ctx, net_weights, flat_nodepin, nodepin_start, pin2net_map, num_nodes):
        if net_weights.is_cuda:
            func = pws_cuda.forward
        else:
            func = pws_cpp.forward
        output = func(net_weights, flat_nodepin, nodepin_start, pin2net_map, num_nodes)
        return output

class PinWeightSum(nn.Module):
    """ 
    @brief Accumulate pin weights of a node. 
    Support one algorithm: node-by-node (TODO: atomic)
    Different parameters are required for different algorithms. 
    """
    def __init__(self,
                 flat_nodepin=None,
                 nodepin_start=None,
                 pin2net_map=None,
                 num_nodes=None,
                 algorithm='node-by-node'):
        """
        @brief initialization 
        @param flat_nodepin flat nodepin map, length of #pins 
        @param nodepin_start starting index in nodepin map for each net, length of #nodes+1,
                the last entry is #pins  
        @param pin2net_map pin2net map, second set of options 
        @param algorithm must be node-by-node
        """
        super(PinWeightSum, self).__init__()
        if algorithm == 'node-by-node':
            assert flat_nodepin is not None and nodepin_start is not None, \
                "flat_nodepin, nodepin_start are requried parameters for algorithm node-by-node"
        self.flat_nodepin = flat_nodepin
        self.nodepin_start = nodepin_start
        self.pin2net_map = pin2net_map
        self.num_nodes = num_nodes
        self.algorithm = algorithm

    def forward(self, net_weights):
        if self.algorithm == 'node-by-node':
            return PinWeightSumFunction.apply(
                net_weights, 
                self.flat_nodepin, self.nodepin_start,
                self.pin2net_map, self.num_nodes)
