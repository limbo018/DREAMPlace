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

import dreamplace.ops.hpwl.hpwl_cpp as hpwl_cpp
import dreamplace.ops.hpwl.hpwl_cpp_atomic as hpwl_cpp_atomic
try: 
    import dreamplace.ops.hpwl.hpwl_cuda as hpwl_cuda
    import dreamplace.ops.hpwl.hpwl_cuda_atomic as hpwl_cuda_atomic
except:
    pass 

class HPWLFunction(Function):
    """compute half-perimeter wirelength.
    @param pos pin location (x array, y array), not cell location 
    @param flat_netpin flat netpin map, length of #pins 
    @param netpin_start starting index in netpin map for each net, length of #nets+1, the last entry is #pins  
    @param net_mask a boolean mask containing whether a net should be computed 
    @param pin2net_map pin2net map, second set of options 
    """
    @staticmethod
    def forward(ctx, pos, flat_netpin, netpin_start, net_mask):
        output = pos.new_empty(1)
        if pos.is_cuda:
            output = hpwl_cuda.forward(pos.view(pos.numel()), flat_netpin, netpin_start, net_mask)
        else:
            output = hpwl_cpp.forward(pos.view(pos.numel()), flat_netpin, netpin_start, net_mask)
        return output 

class HPWLAtomicFunction(Function):
    """compute half-perimeter wirelength using atomic max/min.
    @param pos pin location (x array, y array), not cell location 
    @param pin2net_map pin2net map, second set of options 
    @param net_mask a boolean mask containing whether a net should be computed 
    """
    @staticmethod
    def forward(ctx, pos, pin2net_map, net_mask):
        output = pos.new_empty(1)
        if pos.is_cuda:
            output = hpwl_cuda_atomic.forward(pos.view(pos.numel()), pin2net_map, net_mask)
        else:
            output = hpwl_cpp_atomic.forward(pos.view(pos.numel()), pin2net_map, net_mask)
        return output 

class HPWL(nn.Module):
    """ 
    @brief Compute half-perimeter wirelength. 
    Support two algoriths: net-by-net and atomic. 
    Different parameters are required for different algorithms. 
    """
    def __init__(self, flat_netpin=None, netpin_start=None, pin2net_map=None, net_mask=None, algorithm='atomic'):
        """
        @brief initialization 
        @param flat_netpin flat netpin map, length of #pins 
        @param netpin_start starting index in netpin map for each net, length of #nets+1, the last entry is #pins  
        @param pin2net_map pin2net map 
        @param net_mask whether to compute wirelength, 1 means to compute, 0 means to ignore  
        @param algorithm must be net-by-net | atomic
        """
        super(HPWL, self).__init__()
        assert net_mask is not None, "net_mask is a requried parameter"
        if algorithm == 'net-by-net':
            assert flat_netpin is not None and netpin_start is not None, "flat_netpin, netpin_start are requried parameters for algorithm net-by-net"
        elif algorithm == 'atomic':
            assert pin2net_map is not None, "pin2net_map is required for algorithm atomic"
        self.flat_netpin = flat_netpin 
        self.netpin_start = netpin_start
        self.pin2net_map = pin2net_map 
        self.net_mask = net_mask 
        self.algorithm = algorithm
    def forward(self, pos): 
        if self.algorithm == 'net-by-net': 
            return HPWLFunction.apply(pos, 
                    self.flat_netpin, 
                    self.netpin_start, 
                    self.net_mask
                    )
        elif self.algorithm == 'atomic':
            return HPWLAtomicFunction.apply(pos, 
                    self.pin2net_map, 
                    self.net_mask
                    )
