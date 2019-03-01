##
# @file   hpwl.py
# @author Yibo Lin
# @date   Jun 2018
#

import torch
from torch.autograd import Function
from torch import nn

import hpwl_cpp
import hpwl_cuda
import hpwl_cuda_atomic

class HPWLFunction(Function):
  """compute half-perimeter wirelength.
  @param pos pin location (x array, y array), not cell location 
  @param flat_netpin flat netpin map, length of #pins 
  @param netpin_start starting index in netpin map for each net, length of #nets+1, the last entry is #pins  
  @param ignore_net_degree ignore nets with degree larger than some value 
  @param pin2net_map pin2net map, second set of options 
  @param num_nets #nets, second set of options 
  """
  @staticmethod
  def forward(ctx, pos, flat_netpin, netpin_start, ignore_net_degree, pin2net_map=None, num_nets=None):
      output = pos.new_empty(1)
      if pos.is_cuda:
          if pin2net_map is None: 
              output = hpwl_cuda.forward(pos.view(pos.numel()), flat_netpin, netpin_start, ignore_net_degree)
          else:
              print("use atomic !!!!!!!")
              output = hpwl_cuda_atomic.forward(pos.view(pos.numel()), pin2net_map, num_nets, ignore_net_degree)
      else:
          hpwl_cpp.forward(pos.view(pos.numel()), flat_netpin, netpin_start, ignore_net_degree, output)
      return output 

class HPWL(nn.Module):
    def __init__(self, flat_netpin, netpin_start, ignore_net_degree=None):
        super(HPWL, self).__init__()
        self.flat_netpin = flat_netpin 
        self.netpin_start = netpin_start
        if ignore_net_degree is None: 
            self.ignore_net_degree = self.flat_netpin.numel()
        else:
            self.ignore_net_degree = ignore_net_degree
    def forward(self, pos): 
        return HPWLFunction.apply(pos, 
                self.flat_netpin, 
                self.netpin_start, 
                self.ignore_net_degree
                )

class HPWLAtomicFunction(Function):
  """compute half-perimeter wirelength using atomic max/min.
  @param pos pin location (x array, y array), not cell location 
  @param pin2net_map pin2net map, second set of options 
  @param net_mask a boolean mask containing whether a net should be computed 
  @param ignore_net_degree ignore nets with degree larger than some value 
  """
  @staticmethod
  def forward(ctx, pos, pin2net_map, net_mask):
      output = pos.new_empty(1)
      if pos.is_cuda:
            output = hpwl_cuda_atomic.forward(pos.view(pos.numel()), pin2net_map, net_mask)
      else:
          assert 0, "CPU version NOT IMPLEMENTED"
      return output 

class HPWLAtomic(nn.Module):
    def __init__(self, pin2net_map, net_mask):
        super(HPWLAtomic, self).__init__()
        self.pin2net_map = pin2net_map 
        self.net_mask = net_mask 
    def forward(self, pos): 
        return HPWLAtomicFunction.apply(pos, 
                self.pin2net_map, 
                self.net_mask
                )

