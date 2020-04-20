##
# @file   rmst_wl.py
# @author Yibo Lin
# @date   Jun 2018
#

import torch
from torch.autograd import Function
from torch import nn

import dreamplace.ops.rmst_wl.rmst_wl_cpp as rmst_wl_cpp


class RmstWLFunction(Function):
    """compute half-perimeter wirelength.
  @param pos pin location (x array, y array), not cell location 
  @param flat_netpin flat netpin map, length of #pins 
  @param netpin_start starting index in netpin map for each net, length of #nets+1, the last entry is #pins  
  @param ignore_net_degree ignore nets with degree larger than some value 
  """
    @staticmethod
    def forward(ctx, pos, flat_netpin, netpin_start, ignore_net_degree,
                read_lut_flag, POWVFILE, POSTFILE):
        output = pos.new_empty(netpin_start.numel() - 1)
        if pos.is_cuda:
            assert 0, "CUDA version NOT IMPLEMENTED"
            rmst_wl_cuda.forward(pos.view(pos.numel()), flat_netpin,
                                 netpin_start, ignore_net_degree, output)
        else:
            rmst_wl_cpp.forward(pos.view(pos.numel()), flat_netpin,
                                netpin_start, ignore_net_degree, read_lut_flag,
                                POWVFILE, POSTFILE, output)
        return output


class RmstWL(nn.Module):
    def __init__(self,
                 flat_netpin,
                 netpin_start,
                 ignore_net_degree=None,
                 POWVFILE="POWV9.dat",
                 POSTFILE="POST9.dat"):
        super(RmstWL, self).__init__()
        self.flat_netpin = flat_netpin
        self.netpin_start = netpin_start
        if ignore_net_degree is None:
            self.ignore_net_degree = self.flat_netpin.numel()
        else:
            self.ignore_net_degree = ignore_net_degree
        self.POWVFILE = POWVFILE
        self.POSTFILE = POSTFILE

    def forward(self, pos, read_lut_flag):
        return RmstWLFunction.apply(pos, self.flat_netpin, self.netpin_start,
                                    self.ignore_net_degree, read_lut_flag,
                                    self.POWVFILE, self.POSTFILE)
