##
# @file   logsumexp_wirelength.py
# @author Yibo Lin
# @date   Jun 2018
#

import torch
from torch import nn
from torch.autograd import Function

import logsumexp_wirelength_cpp
import logsumexp_wirelength_cuda
import logsumexp_wirelength_cuda_atomic
import pdb 

class LogSumExpWirelengthFunction(Function):
    """compute weighted average wirelength.
    @param pos pin location (x array, y array), not cell location 
    @param flat_netpin flat netpin map, length of #pins 
    @param netpin_start starting index in netpin map for each net, length of #nets+1, the last entry is #pins  
    @param gamma the smaller, the closer to HPWL 
    """
    @staticmethod
    def forward(ctx, pos, flat_netpin, netpin_start, netpin_values, gamma, ignore_net_degree):
        if pos.is_cuda:
            output = logsumexp_wirelength_cuda.forward(pos.view(pos.numel()), flat_netpin, netpin_start, netpin_values, gamma, ignore_net_degree)
        else:
            output = logsumexp_wirelength_cpp.forward(pos.view(pos.numel()), flat_netpin, netpin_start, netpin_values, gamma, ignore_net_degree)
        ctx.flat_netpin = flat_netpin
        ctx.netpin_start = netpin_start
        ctx.netpin_values = netpin_values
        ctx.gamma = gamma
        ctx.ignore_net_degree = ignore_net_degree
        ctx.exp_xy = output[1]
        ctx.exp_nxy = output[2]
        ctx.exp_xy_sum = output[3];
        ctx.exp_nxy_sum = output[4];
        ctx.pos = pos 
        #if torch.isnan(ctx.exp_xy).any() or torch.isnan(ctx.exp_nxy).any() or torch.isnan(ctx.exp_xy_sum).any() or torch.isnan(ctx.exp_nxy_sum).any() or torch.isnan(output[0]).any():
        #    pdb.set_trace()
        return output[0]

    @staticmethod
    def backward(ctx, grad_pos):
        if grad_pos.is_cuda:
            output = logsumexp_wirelength_cuda.backward(
                    grad_pos, 
                    ctx.pos, 
                    ctx.exp_xy, ctx.exp_nxy, 
                    ctx.exp_xy_sum, ctx.exp_nxy_sum, 
                    ctx.flat_netpin, 
                    ctx.netpin_start, 
                    ctx.netpin_values, 
                    ctx.gamma,
                    ctx.ignore_net_degree
                    )
        else:
            output = logsumexp_wirelength_cpp.backward(
                    grad_pos, 
                    ctx.pos, 
                    ctx.exp_xy, ctx.exp_nxy, 
                    ctx.exp_xy_sum, ctx.exp_nxy_sum, 
                    ctx.flat_netpin, 
                    ctx.netpin_start, 
                    ctx.netpin_values, 
                    ctx.gamma,
                    ctx.ignore_net_degree
                    )
        #if torch.isnan(output).any():
        #    pdb.set_trace()
        return output, None, None, None, None, None

class LogSumExpWirelength(nn.Module):
    def __init__(self, flat_netpin, netpin_start, gamma, ignore_net_degree=None):
        super(LogSumExpWirelength, self).__init__()
        self.flat_netpin = flat_netpin 
        self.netpin_start = netpin_start
        self.netpin_values = None
        self.gamma = gamma
        if ignore_net_degree is None: 
            self.ignore_net_degree = flat_netpin.numel()
        else:
            self.ignore_net_degree = ignore_net_degree
    def forward(self, pos): 
        if self.netpin_values is None: 
            self.netpin_values = torch.ones(self.flat_netpin.numel(), dtype=pos.dtype, device=pos.device)
        return LogSumExpWirelengthFunction.apply(pos, 
                self.flat_netpin, 
                self.netpin_start, 
                self.netpin_values, 
                self.gamma, 
                self.ignore_net_degree
                )

class LogSumExpWirelengthAtomicFunction(Function):
    """compute weighted average wirelength.
    @param pos pin location (x array, y array), not cell location 
    @param pin2net_map pin2net map 
    @param net_mask whether to compute wirelength 
    @param gamma the smaller, the closer to HPWL 
    """
    @staticmethod
    def forward(ctx, pos, pin2net_map, net_mask, gamma):
        if pos.is_cuda:
            output = logsumexp_wirelength_cuda_atomic.forward(pos.view(pos.numel()), pin2net_map, net_mask, gamma)
        else:
            assert 0, "CPU version NOT IMPLEMENTED"
        ctx.pin2net_map = pin2net_map 
        ctx.net_mask = net_mask 
        ctx.gamma = gamma
        ctx.exp_xy = output[1]
        ctx.exp_nxy = output[2]
        ctx.exp_xy_sum = output[3];
        ctx.exp_nxy_sum = output[4];
        ctx.pos = pos 
        #if torch.isnan(ctx.exp_xy).any() or torch.isnan(ctx.exp_nxy).any() or torch.isnan(ctx.exp_xy_sum).any() or torch.isnan(ctx.exp_nxy_sum).any() or torch.isnan(output[0]).any():
        #    pdb.set_trace()
        return output[0]

    @staticmethod
    def backward(ctx, grad_pos):
        if grad_pos.is_cuda:
            output = logsumexp_wirelength_cuda_atomic.backward(
                    grad_pos, 
                    ctx.pos, 
                    ctx.exp_xy.view([-1]), ctx.exp_nxy.view([-1]), 
                    ctx.exp_xy_sum.view([-1]), ctx.exp_nxy_sum.view([-1]), 
                    ctx.pin2net_map, 
                    ctx.net_mask, 
                    ctx.gamma
                    )
        else:
            assert 0, "CPU version NOT IMPLEMENTED"
        #if torch.isnan(output).any():
        #    pdb.set_trace()
        return output, None, None, None

class LogSumExpWirelengthAtomic(nn.Module):
    def __init__(self, pin2net_map, net_mask, gamma):
        super(LogSumExpWirelengthAtomic, self).__init__()
        self.pin2net_map = pin2net_map 
        self.net_mask = net_mask 
        self.gamma = gamma
    def forward(self, pos): 
        return LogSumExpWirelengthAtomicFunction.apply(pos, 
                self.pin2net_map, 
                self.net_mask,
                self.gamma
                )

