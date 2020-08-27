##
# @file   logsumexp_wirelength.py
# @author Yibo Lin
# @date   Jun 2018
# @brief  Compute log-sum-exp wirelength and gradient according to NTUPlace3
#

import time
import torch
from torch import nn
from torch.autograd import Function
import logging

import dreamplace.ops.logsumexp_wirelength.logsumexp_wirelength_cpp_merged as logsumexp_wirelength_cpp_merged
import dreamplace.configure as configure
if configure.compile_configurations["CUDA_FOUND"] == "TRUE":
    import dreamplace.ops.logsumexp_wirelength.logsumexp_wirelength_cuda_merged as logsumexp_wirelength_cuda_merged
    import dreamplace.ops.logsumexp_wirelength.logsumexp_wirelength_cuda_atomic as logsumexp_wirelength_cuda_atomic
import pdb

logger = logging.getLogger(__name__)


class LogSumExpWirelengthAtomicFunction(Function):
    """compute weighted average wirelength.
    @param pos pin location (x array, y array), not cell location 
    @param pin2net_map pin2net map 
    @param net_weights weight of nets 
    @param net_mask whether to compute wirelength 
    @param gamma the smaller, the closer to HPWL 
    """
    @staticmethod
    def forward(ctx, pos, pin2net_map, net_weights, net_mask, gamma):
        if pos.is_cuda:
            output = logsumexp_wirelength_cuda_atomic.forward(
                pos.view(pos.numel()), pin2net_map, net_weights, net_mask,
                gamma)
        else:
            assert 0, "CPU version NOT IMPLEMENTED"
        ctx.pin2net_map = pin2net_map
        ctx.net_weights = net_weights
        ctx.net_mask = net_mask
        ctx.gamma = gamma
        ctx.exp_xy = output[1]
        ctx.exp_nxy = output[2]
        ctx.exp_xy_sum = output[3]
        ctx.exp_nxy_sum = output[4]
        ctx.pos = pos
        #if torch.isnan(ctx.exp_xy).any() or torch.isnan(ctx.exp_nxy).any() or torch.isnan(ctx.exp_xy_sum).any() or torch.isnan(ctx.exp_nxy_sum).any() or torch.isnan(output[0]).any():
        #    pdb.set_trace()
        return output[0]

    @staticmethod
    def backward(ctx, grad_pos):
        if grad_pos.is_cuda:
            output = logsumexp_wirelength_cuda_atomic.backward(
                grad_pos, ctx.pos, ctx.exp_xy.view([-1]),
                ctx.exp_nxy.view([-1]), ctx.exp_xy_sum.view([-1]),
                ctx.exp_nxy_sum.view([-1]), ctx.pin2net_map, ctx.net_weights,
                ctx.net_mask, ctx.gamma)
        else:
            assert 0, "CPU version NOT IMPLEMENTED"
        #if torch.isnan(output).any():
        #    pdb.set_trace()
        return output, None, None, None, None


class LogSumExpWirelengthMergedFunction(Function):
    """
    @brief compute weighted average wirelength.
    """
    @staticmethod
    def forward(ctx, pos, flat_netpin, netpin_start, pin2net_map, net_weights,
                net_mask, pin_mask, gamma):
        """
        @param pos pin location (x array, y array), not cell location
        @param pin2net_map pin2net map
        @param net_weights weight of nets
        @param net_mask whether to compute wirelength
        @param pin_mask whether compute gradient for a pin, 1 means to fill with zero, 0 means to compute
        @param gamma the larger, the closer to HPWL
        """
        tt = time.time()
        if pos.is_cuda:
            func = logsumexp_wirelength_cuda_merged.forward
        else:
            func = logsumexp_wirelength_cpp_merged.forward
        output = func(pos.view(pos.numel()), flat_netpin, netpin_start,
                      pin2net_map, net_weights, net_mask, gamma)
        ctx.pin2net_map = pin2net_map
        ctx.flat_netpin = flat_netpin
        ctx.netpin_start = netpin_start
        ctx.net_weights = net_weights
        ctx.net_mask = net_mask
        ctx.pin_mask = pin_mask
        ctx.gamma = gamma
        ctx.grad_intermediate = output[1]
        ctx.pos = pos
        if pos.is_cuda:
            torch.cuda.synchronize()
        logger.debug("wirelength forward %.3f ms" %
                     ((time.time() - tt) * 1000))
        return output[0]

    @staticmethod
    def backward(ctx, grad_pos):
        tt = time.time()
        if grad_pos.is_cuda:
            func = logsumexp_wirelength_cuda_merged.backward
        else:
            func = logsumexp_wirelength_cpp_merged.backward
        output = func(grad_pos, ctx.pos, ctx.grad_intermediate,
                      ctx.flat_netpin, ctx.netpin_start, ctx.pin2net_map,
                      ctx.net_weights, ctx.net_mask, ctx.gamma)
        output[:int(output.numel() // 2)].masked_fill_(ctx.pin_mask, 0.0)
        output[int(output.numel() // 2):].masked_fill_(ctx.pin_mask, 0.0)
        if grad_pos.is_cuda:
            torch.cuda.synchronize()
        logger.debug("wirelength backward %.3f ms" %
                     ((time.time() - tt) * 1000))
        return output, None, None, None, None, None, None, None


class LogSumExpWirelength(nn.Module):
    """ Compute log-sum-exp wirelength. 
    CPU only supports net-by-net algorithm. 
    GPU supports two algorithms: atomic, sparse. 
    Different parameters are required for different algorithms. 

    @param flat_netpin flat netpin map, length of #pins 
    @param netpin_start starting index in netpin map for each net, length of #nets+1, the last entry is #pins  
    @param pin2net_map pin2net map 
    @param net_weights weight of nets 
    @param net_mask whether to compute wirelength, 1 means to compute, 0 means to ignore  
    @param gamma the smaller, the closer to HPWL 
    @param algorithm must be merged | atomic
    """
    def __init__(self,
                 flat_netpin=None,
                 netpin_start=None,
                 pin2net_map=None,
                 net_weights=None,
                 net_mask=None,
                 pin_mask=None,
                 gamma=None,
                 algorithm='merged'):
        super(LogSumExpWirelength, self).__init__()
        assert net_weights is not None \
                and net_mask is not None \
                and pin_mask is not None \
                and gamma is not None, "net_weights, net_mask, pin_mask, gamma are requried parameters"
        if algorithm == 'merged':
            assert flat_netpin is not None and netpin_start is not None and pin2net_map is not None, "flat_netpin, netpin_start, pin2net_map are requried parameters for algorithm %s" % (
                algorithm)
        elif algorithm == 'atomic':
            assert pin2net_map is not None, "pin2net_map is required for algorithm atomic"
        self.flat_netpin = flat_netpin
        self.netpin_start = netpin_start
        self.netpin_values = None
        self.pin2net_map = pin2net_map
        self.net_weights = net_weights
        self.net_mask = net_mask
        self.pin_mask = pin_mask
        self.gamma = gamma
        self.algorithm = algorithm

    def forward(self, pos):
        if pos.is_cuda:
            if self.algorithm == 'atomic':
                return LogSumExpWirelengthAtomicFunction.apply(
                    pos, self.pin2net_map, self.net_weights, self.net_mask,
                    self.gamma)
            elif self.algorithm == 'merged':
                return LogSumExpWirelengthMergedFunction.apply(
                    pos, self.flat_netpin, self.netpin_start, self.pin2net_map,
                    self.net_weights, self.net_mask, self.pin_mask, self.gamma)
        else:  # only merged for CPU
            return LogSumExpWirelengthMergedFunction.apply(
                pos, self.flat_netpin, self.netpin_start, self.pin2net_map,
                self.net_weights, self.net_mask, self.pin_mask, self.gamma)
