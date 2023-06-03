##
# @file   pinrudy.py
# @author Siting Liu
# @date   Sept 2022
# @brief  Compute Pin Rudy map for each pin
#
import math
import torch
from torch import nn
from torch.autograd import Function
import matplotlib.pyplot as plt
import pdb

import dreamplace.ops.pinrudy.pinrudy_cpp as pinrudy_cpp
import dreamplace.configure as configure
if configure.compile_configurations["CUDA_FOUND"] == "TRUE":
    import dreamplace.ops.pinrudy.pinrudy_cuda as pinrudy_cuda


class PinRudy(nn.Module):
    def __init__(self,
                 netpin_start,
                 flat_netpin,
                 net_weights,
                 xl,
                 xh,
                 yl,
                 yh,
                 num_bins_x,
                 num_bins_y,
                 unit_horizontal_capacity,
                 unit_vertical_capacity,
                 deterministic_flag, 
                 initial_horizontal_utilization_map=None,
                 initial_vertical_utilization_map=None):
        super(PinRudy, self).__init__()
        self.netpin_start = netpin_start
        self.flat_netpin = flat_netpin
        self.net_weights = net_weights
        self.xl = xl
        self.yl = yl
        self.xh = xh
        self.yh = yh
        self.num_bins_x = num_bins_x
        self.num_bins_y = num_bins_y
        self.bin_size_x = (xh - xl) / num_bins_x
        self.bin_size_y = (yh - yl) / num_bins_y

        # initialize parameters
        self.unit_horizontal_capacity = unit_horizontal_capacity
        self.unit_vertical_capacity = unit_vertical_capacity

        self.deterministic_flag = deterministic_flag

        self.initial_horizontal_utilization_map = initial_horizontal_utilization_map
        self.initial_vertical_utilization_map = initial_vertical_utilization_map

        #plt.imsave("rudy_initial.png", (self.initial_horizontal_utilization_map + self.initial_vertical_utilization_map).data.cpu().numpy().T, origin='lower')

    def forward(self, pin_pos):
        horizontal_utilization_map = torch.zeros(
            (self.num_bins_x, self.num_bins_y),
            dtype=pin_pos.dtype,
            device=pin_pos.device)
        vertical_utilization_map = torch.zeros_like(horizontal_utilization_map)
        if pin_pos.is_cuda:
            func = pinrudy_cuda.forward
        else:
            func = pinrudy_cpp.forward
        func(pin_pos, self.netpin_start, self.flat_netpin, self.net_weights,
             self.bin_size_x, self.bin_size_y, self.xl, self.yl, self.xh,
             self.yh, self.num_bins_x, self.num_bins_y, self.deterministic_flag, 
             horizontal_utilization_map, vertical_utilization_map)

        if self.initial_horizontal_utilization_map is not None:
            horizontal_utilization_map.add_(
                self.initial_horizontal_utilization_map)
        if self.initial_vertical_utilization_map is not None:
            vertical_utilization_map.add_(
                self.initial_vertical_utilization_map)

        route_utilization_map = horizontal_utilization_map.abs_() + vertical_utilization_map.abs_()

        return route_utilization_map
