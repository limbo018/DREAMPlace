'''
@Author: Jake Gu
@Date: 2019-12-27 13:53:47
@LastEditors  : Jake Gu
@LastEditTime : 2019-12-27 14:19:16
'''
import math
import torch
from torch import nn
from torch.autograd import Function
import pdb 

import dreamplace.ops.rudy.rudy_cpp as rudy_cpp
try:
    import dreamplace.ops.rudy.rudy_cuda as rudy_cuda
except:
    pass

class Rudy(nn.Module):
    def __init__(self,
                 netpin_start, flat_netpin, net_weights,
                 xl, xh, yl, yh,
                 num_bins_x, num_bins_y,
                 num_horizontal_tracks,
                 num_vertical_tracks,
                 max_route_opt_adjust_rate,
                 num_threads=8
                 ):
        super(Rudy, self).__init__()
        self.netpin_start = netpin_start
        self.flat_netpin = flat_netpin
        self.net_weights = net_weights
        self.xl = xl
        self.xh = xh
        self.yl = yl
        self.yh = yh
        self.num_threads = num_threads
        self.num_bins_x = num_bins_x
        self.num_bins_y = num_bins_y
        self.bin_size_x = (xh - xl) / num_bins_x
        self.bin_size_y = (yh - yl) / num_bins_y

        # initialize parameters
        self.num_horizontal_tracks = num_horizontal_tracks
        self.num_vertical_tracks = num_vertical_tracks
        # maximum and minimum instance area adjustment rate for routability optimization
        self.max_route_opt_adjust_rate = max_route_opt_adjust_rate
        self.min_route_opt_adjust_rate = 1.0 / max_route_opt_adjust_rate

    def forward(self, pin_pos):
        horizontal_utilization_map = torch.zeros((self.num_bins_x, self.num_bins_y), dtype=pin_pos.dtype, device=pin_pos.device)
        vertical_utilization_map = torch.zeros_like(horizontal_utilization_map)
        if pin_pos.is_cuda:
            rudy_cuda.forward(
                    pin_pos,
                    self.netpin_start,
                    self.flat_netpin,
                    self.net_weights,
                    self.bin_size_x,
                    self.bin_size_y,
                    self.xl,
                    self.yl,
                    self.xh,
                    self.yh,
                    self.num_bins_x,
                    self.num_bins_y, 
                    horizontal_utilization_map, 
                    vertical_utilization_map
                    )
        else:
            rudy_cpp.forward(
                    pin_pos,
                    self.netpin_start,
                    self.flat_netpin,
                    self.net_weights,
                    self.bin_size_x,
                    self.bin_size_y,
                    self.xl,
                    self.yl,
                    self.xh,
                    self.yh,
                    self.num_bins_x,
                    self.num_bins_y,
                    self.num_threads, 
                    horizontal_utilization_map, 
                    vertical_utilization_map
                    )

        # convert demand to utilization in each bin
        horizontal_utilization_map.mul_(1 / (self.bin_size_x * self.num_horizontal_tracks))
        vertical_utilization_map.mul_(1 / (self.bin_size_y * self.num_vertical_tracks))
        # infinity norm
        route_utilization_map = torch.max(horizontal_utilization_map.abs_(), vertical_utilization_map.abs_())
        # clamp the routing square of routing utilization map
        route_utilization_map.pow_(2).clamp_(min=self.min_route_opt_adjust_rate, max=self.max_route_opt_adjust_rate)

        return route_utilization_map

