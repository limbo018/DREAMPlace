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
                 unit_horizontal_routing_capacity,
                 unit_vertical_routing_capacity,
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
        self.unit_horizontal_routing_capacity = unit_horizontal_routing_capacity
        self.unit_vertical_routing_capacity = unit_vertical_routing_capacity
        # maximum and minimum instance area adjustment rate for routability optimization
        self.max_route_opt_adjust_rate = max_route_opt_adjust_rate
        self.min_route_opt_adjust_rate = 1.0 / max_route_opt_adjust_rate

    def forward(self, pin_pos):
        if pin_pos.is_cuda:
            output = rudy_cuda.forward(
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
                    self.unit_horizontal_routing_capacity,
                    self.unit_vertical_routing_capacity,
                    self.max_route_opt_adjust_rate,
                    self.min_route_opt_adjust_rate,
                    self.num_bins_x,
                    self.num_bins_y
                    )
        else:
            output = rudy_cpp.forward(
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
                    self.unit_horizontal_routing_capacity,
                    self.unit_vertical_routing_capacity,
                    self.max_route_opt_adjust_rate,
                    self.min_route_opt_adjust_rate,
                    self.num_bins_x,
                    self.num_bins_y,
                    self.num_threads
                    )

        return output

