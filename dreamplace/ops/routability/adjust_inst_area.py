import math
import torch
from torch import nn
from torch.autograd import Function

from dreamplace.ops.routability.rudy import InstanceRouteOptimizationArea

import dreamplace.ops.routability.adjust_instance_area_cpp as adjust_instance_area_cpp
try:
    import dreamplace.ops.routability.adjust_instance_area_cuda as adjust_instance_area_cuda
except:
    pass


class AdjustInstanceAreaFunction(Function):
    @staticmethod
    def forward(ctx,
                pos,
                pin_pos,
                node_size_x,
                node_size_y,
                netpin_start,
                flat_netpin,
                num_nodes,
                num_movable_nodes,
                num_filler_nodes,
                instance_route_area,
                max_total_area,
                num_threads
                ):
        if pos.is_cuda:
            adjust_instance_area_cuda(pos,
                                      pin_pos,
                                      node_size_x,
                                      node_size_y,
                                      netpin_start,
                                      flat_netpin,
                                      num_nodes,
                                      num_movable_nodes,
                                      num_filler_nodes,
                                      instance_route_area,
                                      max_total_area)
        else:
            adjust_instance_area_cpp(pos,
                                     pin_pos,
                                     node_size_x,
                                     node_size_y,
                                     netpin_start,
                                     flat_netpin,
                                     num_nodes,
                                     num_movable_nodes,
                                     num_filler_nodes,
                                     instance_route_area,
                                     max_total_area,
                                     num_threads)


class AdjustInstanceArea(nn.Module):
    def __init__(self,
                 bin_size_x,
                 bin_size_y,
                 node_size_x,
                 node_size_y,
                 netpin_start,
                 flat_netpin,
                 xl,
                 xh,
                 yl,
                 yh,
                 num_nets,
                 num_nodes,
                 num_movable_nodes,
                 num_filler_nodes,
                 num_threads=8):
        super(AdjustInstanceArea, self).__init__()
        self.bin_size_x = bin_size_x
        self.bin_size_y = bin_size_y
        self.node_size_x = node_size_x
        self.node_size_y = node_size_y
        self.netpin_start = netpin_start,
        self.flat_netpin = flat_netpin,
        self.xl = xl
        self.xh = xh
        self.yl = yl
        self.yh = yh
        self.num_bins_x = int(math.ceil((xh - xl) / bin_size_x))
        self.num_bins_y = int(math.ceil((yh - yl) / bin_size_y))
        # initialize parameters
        self.unit_horizontal_routing_capacity = None
        self.unit_vertical_routing_capacity = None
        # maximum and minimum instance area adjustment rate for routability optimization
        self.max_route_opt_adjust_rate = 2.0
        self.min_route_opt_adjust_rate = 1.0 / max_route_opt_adjust_rate
        self.num_nets = num_nets
        self.num_nodes = num_nodes
        self.num_movable_nodes = num_movable_nodes
        self.num_filler_nodes = num_filler_nodes
        self.num_threads = num_threads

        self.max_total_area = (node_size_x[:num_movable_nodes] * node_size_y[:num_movable_nodes]
                               ).sum() + (node_size_x[-num_filler_nodes:] * node_size_y[-num_filler_nodes:]).sum()

        self.instance_route_optimization_area_estimator = InstanceRouteOptimizationArea(
            bin_size_x=self.bin_size_x,
            bin_size_y=self.bin_size_y,
            node_size_x=self.node_size_x,
            node_size_y=self.node_size_y,
            netpin_start=self.netpin_start,
            flat_netpin=self.flat_netpin,
            xl=self.xl,
            xh=self.xh,
            yl=self.yl,
            yh=self.yh,
            num_nets=self.num_nets,
            num_nodes=self.num_nodes,
            num_movable_nodes=self.num_movable_nodes,
            num_filler_nodes=self.num_filler_nodes,
            num_threads=self.num_threads
        )

    def forward(self, pos, pin_pos):
        instance_route_area = self.instance_route_optimization_area_estimator(pos, pin_pos)
        return AdjustInstanceAreaFunction.apply()

