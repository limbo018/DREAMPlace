import math
import torch
from torch import nn
from torch.autograd import Function

import dreamplace.ops.routability.instance_route_optimization_area_cpp as instance_route_optimization_area_cpp
try:
    import dreamplace.ops.routability.instance_route_optimization_area_cuda as instance_route_optimization_area_cuda
except:
    pass


class InstanceRouteOptimizationAreaFunction(Function):
    @staticmethod
    def forward(ctx,
                pos,
                pin_pos,
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
                num_bins_x,
                num_bins_y,
                unit_horizontal_routing_capacity,
                unit_vertical_routing_capacity,
                max_route_opt_adjust_rate,
                min_route_opt_adjust_rate,
                num_nets,
                num_nodes,
                num_movable_nodes,
                num_threads):
        '''
        @param pin_pos bin height
        @param bin_size_x bin width
        @param bin_size_y bin height
        @param xl left boundary
        @param yl bottom boundary
        @param xh right boundary
        @param yh top boundary

        return: instance_route_area
        '''

        instance_route_area = torch.zeros(num_movable_nodes, dtype=pin_pos.dtype, device=pin_pos.device)

        if pin_pos.is_cuda:
            instance_route_optimization_area_cuda(instance_route_area,
                                                  pos,
                                                  pin_pos,
                                                  netpin_start,
                                                  flat_netpin,
                                                  num_bins_x,
                                                  num_bins_y,
                                                  bin_size_x,
                                                  bin_size_y,
                                                  node_size_x,
                                                  node_size_y,
                                                  xl,
                                                  xh,
                                                  yl,
                                                  yh,
                                                  num_nets,
                                                  num_nodes,
                                                  num_movable_nodes,
                                                  unit_horizontal_routing_capacity,
                                                  unit_vertical_routing_capacity,
                                                  max_route_opt_adjust_rate,
                                                  min_route_opt_adjust_rate
                                                  )
        else:
            instance_route_optimization_area_cpp(instance_route_area,
                                                 pos,
                                                 pin_pos,
                                                 netpin_start,
                                                 flat_netpin,
                                                 num_bins_x,
                                                 num_bins_y,
                                                 bin_size_x,
                                                 bin_size_y,
                                                 node_size_x,
                                                 node_size_y,
                                                 xl,
                                                 xh,
                                                 yl,
                                                 yh,
                                                 num_nets,
                                                 num_nodes,
                                                 num_movable_nodes,
                                                 unit_horizontal_routing_capacity,
                                                 unit_vertical_routing_capacity,
                                                 max_route_opt_adjust_rate,
                                                 min_route_opt_adjust_rate,
                                                 num_threads)

        return instance_route_area


class InstanceRouteOptimizationArea(nn.Module):
    def __init__(self,
                 num_bins_x,
                 num_bins_y,
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
                 num_threads=8):
        super(InstanceRouteOptimizationArea, self).__init__()
        self.num_bins_x = num_bins_x
        self.num_bins_y = num_bins_y
        self.bin_size_x = (xh - xl) / num_bins_x
        self.bin_size_y = (yh - yl) / num_bins_y
        self.node_size_x = node_size_x
        self.node_size_y = node_size_y
        self.netpin_start = netpin_start,
        self.flat_netpin = flat_netpin,
        self.xl = xl
        self.xh = xh
        self.yl = yl
        self.yh = yh
        
        # initialize parameters
        self.unit_horizontal_routing_capacity = None
        self.unit_vertical_routing_capacity = None
        # maximum and minimum instance area adjustment rate for routability optimization
        self.max_route_opt_adjust_rate = 2.0
        self.min_route_opt_adjust_rate = 1.0 / max_route_opt_adjust_rate
        self.num_nets = num_nets
        self.num_nodes = num_nodes
        self.num_movable_nodes = num_movable_nodes
        self.num_threads = num_threads

    def forward(self, pos, pin_pos):
        return InstanceRouteOptimizationAreaFunction.apply(pos,
                                                           pin_pos,
                                                           bin_size_x=self.bin_size_x,
                                                           bin_size_y=self.bin_size_y,
                                                           netpin_start=self.netpin_start,
                                                           flat_netpin=self.flat_netpin,
                                                           xl=self.xl,
                                                           xh=self.xh,
                                                           yl=self.yl,
                                                           yh=self.yh,
                                                           num_bin_x=self.num_bins_x,
                                                           num_bin_y=self.num_bins_y,
                                                           unit_horizontal_routing_capacity=self.unit_horizontal_routing_capacity,
                                                           unit_vertical_routing_capacity=self.unit_vertical_routing_capacity,
                                                           max_route_opt_adjust_rate=self.max_route_opt_adjust_rate,
                                                           min_route_opt_adjust_rate=self.min_route_opt_adjust_rate,
                                                           num_nets=self.num_nets,
                                                           num_nodes=self.num_nodes,
                                                           num_movable_nodes=self.num_movable_nodes,
                                                           num_threads=self.num_threads
                                                           )
