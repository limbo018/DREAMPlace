import math
import torch
from torch import nn
from torch.autograd import Function

import dreamplace.ops.routability.instance_route_optimization_area_cpp as instance_route_optimization_area_cpp
import dreamplace.ops.routability.instance_pin_optimization_area_cpp as instance_pin_optimization_area_cpp
try:
    import dreamplace.ops.routability.instance_route_optimization_area_cuda as instance_route_optimization_area_cuda
    import dreamplace.ops.routability.instance_pin_optimization_area_cuda as instance_pin_optimization_area_cuda
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
                net_weights,
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
                                                  net_weights,
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
                                                 net_weights,
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
                 net_weights,
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
                 num_threads=8):
        super(InstanceRouteOptimizationArea, self).__init__()
        self.num_bins_x = num_bins_x
        self.num_bins_y = num_bins_y
        self.bin_size_x = (xh - xl) / num_bins_x
        self.bin_size_y = (yh - yl) / num_bins_y
        self.node_size_x = node_size_x
        self.node_size_y = node_size_y
        self.netpin_start = netpin_start
        self.flat_netpin = flat_netpin
        self.net_weights = net_weights
        self.xl = xl
        self.xh = xh
        self.yl = yl
        self.yh = yh

        # initialize parameters
        self.unit_horizontal_routing_capacity = unit_horizontal_routing_capacity
        self.unit_vertical_routing_capacity = unit_vertical_routing_capacity
        # maximum and minimum instance area adjustment rate for routability optimization
        self.max_route_opt_adjust_rate = max_route_opt_adjust_rate
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
                                                           net_weights=self.net_weights,
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


class InstancePinOptimizationAreaFunction(Function):
    @staticmethod
    def forward(ctx,
                pos,
                node_center_x,
                node_center_y,
                half_bin_size_stretch_x,
                half_bin_size_stretch_y,
                pin_weights,
                num_bins_x,
                num_bins_y,
                bin_size_x,
                bin_size_y,
                node_size_x,
                node_size_y,
                xl,
                yl,
                xh,
                yh,
                num_nodes,
                num_movable_nodes,
                num_physical_nodes,
                unit_pin_capacity,
                max_pin_opt_adjust_rate,
                min_pin_opt_adjust_rate,
                num_threads):

        instance_pin_area = torch.zeros(num_movable_nodes, dtype=pos.dtype, device=pos.device)

        if pos.is_cuda:
            instance_route_optimization_area_cuda(pos,
                                                  node_center_x,
                                                  node_center_y,
                                                  half_bin_size_stretch_x,
                                                  half_bin_size_stretch_y,
                                                  pin_weights,
                                                  num_bins_x,
                                                  num_bins_y,
                                                  bin_size_x,
                                                  bin_size_y,
                                                  node_size_x,
                                                  node_size_y,
                                                  xl,
                                                  yl,
                                                  xh,
                                                  yh,
                                                  num_nodes,
                                                  num_movable_nodes,
                                                  num_physical_nodes,
                                                  unit_pin_capacity,
                                                  max_pin_opt_adjust_rate,
                                                  min_pin_opt_adjust_rate,
                                                  instance_pin_area
                                                  )
        else:
            instance_route_optimization_area_cpp(pos,
                                                 node_center_x,
                                                 node_center_y,
                                                 half_bin_size_stretch_x,
                                                 half_bin_size_stretch_y,
                                                 pin_weights,
                                                 num_bins_x,
                                                 num_bins_y,
                                                 bin_size_x,
                                                 bin_size_y,
                                                 node_size_x,
                                                 node_size_y,
                                                 xl,
                                                 yl,
                                                 xh,
                                                 yh,
                                                 num_nodes,
                                                 num_movable_nodes,
                                                 num_physical_nodes,
                                                 unit_pin_capacity,
                                                 max_pin_opt_adjust_rate,
                                                 min_pin_opt_adjust_rate,
                                                 instance_pin_area,
                                                 num_threads)

        return instance_pin_area


class InstancePinOptimizationArea(nn.Module):
    def __init__(self,
                 num_bins_x,
                 num_bins_y,
                 node_size_x,
                 node_size_y,
                 xl,
                 xh,
                 yl,
                 yh,
                 num_nodes,
                 num_movable_nodes,
                 num_filler_nodes,
                 unit_pin_capacity,
                 pin_stretch_ratio,
                 max_pin_opt_adjust_rate,
                 num_threads=8):
        super(InstancePinOptimizationArea, self).__init__()
        self.num_bins_x = num_bins_x
        self.num_bins_y = num_bins_y
        self.bin_size_x = (xh - xl) / num_bins_x
        self.bin_size_y = (yh - yl) / num_bins_y
        self.node_size_x = node_size_x
        self.node_size_y = node_size_y
        self.xl = xl
        self.xh = xh
        self.yl = yl
        self.yh = yh
        self.num_nodes = num_nodes
        self.num_movable_nodes = num_movable_nodes
        self.num_filler_nodes = num_filler_nodes
        self.num_physical_nodes = num_nodes - num_filler_nodes
        self.num_threads = num_threads

        self.unit_pin_capacity = unit_pin_capacity
        self.pin_stretch_ratio = pin_stretch_ratio
        # maximum and minimum instance area adjustment rate for routability optimization
        self.max_pin_opt_adjust_rate = max_pin_opt_adjust_rate
        self.min_pin_opt_adjust_rate = 1.0 / max_pin_opt_adjust_rate

        self.bin_size_stretch_x = torch.Tensor([self.bin_size_x * self.pin_stretch_ratio]).to(node_size_x.device)
        self.bin_size_stretch_y = torch.Tensor([self.bin_size_y * self.pin_stretch_ratio]).to(node_size_y.device)

    def forward(self, pos):
        # for each physical node, we use the pin counts as the weights
        pin_weights = self.flat_node2pin_start_map[1:self.num_physical_nodes +
                                                   1] - self.flat_node2pin_start_map[:self.num_physical_nodes]

        # compute the pin bounding box
        # to make the pin density map smooth, we stretch each pin to a ratio of the pin utilization bin
        half_bin_size_stretch_x = 0.5 * torch.max(self.bin_size_stretch_x, self.node_size_x[:self.num_physical_nodes])
        half_bin_size_stretch_y = 0.5 * torch.max(self.bin_size_stretch_y, self.node_size_y[:self.num_physical_nodes])
        node_center_x = pos[:self.num_physical_nodes] + 0.5 * self.node_size_x[:self.num_physical_nodes]
        node_center_y = pos[self.num_nodes:self._num_ndoes + self.num_physical_nodes] + \
            0.5 * self.node_size_y[:self.num_physical_nodes]

        return InstancePinOptimizationAreaFunction.apply(
            pos=pos,
            node_center_x=node_center_x,
            node_center_y=node_center_y,
            half_bin_size_stretch_x=half_bin_size_stretch_x,
            half_bin_size_stretch_y=half_bin_size_stretch_y,
            pin_weights=pin_weights,
            num_bins_x=self.num_bins_x,
            num_bins_y=self.num_bins_y,
            bin_size_x=self.bin_size_x,
            bin_size_y=self.bin_size_y,
            node_size_x=self.node_size_x,
            node_size_y=self.node_size_y,
            xl=self.xl,
            yl=self.yl,
            xh=self.xh,
            yh=self.yh,
            num_nodes=self.num_nodes,
            num_movable_nodes=self.num_movable_nodes,
            num_physical_nodes=self.num_physical_nodes,
            unit_pin_capacity=self.unit_pin_capacity,
            max_pin_opt_adjust_rate=self.max_pin_opt_adjust_rate,
            min_pin_opt_adjust_rate=self.min_pin_opt_adjust_rate,
            num_threads=self.num_threads
        )
