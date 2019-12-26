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
        instance_route_area = torch.zeros(self.num_movable_nodes, dtype=pin_pos.dtype, device=pin_pos.device)

        if pin_pos.is_cuda:
            instance_route_optimization_area_cuda.instance_route_optimization_area(
                instance_route_area,
                pos,
                pin_pos,
                self.netpin_start,
                self.flat_netpin,
                self.num_bins_x,
                self.num_bins_y,
                self.bin_size_x,
                self.bin_size_y,
                self.node_size_x,
                self.node_size_y,
                self.net_weights,
                self.xl,
                self.yl,
                self.xh,
                self.yh,
                self.num_nets,
                self.num_nodes,
                self.num_movable_nodes,
                self.unit_horizontal_routing_capacity,
                self.unit_vertical_routing_capacity,
                self.max_route_opt_adjust_rate,
                self.min_route_opt_adjust_rate
            )
        else:
            instance_route_optimization_area_cpp.instance_route_optimization_area(
                instance_route_area,
                pos,
                pin_pos,
                self.netpin_start,
                self.flat_netpin,
                self.num_bins_x,
                self.num_bins_y,
                self.bin_size_x,
                self.bin_size_y,
                self.node_size_x,
                self.node_size_y,
                self.net_weights,
                self.xl,
                self.yl,
                self.xh,
                self.yh,
                self.num_nets,
                self.num_nodes,
                self.num_movable_nodes,
                self.unit_horizontal_routing_capacity,
                self.unit_vertical_routing_capacity,
                self.max_route_opt_adjust_rate,
                self.min_route_opt_adjust_rate,
                self.num_threads)

        return instance_route_area


class InstancePinOptimizationArea(nn.Module):
    def __init__(self,
                 flat_node2pin_start_map,
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
        self.flat_node2pin_start_map = flat_node2pin_start_map
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
        # for each physical node, we use the pin counts as the weights
        self.pin_weights = (flat_node2pin_start_map[1:self.num_physical_nodes + 1] - flat_node2pin_start_map[:self.num_physical_nodes]).to(self.node_size_x.dtype)

    def forward(self, pos):
        # compute the pin bounding box
        # to make the pin density map smooth, we stretch each pin to a ratio of the pin utilization bin
        half_node_size_stretch_x = 0.5 * torch.max(self.bin_size_stretch_x, self.node_size_x[:self.num_physical_nodes])
        half_node_size_stretch_y = 0.5 * torch.max(self.bin_size_stretch_y, self.node_size_y[:self.num_physical_nodes])
        node_center_x = pos[:self.num_physical_nodes] + 0.5 * self.node_size_x[:self.num_physical_nodes]
        node_center_y = pos[self.num_nodes:self.num_nodes + self.num_physical_nodes] + 0.5 * self.node_size_y[:self.num_physical_nodes]

        instance_pin_area = torch.zeros(self.num_movable_nodes, dtype=pos.dtype, device=pos.device)

        if pos.is_cuda:
            instance_pin_optimization_area_cuda.instance_pin_optimization_area(
                pos,
                node_center_x,
                node_center_y,
                half_node_size_stretch_x,
                half_node_size_stretch_y,
                self.pin_weights,
                self.num_bins_x,
                self.num_bins_y,
                self.bin_size_x,
                self.bin_size_y,
                self.node_size_x,
                self.node_size_y,
                self.xl,
                self.yl,
                self.xh,
                self.yh,
                self.num_nodes,
                self.num_movable_nodes,
                self.num_physical_nodes,
                self.unit_pin_capacity,
                self.max_pin_opt_adjust_rate,
                self.min_pin_opt_adjust_rate,
                instance_pin_area
            )
        else:
            instance_pin_optimization_area_cpp.instance_pin_optimization_area(
                pos,
                node_center_x,
                node_center_y,
                half_node_size_stretch_x,
                half_node_size_stretch_y,
                self.pin_weights,
                self.num_bins_x,
                self.num_bins_y,
                self.bin_size_x,
                self.bin_size_y,
                self.node_size_x,
                self.node_size_y,
                self.xl,
                self.yl,
                self.xh,
                self.yh,
                self.num_nodes,
                self.num_movable_nodes,
                self.num_physical_nodes,
                self.unit_pin_capacity,
                self.max_pin_opt_adjust_rate,
                self.min_pin_opt_adjust_rate,
                instance_pin_area,
                self.num_threads)

        return instance_pin_area
