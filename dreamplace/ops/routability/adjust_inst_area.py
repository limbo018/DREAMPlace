import math
import torch
from torch import nn
import torch.nn.functional as F

from dreamplace.ops.routability.rudy import InstanceRouteOptimizationArea, InstancePinOptimizationArea

import dreamplace.ops.routability.update_pin_offset_cpp as update_pin_offset_cpp
try:
    import dreamplace.ops.routability.update_pin_offset_cuda as update_pin_offset_cuda
except:
    pass


class AdjustInstanceArea(nn.Module):
    def __init__(self,
                 node_size_x,
                 node_size_y,
                 netpin_start,
                 flat_netpin,
                 flat_node2pin_start_map,
                 flat_node2pin_map,
                 net_weights,
                 xl,
                 xh,
                 yl,
                 yh,
                 num_nets,
                 num_nodes,
                 num_movable_nodes,
                 num_filler_nodes,
                 instance_area_adjust_overflow=0.15,
                 area_adjust_stop_ratio=0.01,
                 route_area_adjust_stop_ratio=0.01,
                 pin_area_adjust_stop_ratio=0.05,
                 route_num_bins_x=512,
                 route_num_bins_y=512,
                 unit_horizontal_routing_capacity=0.0,
                 unit_vertical_routing_capacity=0.0,
                 max_route_opt_adjust_rate=2.0,
                 pin_num_bins_x=512,
                 pin_num_bins_y=512,
                 unit_pin_capacity=0.0,
                 pin_stretch_ratio=math.sqrt(2),
                 max_pin_opt_adjust_rate=1.5,
                 num_threads=8):
        super(AdjustInstanceArea, self).__init__()
        self.node_size_x = node_size_x
        self.node_size_y = node_size_y
        self.netpin_start = netpin_start
        self.flat_netpin = flat_netpin
        self.flat_node2pin_start_map = flat_node2pin_start_map
        self.flat_node2pin_map = flat_node2pin_map
        self.net_weights = net_weights
        self.xl = xl
        self.xh = xh
        self.yl = yl
        self.yh = yh

        self.num_nets = num_nets
        self.num_nodes = num_nodes
        self.num_movable_nodes = num_movable_nodes
        self.num_filler_nodes = num_filler_nodes
        self.num_threads = num_threads

        self.adjust_area_flag = True
        self.adjust_route_area_flag = True
        self.adjust_pin_area_flag = True

        # stop ratio
        self.instance_area_adjust_overflow = instance_area_adjust_overflow
        self.area_adjust_stop_ratio = area_adjust_stop_ratio
        self.route_area_adjust_stop_ratio = route_area_adjust_stop_ratio
        self.pin_area_adjust_stop_ratio = pin_area_adjust_stop_ratio

        # route_opt_area param
        self.route_num_bins_x = route_num_bins_x
        self.route_num_bins_y = route_num_bins_y
        self.unit_horizontal_routing_capacity = unit_horizontal_routing_capacity
        self.unit_vertical_routing_capacity = unit_vertical_routing_capacity
        self.max_route_opt_adjust_rate = max_route_opt_adjust_rate

        # pin_opt_area param
        self.pin_num_bins_x = pin_num_bins_x
        self.pin_num_bins_y = pin_num_bins_y
        self.unit_pin_capacity = unit_pin_capacity
        self.pin_stretch_ratio = pin_stretch_ratio
        self.max_pin_opt_adjust_rate = max_pin_opt_adjust_rate

        self.max_total_area = (node_size_x[:num_movable_nodes] * node_size_y[:num_movable_nodes]
                               ).sum() + (node_size_x[-num_filler_nodes:] * node_size_y[-num_filler_nodes:]).sum()

        self.instance_route_optimization_area_estimator = InstanceRouteOptimizationArea(
            num_bins_x=self.route_num_bins_x,
            num_bins_y=self.route_num_bins_y,
            node_size_x=self.node_size_x,
            node_size_y=self.node_size_y,
            netpin_start=self.netpin_start,
            flat_netpin=self.flat_netpin,
            net_weights=self.net_weights,
            xl=self.xl,
            xh=self.xh,
            yl=self.yl,
            yh=self.yh,
            num_nets=self.num_nets,
            num_nodes=self.num_nodes,
            num_movable_nodes=self.num_movable_nodes,
            unit_horizontal_routing_capacity=self.unit_horizontal_routing_capacity,
            unit_vertical_routing_capacity=self.unit_vertical_routing_capacity,
            max_route_opt_adjust_rate=self.max_route_opt_adjust_rate,
            num_threads=self.num_threads
        )

        self.instance_pin_optimization_area_estimator = InstancePinOptimizationArea(
            flat_node2pin_start_map=self.flat_node2pin_start_map,
            num_bins_x=self.pin_num_bins_x,
            num_bins_y=self.pin_num_bins_y,
            node_size_x=self.node_size_x,
            node_size_y=self.node_size_y,
            xl=self.xl,
            xh=self.xh,
            yl=self.yl,
            yh=self.yh,
            num_nodes=self.num_nodes,
            num_movable_nodes=self.num_movable_nodes,
            num_filler_nodes=self.num_filler_nodes,
            unit_pin_capacity=self.unit_pin_capacity,
            pin_stretch_ratio=self.pin_stretch_ratio,
            max_pin_opt_adjust_rate=self.max_pin_opt_adjust_rate,
            num_threads=self.num_threads
        )

    def forward(self, pos, pin_pos, pin_offset_x, pin_offset_y, cur_metric_overflow):
        # check the instance area adjustment is performed
        if (cur_metric_overflow > self.instance_area_adjust_overflow) or (not self.adjust_area_flag):
            return False

        # compute routability optimized area
        if self.adjust_route_area_flag:
            route_opt_area = self.instance_route_optimization_area_estimator(pos, pin_pos)

        # compute pin density optimized area
        if self.adjust_pin_area_flag:
            pin_opt_area = self.instance_pin_optimization_area_estimator(pos)

        # compute old areas of movable nodes
        node_size_x_movable = self.node_size_x[:self.num_movable_nodes]
        node_size_y_movable = self.node_size_y[:self.num_movable_nodes]
        old_movable_area = node_size_x_movable * node_size_y_movable
        old_movable_area_sum = old_movable_area.sum()

        # compute the extra area max(route_opt_area, pin_opt_area) over the base area for each movable node
        area_increment = F.relu(torch.max(route_opt_area, pin_opt_area) - old_movable_area)
        area_increment_sum = area_increment.sum()

        # check whether the total area is larger than the max area requirement
        # If yes, scale the extra area to meet the requirement
        # We assume the total base area is no greater than the max area requirement
        scale_factor = (self.max_total_area - old_movable_area_sum) / area_increment_sum

        # set the new_movable_area as base_area + scaled area increment
        if scale_factor <= 0:
            new_movable_area = old_movable_area
            area_increment_sum = 0
        elif scale_factor >= 1:
            new_movable_area = old_movable_area + area_increment
        else:
            new_movable_area = old_movable_area + area_increment * scale_factor
            area_increment_sum *= scale_factor
        new_movable_area_sum = old_movable_area_sum + area_increment_sum

        # compute the adjusted area increase ratio
        route_area_increment_ratio = F.relu(route_opt_area - old_movable_area).sum() / old_movable_area_sum
        pin_area_increment_ratio = F.relu(pin_opt_area - old_movable_area).sum() / old_movable_area_sum
        area_increment_ratio = area_increment_sum / old_movable_area_sum

        # disable some of the area adjustment if the condition holds
        self.adjust_route_area_flag &= route_area_increment_ratio.data.item() > self.route_area_adjust_stop_ratio
        self.adjust_pin_area_flag &= pin_area_increment_ratio.data.item() > self.pin_area_adjust_stop_ratio
        self.adjust_area_flag &= (area_increment_ratio.data.item() > self.area_adjust_stop_ratio) and (self.adjust_route_area_flag or self.adjust_pin_area_flag)
        if not self.adjust_area_flag:
            return False

        # adjust the size of movable nodes
        # each movable node have its own inflation ratio, the shape of movable_nodes_ratio is (num_movable_nodes)
        movable_nodes_ratio = torch.sqrt(new_movable_area / old_movable_area)
        node_size_x_movable *= movable_nodes_ratio
        node_size_y_movable *= movable_nodes_ratio

        # finally scale the filler instance areas to let the total area be max_total_area
        # all the filler nodes share the same deflation ratio, filler_nodes_ratio is a scalar
        node_size_x_filler = self.node_size_x[-self.num_filler_nodes:]
        node_size_y_filler = self.node_size_y[-self.num_filler_nodes:]
        old_filler_area_sum = (node_size_x_filler * node_size_y_filler).sum()
        new_filler_area_sum = F.relu(self.max_total_area - new_movable_area_sum)
        filler_nodes_ratio = torch.sqrt(new_filler_area_sum / old_filler_area_sum).data.item()
        node_size_x_filler *= filler_nodes_ratio
        node_size_y_filler *= filler_nodes_ratio

        if pos.is_cuda:
            update_pin_offset_cuda.update_pin_offset(
                self.num_nodes,
                self.num_movable_nodes,
                self.num_filler_nodes,
                self.flat_node2pin_start_map,
                self.flat_node2pin_map,
                movable_nodes_ratio,
                filler_nodes_ratio,
                pin_offset_x,
                pin_offset_y
            )
        else:
            update_pin_offset_cpp.update_pin_offset(
                self.num_nodes,
                self.num_movable_nodes,
                self.num_filler_nodes,
                self.flat_node2pin_start_map,
                self.flat_node2pin_map,
                movable_nodes_ratio,
                filler_nodes_ratio,
                pin_offset_x,
                pin_offset_y,
                self.num_threads
            )

        return True
