import math
import torch
from torch import nn
import torch.nn.functional as F
import logging
import pdb

import dreamplace.ops.adjust_node_area.adjust_node_area_cpp as adjust_node_area_cpp
import dreamplace.ops.adjust_node_area.update_pin_offset_cpp as update_pin_offset_cpp
try:
    import dreamplace.ops.adjust_node_area.adjust_node_area_cuda as adjust_node_area_cuda
    import dreamplace.ops.adjust_node_area.update_pin_offset_cuda as update_pin_offset_cuda
except:
    pass

logger = logging.getLogger(__name__)


class ComputeNodeAreaFromRouteMap(nn.Module):
    def __init__(self, xl, yl, xh, yh, num_movable_nodes, num_bins_x,
                 num_bins_y):
        super(ComputeNodeAreaFromRouteMap, self).__init__()
        self.xl = xl
        self.yl = yl
        self.xh = xh
        self.yh = yh
        self.num_movable_nodes = num_movable_nodes
        self.num_bins_x = num_bins_x
        self.num_bins_y = num_bins_y
        self.bin_size_x = (xh - xl) / num_bins_x
        self.bin_size_y = (yh - yl) / num_bins_y

    def forward(self, pos, node_size_x, node_size_y, utilization_map):
        if pos.is_cuda:
            func = adjust_node_area_cuda.forward
        else:
            func = adjust_node_area_cpp.forward
        output = func(pos, node_size_x, node_size_y, utilization_map,
                      self.bin_size_x, self.bin_size_y, self.xl, self.yl,
                      self.xh, self.yh, self.num_movable_nodes,
                      self.num_bins_x, self.num_bins_y)
        return output


class ComputeNodeAreaFromPinMap(ComputeNodeAreaFromRouteMap):
    def __init__(self, pin_weights, flat_node2pin_start_map, xl, yl, xh, yh,
                 num_movable_nodes, num_bins_x, num_bins_y, unit_pin_capacity):
        super(ComputeNodeAreaFromPinMap,
              self).__init__(xl, yl, xh, yh, num_movable_nodes, num_bins_x,
                             num_bins_y)
        bin_area = (xh - xl) / num_bins_x * (yh - yl) / num_bins_y
        self.unit_pin_capacity = unit_pin_capacity
        # for each physical node, we use the pin counts as the weights
        if pin_weights is not None:
            self.pin_weights = pin_weights
        elif flat_node2pin_start_map is not None:
            self.pin_weights = flat_node2pin_start_map[
                1:self.num_movable_nodes +
                1] - flat_node2pin_start_map[:self.num_movable_nodes]
        else:
            assert "either pin_weights or flat_node2pin_start_map is required"

    def forward(self, pos, node_size_x, node_size_y, utilization_map):
        output = super(ComputeNodeAreaFromPinMap,
                       self).forward(pos, node_size_x, node_size_y,
                                     utilization_map)
        #output.mul_(self.pin_weights[:self.num_movable_nodes].to(node_size_x.dtype) / (node_size_x[:self.num_movable_nodes] * node_size_y[:self.num_movable_nodes] * self.unit_pin_capacity))
        return output


class AdjustNodeArea(nn.Module):
    def __init__(
        self,
        flat_node2pin_map,
        flat_node2pin_start_map,
        pin_weights,  # only one of them needed
        xl,
        yl,
        xh,
        yh,
        num_movable_nodes,
        num_filler_nodes,
        route_num_bins_x,
        route_num_bins_y,
        pin_num_bins_x,
        pin_num_bins_y,
        total_place_area,  # total placement area excluding fixed cells
        total_whitespace_area,  # total white space area excluding movable and fixed cells
        max_route_opt_adjust_rate,
        route_opt_adjust_exponent=2.5,
        max_pin_opt_adjust_rate=2.5,
        area_adjust_stop_ratio=0.01,
        route_area_adjust_stop_ratio=0.01,
        pin_area_adjust_stop_ratio=0.05,
        unit_pin_capacity=0.0):
        super(AdjustNodeArea, self).__init__()
        self.flat_node2pin_start_map = flat_node2pin_start_map
        self.flat_node2pin_map = flat_node2pin_map
        self.pin_weights = pin_weights
        self.xl = xl
        self.xh = xh
        self.yl = yl
        self.yh = yh

        self.num_movable_nodes = num_movable_nodes
        self.num_filler_nodes = num_filler_nodes

        # maximum and minimum instance area adjustment rate for routability optimization
        self.max_route_opt_adjust_rate = max_route_opt_adjust_rate
        self.min_route_opt_adjust_rate = 1.0 / max_route_opt_adjust_rate
        # exponent for adjusting the utilization map
        self.route_opt_adjust_exponent = route_opt_adjust_exponent
        # maximum and minimum instance area adjustment rate for routability optimization
        self.max_pin_opt_adjust_rate = max_pin_opt_adjust_rate
        self.min_pin_opt_adjust_rate = 1.0 / max_pin_opt_adjust_rate

        # stop ratio
        self.area_adjust_stop_ratio = area_adjust_stop_ratio
        self.route_area_adjust_stop_ratio = route_area_adjust_stop_ratio
        self.pin_area_adjust_stop_ratio = pin_area_adjust_stop_ratio

        self.compute_node_area_route = ComputeNodeAreaFromRouteMap(
            xl=self.xl,
            yl=self.yl,
            xh=self.xh,
            yh=self.yh,
            num_movable_nodes=self.num_movable_nodes,
            num_bins_x=route_num_bins_x,
            num_bins_y=route_num_bins_y)
        self.compute_node_area_pin = ComputeNodeAreaFromPinMap(
            pin_weights=self.pin_weights,
            flat_node2pin_start_map=self.flat_node2pin_start_map,
            xl=self.xl,
            yl=self.yl,
            xh=self.xh,
            yh=self.yh,
            num_movable_nodes=self.num_movable_nodes,
            num_bins_x=pin_num_bins_x,
            num_bins_y=pin_num_bins_y,
            unit_pin_capacity=unit_pin_capacity)

        # placement area excluding fixed cells
        self.total_place_area = total_place_area
        # placement area excluding movable and fixed cells
        self.total_whitespace_area = total_whitespace_area

    def forward(self, pos, node_size_x, node_size_y, pin_offset_x,
                pin_offset_y, target_density, route_utilization_map,
                pin_utilization_map):

        with torch.no_grad():
            adjust_area_flag = True
            adjust_route_area_flag = route_utilization_map is not None
            adjust_pin_area_flag = pin_utilization_map is not None

            if not (adjust_pin_area_flag or adjust_route_area_flag):
                return False, False, False

            # compute old areas of movable nodes
            node_size_x_movable = node_size_x[:self.num_movable_nodes]
            node_size_y_movable = node_size_y[:self.num_movable_nodes]
            old_node_size_x_movable = node_size_x_movable.clone()
            old_node_size_y_movable = node_size_y_movable.clone()
            
            old_movable_area = node_size_x_movable * node_size_y_movable
            old_movable_area_sum = old_movable_area.sum()
            # compute old areas of filler nodes
            node_size_x_filler = node_size_x[-self.num_filler_nodes:]
            node_size_y_filler = node_size_y[-self.num_filler_nodes:]
            old_filler_area_sum = (node_size_x_filler *
                                   node_size_y_filler).sum()

            # compute routability optimized area
            if adjust_route_area_flag:
                # clamp the routing square of routing utilization map
                #topk, indices = route_utilization_map.view(-1).topk(int(0.1 * route_utilization_map.numel()))
                #route_utilization_map_clamp = route_utilization_map.mul(1.0 / max(topk.min(), 1))
                #route_utilization_map_clamp.pow_(2.5).clamp_(min=self.min_route_opt_adjust_rate, max=self.max_route_opt_adjust_rate)
                route_utilization_map_clamp = route_utilization_map.pow(
                    self.route_opt_adjust_exponent).clamp_(
                        min=self.min_route_opt_adjust_rate,
                        max=self.max_route_opt_adjust_rate)
                route_opt_area = self.compute_node_area_route(
                    pos, node_size_x, node_size_y, route_utilization_map_clamp)
            # compute pin density optimized area
            if adjust_pin_area_flag:
                pin_opt_area = self.compute_node_area_pin(
                    pos,
                    node_size_x,
                    node_size_y,
                    # clamp the pin utilization map
                    pin_utilization_map.clamp(
                        min=self.min_pin_opt_adjust_rate,
                        max=self.max_pin_opt_adjust_rate))

            # compute the extra area max(route_opt_area, pin_opt_area) over the base area for each movable node
            if adjust_route_area_flag and adjust_pin_area_flag:
                area_increment = F.relu(
                    torch.max(route_opt_area, pin_opt_area) - old_movable_area)
            elif adjust_route_area_flag:
                area_increment = F.relu(route_opt_area - old_movable_area)
            else:
                area_increment = F.relu(pin_opt_area - old_movable_area)
            area_increment_sum = area_increment.sum()

            # check whether the total area is larger than the max area requirement
            # If yes, scale the extra area to meet the requirement
            # We assume the total base area is no greater than the max area requirement
            scale_factor = (min(0.1 * self.total_whitespace_area,
                                self.total_place_area - old_movable_area_sum) /
                            area_increment_sum).item()

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
            area_increment_ratio = area_increment_sum / old_movable_area_sum
            logger.info(
                "area_increment = %E, area_increment / movable = %g, area_adjust_stop_ratio = %g"
                % (area_increment_sum, area_increment_ratio,
                   self.area_adjust_stop_ratio))
            logger.info(
                "area_increment / total_place_area = %g, area_increment / filler = %g, area_increment / total_whitespace_area = %g"
                % (area_increment_sum / self.total_place_area,
                   area_increment_sum / old_filler_area_sum,
                   area_increment_sum / self.total_whitespace_area))

            # compute the adjusted area increase ratio
            # disable some of the area adjustment if the condition holds
            if adjust_route_area_flag:
                route_area_increment_ratio = F.relu(
                    route_opt_area -
                    old_movable_area).sum() / old_movable_area_sum
                adjust_route_area_flag = route_area_increment_ratio.data.item(
                ) > self.route_area_adjust_stop_ratio
                logger.info(
                    "route_area_increment_ratio = %g, route_area_adjust_stop_ratio = %g"
                    % (route_area_increment_ratio,
                       self.route_area_adjust_stop_ratio))
            if adjust_pin_area_flag:
                pin_area_increment_ratio = F.relu(
                    pin_opt_area -
                    old_movable_area).sum() / old_movable_area_sum
                adjust_pin_area_flag = pin_area_increment_ratio.data.item(
                ) > self.pin_area_adjust_stop_ratio
                logger.info(
                    "pin_area_increment_ratio = %g, pin_area_adjust_stop_ratio = %g"
                    % (pin_area_increment_ratio,
                       self.pin_area_adjust_stop_ratio))
            adjust_area_flag = (
                area_increment_ratio.data.item() > self.area_adjust_stop_ratio
            ) and (adjust_route_area_flag or adjust_pin_area_flag)

            if not adjust_area_flag:
                return adjust_area_flag, adjust_route_area_flag, adjust_pin_area_flag

            num_nodes = int(pos.numel() / 2)
            # adjust the size and positions of movable nodes
            # each movable node have its own inflation ratio, the shape of movable_nodes_ratio is (num_movable_nodes)
            # we keep the centers the same
            movable_nodes_ratio = new_movable_area / old_movable_area
            logger.info(
                "inflation ratio for movable nodes: avg/max %g/%g" %
                (movable_nodes_ratio.mean(), movable_nodes_ratio.max()))
            movable_nodes_ratio.sqrt_()
            # convert positions to centers
            pos.data[:self.num_movable_nodes] += node_size_x_movable * 0.5
            pos.data[num_nodes:num_nodes +
                     self.num_movable_nodes] += node_size_y_movable * 0.5
            # scale size
            node_size_x_movable *= movable_nodes_ratio
            node_size_y_movable *= movable_nodes_ratio
            # convert back to lower left corners
            pos.data[:self.num_movable_nodes] -= node_size_x_movable * 0.5
            pos.data[num_nodes:num_nodes +
                     self.num_movable_nodes] -= node_size_y_movable * 0.5

            # finally scale the filler instance areas to let the total area be self.total_place_area
            # all the filler nodes share the same deflation ratio, filler_nodes_ratio is a scalar
            # we keep the centers the same
            if new_movable_area_sum + old_filler_area_sum > self.total_place_area:
                new_filler_area_sum = F.relu(self.total_place_area -
                                             new_movable_area_sum)
                filler_nodes_ratio = new_filler_area_sum / old_filler_area_sum
                logger.info("inflation ratio for filler nodes: %g" %
                            (filler_nodes_ratio))
                filler_nodes_ratio.sqrt_()
                # convert positions to centers
                pos.data[num_nodes - self.num_filler_nodes:
                         num_nodes] += node_size_x_filler * 0.5
                pos.data[-self.num_filler_nodes:] += node_size_y_filler * 0.5
                # scale size
                node_size_x_filler *= filler_nodes_ratio
                node_size_y_filler *= filler_nodes_ratio
                # convert back to lower left corners
                pos.data[num_nodes - self.num_filler_nodes:
                         num_nodes] -= node_size_x_filler * 0.5
                pos.data[-self.num_filler_nodes:] -= node_size_y_filler * 0.5
            else:
                new_filler_area_sum = old_filler_area_sum

            logger.info(
                "old total movable nodes area %.3E, filler area %.3E, total movable + filler area %.3E, total_place_area %.3E"
                % (old_movable_area_sum, old_filler_area_sum,
                   old_movable_area_sum + old_filler_area_sum,
                   self.total_place_area))
            logger.info(
                "new total movable nodes area %.3E, filler area %.3E, total movable + filler area %.3E, total_place_area %.3E"
                % (new_movable_area_sum, new_filler_area_sum,
                   new_movable_area_sum + new_filler_area_sum,
                   self.total_place_area))
            target_density.data.copy_(
                (new_movable_area_sum + new_filler_area_sum) /
                self.total_place_area)
            logger.info("new target_density %g" % (target_density))

            if pos.is_cuda:
                func = update_pin_offset_cuda.forward
            else:
                func = update_pin_offset_cpp.forward
            # update_pin_offset requires node_size before adjustment
            # update_pin_offset makes sure the absolute pin locations remain the same after inflation
            func(old_node_size_x_movable , old_node_size_y_movable , self.flat_node2pin_start_map,
                 self.flat_node2pin_map, movable_nodes_ratio,
                 self.num_movable_nodes, pin_offset_x, pin_offset_y)
            return adjust_area_flag, adjust_route_area_flag, adjust_pin_area_flag
