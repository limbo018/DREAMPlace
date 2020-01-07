import math
import torch
from torch import nn
from torch.autograd import Function

import dreamplace.ops.pin_utilization.pin_utilization_cpp as pin_utilization_cpp
try:
    import dreamplace.ops.pin_utilization.pin_utilization_cuda as pin_utilization_cuda 
except:
    pass

class PinUtilization(nn.Module):
    def __init__(self,
            node_size_x, node_size_y,
            pin_weights, 
            flat_node2pin_start_map,
            xl, xh, yl, yh,
            num_movable_nodes, num_filler_nodes, 
            num_bins_x, num_bins_y,
            unit_pin_capacity,
            pin_stretch_ratio,
            max_pin_opt_adjust_rate,
            num_threads=8
            ):
        super(PinUtilization, self).__init__()
        self.node_size_x = node_size_x
        self.node_size_y = node_size_y
        self.xl = xl
        self.xh = xh
        self.yl = yl
        self.yh = yh
        self.num_nodes = len(node_size_x)
        self.num_movable_nodes = num_movable_nodes
        self.num_filler_nodes = num_filler_nodes
        self.num_physical_nodes = num_nodes - num_filler_nodes
        self.num_bins_x = num_bins_x
        self.num_bins_y = num_bins_y
        self.bin_size_x = (xh - xl) / num_bins_x
        self.bin_size_y = (yh - yl) / num_bins_y
        self.num_threads = num_threads

        self.unit_pin_capacity = unit_pin_capacity
        # maximum and minimum instance area adjustment rate for routability optimization
        self.max_pin_opt_adjust_rate = max_pin_opt_adjust_rate
        self.min_pin_opt_adjust_rate = 1.0 / max_pin_opt_adjust_rate

        # to make the pin density map smooth, we stretch each pin to a ratio of the pin utilization bin
        self.half_node_size_stretch_x = 0.5 * self.node_size_x[:self.num_physical_nodes].clamp(min=self.bin_size_x * pin_stretch_ratio) 
        self.half_node_size_stretch_y = 0.5 * self.node_size_y[:self.num_physical_nodes].clamp(min=self.bin_size_y * pin_stretch_ratio)
        # for each physical node, we use the pin counts as the weights
        if pin_weights is not None:
            self.pin_weights = pin_weights 
        elif flat_node2pin_start_map is not None: 
            self.pin_weights = (flat_node2pin_start_map[1:self.num_physical_nodes + 1] - flat_node2pin_start_map[:self.num_physical_nodes]).to(self.node_size_x.dtype)
        else:
            assert "either pin_weights or flat_node2pin_start_map is required"

    def forward(self, pos):
        if pos.is_cuda:
            output = pin_utilization_cuda.pin_utilization(
                    pos,
                    self.node_size_x,
                    self.node_size_y,
                    self.half_node_size_stretch_x,
                    self.half_node_size_stretch_y,
                    self.pin_weights,
                    self.xl,
                    self.yl,
                    self.xh,
                    self.yh,
                    self.bin_size_x,
                    self.bin_size_y,
                    self.unit_pin_capacity,
                    self.max_pin_opt_adjust_rate,
                    self.min_pin_opt_adjust_rate,
                    self.num_physical_nodes,
                    self.num_bins_x,
                    self.num_bins_y
                    )
        else:
            output = pin_utilization_cpp.pin_utilization(
                    pos,
                    self.node_size_x,
                    self.node_size_y,
                    self.half_node_size_stretch_x,
                    self.half_node_size_stretch_y,
                    self.pin_weights,
                    self.xl,
                    self.yl,
                    self.xh,
                    self.yh,
                    self.bin_size_x,
                    self.bin_size_y,
                    self.unit_pin_capacity,
                    self.max_pin_opt_adjust_rate,
                    self.min_pin_opt_adjust_rate,
                    self.num_physical_nodes,
                    self.num_bins_x,
                    self.num_bins_y,
                    self.num_threads
                    )

        return output
