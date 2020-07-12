##
# @file   adjust_node_area_unitest.py
# @author Zixuan Jiang, Jiaqi Gu
# @date   Dec 2019
#

import os
import sys
import unittest
import torch
import numpy as np

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
from dreamplace.ops.adjust_node_area import adjust_node_area
sys.path.pop()


class AdjustNodeAreaUnittest(unittest.TestCase):
    def test_adjust_node_area(self):
        dtype = torch.float32
        pos = torch.Tensor([[1, 10], [2, 20], [3, 30]]).to(dtype)
        pin_offset_x = torch.Tensor([0.1, 0.2, 0.3, 0.4, 0.5]).to(dtype)
        pin_offset_y = torch.Tensor([0.01, 0.02, 0.03, 0.04, 0.05]).to(dtype)
        node_size_x = torch.Tensor([0.5, 0.5, 0.5]).to(dtype)
        node_size_y = torch.Tensor([0.05, 0.05, 0.05]).to(dtype)

        node2pin_map = np.array([np.array([0, 4]), np.array([1, 2, 3])])
        num_movable_nodes = len(node2pin_map)
        num_filler_nodes = 1
        # assume no terminals

        num_pins = 0
        for pins in node2pin_map:
            num_pins += len(pins)
        pin2node_map = np.zeros(num_pins, dtype=np.int32)
        for node_id, pins in enumerate(node2pin_map):
            for pin in pins:
                pin2node_map[pin] = node_id

        # construct flat_node2pin_map and flat_node2pin_start_map
        flat_node2pin_map = np.zeros(num_pins, dtype=np.int32)
        # starting index in nodepin map for each node, length of #nodes+1, the last entry is #pins
        flat_node2pin_start_map = np.zeros(len(node2pin_map) + 1,
                                           dtype=np.int32)
        count = 0
        for i in range(len(node2pin_map)):
            flat_node2pin_map[count:count +
                              len(node2pin_map[i])] = node2pin_map[i]
            flat_node2pin_start_map[i] = count
            count += len(node2pin_map[i])
        flat_node2pin_start_map[len(node2pin_map)] = len(pin2node_map)

        flat_node2pin_start_map = torch.from_numpy(flat_node2pin_start_map)
        flat_node2pin_map = torch.from_numpy(flat_node2pin_map)

        xl, xh = 0, 8
        yl, yh = 0, 64
        route_num_bins_x, route_num_bins_y = 8, 8
        pin_num_bins_x, pin_num_bins_y = 16, 16

        total_place_area = (xh - xl) * (yh - yl)
        total_whitespace_area = (
            total_place_area - (node_size_x[:num_movable_nodes] *
                                node_size_y[:num_movable_nodes]).sum()).item()

        route_utilization_map = torch.ones(
            [route_num_bins_x, route_num_bins_y]).uniform_(0.5, 2)
        pin_utilization_map = torch.ones([pin_num_bins_x,
                                          pin_num_bins_y]).uniform_(0.5, 2)

        area_adjust_stop_ratio = 0.01
        route_area_adjust_stop_ratio = 0.01
        pin_area_adjust_stop_ratio = 0.05
        unit_pin_capacity = 0.5
        pin_weights = None

        max_route_opt_adjust_rate = 3.0
        max_pin_opt_adjust_rate = 10.0

        target_density = torch.Tensor([0.9])

        # test cpu
        adjust_node_area_op = adjust_node_area.AdjustNodeArea(
            flat_node2pin_map=flat_node2pin_map,
            flat_node2pin_start_map=flat_node2pin_start_map,
            pin_weights=pin_weights,
            xl=xl,
            yl=yl,
            xh=xh,
            yh=yh,
            num_movable_nodes=num_movable_nodes,
            num_filler_nodes=num_filler_nodes,
            route_num_bins_x=route_num_bins_x,
            route_num_bins_y=route_num_bins_y,
            pin_num_bins_x=pin_num_bins_x,
            pin_num_bins_y=pin_num_bins_y,
            total_place_area=total_place_area,
            total_whitespace_area=total_whitespace_area,
            max_route_opt_adjust_rate=max_route_opt_adjust_rate,
            max_pin_opt_adjust_rate=max_pin_opt_adjust_rate,
            area_adjust_stop_ratio=area_adjust_stop_ratio,
            route_area_adjust_stop_ratio=route_area_adjust_stop_ratio,
            pin_area_adjust_stop_ratio=pin_area_adjust_stop_ratio,
            unit_pin_capacity=unit_pin_capacity)

        pos_cpu = pos.clone().t().contiguous().view(-1)
        node_size_x_cpu = node_size_x.clone()
        node_size_y_cpu = node_size_y.clone()
        pin_offset_x_cpu = pin_offset_x.clone()
        pin_offset_y_cpu = pin_offset_y.clone()
        flag1_cpu, flag2_cpu, flag3_cpu = adjust_node_area_op.forward(
            pos_cpu, node_size_x_cpu, node_size_y_cpu, pin_offset_x_cpu,
            pin_offset_y_cpu, target_density, route_utilization_map.clone(),
            pin_utilization_map.clone())

        if torch.cuda.device_count():
            adjust_node_area_op_cuda = adjust_node_area.AdjustNodeArea(
                flat_node2pin_map=flat_node2pin_map.cuda(),
                flat_node2pin_start_map=flat_node2pin_start_map.cuda(),
                pin_weights=pin_weights,
                xl=xl,
                yl=yl,
                xh=xh,
                yh=yh,
                num_movable_nodes=num_movable_nodes,
                num_filler_nodes=num_filler_nodes,
                route_num_bins_x=route_num_bins_x,
                route_num_bins_y=route_num_bins_y,
                pin_num_bins_x=pin_num_bins_x,
                pin_num_bins_y=pin_num_bins_y,
                total_place_area=total_place_area,
                total_whitespace_area=total_whitespace_area,
                max_route_opt_adjust_rate=max_route_opt_adjust_rate,
                max_pin_opt_adjust_rate=max_pin_opt_adjust_rate,
                area_adjust_stop_ratio=area_adjust_stop_ratio,
                route_area_adjust_stop_ratio=route_area_adjust_stop_ratio,
                pin_area_adjust_stop_ratio=pin_area_adjust_stop_ratio,
                unit_pin_capacity=unit_pin_capacity)
            pos_cuda = pos.t().contiguous().view(-1).cuda()
            node_size_x_cuda = node_size_x.cuda()
            node_size_y_cuda = node_size_y.cuda()
            pin_offset_x_cuda = pin_offset_x.cuda()
            pin_offset_y_cuda = pin_offset_y.cuda()
            flag1_cuda, flag2_cuda, flag3_cuda = adjust_node_area_op_cuda.forward(
                pos_cuda, node_size_x_cuda, node_size_y_cuda,
                pin_offset_x_cuda, pin_offset_y_cuda, target_density.cuda(),
                route_utilization_map.cuda(), pin_utilization_map.cuda())

            assert (flag1_cpu == flag1_cuda) and \
                   (flag2_cpu == flag2_cuda) and \
                   (flag3_cpu == flag3_cuda), "the flags via CPU and GPU are different"

            if flag1_cpu:
                assert torch.allclose(pos_cuda.cpu(), pos_cpu) and \
                    torch.allclose(node_size_x_cuda.cpu(), node_size_x_cpu) and \
                    torch.allclose(node_size_y_cuda.cpu(), node_size_y_cpu) and \
                    torch.allclose(pin_offset_x_cuda.cpu(), pin_offset_x_cpu) and \
                    torch.allclose(pin_offset_y_cuda.cpu(), pin_offset_y_cpu), \
                    "the results via CPU and GPU are different"


if __name__ == '__main__':
    unittest.main()
