##
# @file   pin_utilization_unitest.py
# @author Zixuan Jiang, Jiaqi Gu
# @date   Dec 2019
#

import os
import sys
import unittest
import torch
import numpy as np
import math

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
from dreamplace.ops.pin_utilization import pin_utilization
sys.path.pop()


class PinUtilizationUnittest(unittest.TestCase):
    def test_pin_utilization(self):
        # the data of nodes are from unitest/ops/pin_pos_unitest.py
        dtype = torch.float32

        pos = torch.Tensor([[1, 10], [2, 20], [3, 30]]).to(dtype)
        node2pin_map = np.array([np.array([0, 4]), np.array([1, 2, 3])])
        num_movable_nodes = len(node2pin_map)
        num_filler_nodes = 1
        num_nodes = num_movable_nodes + num_filler_nodes

        num_pins = 0
        for pins in node2pin_map:
            num_pins += len(pins)
        pin2node_map = np.zeros(num_pins, dtype=np.int32)
        for node_id, pins in enumerate(node2pin_map):
            for pin in pins:
                pin2node_map[pin] = node_id

        # construct flat_node2pin_map and flat_node2pin_start_map
        # flat nodepin map, length of #pins
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

        node_size_x = torch.Tensor([3, 3, 3]).to(dtype)
        node_size_y = torch.Tensor([6, 6, 3]).to(dtype)
        xl, xh = 0, 8
        yl, yh = 0, 64
        num_bins_x, num_bins_y = 2, 16
        bin_size_x = (xh - xl) / num_bins_x
        bin_size_y = (yh - yl) / num_bins_y

        pin_weights = None
        unit_pin_capacity = 0.5
        pin_stretch_ratio = math.sqrt(2)
        deterministic_flag = 1

        # test cpu
        pin_utilization_op = pin_utilization.PinUtilization(
            node_size_x=node_size_x,
            node_size_y=node_size_y,
            pin_weights=pin_weights,
            flat_node2pin_start_map=flat_node2pin_start_map,
            xl=xl,
            xh=xh,
            yl=yl,
            yh=yh,
            num_movable_nodes=num_movable_nodes,
            num_filler_nodes=num_filler_nodes,
            num_bins_x=num_bins_x,
            num_bins_y=num_bins_y,
            unit_pin_capacity=unit_pin_capacity,
            pin_stretch_ratio=pin_stretch_ratio,
            deterministic_flag=deterministic_flag)

        result_cpu = pin_utilization_op.forward(pos.t().contiguous().view(-1))
        print("Test on CPU. pin_utilization map = ", result_cpu)

        if torch.cuda.device_count():
            # test gpu
            pin_utilization_op_cuda = pin_utilization.PinUtilization(
                node_size_x=node_size_x.cuda(),
                node_size_y=node_size_y.cuda(),
                pin_weights=pin_weights,
                flat_node2pin_start_map=flat_node2pin_start_map.cuda(),
                xl=xl,
                xh=xh,
                yl=yl,
                yh=yh,
                num_movable_nodes=num_movable_nodes,
                num_filler_nodes=num_filler_nodes,
                num_bins_x=num_bins_x,
                num_bins_y=num_bins_y,
                unit_pin_capacity=unit_pin_capacity,
                pin_stretch_ratio=pin_stretch_ratio,
                deterministic_flag=deterministic_flag)

            result_cuda = pin_utilization_op_cuda.forward(
                pos.t().contiguous().view(-1).cuda())
            print("Test on GPU. pin_utilization map = ", result_cuda)
            np.testing.assert_allclose(result_cpu, result_cuda.cpu())


if __name__ == '__main__':
    unittest.main()
