##
# @file   rudy_unitest.py
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
from dreamplace.ops.rudy import rudy
sys.path.pop()


class RudyUnittest(unittest.TestCase):
    def test_rudy(self):
        # the data of net and pin are from unitest/ops/weighted_average_wirelength_unitest.py
        dtype = torch.float32
        pin_pos = torch.Tensor([[0.0, 0.0], [1.0, 2.0], [1.5, 0.2], [0.5, 3.1],
                                [0.6, 1.1]]).to(dtype)
        net2pin_map = np.array([np.array([0, 4]), np.array([1, 2, 3])])
        net_weights = torch.Tensor([1, 2]).to(dtype)

        # construct flat_net2pin_map and flat_net2pin_start_map
        # flat netpin map, length of #pins
        flat_net2pin_map = np.zeros(len(pin_pos), dtype=np.int32)
        # starting index in netpin map for each net, length of #nets+1, the last entry is #pins
        flat_net2pin_start_map = np.zeros(len(net2pin_map) + 1, dtype=np.int32)
        count = 0
        for i in range(len(net2pin_map)):
            flat_net2pin_map[count:count +
                             len(net2pin_map[i])] = net2pin_map[i]
            flat_net2pin_start_map[i] = count
            count += len(net2pin_map[i])
        flat_net2pin_start_map[len(net2pin_map)] = len(pin_pos)

        print("flat_net2pin_map = ", flat_net2pin_map)
        print("flat_net2pin_start_map = ", flat_net2pin_start_map)
        flat_net2pin_map = torch.from_numpy(flat_net2pin_map)
        flat_net2pin_start_map = torch.from_numpy(flat_net2pin_start_map)

        # parameters for this test
        xl, xh = 0.0, 2.0
        yl, yh = 0.0, 4.0
        num_bins_x = 8
        num_bins_y = 8
        unit_horizontal_capacity = 0.1
        unit_vertical_capacity = 0.2

        # test cpu
        rudy_op = rudy.Rudy(netpin_start=flat_net2pin_start_map,
                            flat_netpin=flat_net2pin_map,
                            net_weights=net_weights,
                            xl=xl,
                            xh=xh,
                            yl=yl,
                            yh=yh,
                            num_bins_x=num_bins_x,
                            num_bins_y=num_bins_y,
                            unit_horizontal_capacity=unit_horizontal_capacity,
                            unit_vertical_capacity=unit_vertical_capacity,
                            deterministic_flag=1)

        result_cpu = rudy_op.forward(pin_pos.t().contiguous().view(-1))
        print("Test on CPU. rudy map = ", result_cpu)

        if torch.cuda.device_count():
            # test gpu
            rudy_op_cuda = rudy.Rudy(
                netpin_start=flat_net2pin_start_map.cuda(),
                flat_netpin=flat_net2pin_map.cuda(),
                net_weights=net_weights.cuda(),
                xl=xl,
                xh=xh,
                yl=yl,
                yh=yh,
                num_bins_x=num_bins_x,
                num_bins_y=num_bins_y,
                unit_horizontal_capacity=unit_horizontal_capacity,
                unit_vertical_capacity=unit_vertical_capacity,
                deterministic_flag=1)

            result_cuda = rudy_op_cuda.forward(
                pin_pos.t().contiguous().view(-1).cuda())
            print("Test on GPU. rudy map = ", result_cuda)

            np.testing.assert_allclose(result_cpu, result_cuda.cpu())


if __name__ == '__main__':
    unittest.main()
