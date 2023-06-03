##
# @file   rmst_wl_unitest.py
# @author Yibo Lin
# @date   Mar 2019
#

import sys
import os
import numpy as np
import unittest

import torch
from torch.autograd import Function, Variable

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
from dreamplace.ops.rmst_wl import rmst_wl
sys.path.pop()

import pdb


class RmstWLOpTest(unittest.TestCase):
    def test_rmst_wlRandom(self):
        dtype = np.float64
        pin_pos = np.array(
            [[0.0, 0.0], [1.0, 2.0], [1.5, 0.2], [0.5, 3.1], [0.6, 1.1]],
            dtype=dtype)
        net2pin_map = np.array([np.array([0, 4]), np.array([1, 2, 3])])

        pin_x = pin_pos[:, 0]
        pin_y = pin_pos[:, 1]

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

        # test cpu
        print(np.transpose(pin_pos))
        pin_pos_var = Variable(torch.from_numpy(pin_pos))
        print(pin_pos_var)
        # clone is very important, because the custom op cannot deep copy the data
        pin_pos_var = torch.t(pin_pos_var).contiguous()
        #pdb.set_trace()
        project_path = os.path.abspath(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        POWVFILE = os.path.join(project_path, "thirdparty/flute/POWV9.dat")
        POSTFILE = os.path.join(project_path, "thirdparty/flute/POST9.dat")
        custom = rmst_wl.RmstWL(torch.from_numpy(flat_net2pin_map),
                                torch.from_numpy(flat_net2pin_start_map),
                                torch.tensor(len(flat_net2pin_map)),
                                POWVFILE=POWVFILE,
                                POSTFILE=POSTFILE)
        rmst_wl_value = custom.forward(pin_pos_var, read_lut_flag=True)
        print("rmst_wl_value = ", rmst_wl_value.data.numpy())
        rmst_wl_value = custom.forward(pin_pos_var, read_lut_flag=False)
        print("rmst_wl_value = ", rmst_wl_value.data.numpy())
        #np.testing.assert_allclose(rmst_wl_value.data.numpy(), golden_value)


if __name__ == '__main__':
    unittest.main()
