import sys
import numpy as np
import unittest

import torch
from torch.autograd import Function, Variable
if sys.version_info[0] < 3: 
    import src.hpwl as hpwl
else:
    from .src import hpwl
import pdb 

"""
return hpwl of a net 
"""
def net_hpwl(x, y, net2pin_map, net_id): 
    pins = net2pin_map[net_id]
    hpwl_x = np.amax(x[pins]) - np.amin(x[pins])
    hpwl_y = np.amax(y[pins]) - np.amin(y[pins])

    return hpwl_x+hpwl_y

"""
return hpwl of all nets
"""
def all_hpwl(x, y, net2pin_map):
    wl = 0
    for net_id in range(len(net2pin_map)):
        wl += net_hpwl(x, y, net2pin_map, net_id)
    return wl 

class HPWLOpTest(unittest.TestCase):
    def test_hpwlRandom(self):
        pin_pos = np.array([[0.0, 0.0], [1.0, 2.0], [1.5, 0.2], [0.5, 3.1], [0.6, 1.1]], dtype=np.float32)
        net2pin_map = np.array([np.array([0, 4]), np.array([1, 2, 3])])

        pin_x = pin_pos[:, 0]
        pin_y = pin_pos[:, 1]

        # construct flat_net2pin_map and flat_net2pin_start_map
        # flat netpin map, length of #pins
        flat_net2pin_map = np.zeros(len(pin_pos), dtype=np.int32)
        # starting index in netpin map for each net, length of #nets+1, the last entry is #pins  
        flat_net2pin_start_map = np.zeros(len(net2pin_map)+1, dtype=np.int32)
        count = 0
        for i in range(len(net2pin_map)):
            flat_net2pin_map[count:count+len(net2pin_map[i])] = net2pin_map[i]
            flat_net2pin_start_map[i] = count 
            count += len(net2pin_map[i])
        flat_net2pin_start_map[len(net2pin_map)] = len(pin_pos)
        
        print("flat_net2pin_map = ", flat_net2pin_map)
        print("flat_net2pin_start_map = ", flat_net2pin_start_map)

        # construct pin2net_map 
        pin2net_map = np.zeros(len(pin_pos), dtype=np.int32)
        for i in range(len(net2pin_map)):
            for pin_id in net2pin_map[i]:
                pin2net_map[pin_id] = i 
        print("pin2net_map = ", pin2net_map)

        # net degrees 
        net_degrees = np.array([len(net2pin) for net2pin in net2pin_map])

        golden_value = all_hpwl(pin_x, pin_y, net2pin_map)
        print("golden_value = ", golden_value)

        # test cpu 
        print(np.transpose(pin_pos))
        pin_pos_var = Variable(torch.from_numpy(pin_pos))
        print(pin_pos_var)
        # clone is very important, because the custom op cannot deep copy the data 
        pin_pos_var = torch.t(pin_pos_var).contiguous()
        #pdb.set_trace()
        custom = hpwl.HPWL(
                torch.from_numpy(flat_net2pin_map), 
                torch.from_numpy(flat_net2pin_start_map),
                torch.tensor(len(flat_net2pin_map))
                )
        hpwl_value = custom.forward(pin_pos_var)
        print("hpwl_value = ", hpwl_value.data.numpy())
        np.testing.assert_allclose(hpwl_value.data.numpy(), golden_value)

        # test gpu 
        custom_cuda = hpwl.HPWL(
                torch.from_numpy(flat_net2pin_map).cuda(), 
                torch.from_numpy(flat_net2pin_start_map).cuda(),
                torch.tensor(len(flat_net2pin_map)).cuda()
                )
        hpwl_value = custom_cuda.forward(pin_pos_var.cuda())
        print("hpwl_value cuda = ", hpwl_value.data.cpu().numpy())
        np.testing.assert_allclose(hpwl_value.data.cpu().numpy(), golden_value)

        # test atomic cpu 
        net_mask = (net_degrees <= np.amax(net_degrees)).astype(np.uint8)
        print("net_mask = ", net_mask)
        custom_atomic = hpwl.HPWLAtomic(
                torch.from_numpy(pin2net_map), 
                torch.from_numpy(net_mask)
                )
        hpwl_value = custom_atomic.forward(pin_pos_var)
        print("hpwl_value atomic = ", hpwl_value.data.numpy())
        np.testing.assert_allclose(hpwl_value.data.numpy(), golden_value)

        # test atomic gpu 
        custom_cuda_atomic = hpwl.HPWLAtomic(
                torch.from_numpy(pin2net_map).cuda(), 
                torch.from_numpy(net_mask).cuda()
                )
        hpwl_value = custom_cuda_atomic.forward(pin_pos_var.cuda())
        print("hpwl_value cuda atomic = ", hpwl_value.data.cpu().numpy())
        np.testing.assert_allclose(hpwl_value.data.cpu().numpy(), golden_value)

if __name__ == '__main__':
    unittest.main()
