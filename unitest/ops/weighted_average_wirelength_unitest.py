##
# @file   weighted_average_wirelength_unitest.py
# @author Yibo Lin
# @date   Mar 2019
#

import os
import sys
import time
import numpy as np
import unittest
#import pickle
import gzip
if sys.version_info[0] < 3:
    import cPickle as pickle
else:
    import _pickle as pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dreamplace.ops.weighted_average_wirelength import weighted_average_wirelength
sys.path.pop()

import pdb

import torch
from torch.autograd import Function, Variable

def unsorted_segment_sum(pin_x, pin2net_map, num_nets):
    result = np.zeros(num_nets, dtype=pin_x.dtype)
    for i in range(len(pin2net_map)):
        result[pin2net_map[i]] += pin_x[i]
    return result

def build_wirelength(pin_x, pin_y, pin2net_map, net2pin_map, gamma, ignore_net_degree):
    # wirelength cost
    # weighted-average

    # temporily store exp(x)
    scaled_pin_x = pin_x/gamma
    scaled_pin_y = pin_y/gamma

    exp_pin_x = np.exp(scaled_pin_x)
    exp_pin_y = np.exp(scaled_pin_y)
    nexp_pin_x = np.exp(-scaled_pin_x)
    nexp_pin_y = np.exp(-scaled_pin_y)

    # sum of exp(x)
    sum_exp_pin_x = unsorted_segment_sum(exp_pin_x, pin2net_map, len(net2pin_map))
    sum_exp_pin_y = unsorted_segment_sum(exp_pin_y, pin2net_map, len(net2pin_map))
    sum_nexp_pin_x = unsorted_segment_sum(nexp_pin_x, pin2net_map, len(net2pin_map))
    sum_nexp_pin_y = unsorted_segment_sum(nexp_pin_y, pin2net_map, len(net2pin_map))

    # sum of x*exp(x)
    sum_x_exp_pin_x = unsorted_segment_sum(pin_x*exp_pin_x, pin2net_map, len(net2pin_map))
    sum_y_exp_pin_y = unsorted_segment_sum(pin_y*exp_pin_y, pin2net_map, len(net2pin_map))
    sum_x_nexp_pin_x = unsorted_segment_sum(pin_x*nexp_pin_x, pin2net_map, len(net2pin_map))
    sum_y_nexp_pin_y = unsorted_segment_sum(pin_y*nexp_pin_y, pin2net_map, len(net2pin_map))

    sum_exp_pin_x = sum_exp_pin_x
    sum_x_exp_pin_x = sum_x_exp_pin_x

    wl = sum_x_exp_pin_x / sum_exp_pin_x - sum_x_nexp_pin_x / sum_nexp_pin_x \
            + sum_y_exp_pin_y / sum_exp_pin_y - sum_y_nexp_pin_y / sum_nexp_pin_y

    for i in range(len(net2pin_map)):
        if len(net2pin_map[i]) >= ignore_net_degree:
            wl[i] = 0

    wirelength = np.sum(wl)

    return wirelength

class WeightedAverageWirelengthOpTest(unittest.TestCase):
    def test_weighted_average_wirelength_random(self):
        dtype = torch.float32
        pin_pos = np.array([[0.0, 0.0], [1.0, 2.0], [1.5, 0.2], [0.5, 3.1], [0.6, 1.1]], dtype=np.float32)
        net2pin_map = np.array([np.array([0, 4]), np.array([1, 2, 3])])
        pin2net_map = np.zeros(len(pin_pos), dtype=np.int32)
        for net_id, pins in enumerate(net2pin_map):
            for pin in pins:
                pin2net_map[pin] = net_id

        pin_x = pin_pos[:, 0]
        pin_y = pin_pos[:, 1]
        gamma = 0.5
        ignore_net_degree = 4
        pin_mask = np.zeros(len(pin2net_map), dtype=np.uint8)

        # net mask
        net_mask = np.ones(len(net2pin_map), dtype=np.uint8)
        for i in range(len(net2pin_map)):
            if len(net2pin_map[i]) >= ignore_net_degree:
                net_mask[i] = 0

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

        golden_value = np.array([build_wirelength(pin_x, pin_y, pin2net_map, net2pin_map, gamma, ignore_net_degree)])
        print("golden_value = ", golden_value)

        # test cpu
        print(np.transpose(pin_pos))
        pin_pos_var = Variable(torch.tensor(np.transpose(pin_pos), dtype=dtype).reshape([-1]), requires_grad=True)
        #pin_pos_var = torch.nn.Parameter(torch.from_numpy(np.transpose(pin_pos)).reshape([-1]))
        print(pin_pos_var)
        # clone is very important, because the custom op cannot deep copy the data
        custom = weighted_average_wirelength.WeightedAverageWirelength(
                flat_netpin=torch.from_numpy(flat_net2pin_map),
                netpin_start=torch.from_numpy(flat_net2pin_start_map),
                pin2net_map=torch.from_numpy(pin2net_map),
                net_mask=torch.from_numpy(net_mask),
                pin_mask=torch.from_numpy(pin_mask),
                gamma=torch.tensor(gamma, dtype=dtype),
                algorithm='net-by-net'
                )
        result = custom.forward(pin_pos_var)
        print("custom = ", result)
        result.backward()
        grad = pin_pos_var.grad.clone()
        print("custom_grad = ", grad)

        np.testing.assert_allclose(result.data.numpy(), golden_value, atol=1e-6)

        # test gpu
        pin_pos_var.grad.zero_()
        custom_cuda = weighted_average_wirelength.WeightedAverageWirelength(
                flat_netpin=Variable(torch.from_numpy(flat_net2pin_map)).cuda(),
                netpin_start=Variable(torch.from_numpy(flat_net2pin_start_map)).cuda(),
                pin2net_map=torch.from_numpy(pin2net_map).cuda(),
                net_mask=torch.from_numpy(net_mask).cuda(),
                pin_mask=torch.from_numpy(pin_mask).cuda(),
                gamma=torch.tensor(gamma, dtype=dtype).cuda(),
                algorithm='net-by-net'
                )
        result_cuda = custom_cuda.forward(pin_pos_var.cuda())
        print("custom_cuda_result = ", result_cuda.data.cpu())
        result_cuda.backward()
        grad_cuda = pin_pos_var.grad.clone()
        print("custom_grad_cuda = ", grad_cuda.data.cpu())

        np.testing.assert_allclose(result_cuda.data.cpu().numpy(), golden_value, atol=1e-6)
        np.testing.assert_allclose(grad_cuda.data.cpu().numpy(), grad.data.numpy(), rtol=1e-6, atol=1e-7)

        # test gpu atomic
        pin_pos_var.grad.zero_()
        custom_cuda = weighted_average_wirelength.WeightedAverageWirelength(
                flat_netpin=Variable(torch.from_numpy(flat_net2pin_map)).cuda(),
                netpin_start=Variable(torch.from_numpy(flat_net2pin_start_map)).cuda(),
                pin2net_map=torch.from_numpy(pin2net_map).cuda(),
                net_mask=torch.from_numpy(net_mask).cuda(),
                pin_mask=torch.from_numpy(pin_mask).cuda(),
                gamma=torch.tensor(gamma, dtype=dtype).cuda(),
                algorithm='atomic'
                )
        result_cuda = custom_cuda.forward(pin_pos_var.cuda())
        print("custom_cuda_result atomic = ", result_cuda.data.cpu())
        result_cuda.backward()
        grad_cuda = pin_pos_var.grad.clone()
        print("custom_grad_cuda atomic = ", grad_cuda.data.cpu())

        np.testing.assert_allclose(result_cuda.data.cpu().numpy(), golden_value, atol=1e-6)
        np.testing.assert_allclose(grad_cuda.data.cpu().numpy(), grad.data.numpy(), rtol=1e-6, atol=1e-7)

        # test gpu sparse
        pin_pos_var.grad.zero_()
        custom_cuda = weighted_average_wirelength.WeightedAverageWirelength(
                flat_netpin=Variable(torch.from_numpy(flat_net2pin_map)).cuda(),
                netpin_start=Variable(torch.from_numpy(flat_net2pin_start_map)).cuda(),
                pin2net_map=torch.from_numpy(pin2net_map).cuda(),
                net_mask=torch.from_numpy(net_mask).cuda(),
                pin_mask=torch.from_numpy(pin_mask).cuda(),
                gamma=torch.tensor(gamma, dtype=dtype).cuda(),
                algorithm='sparse'
                )
        result_cuda = custom_cuda.forward(pin_pos_var.cuda())
        print("custom_cuda_result sparse = ", result_cuda.data.cpu())
        result_cuda.backward()
        grad_cuda = pin_pos_var.grad.clone()
        print("custom_grad_cuda sparse = ", grad_cuda.data.cpu())

        np.testing.assert_allclose(result_cuda.data.cpu().numpy(), golden_value, atol=1e-6)
        np.testing.assert_allclose(grad_cuda.data.cpu().numpy(), grad.data.numpy(), rtol=1e-6, atol=1e-7)

def eval_runtime(design):
    with gzip.open("../../benchmarks/ispd2005/wirelength/%s_wirelength.pklz" % (design), "rb") as f:
        if sys.version_info[0] < 3:
            flat_net2pin_map, flat_net2pin_start_map, pin2net_map, net_mask, pin_mask, gamma = pickle.load(f)
        else:
            flat_net2pin_map, flat_net2pin_start_map, pin2net_map, net_mask, pin_mask, gamma = pickle.load(f, encoding='bytes')

    dtype = torch.float64
    pin_pos_var = Variable(torch.empty(len(pin2net_map)*2, dtype=dtype).uniform_(0, 1000), requires_grad=True).cuda()
    custom_net_by_net = weighted_average_wirelength.WeightedAverageWirelength(
            flat_netpin=Variable(torch.from_numpy(flat_net2pin_map)).cuda(),
            netpin_start=Variable(torch.from_numpy(flat_net2pin_start_map)).cuda(),
            pin2net_map=torch.from_numpy(pin2net_map).cuda(),
            net_mask=torch.from_numpy(net_mask).cuda(),
            pin_mask=torch.from_numpy(pin_mask).cuda(),
            gamma=gamma.clone().detach().to(dtype).cuda(),
            algorithm='net-by-net'
            )
    custom_atomic = weighted_average_wirelength.WeightedAverageWirelength(
            flat_netpin=Variable(torch.from_numpy(flat_net2pin_map)).cuda(),
            netpin_start=Variable(torch.from_numpy(flat_net2pin_start_map)).cuda(),
            pin2net_map=torch.from_numpy(pin2net_map).cuda(),
            net_mask=torch.from_numpy(net_mask).cuda(),
            pin_mask=torch.from_numpy(pin_mask).cuda(),
            gamma=gamma.clone().detach().to(dtype).cuda(),
            algorithm='atomic'
            )
    custom_sparse = weighted_average_wirelength.WeightedAverageWirelength(
            flat_netpin=Variable(torch.from_numpy(flat_net2pin_map)).cuda(),
            netpin_start=Variable(torch.from_numpy(flat_net2pin_start_map)).cuda(),
            pin2net_map=torch.from_numpy(pin2net_map).cuda(),
            net_mask=torch.from_numpy(net_mask).cuda(),
            pin_mask=torch.from_numpy(pin_mask).cuda(),
            gamma=gamma.clone().detach().to(dtype).cuda(),
            algorithm='sparse'
            )

    torch.cuda.synchronize()
    iters = 100
    # tt = time.time()
    # for i in range(iters):
    #     result = custom_net_by_net.forward(pin_pos_var)
    #     result.backward()
    # torch.cuda.synchronize()
    # print("custom_net_by_net takes %.3f ms" % ((time.time()-tt)/iters*1000))

    tt = time.time()
    for i in range(iters):
        result = custom_atomic.forward(pin_pos_var)
        result.backward()
    torch.cuda.synchronize()
    print("custom_atomic takes %.3f ms" % ((time.time()-tt)/iters*1000))

    # tt = time.time()
    # for i in range(iters):
    #     result = custom_sparse.forward(pin_pos_var)
    #     result.backward()
    # torch.cuda.synchronize()
    # print("custom_sparse takes %.3f ms" % ((time.time()-tt)/iters*1000))


if __name__ == '__main__':
    #unittest.main()

    design = sys.argv[1]
    eval_runtime(design)
