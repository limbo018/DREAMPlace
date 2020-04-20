##
# @file   k_reorder_unitest.py
# @author Yibo Lin
# @date   Mar 2019
#

import os
import sys
import math
import numpy as np
import unittest
import cairocffi as cairo
import time
import math

import torch
from torch.autograd import Function, Variable
from scipy.optimize import linear_sum_assignment
import gzip
if sys.version_info[0] < 3:
    import cPickle as pickle
else:
    import _pickle as pickle

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from dreamplace.ops.k_reorder import k_reorder
#from dreamplace.ops.k_reorder_taskflow import k_reorder_taskflow
sys.path.pop()

import pdb


def test_ispd2005(design, device_str):
    print("run design %s" % (design))
    with gzip.open(design, "rb") as f:
        if sys.version_info[0] < 3:
            data_collections = pickle.load(f)
        else:
            data_collections = pickle.load(f, encoding='bytes')
        node_size_x = data_collections[0]
        node_size_y = data_collections[1]
        flat_net2pin_map = data_collections[2]
        flat_net2pin_start_map = data_collections[3]
        pin2net_map = data_collections[4]
        flat_node2pin_map = data_collections[5]
        flat_node2pin_start_map = data_collections[6]
        pin2node_map = data_collections[7]
        pin_offset_x = data_collections[8]
        pin_offset_y = data_collections[9]
        net_mask_ignore_large_degrees = data_collections[10]
        xl = data_collections[11]
        yl = data_collections[12]
        xh = data_collections[13]
        yh = data_collections[14]
        site_width = data_collections[15]
        row_height = data_collections[16]
        num_bins_x = data_collections[17]
        num_bins_y = data_collections[18]
        num_movable_nodes = data_collections[19]
        num_terminal_NIs = data_collections[20]
        num_filler_nodes = data_collections[21]
        pos = data_collections[22]

        #net_mask = net_mask_ignore_large_degrees
        net_mask = np.ones_like(net_mask_ignore_large_degrees)
        for i in range(1, len(flat_net2pin_start_map)):
            degree = flat_net2pin_start_map[i] - flat_net2pin_start_map[i - 1]
            if degree > 100:
                net_mask[i - 1] = 0
        net_mask = torch.from_numpy(net_mask)

        #max_node_degree = 0
        #for i in range(1, len(flat_node2pin_start_map)):
        #    if i <= num_movable_nodes:
        #        max_node_degree = max(max_node_degree, flat_node2pin_start_map[i]-flat_node2pin_start_map[i-1])
        #print("max node degree %d" % (max_node_degree))

        device = torch.device(device_str)

        print("num_movable_nodes %d, num_nodes %d" %
              (num_movable_nodes,
               node_size_x.numel() - num_filler_nodes - num_terminal_NIs))

        torch.set_num_threads(20)
        # test cpu/cuda
        custom = k_reorder.KReorder(
            node_size_x=node_size_x.float().to(device),
            node_size_y=node_size_y.float().to(device),
            flat_net2pin_map=flat_net2pin_map.to(device),
            flat_net2pin_start_map=flat_net2pin_start_map.to(device),
            pin2net_map=pin2net_map.to(device),
            flat_node2pin_map=flat_node2pin_map.to(device),
            flat_node2pin_start_map=flat_node2pin_start_map.to(device),
            pin2node_map=pin2node_map.to(device),
            pin_offset_x=pin_offset_x.float().to(device),
            pin_offset_y=pin_offset_y.float().to(device),
            net_mask=net_mask.to(device),
            xl=xl,
            yl=yl,
            xh=xh,
            yh=yh,
            site_width=site_width,
            row_height=row_height,
            num_bins_x=num_bins_x,
            num_bins_y=num_bins_y,
            num_movable_nodes=num_movable_nodes,
            num_terminal_NIs=num_terminal_NIs,
            num_filler_nodes=num_filler_nodes,
            K=4,
            max_iters=2)

        result = custom(pos.float().to(device))
        #print("initial result = ", np.concatenate([xx, yy]))
        #print("custom_result = ", result)

        #with gzip.open("bigblue2.dp.swap.pklz", "wb") as f:
        #    pickle.dump((node_size_x.cpu(), node_size_y.cpu(),
        #        flat_net2pin_map.cpu(), flat_net2pin_start_map.cpu(), pin2net_map.cpu(),
        #        flat_node2pin_map.cpu(), flat_node2pin_start_map.cpu(), pin2node_map.cpu(),
        #        pin_offset_x.cpu(), pin_offset_y.cpu(),
        #        net_mask_ignore_large_degrees.cpu(),
        #        xl, yl, xh, yh,
        #        site_width, row_height,
        #        num_bins_x, num_bins_y,
        #        num_movable_nodes,
        #        num_terminal_NIs,
        #        num_filler_nodes,
        #        result.cpu()
        #        ), f)
        #    exit()


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("usage: python script.py design.pklz cpu|cuda")
    else:
        design = sys.argv[1]
        device_str = sys.argv[2]
        test_ispd2005(design, device_str)
