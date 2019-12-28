'''
@Author: Jake Gu
@Date: 2019-12-27 21:23:24
@LastEditors  : Jake Gu
@LastEditTime : 2019-12-27 21:33:27
'''
import unittest
import torch
import numpy as np

from dreamplace.ops.adjust_node_area import adjust_node_area

class AdjustNodeAreaUnittest(unittest.TestCase):
    def test_adjust_node_area(self):
        dtype = torch.float32
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
        flat_node2pin_start_map = np.zeros(len(node2pin_map)+1, dtype=np.int32)
        count = 0
        for i in range(len(node2pin_map)):
            flat_node2pin_map[count:count+len(node2pin_map[i])] = node2pin_map[i]
            flat_node2pin_start_map[i] = count
            count += len(node2pin_map[i])
        flat_node2pin_start_map[len(node2pin_map)] = len(pin2node_map)

        flat_node2pin_start_map = torch.from_numpy(flat_node2pin_start_map)
        flat_node2pin_map = torch.from_numpy(flat_node2pin_map)

        xl, xh = 0, 8
        yl, yh = 0, 64
        route_num_bins_x, route_num_bins_y = 8, 8
        pin_num_bins_x, pin_num_bins_y = 16, 16
        
        route_utilization_map = torch.ones([route_num_bins_x, route_num_bins_y]).uniform_(0.5, 2)
        pin_utilization_map = torch.ones([pin_num_bins_x, pin_num_bins_y]).uniform_(0.5, 2)
        

        area_adjust_stop_ratio=0.01
        route_area_adjust_stop_ratio=0.01
        pin_area_adjust_stop_ratio=0.05
        unit_pin_capacity=0.5
        pin_weights = None

        pos = torch.Tensor([[1, 10], [2, 20], [3, 30]]).to(dtype)
        pin_offset_x = torch.Tensor([0.1, 0.2, 0.3, 0.4, 0.5]).to(dtype)
        pin_offset_y = torch.Tensor([0.01, 0.02, 0.03, 0.04, 0.05]).to(dtype)
        node_size_x = torch.Tensor([0.5, 0.5, 0.5]).to(dtype)
        node_size_y = torch.Tensor([0.05, 0.05, 0.05]).to(dtype)

        # test cpu
        adjust_node_area_op = adjust_node_area.AdjustNodeArea(
                 flat_node2pin_map=flat_node2pin_map,
                 flat_node2pin_start_map=flat_node2pin_start_map,
                 pin_weights=pin_weights,  # only one of them needed
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
                 area_adjust_stop_ratio=area_adjust_stop_ratio,
                 route_area_adjust_stop_ratio=route_area_adjust_stop_ratio,
                 pin_area_adjust_stop_ratio=pin_area_adjust_stop_ratio,
                 unit_pin_capacity=unit_pin_capacity,
                 num_threads=8
                 )
        pos_clone = pos.clone().t().contiguous().view(-1)
        node_size_x_clone = node_size_x.clone()
        node_size_y_clone = node_size_y.clone()
        pin_offset_x_clone = pin_offset_x.clone()
        pin_offset_y_clone = pin_offset_y.clone()
        flag1, flag2, flag3 = adjust_node_area_op.forward(
                pos_clone,
                node_size_x_clone, 
                node_size_y_clone,
                pin_offset_x_clone, 
                pin_offset_y_clone,
                route_utilization_map.clone(),
                pin_utilization_map.clone())
        print("Test on CPU. adjust_node_area = ", flag1, flag2, flag3)
        if flag1:
            print(node_size_x_clone, node_size_y_clone, pin_offset_x_clone, pin_offset_y_clone)

        if torch.cuda.device_count():
            # test gpu
            adjust_node_area_op_cuda = adjust_node_area.AdjustNodeArea(
                    flat_node2pin_map=flat_node2pin_map.cuda(),
                    flat_node2pin_start_map=flat_node2pin_start_map.cuda(),
                    pin_weights=pin_weights,  # only one of them needed
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
                    area_adjust_stop_ratio=area_adjust_stop_ratio,
                    route_area_adjust_stop_ratio=route_area_adjust_stop_ratio,
                    pin_area_adjust_stop_ratio=pin_area_adjust_stop_ratio,
                    unit_pin_capacity=unit_pin_capacity
                    )
            pos = pos.t().contiguous().view(-1).cuda()
            node_size_x = node_size_x.cuda()
            node_size_y = node_size_y.cuda()
            pin_offset_x = pin_offset_x.cuda()
            pin_offset_y = pin_offset_y.cuda()
            flag1, flag2, flag3 = adjust_node_area_op_cuda.forward(
                                pos,
                                node_size_x, 
                                node_size_y,
                                pin_offset_x, 
                                pin_offset_y,
                                route_utilization_map.cuda(),
                                pin_utilization_map.cuda())
            print("Test on GPU. adjust_node_area = ", flag1, flag2, flag3)
            if flag1:
                print(node_size_x, node_size_y, pin_offset_x, pin_offset_y)
        

if __name__ == '__main__':
    unittest.main()
