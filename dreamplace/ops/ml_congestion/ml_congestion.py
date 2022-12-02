##
# @file   ml_congestion.py
# @author Yibo Lin
# @date   Oct 2022
#

import math
import torch
from torch import nn
from torch.autograd import Function
import matplotlib.pyplot as plt
import pdb

import dreamplace.ops.rudy.rudy as rudy
import dreamplace.ops.pinrudy.pinrudy as pinrudy
############## Your code block begins here ##############
import dreamplace.ops.ml_congestion.gpdl as gpdl
############## Your code block ends here ################

class MLCongestion(nn.Module):
    """
    @brief compute congestion map based on a neural network model 
    @param fixed_node_map_op an operator to compute fixed macro map given node positions 
    @param rudy_utilization_map_op an operator to compute RUDY map given node positions
    @param pinrudy_utilization_map_op an operator to compute pin RUDY map given node positions 
    @param pin_pos_op an operator to compute pin positions given node positions 
    @param xl left boundary 
    @param yl bottom boundary 
    @param xh right boundary 
    @param yh top boundary 
    @param num_bins_x #bins in horizontal direction, assume to be the same as horizontal routing grids 
    @param num_bins_y #bins in vertical direction, assume to be the same as vertical routing grids 
    @param unit_horizontal_capacity amount of routing resources in horizontal direction in unit distance
    @param unit_vertical_capacity amount of routing resources in vertical direction in unit distance
    @param pretrained_ml_congestion_weight_file file path for pretrained weights of the machine learning model 
    """
    def __init__(self,
                 fixed_node_map_op,
                 rudy_utilization_map_op, 
                 pinrudy_utilization_map_op, 
                 pin_pos_op, 
                 xl,
                 xh,
                 yl,
                 yh,
                 num_bins_x,
                 num_bins_y,
                 unit_horizontal_capacity,
                 unit_vertical_capacity,
                 pretrained_ml_congestion_weight_file):
        super(MLCongestion, self).__init__()
        ############## Your code block begins here ##############
        # init parameters
        self.fixed_node_map_op = fixed_node_map_op
        self.rudy_utilization_map_op = rudy_utilization_map_op
        self.pinrudy_utilization_map_op = pinrudy_utilization_map_op
        
        self.pin_pos_op = pin_pos_op
        self.xl = xl
        self.xh = xh
        self.yl = yl
        self.yh = yh

        self.num_bins_x = num_bins_x
        self.num_bins_y = num_bins_y
        self.unit_horizontal_capacity = unit_horizontal_capacity
        self.unit_vertical_capacity = unit_vertical_capacity
        self.pretrained_ml_congestion_weight_file = pretrained_ml_congestion_weight_file

        # build model
        self.ml_congestion_model = gpdl.GPDL(out_channels=1)
        self.ml_congestion_model.load_state_dict(torch.load(self.pretrained_ml_congestion_weight_file)['state_dict'])
        ############## Your code block ends here ################

    def __call__(self, pos):
        return self.forward(pos)

    def forward(self, pos):
        ############## Your code block begins here ##############
        macro_map = self.fixed_node_map_op(pos)
        rudy_map = self.rudy_utilization_map_op(pos)
        pinrudy_map = self.pinrudy_utilization_map_op(pos)

        ml_input = torch.stack([macro_map, rudy_map, pinrudy_map], dim=0).unsqueeze(0)
        self.ml_congestion_model = self.ml_congestion_model.to(ml_input.device)
        congestion_map = torch.squeeze(self.ml_congestion_model(ml_input))
        return congestion_map
        ############## Your code block ends here ################
