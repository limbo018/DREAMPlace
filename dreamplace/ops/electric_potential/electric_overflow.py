##
# @file   electric_overflow.py
# @author Yibo Lin
# @date   Aug 2018
#

import math 
import numpy as np 
import torch
from torch import nn
from torch.autograd import Function
from torch.nn import functional as F

import dreamplace.ops.electric_potential.electric_potential_cpp as electric_potential_cpp
try: 
    import dreamplace.ops.electric_potential.electric_potential_cuda as electric_potential_cuda 
except:
    pass

import pdb 
import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt 

class ElectricOverflowFunction(Function):
    """compute density overflow.
    """

    @staticmethod
    def forward(
          ctx, 
          pos,
          node_size_x, node_size_y,
          bin_center_x, bin_center_y, 
          initial_density_map, 
          target_density, 
          xl, yl, xh, yh, 
          bin_size_x, bin_size_y, 
          num_movable_nodes, 
          num_filler_nodes, 
          padding, 
          padding_mask, # same dimensions as density map, with padding regions to be 1 
          num_bins_x, 
          num_bins_y, 
          num_movable_impacted_bins_x, 
          num_movable_impacted_bins_y, 
          num_filler_impacted_bins_x, 
          num_filler_impacted_bins_y, 
          num_threads
          ):
        
        if pos.is_cuda:
            output = electric_potential_cuda.density_map(
                    pos.view(pos.numel()), 
                    node_size_x, node_size_y,
                    bin_center_x, bin_center_y, 
                    initial_density_map, 
                    target_density, 
                    xl, yl, xh, yh, 
                    bin_size_x, bin_size_y, 
                    num_movable_nodes, 
                    num_filler_nodes, 
                    padding, 
                    padding_mask, 
                    num_bins_x, 
                    num_bins_y, 
                    num_movable_impacted_bins_x, 
                    num_movable_impacted_bins_y,
                    num_filler_impacted_bins_x, 
                    num_filler_impacted_bins_y
                    ) 
        else:
            output = electric_potential_cpp.density_map(
                    pos.view(pos.numel()), 
                    node_size_x, node_size_y,
                    bin_center_x, bin_center_y, 
                    initial_density_map, 
                    target_density, 
                    xl, yl, xh, yh, 
                    bin_size_x, bin_size_y, 
                    num_movable_nodes, 
                    num_filler_nodes, 
                    padding, 
                    padding_mask, 
                    num_bins_x, 
                    num_bins_y, 
                    num_movable_impacted_bins_x, 
                    num_movable_impacted_bins_y,
                    num_filler_impacted_bins_x, 
                    num_filler_impacted_bins_y, 
                    num_threads
                    ) 

        bin_area = bin_size_x*bin_size_y
        density_map = output.view([num_bins_x, num_bins_y])
        density_cost = (density_map-target_density*bin_area).clamp_(min=0.0).sum()

        #torch.set_printoptions(precision=10)
        #print("initial_density_map")
        #print(initial_density_map/bin_area)
        #print("density_map") 
        #print(density_map/bin_area)

        return density_cost, density_map.max()/bin_area

class ElectricOverflow(nn.Module):
    def __init__(self, 
            node_size_x, node_size_y,
            bin_center_x, bin_center_y, 
            target_density, 
            xl, yl, xh, yh, 
            bin_size_x, bin_size_y, 
            num_movable_nodes, 
            num_terminals, 
            num_filler_nodes,
            padding, 
            num_threads=8
            ):
        super(ElectricOverflow, self).__init__()
        self.node_size_x = node_size_x
        self.node_size_y = node_size_y
        self.bin_center_x = bin_center_x
        self.bin_center_y = bin_center_y
        self.target_density = target_density
        self.xl = xl 
        self.yl = yl
        self.xh = xh 
        self.yh = yh 
        self.bin_size_x = torch.tensor(bin_size_x, dtype=node_size_x.dtype, device=node_size_x.device)
        self.bin_size_y = torch.tensor(bin_size_y, dtype=node_size_y.dtype, device=node_size_y.device)
        self.num_movable_nodes = num_movable_nodes
        self.num_terminals = num_terminals
        self.num_filler_nodes = num_filler_nodes
        self.padding = padding
        # compute maximum impacted bins 
        self.num_bins_x = int(math.ceil((xh-xl)/bin_size_x))
        self.num_bins_y = int(math.ceil((yh-yl)/bin_size_y))
        sqrt2 = 1.414213562
        self.num_movable_impacted_bins_x = ((node_size_x[:num_movable_nodes].max()+2*sqrt2*self.bin_size_x)/self.bin_size_x).ceil().clamp(max=self.num_bins_x);
        self.num_movable_impacted_bins_y = ((node_size_y[:num_movable_nodes].max()+2*sqrt2*self.bin_size_y)/self.bin_size_y).ceil().clamp(max=self.num_bins_y);
        if num_filler_nodes: 
            self.num_filler_impacted_bins_x = ((node_size_x[-num_filler_nodes:].max()+2*sqrt2*self.bin_size_x)/self.bin_size_x).ceil().clamp(max=self.num_bins_x);
            self.num_filler_impacted_bins_y = ((node_size_y[-num_filler_nodes:].max()+2*sqrt2*self.bin_size_y)/self.bin_size_y).ceil().clamp(max=self.num_bins_y);
        else:
            self.num_filler_impacted_bins_x = 0
            self.num_filler_impacted_bins_y = 0
        if self.padding > 0: 
            self.padding_mask = torch.ones(self.num_bins_x, self.num_bins_y, dtype=torch.uint8, device=node_size_x.device)
            self.padding_mask[self.padding:self.num_bins_x-self.padding, self.padding:self.num_bins_y-self.padding].fill_(0)
        else:
            self.padding_mask = torch.zeros(self.num_bins_x, self.num_bins_y, dtype=torch.uint8, device=node_size_x.device)

        self.num_threads = num_threads

        # initial density_map due to fixed cells 
        self.initial_density_map = None

    def forward(self, pos): 
        if self.initial_density_map is None: 
            if self.num_terminals == 0:
                num_fixed_impacted_bins_x = 0 
                num_fixed_impacted_bins_y = 0 
            else:
                num_fixed_impacted_bins_x = ((self.node_size_x[self.num_movable_nodes:self.num_movable_nodes+self.num_terminals].max()+self.bin_size_x)/self.bin_size_x).ceil().clamp(max=self.num_bins_x)
                num_fixed_impacted_bins_y = ((self.node_size_y[self.num_movable_nodes:self.num_movable_nodes+self.num_terminals].max()+self.bin_size_y)/self.bin_size_y).ceil().clamp(max=self.num_bins_y)
            if pos.is_cuda:
                self.initial_density_map = electric_potential_cuda.fixed_density_map(
                        pos.view(pos.numel()), 
                        self.node_size_x, self.node_size_y,
                        self.bin_center_x, self.bin_center_y, 
                        self.xl, self.yl, self.xh, self.yh, 
                        self.bin_size_x, self.bin_size_y, 
                        self.num_movable_nodes, 
                        self.num_terminals, 
                        self.num_bins_x, 
                        self.num_bins_y, 
                        num_fixed_impacted_bins_x, 
                        num_fixed_impacted_bins_y
                        ) 
            else:
                self.initial_density_map = electric_potential_cpp.fixed_density_map(
                        pos.view(pos.numel()), 
                        self.node_size_x, self.node_size_y,
                        self.bin_center_x, self.bin_center_y, 
                        self.xl, self.yl, self.xh, self.yh, 
                        self.bin_size_x, self.bin_size_y, 
                        self.num_movable_nodes, 
                        self.num_terminals, 
                        self.num_bins_x, 
                        self.num_bins_y, 
                        num_fixed_impacted_bins_x, 
                        num_fixed_impacted_bins_y, 
                        self.num_threads
                        ) 
            #plot(0, self.initial_density_map.clone().div(self.bin_size_x*self.bin_size_y).cpu().numpy(), self.padding, 'summary/initial_potential_map')
            # scale density of fixed macros 
            self.initial_density_map.mul_(self.target_density)

        return ElectricOverflowFunction.apply(
                pos,
                self.node_size_x, self.node_size_y,
                self.bin_center_x, self.bin_center_y, 
                self.initial_density_map,
                self.target_density, 
                self.xl, self.yl, self.xh, self.yh, 
                self.bin_size_x, self.bin_size_y, 
                self.num_movable_nodes, 
                self.num_filler_nodes, 
                self.padding, 
                self.padding_mask, 
                self.num_bins_x, 
                self.num_bins_y, 
                self.num_movable_impacted_bins_x, 
                self.num_movable_impacted_bins_y, 
                self.num_filler_impacted_bins_x, 
                self.num_filler_impacted_bins_y, 
                self.num_threads
                )

