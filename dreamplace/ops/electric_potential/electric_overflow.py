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
import dreamplace.configure as configure
if configure.compile_configurations["CUDA_FOUND"] == "TRUE": 
    import dreamplace.ops.electric_potential.electric_potential_cuda as electric_potential_cuda

import pdb
import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class ElectricOverflowFunction(Function):
    """
    @brief compute density overflow.
    @param ctx pytorch API to store data for backward proporgation 
    @param pos location of cells, x and then y
    @param node_size_x_clamped stretched size, max(bin_size*sqrt2, node_size)
    @param node_size_y_clamped stretched size, max(bin_size*sqrt2, node_size)
    @param offset_x (stretched size - node_size) / 2
    @param offset_y (stretched size - node_size) / 2
    @param ratio original area / stretched area 
    @param initial_density_map density_map for fixed cells 
    @param buf buffer for deterministic density map computation on CPU 
    @param target_density target density 
    @param xl left boundary 
    @param yl lower boundary 
    @param xh right boundary 
    @param yh upper boundary 
    @param bin_size_x bin width
    @param bin_size_x bin height
    @param num_movable_nodes number of movable cells 
    @param num_filler_nodes number of filler cells 
    @param padding bin padding to boundary of placement region 
    @param padding_mask padding mask with 0 and 1 to indicate padding bins with padding regions to be 1  
    @param num_bins_x number of bins in horizontal direction
    @param num_bins_y number of bins in vertical direction
    @param num_movable_impacted_bins_x number of impacted bins for any movable cell in x direction 
    @param num_movable_impacted_bins_y number of impacted bins for any movable cell in y direction 
    @param num_filler_impacted_bins_x number of impacted bins for any filler cell in x direction 
    @param num_filler_impacted_bins_y number of impacted bins for any filler cell in y direction 
    @param sorted_node_map the indices of the movable node map
    @param num_threads number of threads 
    """

    @staticmethod
    def forward(
          ctx,
          pos,
          node_size_x_clamped, node_size_y_clamped,
          offset_x, offset_y,
          ratio,
          bin_center_x, bin_center_y,
          initial_density_map,
          buf, 
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
          deterministic_flag, 
          sorted_node_map,
          num_threads
          ):

        if pos.is_cuda:
            output = electric_potential_cuda.density_map(
                    pos.view(pos.numel()),
                    node_size_x_clamped, node_size_y_clamped,
                    offset_x, offset_y,
                    ratio,
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
                    deterministic_flag, 
                    sorted_node_map
                    )
        else:
            output = electric_potential_cpp.density_map(
                    pos.view(pos.numel()),
                    node_size_x_clamped, node_size_y_clamped,
                    offset_x, offset_y,
                    ratio,
                    bin_center_x, bin_center_y,
                    initial_density_map,
                    buf, 
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
        # logger.debug("initial_density_map")
        # logger.debug(initial_density_map/bin_area)
        # logger.debug("density_map")
        # logger.debug(density_map/bin_area)

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
            deterministic_flag, # control whether to use deterministic routine 
            sorted_node_map,
            num_threads=8
            ):
        super(ElectricOverflow, self).__init__()
        sqrt2 = math.sqrt(2)
        self.node_size_x = node_size_x
        # clamped means stretch a cell to bin size 
        # clamped = max(bin_size*sqrt2, node_size)
        # offset means half of the stretch size 
        # ratio means the original area over the stretched area 
        self.node_size_x_clamped = node_size_x.clamp(min=bin_size_x*sqrt2)
        self.offset_x = (node_size_x - self.node_size_x_clamped).mul(0.5)
        self.node_size_y = node_size_y
        self.node_size_y_clamped = node_size_y.clamp(min=bin_size_y*sqrt2)
        self.offset_y = (node_size_y - self.node_size_y_clamped).mul(0.5)
        node_area = node_size_x * node_size_y
        self.ratio = node_area / (self.node_size_x_clamped * self.node_size_y_clamped)

        # detect movable macros and scale down the density to avoid halos 
        # the definition of movable macros should be different according to algorithms 
        # so I prefer to code it inside an operator 
        # I use a heuristic that cells whose areas are 10x of the mean area will be regarded movable macros in global placement 
        if target_density < 1: 
            mean_area = node_area[:num_movable_nodes].mean().mul_(10)
            row_height = node_size_y[:num_movable_nodes].min().mul_(2)
            movable_macro_mask = (node_area[:num_movable_nodes] > mean_area) & (self.node_size_y[:num_movable_nodes] > row_height)
            self.ratio[:num_movable_nodes][movable_macro_mask] = target_density

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
        self.sorted_node_map = sorted_node_map
        # compute maximum impacted bins
        self.num_bins_x = int(math.ceil((xh-xl)/bin_size_x))
        self.num_bins_y = int(math.ceil((yh-yl)/bin_size_y))
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

        self.deterministic_flag = deterministic_flag
        # initial density_map due to fixed cells
        self.initial_density_map = None
        # buffer for deterministic density map computation on CPU 
        self.buf = torch.Tensor() 

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
                        num_fixed_impacted_bins_y, 
                        self.deterministic_flag
                        )
            else:
                self.buf = torch.empty(self.num_threads * self.num_bins_x * self.num_bins_y, dtype=pos.dtype, device=pos.device)
                self.initial_density_map = electric_potential_cpp.fixed_density_map(
                        pos.view(pos.numel()),
                        self.node_size_x, self.node_size_y,
                        self.bin_center_x, self.bin_center_y,
                        self.buf, 
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
                self.node_size_x_clamped, self.node_size_y_clamped,
                self.offset_x, self.offset_y,
                self.ratio,
                self.bin_center_x, self.bin_center_y,
                self.initial_density_map,
                self.buf, 
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
                self.deterministic_flag, 
                self.sorted_node_map,
                self.num_threads
                )

