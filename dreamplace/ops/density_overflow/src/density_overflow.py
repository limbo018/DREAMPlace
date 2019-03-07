##
# @file   density_overflow.py
# @author Yibo Lin
# @date   Jun 2018
# @brief  Compute density overflow 
#

import math 
import torch
from torch import nn
from torch.autograd import Function

import density_overflow_cpp
import density_overflow_cuda_thread_map
import density_overflow_cuda_by_node

import numpy as np 
import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt 

import pdb 

class DensityOverflowFunction(Function):
    """
    @brief compute density overflow.
    """
    @staticmethod
    def forward(
          pos,
          node_size_x,
          node_size_y,
          bin_center_x, 
          bin_center_y, 
          initial_density_map, 
          thread2node_map, 
          thread2bin_x_map,
          thread2bin_y_map, 
          target_density, 
          xl, 
          yl, 
          xh, 
          yh, 
          bin_size_x, 
          bin_size_y, 
          num_movable_nodes, 
          num_filler_nodes, 
          algorithm
          ):
        if pos.is_cuda:
            if algorithm == 'threadmap': 
                output = density_overflow_cuda_thread_map.forward(
                        pos.view(pos.numel()), 
                        node_size_x,
                        node_size_y,
                        bin_center_x, 
                        bin_center_y, 
                        initial_density_map, 
                        thread2node_map, 
                        thread2bin_x_map,
                        thread2bin_y_map, 
                        target_density, 
                        xl, 
                        yl, 
                        xh, 
                        yh, 
                        bin_size_x, 
                        bin_size_y,
                        num_movable_nodes, 
                        num_filler_nodes)
            elif algorithm == 'by-node': 
                output = density_overflow_cuda_by_node.forward(
                        pos.view(pos.numel()), 
                        node_size_x,
                        node_size_y,
                        bin_center_x, 
                        bin_center_y, 
                        initial_density_map, 
                        target_density, 
                        xl, 
                        yl, 
                        xh, 
                        yh, 
                        bin_size_x, 
                        bin_size_y,
                        num_movable_nodes, 
                        num_filler_nodes)
        else:
            output = density_overflow_cpp.forward(
                    pos.view(pos.numel()), 
                    node_size_x,
                    node_size_y,
                    bin_center_x, 
                    bin_center_y, 
                    initial_density_map, 
                    target_density, 
                    xl, 
                    yl, 
                    xh, 
                    yh, 
                    bin_size_x, 
                    bin_size_y, 
                    num_movable_nodes, 
                    num_filler_nodes
                    )
        #print("overflow initial_density_map")
        #print(initial_density_map/(bin_size_x*bin_size_y))
        #print("overflow density_map")
        #print(output[1]/(bin_size_x*bin_size_y))
        #plot(output[1].clone().div(bin_size_x*bin_size_y).cpu().numpy(), 'density_map')
        # output consists of (overflow, density_map, max_density)
        return output[0], output[2]

class DensityOverflow(Function):
    """
    @brief Compute density overflow for both movable and fixed cells.
    The density map for fixed cells is pre-computed. 
    Each call will only compute the density map for movable cells. 
    """
    def __init__(self, node_size_x, node_size_y, bin_center_x, bin_center_y, target_density, xl, yl, xh, yh, bin_size_x, bin_size_y, num_movable_nodes, num_terminals, num_filler_nodes, algorithm='by-node'):
        """
        @brief initialization 
        @param node_size_x cell width array consisting of movable cells, fixed cells, and filler cells in order  
        @param node_size_y cell height array consisting of movable cells, fixed cells, and filler cells in order   
        @param bin_center_x_tensor bin center x locations 
        @param bin_center_y_tensor bin center y locations 
        @param target_density target density 
        @param xl left boundary 
        @param yl bottom boundary 
        @param xh right boundary 
        @param yh top boundary 
        @param bin_size_x bin width 
        @param bin_size_y bin height 
        @param num_movable_nodes number of movable cells 
        @param num_terminals number of fixed cells 
        @param num_filler_nodes number of filler cells 
        @param algorithm must be by-node | threadmap
        """
        super(DensityOverflow, self).__init__()
        self.node_size_x = node_size_x
        self.node_size_y = node_size_y
        self.bin_center_x = bin_center_x
        self.bin_center_y = bin_center_y
        self.target_density = target_density
        self.xl = xl 
        self.yl = yl
        self.xh = xh 
        self.yh = yh 
        self.bin_size_x = bin_size_x
        self.bin_size_y = bin_size_y
        self.num_movable_nodes = num_movable_nodes
        self.num_terminals = num_terminals
        self.num_filler_nodes = num_filler_nodes
        self.algorithm = algorithm 
        # compute maximum impacted bins 
        if algorithm == 'threadmap' and node_size_x.is_cuda: 
            thread_map = density_overflow_cuda_thread_map.thread_map(
                    self.node_size_x.cpu(), 
                    self.node_size_y.cpu(), 
                    self.xl, self.yl, self.xh, self.yh, 
                    self.bin_size_x, self.bin_size_y, 
                    self.num_movable_nodes, self.num_filler_nodes
                    )
            self.thread2node_map = thread_map[0].cuda()
            self.thread2bin_x_map = thread_map[1].cuda()
            self.thread2bin_y_map = thread_map[2].cuda()
        else:
            self.thread2node_map = None
            self.thread2bin_x_map = None
            self.thread2bin_y_map = None
        self.initial_density_map = None
    def forward(self, pos): 
        """
        @brief API 
        @param pos cell locations. The array consists of x locations of movable cells, fixed cells, and filler cells, then y locations of them 
        """
        if self.initial_density_map is None:
            if self.num_terminals == 0:
                num_impacted_bins_x = 0 
                num_impacted_bins_y = 0 
            else:
                num_bins_x = int(math.ceil((self.xh-self.xl)/self.bin_size_x))
                num_bins_y = int(math.ceil((self.yh-self.yl)/self.bin_size_y))
                num_impacted_bins_x = ((self.node_size_x[self.num_movable_nodes:self.num_movable_nodes+self.num_terminals].max()+self.bin_size_x)/self.bin_size_x).ceil().clamp(max=num_bins_x)
                num_impacted_bins_y = ((self.node_size_y[self.num_movable_nodes:self.num_movable_nodes+self.num_terminals].max()+self.bin_size_y)/self.bin_size_y).ceil().clamp(max=num_bins_y)
            if pos.is_cuda:
                self.initial_density_map = density_overflow_cuda_thread_map.fixed_density_map(
                        pos.view(pos.numel()), 
                        self.node_size_x,
                        self.node_size_y,
                        self.bin_center_x, 
                        self.bin_center_y, 
                        self.xl, 
                        self.yl, 
                        self.xh, 
                        self.yh, 
                        self.bin_size_x, 
                        self.bin_size_y,
                        self.num_movable_nodes, 
                        self.num_terminals, 
                        num_impacted_bins_x, num_impacted_bins_y)
            else:
                self.initial_density_map = density_overflow_cpp.fixed_density_map(
                        pos.view(pos.numel()), 
                        self.node_size_x,
                        self.node_size_y,
                        self.bin_center_x, 
                        self.bin_center_y, 
                        self.xl, 
                        self.yl, 
                        self.xh, 
                        self.yh, 
                        self.bin_size_x, 
                        self.bin_size_y, 
                        self.num_movable_nodes, 
                        self.num_terminals
                        )
            #plot(self.initial_density_map.clone().div(self.bin_size_x*self.bin_size_y).cpu().numpy(), 'initial_density_map')

        return DensityOverflowFunction.forward(
                pos,
                node_size_x=self.node_size_x,
                node_size_y=self.node_size_y,
                bin_center_x=self.bin_center_x, 
                bin_center_y=self.bin_center_y, 
                initial_density_map=self.initial_density_map, 
                thread2node_map=self.thread2node_map, 
                thread2bin_x_map=self.thread2bin_x_map, 
                thread2bin_y_map=self.thread2bin_y_map, 
                target_density=self.target_density, 
                xl=self.xl, 
                yl=self.yl, 
                xh=self.xh, 
                yh=self.yh, 
                bin_size_x=self.bin_size_x, 
                bin_size_y=self.bin_size_y, 
                num_movable_nodes=self.num_movable_nodes, 
                num_filler_nodes=self.num_filler_nodes, 
                algorithm=self.algorithm
                )

def plot(density_map, name):
    """
    @brief density map contour and heat map 
    """
    print(np.amax(density_map))
    print(np.mean(density_map))
    fig = plt.figure(figsize=(4, 3))
    ax = fig.gca(projection='3d')

    x = np.arange(density_map.shape[0])
    y = np.arange(density_map.shape[1])

    x, y = np.meshgrid(x, y)
    ax.plot_surface(x, y, density_map, alpha=0.8)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('density')

    #plt.tight_layout()
    plt.savefig(name+".3d.png")

    plt.clf()

    fig, ax = plt.subplots()

    ax.pcolor(density_map)

    # Loop over data dimensions and create text annotations.
    #for i in range(density_map.shape[0]):
    #    for j in range(density_map.shape[1]):
    #        text = ax.text(j, i, density_map[i, j],
    #                ha="center", va="center", color="w")
    fig.tight_layout()
    plt.savefig(name+".2d.png")
