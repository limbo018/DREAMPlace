##
# @file   density_potential.py
# @author Yibo Lin
# @date   Jun 2018
# @brief  Compute density potential according to NTUPlace3 (https://doi.org/10.1109/TCAD.2008.923063)
#

import math
import numpy as np
import torch
from torch import nn
from torch.autograd import Function
from torch.nn import functional as F

import dreamplace.ops.density_potential.density_potential_cpp as density_potential_cpp
import dreamplace.configure as configure
if configure.compile_configurations["CUDA_FOUND"] == "TRUE":
    import dreamplace.ops.density_potential.density_potential_cuda as density_potential_cuda

import pdb
import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# global variable for plot
#plot_count = 0


class DensityPotentialFunction(Function):
    """
    @brief compute density potential.
    """
    @staticmethod
    def forward(
        ctx,
        pos,
        node_size_x,
        node_size_y,
        ax,
        bx,
        cx,
        ay,
        by,
        cy,
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
        num_filler_nodes,
        padding,
        padding_mask,  # same dimensions as density map, with padding regions to be 1 
        num_bins_x,
        num_bins_y,
        num_impacted_bins_x,
        num_impacted_bins_y):
        if pos.is_cuda:
            func = density_potential_cuda.forward
        else:
            func = density_potential_cpp.forward
        output = func(pos.view(pos.numel()), node_size_x, node_size_y, ax, bx,
                      cx, ay, by, cy, bin_center_x, bin_center_y,
                      initial_density_map, target_density, xl, yl, xh, yh,
                      bin_size_x, bin_size_y, num_movable_nodes,
                      num_filler_nodes, padding, num_bins_x, num_bins_y,
                      num_impacted_bins_x, num_impacted_bins_y)

        # output consists of (density_cost, density_map, max_density)
        ctx.node_size_x = node_size_x
        ctx.node_size_y = node_size_y
        ctx.ax = ax
        ctx.bx = bx
        ctx.cx = cx
        ctx.ay = ay
        ctx.by = by
        ctx.cy = cy
        ctx.bin_center_x = bin_center_x
        ctx.bin_center_y = bin_center_y
        ctx.target_density = target_density
        ctx.xl = xl
        ctx.yl = yl
        ctx.xh = xh
        ctx.yh = yh
        ctx.bin_size_x = bin_size_x
        ctx.bin_size_y = bin_size_y
        ctx.num_movable_nodes = num_movable_nodes
        ctx.num_filler_nodes = num_filler_nodes
        ctx.padding = padding
        ctx.num_bins_x = num_bins_x
        ctx.num_bins_y = num_bins_y
        ctx.num_impacted_bins_x = num_impacted_bins_x
        ctx.num_impacted_bins_y = num_impacted_bins_y
        ctx.pos = pos
        ctx.density_map = output[1]

        # set padding density
        if padding > 0:
            ctx.density_map.masked_fill_(
                padding_mask, target_density * bin_size_x * bin_size_y)

        #global plot_count
        #if plot_count % 100 == 0:
        #    plot(plot_count, output[1].clone().div(bin_size_x*bin_size_y).cpu().numpy(), padding, 'summary/potential_map')
        #plot_count += 1

        return output[0]

    @staticmethod
    def backward(ctx, grad_pos):
        if grad_pos.is_cuda:
            func = density_potential_cuda.backward
        else:
            func = density_potential_cpp.backward
        output = func(grad_pos, ctx.num_bins_x, ctx.num_bins_y,
                      ctx.num_impacted_bins_x, ctx.num_impacted_bins_y,
                      ctx.density_map, ctx.pos, ctx.node_size_x,
                      ctx.node_size_y, ctx.ax, ctx.bx, ctx.cx, ctx.ay, ctx.by,
                      ctx.cy, ctx.bin_center_x, ctx.bin_center_y,
                      ctx.target_density, ctx.xl, ctx.yl, ctx.xh, ctx.yh,
                      ctx.bin_size_x, ctx.bin_size_y, ctx.num_movable_nodes,
                      ctx.num_filler_nodes, ctx.padding)
        return output, None, None, None, \
                None, None, None, None, \
                None, None, None, None, \
                None, None, None, None, \
                None, None, None, None, \
                None, None, None, None, \
                None, None, None


class DensityPotential(nn.Module):
    """
    @brief Compute density potential according to NTUPlace3 
    """
    def __init__(self, node_size_x, node_size_y, ax, bx, cx, ay, by, cy,
                 bin_center_x, bin_center_y, target_density, xl, yl, xh, yh,
                 bin_size_x, bin_size_y, num_movable_nodes, num_terminals,
                 num_filler_nodes, padding, sigma, delta):
        """
        @brief initialization 
        @param node_size_x cell width array consisting of movable cells, fixed cells, and filler cells in order  
        @param node_size_y cell height array consisting of movable cells, fixed cells, and filler cells in order   
        @param ax 
        @param bx 
        @param cx 
        @param ay 
        @param by 
        @param cy see the a, b, c defined in NTUPlace3 
        @param bin_center_x bin center x locations 
        @param bin_center_y bin center y locations 
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
        @param padding bin padding to boundary of placement region 
        @param sigma parameter for density map of fixed cells according to NTUPlace3 
        @param delta parameter for density map of fixed cells according to NTUPlace3  
        """
        super(DensityPotential, self).__init__()
        self.node_size_x = node_size_x
        self.node_size_y = node_size_y
        self.ax = ax
        self.bx = bx
        self.cx = cx
        self.ay = ay
        self.by = by
        self.cy = cy
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
        self.padding = padding
        # compute maximum impacted bins
        self.num_bins_x = int(math.ceil((xh - xl) / bin_size_x))
        self.num_bins_y = int(math.ceil((yh - yl) / bin_size_y))
        self.num_impacted_bins_x = (
            (node_size_x[:num_movable_nodes].max() + 4 * self.bin_size_x) /
            self.bin_size_x).ceil().clamp(max=self.num_bins_x)
        self.num_impacted_bins_y = (
            (node_size_y[:num_movable_nodes].max() + 4 * self.bin_size_y) /
            self.bin_size_y).ceil().clamp(max=self.num_bins_y)
        if self.padding > 0:
            self.padding_mask = torch.ones(self.num_bins_x,
                                           self.num_bins_y,
                                           dtype=torch.uint8,
                                           device=node_size_x.device)
            self.padding_mask[self.padding:self.num_bins_x - self.padding,
                              self.padding:self.num_bins_y -
                              self.padding].fill_(0)
        else:
            self.padding_mask = torch.zeros(self.num_bins_x,
                                            self.num_bins_y,
                                            dtype=torch.uint8,
                                            device=node_size_x.device)

        # parameters for initial density map
        self.sigma = sigma
        self.delta = delta
        # initial density_map due to fixed cells
        self.initial_density_map = None

    def forward(self, pos):
        if self.initial_density_map is None:
            if self.num_terminals == 0:
                num_impacted_bins_x = 0
                num_impacted_bins_y = 0
            else:
                num_impacted_bins_x = ((self.node_size_x[
                    self.num_movable_nodes:self.num_movable_nodes +
                    self.num_terminals].max() + self.bin_size_x) /
                                       self.bin_size_x).ceil().clamp(
                                           max=self.num_bins_x)
                num_impacted_bins_y = ((self.node_size_y[
                    self.num_movable_nodes:self.num_movable_nodes +
                    self.num_terminals].max() + self.bin_size_y) /
                                       self.bin_size_y).ceil().clamp(
                                           max=self.num_bins_y)
            if pos.is_cuda:
                func = density_potential_cuda.fixed_density_map
            else:
                func = density_potential_cpp.fixed_density_map
            self.initial_density_map = func(
                pos.view(pos.numel()), self.node_size_x, self.node_size_y,
                self.ax, self.bx, self.cx, self.ay, self.by, self.cy,
                self.bin_center_x, self.bin_center_y, self.xl, self.yl,
                self.xh, self.yh, self.bin_size_x, self.bin_size_y,
                self.num_movable_nodes, self.num_terminals, self.num_bins_x,
                self.num_bins_y, num_impacted_bins_x, num_impacted_bins_y,
                self.sigma, self.delta)
            # there exist fixed cells
            if (self.num_movable_nodes +
                    self.num_filler_nodes) < pos.numel() / 2:
                # convert area to density
                bin_area = self.bin_size_x * self.bin_size_y
                self.initial_density_map.div_(bin_area)
                # gaussian filter
                gaussian_weights = torch.tensor(gaussian_kernel(
                    self.sigma)).to(pos.device)
                self.initial_density_map = F.conv2d(
                    self.initial_density_map.view(
                        [1, 1, self.num_bins_x, self.num_bins_y]),
                    gaussian_weights.view([
                        1, 1,
                        gaussian_weights.size(0),
                        gaussian_weights.size(1)
                    ]),
                    padding=[
                        gaussian_weights.size(0) / 2,
                        gaussian_weights.size(1) / 2
                    ]).view([self.num_bins_x, self.num_bins_y])
                ## level smoothing
                #self.initial_density_map.div_(self.initial_density_map.max())
                #density_mean = self.initial_density_map.mean()
                #delta_map = self.initial_density_map - density_mean
                #self.initial_density_map = density_mean + delta_map.sign().mul_(delta_map.abs().pow_(self.delta))
                # convert density to area
                self.initial_density_map.mul_(bin_area)

                #plot(self.initial_density_map.clone().div(self.bin_size_x*self.bin_size_y).cpu().numpy(), self.padding, 'initial_potential_map')

        return DensityPotentialFunction.apply(
            pos, self.node_size_x, self.node_size_y, self.ax, self.bx, self.cx,
            self.ay, self.by, self.cy, self.bin_center_x, self.bin_center_y,
            self.initial_density_map, self.target_density, self.xl, self.yl,
            self.xh, self.yh, self.bin_size_x, self.bin_size_y,
            self.num_movable_nodes, self.num_filler_nodes, self.padding,
            self.padding_mask, self.num_bins_x, self.num_bins_y,
            self.num_impacted_bins_x, self.num_impacted_bins_y)


def gaussian_kernel(sigma, truncate=4.0):
    """
    Return Gaussian that truncates at the given number of standard deviations. 
    """

    sigma = float(sigma)
    radius = int(truncate * sigma + 0.5)

    x, y = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    sigma = sigma**2

    k = 2 * np.exp(-0.5 * (x**2 + y**2) / sigma)
    k = k / np.sum(k)

    return k


def plot(plot_count, density_map, padding, name):
    """
    density map contour and heat map 
    """
    density_map = density_map[padding:-1 - padding, padding:-1 - padding]
    print("max density = %g" % (np.amax(density_map)))
    print("mean density = %g" % (np.mean(density_map)))

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    x = np.arange(density_map.shape[0])
    y = np.arange(density_map.shape[1])

    x, y = np.meshgrid(x, y)
    ax.plot_surface(x, y, density_map, alpha=0.8)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('density')

    #plt.tight_layout()
    plt.savefig(name + ".3d.%d.png" % (plot_count))
    plt.close()

    #plt.clf()

    #fig, ax = plt.subplots()

    #ax.pcolor(density_map)

    ## Loop over data dimensions and create text annotations.
    ##for i in range(density_map.shape[0]):
    ##    for j in range(density_map.shape[1]):
    ##        text = ax.text(j, i, density_map[i, j],
    ##                ha="center", va="center", color="w")
    #fig.tight_layout()
    #plt.savefig(name+".2d.%d.png" % (plot_count))
    #plt.close()
