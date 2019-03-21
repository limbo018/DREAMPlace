##
# @file   electric_potential.py
# @author Yibo Lin
# @date   Jun 2018
# @brief  electric potential according to e-place (http://cseweb.ucsd.edu/~jlu/papers/eplace-todaes14/paper.pdf) 
#

import os 
import sys
import math 
import numpy as np 
import time
import torch
from torch import nn
from torch.autograd import Function
from torch.nn import functional as F

import dreamplace.ops.dct.dct as dct 
import dreamplace.ops.dct.discrete_spectral_transform as discrete_spectral_transform

import lib.dreamplace.ops.electric_potential.electric_potential_cpp as electric_potential_cpp
import lib.dreamplace.ops.electric_potential.electric_potential_cuda as electric_potential_cuda 

import pdb 
import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt 

# global variable for plot 
plot_count = 0

class ElectricPotentialFunction(Function):
    """
    @brief compute electric potential according to e-place.
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
          perm_M=None, # permutation 
          perm_N=None, # permutation
          expk_M=None, # 2*exp(j*pi*k/M)
          expk_N=None, # 2*exp(j*pi*k/N)
          inv_wu2_plus_wv2_2X=None, # 2.0/(wu^2 + wv^2)
          wu_by_wu2_plus_wv2_2X=None, # 2*wu/(wu^2 + wv^2)
          wv_by_wu2_plus_wv2_2X=None, # 2*wv/(wu^2 + wv^2)
          fast_mode=True # fast mode will discard some computation  
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
                    num_filler_impacted_bins_y
                    ) 

        # output consists of (density_cost, density_map, max_density)
        ctx.node_size_x = node_size_x
        ctx.node_size_y = node_size_y
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
        ctx.num_movable_impacted_bins_x = num_movable_impacted_bins_x
        ctx.num_movable_impacted_bins_y = num_movable_impacted_bins_y
        ctx.num_filler_impacted_bins_x = num_filler_impacted_bins_x
        ctx.num_filler_impacted_bins_y = num_filler_impacted_bins_y
        ctx.pos = pos 
        density_map = output.view([ctx.num_bins_x, ctx.num_bins_y])
        #density_map = torch.ones([ctx.num_bins_x, ctx.num_bins_y], dtype=pos.dtype, device=pos.device)
        #ctx.field_map_x = torch.ones([ctx.num_bins_x, ctx.num_bins_y], dtype=pos.dtype, device=pos.device)
        #ctx.field_map_y = torch.ones([ctx.num_bins_x, ctx.num_bins_y], dtype=pos.dtype, device=pos.device)
        #return torch.zeros(1, dtype=pos.dtype, device=pos.device)

        # for DCT 
        M = num_bins_x
        N = num_bins_y
        if expk_M is None: 
            perm_M = discrete_spectral_transform.get_perm(M, dtype=torch.int64, device=density_map.device)
            perm_N = discrete_spectral_transform.get_perm(N, dtype=torch.int64, device=density_map.device)
            expk_M = discrete_spectral_transform.get_expk(M, dtype=density_map.dtype, device=density_map.device)
            expk_N = discrete_spectral_transform.get_expk(N, dtype=density_map.dtype, device=density_map.device)
        # wu and wv 
        if inv_wu2_plus_wv2_2X is None: 
            wu = torch.arange(M, dtype=density_map.dtype, device=density_map.device).mul(2*np.pi/M).view([M, 1])
            wv = torch.arange(N, dtype=density_map.dtype, device=density_map.device).mul(2*np.pi/N).view([1, N])
            wu2_plus_wv2 = wu.pow(2) + wv.pow(2)
            wu2_plus_wv2[0, 0] = 1.0 # avoid zero-division, it will be zeroed out 
            inv_wu2_plus_wv2_2X = 2.0 / wu2_plus_wv2
            inv_wu2_plus_wv2_2X[0, 0] = 0.0 
            wu_by_wu2_plus_wv2_2X = wu.mul(inv_wu2_plus_wv2_2X)
            wv_by_wu2_plus_wv2_2X = wv.mul(inv_wu2_plus_wv2_2X)
        
        # compute auv 
        density_map.mul_(1.0/(ctx.bin_size_x*ctx.bin_size_y))
        #auv = discrete_spectral_transform.dct2_2N(density_map, expk0=expk_M, expk1=expk_N)
        auv = dct.dct2(density_map, expk0=expk_M, expk1=expk_N)
        auv[0, :].mul_(0.5)
        auv[:, 0].mul_(0.5)

        # compute field xi 
        auv_by_wu2_plus_wv2_wu = auv.mul(wu_by_wu2_plus_wv2_2X)
        auv_by_wu2_plus_wv2_wv = auv.mul(wv_by_wu2_plus_wv2_2X)
        #ctx.field_map_x = discrete_spectral_transform.idsct2(auv_by_wu2_plus_wv2_wu, expk_M, expk_N).contiguous()
        ctx.field_map_x = dct.idsct2(auv_by_wu2_plus_wv2_wu, expk_M, expk_N)
        #ctx.field_map_y = discrete_spectral_transform.idcst2(auv_by_wu2_plus_wv2_wv, expk_M, expk_N).contiguous()
        ctx.field_map_y = dct.idcst2(auv_by_wu2_plus_wv2_wv, expk_M, expk_N)

        # energy = \sum q*phi
        # it takes around 80% of the computation time 
        # so I will not always evaluate it 
        if fast_mode: # dummy for invoking backward propagation 
            energy = torch.zeros(1, dtype=pos.dtype, device=pos.device)
        else: 
            # compute potential phi 
            # auv / (wu**2 + wv**2)
            auv_by_wu2_plus_wv2 = auv.mul(inv_wu2_plus_wv2_2X).mul_(2)
            #potential_map = discrete_spectral_transform.idcct2(auv_by_wu2_plus_wv2, expk_M, expk_N) 
            potential_map = dct.idcct2(auv_by_wu2_plus_wv2, expk_M, expk_N) 
            # compute energy 
            energy = potential_map.mul_(density_map).sum()

        #torch.set_printoptions(precision=10)
        #print("initial_density_map")
        #print(initial_density_map/(ctx.bin_size_x*ctx.bin_size_y))
        #print("density_map") 
        #print(density_map/(ctx.bin_size_x*ctx.bin_size_y))
        #print("auv_by_wu2_plus_wv2")
        #print(auv_by_wu2_plus_wv2)
        #print("potential_map") 
        #print(potential_map)
        #print("field_map_x")
        #print(ctx.field_map_x)
        #print("field_map_y")
        #print(ctx.field_map_y)

        #global plot_count 
        #if plot_count >= 600 and plot_count % 1 == 0: 
        #    print("density_map")
        #    plot(plot_count, density_map.clone().div(bin_size_x*bin_size_y).cpu().numpy(), padding, "summary/%d.density_map" % (plot_count))
        #    print("potential_map")
        #    plot(plot_count, potential_map.clone().cpu().numpy(), padding, "summary/%d.potential_map" % (plot_count))
        #    print("field_map_x")
        #    plot(plot_count, ctx.field_map_x.clone().cpu().numpy(), padding, "summary/%d.field_map_x" % (plot_count))
        #    print("field_map_y")
        #    plot(plot_count, ctx.field_map_y.clone().cpu().numpy(), padding, "summary/%d.field_map_y" % (plot_count))
        #plot_count += 1

        torch.cuda.synchronize()
        return energy 

    @staticmethod
    def backward(ctx, grad_pos):
        #tt = time.time()
        if grad_pos.is_cuda:
            output = -electric_potential_cuda.electric_force(
                    grad_pos, 
                    ctx.num_bins_x, ctx.num_bins_y, 
                    ctx.num_movable_impacted_bins_x, ctx.num_movable_impacted_bins_y, 
                    ctx.num_filler_impacted_bins_x, ctx.num_filler_impacted_bins_y, 
                    ctx.field_map_x.view([-1]), ctx.field_map_y.view([-1]), 
                    ctx.pos, 
                    ctx.node_size_x, ctx.node_size_y,
                    ctx.bin_center_x, ctx.bin_center_y,
                    ctx.xl, ctx.yl, ctx.xh, ctx.yh, 
                    ctx.bin_size_x, ctx.bin_size_y, 
                    ctx.num_movable_nodes, 
                    ctx.num_filler_nodes
                    )
        else:
            output = -electric_potential_cpp.electric_force(
                    grad_pos, 
                    ctx.num_bins_x, ctx.num_bins_y, 
                    ctx.num_movable_impacted_bins_x, ctx.num_movable_impacted_bins_y, 
                    ctx.num_filler_impacted_bins_x, ctx.num_filler_impacted_bins_y, 
                    ctx.field_map_x.view([-1]), ctx.field_map_y.view([-1]), 
                    ctx.pos, 
                    ctx.node_size_x, ctx.node_size_y,
                    ctx.bin_center_x, ctx.bin_center_y,
                    ctx.xl, ctx.yl, ctx.xh, ctx.yh, 
                    ctx.bin_size_x, ctx.bin_size_y, 
                    ctx.num_movable_nodes, 
                    ctx.num_filler_nodes
                    )

        #global plot_count 
        #if plot_count >= 300: 
        #    indices = (ctx.pos[ctx.pos.numel()/2-ctx.num_filler_nodes:ctx.pos.numel()/2] < ctx.xl+ctx.bin_size_x).nonzero()
        #    pdb.set_trace()

        #pgradx = []
        #pgrady = []
        #with open("/home/polaris/yibolin/Libraries/RePlAce/output/ispd/adaptec1.eplace/gradient.csv", "r") as f:
        #    for line in f:
        #        tokens = line.strip().split(" ")
        #        pgradx.append(float(tokens[3].strip()))
        #        pgrady.append(float(tokens[4].strip()))
        #pgrad = np.concatenate([np.array(pgradx), np.array(pgrady)])

        #output = torch.empty_like(ctx.pos).uniform_(0.0, 0.1)
        torch.cuda.synchronize()
        #print("\t\tdensity backward %.3f ms" % ((time.time()-tt)*1000))
        return output, \
                None, None, None, None, \
                None, None, None, None, \
                None, None, None, None, \
                None, None, None, None, \
                None, None, None, None, \
                None, None, None, None, \
                None, None, None, None, \
                None, None

class ElectricPotential(nn.Module):
    """
    @brief Compute electric potential according to e-place 
    """
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
            fast_mode=False
            ):
        """ 
        @brief initialization 
        Be aware that all scalars must be python type instead of tensors. 
        Otherwise, GPU version can be weirdly slow. 
        @param node_size_x cell width array consisting of movable cells, fixed cells, and filler cells in order  
        @param node_size_y cell height array consisting of movable cells, fixed cells, and filler cells in order   
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
        @param fast_mode if true, only gradient is computed, while objective computation is skipped 
        """
        super(ElectricPotential, self).__init__()
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
        self.padding = padding
        # compute maximum impacted bins 
        self.num_bins_x = int(math.ceil((xh-xl)/bin_size_x))
        self.num_bins_y = int(math.ceil((yh-yl)/bin_size_y))
        sqrt2 = 1.414213562
        self.num_movable_impacted_bins_x = int(((node_size_x[:num_movable_nodes].max()+2*sqrt2*self.bin_size_x)/self.bin_size_x).ceil().clamp(max=self.num_bins_x));
        self.num_movable_impacted_bins_y = int(((node_size_y[:num_movable_nodes].max()+2*sqrt2*self.bin_size_y)/self.bin_size_y).ceil().clamp(max=self.num_bins_y));
        if num_filler_nodes: 
            self.num_filler_impacted_bins_x = int(((node_size_x[-num_filler_nodes:].max()+2*sqrt2*self.bin_size_x)/self.bin_size_x).ceil().clamp(max=self.num_bins_x));
            self.num_filler_impacted_bins_y = int(((node_size_y[-num_filler_nodes:].max()+2*sqrt2*self.bin_size_y)/self.bin_size_y).ceil().clamp(max=self.num_bins_y));
        else:
            self.num_filler_impacted_bins_x = 0
            self.num_filler_impacted_bins_y = 0
        if self.padding > 0: 
            self.padding_mask = torch.ones(self.num_bins_x, self.num_bins_y, dtype=torch.uint8, device=node_size_x.device)
            self.padding_mask[self.padding:self.num_bins_x-self.padding, self.padding:self.num_bins_y-self.padding].fill_(0)
        else:
            self.padding_mask = torch.zeros(self.num_bins_x, self.num_bins_y, dtype=torch.uint8, device=node_size_x.device)

        # initial density_map due to fixed cells 
        self.initial_density_map = None
        self.perm_M = None
        self.perm_N = None
        self.expk_M = None
        self.expk_N = None
        self.inv_wu2_plus_wv2_2X = None 
        self.wu_by_wu2_plus_wv2_2X = None 
        self.wv_by_wu2_plus_wv2_2X = None 

        # whether really evaluate potential_map and energy or use dummy 
        self.fast_mode = fast_mode 

    def forward(self, pos): 
        if self.initial_density_map is None: 
            if self.num_terminals == 0:
                num_fixed_impacted_bins_x = 0 
                num_fixed_impacted_bins_y = 0 
            else:
                num_fixed_impacted_bins_x = int(((self.node_size_x[self.num_movable_nodes:self.num_movable_nodes+self.num_terminals].max()+self.bin_size_x)/self.bin_size_x).ceil().clamp(max=self.num_bins_x))
                num_fixed_impacted_bins_y = int(((self.node_size_y[self.num_movable_nodes:self.num_movable_nodes+self.num_terminals].max()+self.bin_size_y)/self.bin_size_y).ceil().clamp(max=self.num_bins_y))
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
                        num_fixed_impacted_bins_y
                        ) 
            #plot(0, self.initial_density_map.clone().div(self.bin_size_x*self.bin_size_y).cpu().numpy(), self.padding, 'summary/initial_potential_map')
            # scale density of fixed macros 
            self.initial_density_map.mul_(self.target_density)
            # expk 
            M = self.num_bins_x
            N = self.num_bins_y
            self.perm_M = discrete_spectral_transform.get_perm(M, dtype=torch.int64, device=pos.device)
            self.perm_N = discrete_spectral_transform.get_perm(N, dtype=torch.int64, device=pos.device)
            self.expk_M = discrete_spectral_transform.get_expk(M, dtype=pos.dtype, device=pos.device)
            self.expk_N = discrete_spectral_transform.get_expk(N, dtype=pos.dtype, device=pos.device)
            # wu and wv 
            wu = torch.arange(M, dtype=pos.dtype, device=pos.device).mul(2*np.pi/M).view([M, 1])
            # scale wv because the aspect ratio of a bin may not be 1 
            wv = torch.arange(N, dtype=pos.dtype, device=pos.device).mul(2*np.pi/N).view([1, N]).mul_(self.bin_size_x/self.bin_size_y)
            wu2_plus_wv2 = wu.pow(2) + wv.pow(2)
            wu2_plus_wv2[0, 0] = 1.0 # avoid zero-division, it will be zeroed out 
            self.inv_wu2_plus_wv2_2X = 2.0 / wu2_plus_wv2
            self.inv_wu2_plus_wv2_2X[0, 0] = 0.0 
            self.wu_by_wu2_plus_wv2_2X = wu.mul(self.inv_wu2_plus_wv2_2X)
            self.wv_by_wu2_plus_wv2_2X = wv.mul(self.inv_wu2_plus_wv2_2X)

        return ElectricPotentialFunction.apply(
                pos,
                self.node_size_x, self.node_size_y,
                self.bin_center_x, self.bin_center_y, 
                self.initial_density_map,
                self.target_density, 
                self.xl, self.yl, self.xh, self.yh, 
                self.bin_size_x, self.bin_size_y, 
                self.num_movable_nodes, self.num_filler_nodes,
                self.padding, 
                self.padding_mask, 
                self.num_bins_x, 
                self.num_bins_y, 
                self.num_movable_impacted_bins_x, 
                self.num_movable_impacted_bins_y,
                self.num_filler_impacted_bins_x, 
                self.num_filler_impacted_bins_y, 
                self.perm_M, self.perm_N, 
                self.expk_M, self.expk_N, 
                self.inv_wu2_plus_wv2_2X, 
                self.wu_by_wu2_plus_wv2_2X, self.wv_by_wu2_plus_wv2_2X, 
                self.fast_mode
                )

def plot(plot_count, density_map, padding, name):
    """
    density map contour and heat map 
    """
    density_map = density_map[padding:density_map.shape[0]-padding, padding:density_map.shape[1]-padding]
    print("max density = %g @ %s" % (np.amax(density_map), np.unravel_index(np.argmax(density_map), density_map.shape)))
    print("mean density = %g" % (np.mean(density_map)))

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    x = np.arange(density_map.shape[0])
    y = np.arange(density_map.shape[1])

    x, y = np.meshgrid(x, y)
    # looks like x and y should be swapped 
    ax.plot_surface(y, x, density_map, alpha=0.8)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('density')

    #plt.tight_layout()
    plt.savefig(name+".3d.png")
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
