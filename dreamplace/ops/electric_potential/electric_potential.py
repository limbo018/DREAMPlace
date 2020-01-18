##
# @file   electric_potential.py
# @author Yibo Lin
# @date   Jun 2018
# @brief  electric potential according to e-place (http://cseweb.ucsd.edu/~jlu/papers/eplace-todaes14/paper.pdf)
#

#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import os
import sys
import math
import numpy as np
import time
import torch
from torch import nn
from torch.autograd import Function
from torch.nn import functional as F
import logging

import dreamplace.ops.dct.discrete_spectral_transform as discrete_spectral_transform

import dreamplace.ops.dct.dct2_fft2 as dct
from dreamplace.ops.dct.discrete_spectral_transform import get_exact_expk as precompute_expk

#import dreamplace.ops.dct.dct as dct
#from dreamplace.ops.dct.discrete_spectral_transform import get_expk as precompute_expk

from dreamplace.ops.electric_potential.electric_overflow import ElectricDensityMapFunction as ElectricDensityMapFunction
from dreamplace.ops.electric_potential.electric_overflow import ElectricOverflow as ElectricOverflow

import dreamplace.ops.electric_potential.electric_potential_cpp as electric_potential_cpp
try:
    import dreamplace.ops.electric_potential.electric_potential_cuda as electric_potential_cuda
except:
    pass

import pdb
import matplotlib
matplotlib.use('Agg')

logger = logging.getLogger(__name__)

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
        padding_mask,  # same dimensions as density map, with padding regions to be 1
        num_bins_x,
        num_bins_y,
        num_movable_impacted_bins_x,
        num_movable_impacted_bins_y,
        num_filler_impacted_bins_x,
        num_filler_impacted_bins_y,
        deterministic_flag, 
        sorted_node_map,
        exact_expkM=None,  # exp(-j*pi*k/M)
        exact_expkN=None,  # exp(-j*pi*k/N)
        inv_wu2_plus_wv2=None,  # 1.0/(wu^2 + wv^2)
        wu_by_wu2_plus_wv2_half=None,  # wu/(wu^2 + wv^2)/2
        wv_by_wu2_plus_wv2_half=None,  # wv/(wu^2 + wv^2)/2
        dct2=None,
        idct2=None,
        idct_idxst=None,
        idxst_idct=None,
        fast_mode=True,  # fast mode will discard some computation
        num_threads=8
    ):

        tt = time.time()
        density_map = ElectricDensityMapFunction.forward(
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
                padding_mask,
                num_bins_x,
                num_bins_y,
                num_movable_impacted_bins_x,
                num_movable_impacted_bins_y,
                num_filler_impacted_bins_x,
                num_filler_impacted_bins_y,
                deterministic_flag, 
                sorted_node_map,
                num_threads
                )

        # output consists of (density_cost, density_map, max_density)
        ctx.node_size_x_clamped = node_size_x_clamped
        ctx.node_size_y_clamped = node_size_y_clamped
        ctx.offset_x = offset_x
        ctx.offset_y = offset_y
        ctx.ratio = ratio
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
        ctx.sorted_node_map = sorted_node_map
        ctx.num_threads = num_threads
        #density_map = torch.ones([ctx.num_bins_x, ctx.num_bins_y], dtype=pos.dtype, device=pos.device)
        #ctx.field_map_x = torch.ones([ctx.num_bins_x, ctx.num_bins_y], dtype=pos.dtype, device=pos.device)
        #ctx.field_map_y = torch.ones([ctx.num_bins_x, ctx.num_bins_y], dtype=pos.dtype, device=pos.device)
        # return torch.zeros(1, dtype=pos.dtype, device=pos.device)

        # for DCT
        M = num_bins_x
        N = num_bins_y

        # wu and wv
        if inv_wu2_plus_wv2 is None:
            wu = torch.arange(M, dtype=density_map.dtype, device=density_map.device).mul(2 * np.pi / M).view([M, 1])
            wv = torch.arange(N, dtype=density_map.dtype, device=density_map.device).mul(2 * np.pi / N).view([1, N])
            wu2_plus_wv2 = wu.pow(2) + wv.pow(2)
            wu2_plus_wv2[0, 0] = 1.0  # avoid zero-division, it will be zeroed out
            inv_wu2_plus_wv2 = 1.0 / wu2_plus_wv2
            inv_wu2_plus_wv2[0, 0] = 0.0
            wu_by_wu2_plus_wv2_half = wu.mul(inv_wu2_plus_wv2).mul_(1./ 2)
            wv_by_wu2_plus_wv2_half = wv.mul(inv_wu2_plus_wv2).mul_(1./ 2)

        # compute auv
        density_map.mul_(1.0 / (ctx.bin_size_x * ctx.bin_size_y))

        #auv = discrete_spectral_transform.dct2_2N(density_map, expk0=exact_expkM, expk1=exact_expkN)
        auv = dct2.forward(density_map)

        # compute field xi
        auv_by_wu2_plus_wv2_wu = auv.mul(wu_by_wu2_plus_wv2_half)
        auv_by_wu2_plus_wv2_wv = auv.mul(wv_by_wu2_plus_wv2_half)

        #ctx.field_map_x = discrete_spectral_transform.idsct2(auv_by_wu2_plus_wv2_wu, exact_expkM, exact_expkN).contiguous()
        ctx.field_map_x = idxst_idct.forward(auv_by_wu2_plus_wv2_wu)
        #ctx.field_map_y = discrete_spectral_transform.idcst2(auv_by_wu2_plus_wv2_wv, exact_expkM, exact_expkN).contiguous()
        ctx.field_map_y = idct_idxst.forward(auv_by_wu2_plus_wv2_wv)

        # energy = \sum q*phi
        # it takes around 80% of the computation time
        # so I will not always evaluate it
        if fast_mode:  # dummy for invoking backward propagation
            energy = torch.zeros(1, dtype=pos.dtype, device=pos.device)
        else:
            # compute potential phi
            # auv / (wu**2 + wv**2)
            auv_by_wu2_plus_wv2 = auv.mul(inv_wu2_plus_wv2)
            #potential_map = discrete_spectral_transform.idcct2(auv_by_wu2_plus_wv2, exact_expkM, exact_expkN)
            potential_map = idct2.forward(auv_by_wu2_plus_wv2)
            # compute energy
            energy = potential_map.mul(density_map).sum()

        # torch.set_printoptions(precision=10)
        # logger.debug("initial_density_map")
        # logger.debug(initial_density_map/(ctx.bin_size_x*ctx.bin_size_y))
        # logger.debug("density_map")
        # logger.debug(density_map/(ctx.bin_size_x*ctx.bin_size_y))
        # logger.debug("auv_by_wu2_plus_wv2")
        # logger.debug(auv_by_wu2_plus_wv2)
        # logger.debug("potential_map")
        # logger.debug(potential_map)
        # logger.debug("field_map_x")
        # logger.debug(ctx.field_map_x)
        # logger.debug("field_map_y")
        # logger.debug(ctx.field_map_y)

        #global plot_count
        # if plot_count >= 600 and plot_count % 1 == 0:
        #    logger.debug("density_map")
        #    plot(plot_count, density_map.clone().div(bin_size_x*bin_size_y).cpu().numpy(), padding, "summary/%d.density_map" % (plot_count))
        #    logger.debug("potential_map")
        #    plot(plot_count, potential_map.clone().cpu().numpy(), padding, "summary/%d.potential_map" % (plot_count))
        #    logger.debug("field_map_x")
        #    plot(plot_count, ctx.field_map_x.clone().cpu().numpy(), padding, "summary/%d.field_map_x" % (plot_count))
        #    logger.debug("field_map_y")
        #    plot(plot_count, ctx.field_map_y.clone().cpu().numpy(), padding, "summary/%d.field_map_y" % (plot_count))
        #plot_count += 1

        if pos.is_cuda:
            torch.cuda.synchronize()
        logger.debug("density forward %.3f ms" % ((time.time()-tt)*1000))
        return energy 

    @staticmethod
    def backward(ctx, grad_pos):
        tt = time.time()
        if grad_pos.is_cuda:
            output = -electric_potential_cuda.electric_force(
                grad_pos,
                ctx.num_bins_x, ctx.num_bins_y,
                ctx.num_movable_impacted_bins_x, ctx.num_movable_impacted_bins_y,
                ctx.num_filler_impacted_bins_x, ctx.num_filler_impacted_bins_y,
                ctx.field_map_x.view([-1]), ctx.field_map_y.view([-1]),
                ctx.pos,
                ctx.node_size_x_clamped, ctx.node_size_y_clamped,
                ctx.offset_x, ctx.offset_y,
                ctx.ratio,
                ctx.bin_center_x, ctx.bin_center_y,
                ctx.xl, ctx.yl, ctx.xh, ctx.yh,
                ctx.bin_size_x, ctx.bin_size_y,
                ctx.num_movable_nodes,
                ctx.num_filler_nodes,
                ctx.sorted_node_map
            )
        else:
            output = -electric_potential_cpp.electric_force(
                grad_pos,
                ctx.num_bins_x, ctx.num_bins_y,
                ctx.num_movable_impacted_bins_x, ctx.num_movable_impacted_bins_y,
                ctx.num_filler_impacted_bins_x, ctx.num_filler_impacted_bins_y,
                ctx.field_map_x.view([-1]), ctx.field_map_y.view([-1]),
                ctx.pos,
                ctx.node_size_x_clamped, ctx.node_size_y_clamped,
                ctx.offset_x, ctx.offset_y,
                ctx.ratio,
                ctx.bin_center_x, ctx.bin_center_y,
                ctx.xl, ctx.yl, ctx.xh, ctx.yh,
                ctx.bin_size_x, ctx.bin_size_y,
                ctx.num_movable_nodes,
                ctx.num_filler_nodes,
                ctx.num_threads
            )

        #global plot_count
        # if plot_count >= 300:
        #    indices = (ctx.pos[ctx.pos.numel()/2-ctx.num_filler_nodes:ctx.pos.numel()/2] < ctx.xl+ctx.bin_size_x).nonzero()
        #    pdb.set_trace()

        #pgradx = []
        #pgrady = []
        # with open("/home/polaris/yibolin/Libraries/RePlAce/output/ispd/adaptec1.eplace/gradient.csv", "r") as f:
        #    for line in f:
        #        tokens = line.strip().split(" ")
        #        pgradx.append(float(tokens[3].strip()))
        #        pgrady.append(float(tokens[4].strip()))
        #pgrad = np.concatenate([np.array(pgradx), np.array(pgrady)])

        #output = torch.empty_like(ctx.pos).uniform_(0.0, 0.1)
        if grad_pos.is_cuda:
            torch.cuda.synchronize()
        logger.debug("density backward %.3f ms" % ((time.time()-tt)*1000))
        return output, \
            None, None, None, None, \
            None, None, None, None, \
            None, None, None, None, \
            None, None, None, None, \
            None, None, None, None, \
            None, None, None, None, \
            None, None, None, None, \
            None, None, None, None, \
            None, None, None, None, \
            None, None, None

class ElectricPotential(ElectricOverflow):
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
                 deterministic_flag, # control whether to use deterministic routine 
                 sorted_node_map,
                 movable_macro_mask=None, 
                 fast_mode=False,
                 num_threads=8
                 ):
        """
        @brief initialization
        Be aware that all scalars must be python type instead of tensors.
        Otherwise, GPU version can be weirdly slow.
        @param node_size_x cell width array consisting of movable cells, fixed cells, and filler cells in order
        @param node_size_y cell height array consisting of movable cells, fixed cells, and filler cells in order
        @param movable_macro_mask some large movable macros need to be scaled to avoid halos
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
        @param deterministic_flag control whether to use deterministic routine 
        @param fast_mode if true, only gradient is computed, while objective computation is skipped
        @param num_threads number of threads
        """
        super(ElectricPotential, self).__init__(
                node_size_x=node_size_x, node_size_y=node_size_y,
                bin_center_x=bin_center_x, bin_center_y=bin_center_y,
                target_density=target_density,
                xl=xl, yl=yl, xh=xh, yh=yh,
                bin_size_x=bin_size_x, bin_size_y=bin_size_y,
                num_movable_nodes=num_movable_nodes,
                num_terminals=num_terminals,
                num_filler_nodes=num_filler_nodes,
                padding=padding,
                deterministic_flag=deterministic_flag, 
                sorted_node_map=sorted_node_map,
                movable_macro_mask=movable_macro_mask, 
                num_threads=num_threads
                )
        self.fast_mode = fast_mode

    def reset(self): 
        """ Compute members derived from input 
        """
        super(ElectricPotential, self).reset()
        logger.info("regard %d cells as movable macros in global placement" % (self.num_movable_macros))

        self.exact_expkM = None
        self.exact_expkN = None
        self.inv_wu2_plus_wv2 = None
        self.wu_by_wu2_plus_wv2_half = None
        self.wv_by_wu2_plus_wv2_half = None

        # dct2, idct2, idct_idxst, idxst_idct functions
        self.dct2 = None
        self.idct2 = None
        self.idct_idxst = None
        self.idxst_idct = None

    def forward(self, pos):
        if self.initial_density_map is None:
            self.compute_initial_density_map(pos)
            # plot(0, self.initial_density_map.clone().div(self.bin_size_x*self.bin_size_y).cpu().numpy(), self.padding, 'summary/initial_potential_map')
            logger.info("fixed density map: average %g, max %g, bin area %g" % (self.initial_density_map.mean(), self.initial_density_map.max(), self.bin_size_x*self.bin_size_y))

            # expk
            M = self.num_bins_x
            N = self.num_bins_y
            self.exact_expkM = precompute_expk(M, dtype=pos.dtype, device=pos.device)
            self.exact_expkN = precompute_expk(N, dtype=pos.dtype, device=pos.device)

            # init dct2, idct2, idct_idxst, idxst_idct with expkM and expkN
            self.dct2 = dct.DCT2(self.exact_expkM, self.exact_expkN)
            if not self.fast_mode:
                self.idct2 = dct.IDCT2(self.exact_expkM, self.exact_expkN)
            self.idct_idxst = dct.IDCT_IDXST(self.exact_expkM, self.exact_expkN)
            self.idxst_idct = dct.IDXST_IDCT(self.exact_expkM, self.exact_expkN)

            # wu and wv
            wu = torch.arange(M, dtype=pos.dtype, device=pos.device).mul(2 * np.pi / M).view([M, 1])
            # scale wv because the aspect ratio of a bin may not be 1
            wv = torch.arange(N, dtype=pos.dtype, device=pos.device).mul(2 * np.pi / N).view([1, N]).mul_(self.bin_size_x / self.bin_size_y)
            wu2_plus_wv2 = wu.pow(2) + wv.pow(2)
            wu2_plus_wv2[0, 0] = 1.0  # avoid zero-division, it will be zeroed out
            self.inv_wu2_plus_wv2 = 1.0 / wu2_plus_wv2
            self.inv_wu2_plus_wv2[0, 0] = 0.0
            self.wu_by_wu2_plus_wv2_half = wu.mul(self.inv_wu2_plus_wv2).mul_(1./ 2)
            self.wv_by_wu2_plus_wv2_half = wv.mul(self.inv_wu2_plus_wv2).mul_(1./ 2)

        return ElectricPotentialFunction.apply(
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
            self.num_movable_nodes, self.num_filler_nodes,
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
            self.exact_expkM, self.exact_expkN,
            self.inv_wu2_plus_wv2,
            self.wu_by_wu2_plus_wv2_half, self.wv_by_wu2_plus_wv2_half,
            self.dct2, self.idct2, self.idct_idxst, self.idxst_idct,
            self.fast_mode,
            self.num_threads
        )
