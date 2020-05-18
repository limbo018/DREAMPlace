##
# @file   PlaceObj.py
# @author Yibo Lin
# @date   Jul 2018
# @brief  Placement model class defining the placement objective.
#

import os
import sys
import time
import numpy as np
import itertools
import logging
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import pdb
import gzip
if sys.version_info[0] < 3:
    import cPickle as pickle
else:
    import _pickle as pickle
import dreamplace.ops.weighted_average_wirelength.weighted_average_wirelength as weighted_average_wirelength
import dreamplace.ops.logsumexp_wirelength.logsumexp_wirelength as logsumexp_wirelength
import dreamplace.ops.electric_potential.electric_potential as electric_potential
import dreamplace.ops.density_potential.density_potential as density_potential
import dreamplace.ops.rudy.rudy as rudy
import dreamplace.ops.pin_utilization.pin_utilization as pin_utilization
import dreamplace.ops.nctugr_binary.nctugr_binary as nctugr_binary
import dreamplace.ops.adjust_node_area.adjust_node_area as adjust_node_area


class PreconditionOp:
    """Preconditioning engine is critical for convergence.
    Need to be carefully designed.
    """
    def __init__(self, placedb, data_collections):
        self.placedb = placedb
        self.data_collections = data_collections
        self.iteration = 0
        self.alpha = 1.0
        self.best_overflow = None
        self.overflows = []

    def set_overflow(self, overflow):
        self.overflows.append(overflow)
        if self.best_overflow is None:
            self.best_overflow = overflow
        else:
            self.best_overflow = min(self.best_overflow, overflow)

    def __call__(self, grad, density_weight):
        """Introduce alpha parameter to avoid divergence.
        It is tricky for this parameter to increase.
        """
        with torch.no_grad():
            #### TODO vectorized preconditioning
            if(density_weight.size(0) == 1):
                precond = self.data_collections.num_pins_in_nodes + self.alpha * density_weight * self.data_collections.node_areas
            else:
                for i in range(density_weight.size(0)):
                    node_areas = self.data_collections.node_areas.clone()
                    mask = self.data_collections.node2fence_region_map == i
                    node_areas[:self.placedb.num_movable_nodes].masked_scatter_(mask, node_areas[:self.placedb.num_movable_nodes][mask]*density_weight[i])
                    filler_beg, filler_end = self.placedb.filler_start_map[i:i+2]
                    node_areas[self.placedb.num_nodes-self.placedb.num_filler_nodes+filler_beg:self.placedb.num_nodes-self.placedb.num_filler_nodes+filler_end] *= density_weight[i]

                precond = self.data_collections.num_pins_in_nodes + self.alpha * node_areas

            precond.clamp_(min=1.0)
            grad[0:self.placedb.num_nodes].div_(precond)
            grad[self.placedb.num_nodes:self.placedb.num_nodes *
                 2].div_(precond)
            self.iteration += 1

            # assume overflow has been updated
            # if self.overflows and self.overflows[-1] < 0.3 and self.alpha < 1024:
            #     if (self.iteration % 20) == 0:
            #         self.alpha *= 2
            #         logging.info(
            #             "preconditioning alpha = %g, best_overflow %g, overflow %g"
            #             % (self.alpha, self.best_overflow, self.overflows[-1]))

        return grad


class PlaceObj(nn.Module):
    """
    @brief Define placement objective:
        wirelength + density_weight * density penalty
    It includes various ops related to global placement as well.
    """
    def __init__(self, density_weight, params, placedb, data_collections,
                 op_collections, global_place_params):
        """
        @brief initialize ops for placement
        @param density_weight density weight in the objective
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of all data and variables required for constructing the ops
        @param op_collections a collection of all ops
        @param global_place_params global placement parameters for current global placement stage
        """
        super(PlaceObj, self).__init__()

        ### quadratic penalty
        self.density_quad_coeff = 2000
        self.init_density = None
        self.quad_penalty = True

        self.params = params
        self.placedb = placedb
        self.data_collections = data_collections
        self.op_collections = op_collections
        self.global_place_params = global_place_params

        self.gpu = params.gpu
        self.data_collections = data_collections
        self.op_collections = op_collections
        if(len(placedb.regions)>0):
            ### different fence region needs different density weights in multi-electric field algorithm
            self.density_weight = torch.tensor(
                [density_weight]*(len(placedb.regions)+1),
                dtype=self.data_collections.pos[0].dtype,
                device=self.data_collections.pos[0].device)
        else:
            self.density_weight = torch.tensor(
                [density_weight],
                dtype=self.data_collections.pos[0].dtype,
                device=self.data_collections.pos[0].device)
        ### Note: even for multi-electric fields, they use the same gamma
        self.gamma = torch.tensor(10 * self.base_gamma(params, placedb),
                                  dtype=self.data_collections.pos[0].dtype,
                                  device=self.data_collections.pos[0].device)

        # compute weighted average wirelength from position
        num_bins_x = global_place_params["num_bins_x"] if global_place_params[
            "num_bins_x"] else placedb.num_bins_x
        num_bins_y = global_place_params["num_bins_y"] if global_place_params[
            "num_bins_y"] else placedb.num_bins_y
        name = "%dx%d bins" % (num_bins_x, num_bins_y)
        self.name = name
        self.num_bins_x = num_bins_x
        self.num_bins_y = num_bins_y
        if global_place_params["wirelength"] == "weighted_average":
            self.op_collections.wirelength_op, self.op_collections.update_gamma_op = self.build_weighted_average_wl(
                params, placedb, self.data_collections,
                self.op_collections.pin_pos_op)
        elif global_place_params["wirelength"] == "logsumexp":
            self.op_collections.wirelength_op, self.op_collections.update_gamma_op = self.build_logsumexp_wl(
                params, placedb, self.data_collections,
                self.op_collections.pin_pos_op)
        else:
            assert 0, "unknown wirelength model %s" % (
                global_place_params["wirelength"])
        #self.op_collections.density_op = self.build_density_potential(params, placedb, self.data_collections, num_bins_x, num_bins_y, padding=1, name)
        self.op_collections.density_op = self.build_electric_potential(
            params,
            placedb,
            self.data_collections,
            num_bins_x,
            num_bins_y,
            padding=0,
            name=name)
        ### build multiple density op for multi-electric field
        if(len(self.placedb.regions)>0):
            self.op_collections.fence_region_density_ops, self.op_collections.fence_region_density_merged_op = self.build_multi_fence_region_density_op()
        self.op_collections.update_density_weight_op = self.build_update_density_weight(
            params, placedb)
        self.op_collections.precondition_op = self.build_precondition(
            params, placedb, self.data_collections)
        self.op_collections.noise_op = self.build_noise(
            params, placedb, self.data_collections)
        if params.routability_opt_flag:
            # compute congestion map, RISA/RUDY congestion map
            self.op_collections.route_utilization_map_op = self.build_route_utilization_map(
                params, placedb, self.data_collections)
            self.op_collections.pin_utilization_map_op = self.build_pin_utilization_map(
                params, placedb, self.data_collections)
            self.op_collections.nctugr_congestion_map_op = self.build_nctugr_congestion_map(
                params, placedb, self.data_collections)
            # adjust instance area with congestion map
            self.op_collections.adjust_node_area_op = self.build_adjust_node_area(
                params, placedb, self.data_collections)

        self.Lgamma_iteration = global_place_params["iteration"]
        if 'Llambda_density_weight_iteration' in global_place_params:
            self.Llambda_density_weight_iteration = global_place_params[
                'Llambda_density_weight_iteration']
        else:
            self.Llambda_density_weight_iteration = 1
        if 'Lsub_iteration' in global_place_params:
            self.Lsub_iteration = global_place_params['Lsub_iteration']
        else:
            self.Lsub_iteration = 1
        if 'routability_Lsub_iteration' in global_place_params:
            self.routability_Lsub_iteration = global_place_params[
                'routability_Lsub_iteration']
        else:
            self.routability_Lsub_iteration = self.Lsub_iteration
        self.start_fence_region_density = False

    def obj_fn(self, pos):
        """
        @brief Compute objective.
            wirelength + density_weight * density penalty
        @param pos locations of cells
        @return objective value
        """
        wirelength = self.op_collections.wirelength_op(pos)
        if(len(self.placedb.regions) > 0):
            density = self.op_collections.fence_region_density_merged_op(pos)
        else:
            density = self.op_collections.density_op(pos)
        if(self.init_density is None):
            ### record initial density
            self.init_density = density.data.clone()
        if(self.quad_penalty):
            ### quadratic density penalty
            density = density + self.density_quad_coeff / 2 / self.init_density * density**2
        if(len(self.placedb.regions) > 0):
            result = wirelength + self.density_weight.dot(density)
        else:
            result = wirelength + self.density_weight * density

        return result

    def obj_and_grad_fn_old(self, pos_w, pos_g=None, admm_multiplier=None):
        """
        @brief compute objective and gradient.
            wirelength + density_weight * density penalty
        @param pos locations of cells
        @return objective value
        """
        #self.check_gradient(pos)
        if(not self.start_fence_region_density):
            obj = self.obj_fn(pos_w, pos_g, admm_multiplier)
            if pos_w.grad is not None:
                pos_w.grad.zero_()
            obj.backward()
        else:
            num_nodes = self.placedb.num_nodes
            num_movable_nodes = self.placedb.num_movable_nodes
            num_filler_nodes = self.placedb.num_filler_nodes


            wl = self.op_collections.wirelength_op(pos_w)
            if pos_w.grad is not None:
                pos_w.grad.zero_()
            wl.backward()
            wl_grad = pos_w.grad.data.clone()
            if pos_w.grad is not None:
                pos_w.grad.zero_()

            if(self.init_density is None):
                self.init_density = self.op_collections.density_op(pos_w.data).data.item()

            if(self.quad_penalty):
                inner_density = self.op_collections.inner_fence_region_density_op(pos_w)
                inner_density = inner_density + self.density_quad_coeff / 2 / self.init_density  * inner_density**2
            else:
                inner_density = self.op_collections.inner_fence_region_density_op(pos_w)

            inner_density.backward()
            inner_density_grad = pos_w.grad.data.clone()
            mask = self.data_collections.node2fence_region_map > 1e3
            inner_density_grad[:num_movable_nodes].masked_fill_(mask, 0)
            inner_density_grad[num_nodes:num_nodes+num_movable_nodes].masked_fill_(mask, 0)
            inner_density_grad[num_nodes-num_filler_nodes:num_nodes].mul_(0.5)
            inner_density_grad[-num_filler_nodes:].mul_(0.5)
            if pos_w.grad is not None:
                pos_w.grad.zero_()

            if(self.quad_penalty):
                outer_density = self.op_collections.outer_fence_region_density_op(pos_w)
                outer_density = outer_density + self.density_quad_coeff / 2 / self.init_density  * outer_density**2
            else:
                outer_density = self.op_collections.outer_fence_region_density_op(pos_w)

            outer_density.backward()
            outer_density_grad = pos_w.grad.data.clone()
            mask = self.data_collections.node2fence_region_map < 1e3
            outer_density_grad[:num_movable_nodes].masked_fill_(mask, 0)
            outer_density_grad[num_nodes:num_nodes+num_movable_nodes].masked_fill_(mask, 0)
            outer_density_grad[num_nodes-num_filler_nodes:num_nodes].mul_(0.5)
            outer_density_grad[-num_filler_nodes:].mul_(0.5)



            # if 0 and pos_g is not None:
            #     if pos_w.grad is not None:
            #         pos_w.grad.zero_()
            #     diff = pos_w - pos_g + admm_multiplier
            #     penalty = self.admm_rho / 2 * torch.dot(diff, diff)
            #     penalty.backward()
            #     admm_grad = pos_w.grad.data.clone()
            #     pos_w.grad.data.copy_(wl_grad + self.density_weight*(inner_density_grad + outer_density_grad) + admm_grad)
            #     obj = wl.data.item() + self.density_weight * self.op_collections.density_op(pos_w.data) + penalty

            # else:

            if(self.quad_penalty):
                density = self.op_collections.density_op(pos_w.data)
                obj = wl.data.item() + self.density_weight * (density + self.density_quad_coeff / 2 / self.init_density * density**2)
            else:
                obj = wl.data.item() + self.density_weight * self.op_collections.density_op(pos_w.data)


            pos_w.grad.data.copy_(wl_grad + self.density_weight*(inner_density_grad + outer_density_grad))


        self.op_collections.precondition_op(pos_w.grad, self.density_weight, 0)

        return obj, pos_w.grad


    def obj_and_grad_fn(self, pos):
        """
        @brief compute objective and gradient.
            wirelength + density_weight * density penalty
        @param pos locations of cells
        @return objective value
        """
        #self.check_gradient(pos)
        obj = self.obj_fn(pos)
        if pos.grad is not None:
            pos.grad.zero_()
        obj.backward()

        self.op_collections.precondition_op(pos.grad, self.density_weight)

        return obj, pos.grad


    def forward(self):
        """
        @brief Compute objective with current locations of cells.
        """
        return self.obj_fn(self.data_collections.pos[0])

    def check_gradient(self, pos):
        """
        @brief check gradient for debug
        @param pos locations of cells
        """
        wirelength = self.op_collections.wirelength_op(pos)

        if pos.grad is not None:
            pos.grad.zero_()
        wirelength.backward()
        wirelength_grad = pos.grad.clone()

        pos.grad.zero_()
        density = self.density_weight * self.op_collections.density_op(pos)
        density.backward()
        density_grad = pos.grad.clone()

        wirelength_grad_norm = wirelength_grad.norm(p=1)
        density_grad_norm = density_grad.norm(p=1)

        logging.info("wirelength_grad norm = %.6E" % (wirelength_grad_norm))
        logging.info("density_grad norm    = %.6E" % (density_grad_norm))
        pos.grad.zero_()

    def estimate_initial_learning_rate(self, x_k, lr):
        """
        @brief Estimate initial learning rate by moving a small step.
        Computed as | x_k - x_k_1 |_2 / | g_k - g_k_1 |_2.
        @param x_k current solution
        @param lr small step
        """
        obj_k, g_k = self.obj_and_grad_fn(x_k)
        x_k_1 = torch.autograd.Variable(x_k - lr * g_k, requires_grad=True)
        obj_k_1, g_k_1 = self.obj_and_grad_fn(x_k_1)

        return (x_k - x_k_1).norm(p=2) / (g_k - g_k_1).norm(p=2)

    def build_weighted_average_wl(self, params, placedb, data_collections,
                                  pin_pos_op):
        """
        @brief build the op to compute weighted average wirelength
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of data and variables required for constructing ops
        @param pin_pos_op the op to compute pin locations according to cell locations
        """

        # use WeightedAverageWirelength atomic
        wirelength_for_pin_op = weighted_average_wirelength.WeightedAverageWirelength(
            flat_netpin=data_collections.flat_net2pin_map,
            netpin_start=data_collections.flat_net2pin_start_map,
            pin2net_map=data_collections.pin2net_map,
            net_weights=data_collections.net_weights,
            net_mask=data_collections.net_mask_ignore_large_degrees,
            pin_mask=data_collections.pin_mask_ignore_fixed_macros,
            gamma=self.gamma,
            algorithm='merged')

        # wirelength for position
        def build_wirelength_op(pos):
            return wirelength_for_pin_op(pin_pos_op(pos))

        # update gamma
        base_gamma = self.base_gamma(params, placedb)

        def build_update_gamma_op(iteration, overflow):
            self.update_gamma(iteration, overflow, base_gamma)
            #logging.debug("update gamma to %g" % (wirelength_for_pin_op.gamma.data))

        return build_wirelength_op, build_update_gamma_op

    def build_logsumexp_wl(self, params, placedb, data_collections,
                           pin_pos_op):
        """
        @brief build the op to compute log-sum-exp wirelength
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of data and variables required for constructing ops
        @param pin_pos_op the op to compute pin locations according to cell locations
        """

        wirelength_for_pin_op = logsumexp_wirelength.LogSumExpWirelength(
            flat_netpin=data_collections.flat_net2pin_map,
            netpin_start=data_collections.flat_net2pin_start_map,
            pin2net_map=data_collections.pin2net_map,
            net_weights=data_collections.net_weights,
            net_mask=data_collections.net_mask_ignore_large_degrees,
            pin_mask=data_collections.pin_mask_ignore_fixed_macros,
            gamma=self.gamma,
            algorithm='merged')

        # wirelength for position
        def build_wirelength_op(pos):
            return wirelength_for_pin_op(pin_pos_op(pos))

        # update gamma
        base_gamma = self.base_gamma(params, placedb)

        def build_update_gamma_op(iteration, overflow):
            self.update_gamma(iteration, overflow, base_gamma)
            #logging.debug("update gamma to %g" % (wirelength_for_pin_op.gamma.data))

        return build_wirelength_op, build_update_gamma_op

    def build_density_potential(self, params, placedb, data_collections,
                                num_bins_x, num_bins_y, padding, name):
        """
        @brief NTUPlace3 density potential
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of data and variables required for constructing ops
        @param num_bins_x number of bins in horizontal direction
        @param num_bins_y number of bins in vertical direction
        @param padding number of padding bins to left, right, bottom, top of the placement region
        @param name string for printing
        """
        bin_size_x = (placedb.xh - placedb.xl) / num_bins_x
        bin_size_y = (placedb.yh - placedb.yl) / num_bins_y

        xl = placedb.xl - padding * bin_size_x
        xh = placedb.xh + padding * bin_size_x
        yl = placedb.yl - padding * bin_size_y
        yh = placedb.yh + padding * bin_size_y
        local_num_bins_x = num_bins_x + 2 * padding
        local_num_bins_y = num_bins_y + 2 * padding
        max_num_bins_x = np.ceil(
            (np.amax(placedb.node_size_x) + 4 * bin_size_x) / bin_size_x)
        max_num_bins_y = np.ceil(
            (np.amax(placedb.node_size_y) + 4 * bin_size_y) / bin_size_y)
        max_num_bins = max(int(max_num_bins_x), int(max_num_bins_y))
        logging.info(
            "%s #bins %dx%d, bin sizes %gx%g, max_num_bins = %d, padding = %d"
            % (name, local_num_bins_x, local_num_bins_y,
               bin_size_x / placedb.row_height,
               bin_size_y / placedb.row_height, max_num_bins, padding))
        if local_num_bins_x < max_num_bins:
            logging.warning("local_num_bins_x (%d) < max_num_bins (%d)" %
                            (local_num_bins_x, max_num_bins))
        if local_num_bins_y < max_num_bins:
            logging.warning("local_num_bins_y (%d) < max_num_bins (%d)" %
                            (local_num_bins_y, max_num_bins))

        node_size_x = placedb.node_size_x
        node_size_y = placedb.node_size_y

        # coefficients
        ax = (4 / (node_size_x + 2 * bin_size_x) /
              (node_size_x + 4 * bin_size_x)).astype(placedb.dtype).reshape(
                  [placedb.num_nodes, 1])
        bx = (2 / bin_size_x / (node_size_x + 4 * bin_size_x)).astype(
            placedb.dtype).reshape([placedb.num_nodes, 1])
        ay = (4 / (node_size_y + 2 * bin_size_y) /
              (node_size_y + 4 * bin_size_y)).astype(placedb.dtype).reshape(
                  [placedb.num_nodes, 1])
        by = (2 / bin_size_y / (node_size_y + 4 * bin_size_y)).astype(
            placedb.dtype).reshape([placedb.num_nodes, 1])

        # bell shape overlap function
        def npfx1(dist):
            # ax will be broadcast from num_nodes*1 to num_nodes*num_bins_x
            return 1.0 - ax.reshape([placedb.num_nodes, 1]) * np.square(dist)

        def npfx2(dist):
            # bx will be broadcast from num_nodes*1 to num_nodes*num_bins_x
            return bx.reshape([
                placedb.num_nodes, 1
            ]) * np.square(dist - node_size_x / 2 - 2 * bin_size_x).reshape(
                [placedb.num_nodes, 1])

        def npfy1(dist):
            # ay will be broadcast from num_nodes*1 to num_nodes*num_bins_y
            return 1.0 - ay.reshape([placedb.num_nodes, 1]) * np.square(dist)

        def npfy2(dist):
            # by will be broadcast from num_nodes*1 to num_nodes*num_bins_y
            return by.reshape([
                placedb.num_nodes, 1
            ]) * np.square(dist - node_size_y / 2 - 2 * bin_size_y).reshape(
                [placedb.num_nodes, 1])

        # should not use integral, but sum; basically sample 5 distances, -2wb, -wb, 0, wb, 2wb; the sum does not change much when shifting cells
        integral_potential_x = npfx1(0) + 2 * npfx1(bin_size_x) + 2 * npfx2(
            2 * bin_size_x)
        cx = (node_size_x.reshape([placedb.num_nodes, 1]) /
              integral_potential_x).reshape([placedb.num_nodes, 1])
        # should not use integral, but sum; basically sample 5 distances, -2wb, -wb, 0, wb, 2wb; the sum does not change much when shifting cells
        integral_potential_y = npfy1(0) + 2 * npfy1(bin_size_y) + 2 * npfy2(
            2 * bin_size_y)
        cy = (node_size_y.reshape([placedb.num_nodes, 1]) /
              integral_potential_y).reshape([placedb.num_nodes, 1])

        return density_potential.DensityPotential(
            node_size_x=data_collections.node_size_x,
            node_size_y=data_collections.node_size_y,
            ax=torch.tensor(ax.ravel(),
                            dtype=data_collections.pos[0].dtype,
                            device=data_collections.pos[0].device),
            bx=torch.tensor(bx.ravel(),
                            dtype=data_collections.pos[0].dtype,
                            device=data_collections.pos[0].device),
            cx=torch.tensor(cx.ravel(),
                            dtype=data_collections.pos[0].dtype,
                            device=data_collections.pos[0].device),
            ay=torch.tensor(ay.ravel(),
                            dtype=data_collections.pos[0].dtype,
                            device=data_collections.pos[0].device),
            by=torch.tensor(by.ravel(),
                            dtype=data_collections.pos[0].dtype,
                            device=data_collections.pos[0].device),
            cy=torch.tensor(cy.ravel(),
                            dtype=data_collections.pos[0].dtype,
                            device=data_collections.pos[0].device),
            bin_center_x=data_collections.bin_center_x_padded(padding),
            bin_center_y=data_collections.bin_center_y_padded(padding),
            target_density=data_collections.target_density,
            num_movable_nodes=placedb.num_movable_nodes,
            num_terminals=placedb.num_terminals,
            num_filler_nodes=placedb.num_filler_nodes,
            xl=xl,
            yl=yl,
            xh=xh,
            yh=yh,
            bin_size_x=bin_size_x,
            bin_size_y=bin_size_y,
            padding=padding,
            sigma=(1.0 / 16) * placedb.width / bin_size_x,
            delta=2.0)

    def build_electric_potential(self, params, placedb, data_collections,
                                 num_bins_x, num_bins_y, padding, name, region_id=None, fence_regions=None):
        """
        @brief e-place electrostatic potential
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of data and variables required for constructing ops
        @param num_bins_x number of bins in horizontal direction
        @param num_bins_y number of bins in vertical direction
        @param padding number of padding bins to left, right, bottom, top of the placement region
        @param name string for printing
        @param fence_regions a [n_subregions, 4] tensor for fence regions potential penalty
        """
        bin_size_x = (placedb.xh - placedb.xl) / num_bins_x
        bin_size_y = (placedb.yh - placedb.yl) / num_bins_y

        xl = placedb.xl - padding * bin_size_x
        xh = placedb.xh + padding * bin_size_x
        yl = placedb.yl - padding * bin_size_y
        yh = placedb.yh + padding * bin_size_y
        local_num_bins_x = num_bins_x + 2 * padding
        local_num_bins_y = num_bins_y + 2 * padding
        max_num_bins_x = np.ceil(
            (np.amax(placedb.node_size_x[0:placedb.num_movable_nodes]) +
             2 * bin_size_x) / bin_size_x)
        max_num_bins_y = np.ceil(
            (np.amax(placedb.node_size_y[0:placedb.num_movable_nodes]) +
             2 * bin_size_y) / bin_size_y)
        max_num_bins = max(int(max_num_bins_x), int(max_num_bins_y))
        logging.info(
            "%s #bins %dx%d, bin sizes %gx%g, max_num_bins = %d, padding = %d"
            % (name, local_num_bins_x, local_num_bins_y,
               bin_size_x / placedb.row_height,
               bin_size_y / placedb.row_height, max_num_bins, padding))
        if local_num_bins_x < max_num_bins:
            logging.warning("local_num_bins_x (%d) < max_num_bins (%d)" %
                            (local_num_bins_x, max_num_bins))
        if local_num_bins_y < max_num_bins:
            logging.warning("local_num_bins_y (%d) < max_num_bins (%d)" %
                            (local_num_bins_y, max_num_bins))
        #### for fence region, the target density is different from different regions
        target_density = data_collections.target_density.item() if fence_regions is None else placedb.target_density_fence_region[region_id]
        return electric_potential.ElectricPotential(
            node_size_x=data_collections.node_size_x,
            node_size_y=data_collections.node_size_y,
            bin_center_x=data_collections.bin_center_x_padded(
                placedb, padding),
            bin_center_y=data_collections.bin_center_y_padded(
                placedb, padding),
            target_density=target_density,
            xl=xl,
            yl=yl,
            xh=xh,
            yh=yh,
            bin_size_x=bin_size_x,
            bin_size_y=bin_size_y,
            num_movable_nodes=placedb.num_movable_nodes,
            num_terminals=placedb.num_terminals,
            num_filler_nodes=placedb.num_filler_nodes,
            padding=padding,
            deterministic_flag=params.deterministic_flag,
            sorted_node_map=data_collections.sorted_node_map,
            movable_macro_mask=data_collections.movable_macro_mask,
            fast_mode=params.RePlAce_skip_energy_flag,
            region_id=region_id,
            fence_regions=fence_regions,
            node2fence_region_map=data_collections.node2fence_region_map,
            placedb=placedb)

    def initialize_density_weight(self, params, placedb):
        """
        @brief compute initial density weight
        @param params parameters
        @param placedb placement database
        """
        wirelength = self.op_collections.wirelength_op(
            self.data_collections.pos[0])
        if self.data_collections.pos[0].grad is not None:
            self.data_collections.pos[0].grad.zero_()
        wirelength.backward()
        wirelength_grad_norm = self.data_collections.pos[0].grad.norm(p=1)

        self.data_collections.pos[0].grad.zero_()
        density_weight = []
        if(len(self.placedb.regions) > 0):
            density = self.op_collections.fence_region_density_merged_op(self.data_collections.pos[0]).sum()
            density.backward()
            density_grad_norm = self.data_collections.pos[0].grad.norm(p=1)
            grad_norm_ratio = wirelength_grad_norm / density_grad_norm
            density_weight = params.density_weight * grad_norm_ratio
            self.data_collections.pos[0].grad.zero_()
            self.density_weight = torch.tensor([density_weight]*(len(self.placedb.regions)+1), device=self.data_collections.pos[0].device)
        else:
            density = self.op_collections.density_op(self.data_collections.pos[0])
            density.backward()
            density_grad_norm = self.data_collections.pos[0].grad.norm(p=1)

            grad_norm_ratio = wirelength_grad_norm / density_grad_norm
            self.density_weight = torch.tensor(
                [params.density_weight * grad_norm_ratio],
                dtype=self.data_collections.pos[0].dtype,
                device=self.data_collections.pos[0].device)

        return self.density_weight

    def build_update_density_weight(self, params, placedb, algo="overflow"):
        """
        @brief update density weight
        @param params parameters
        @param placedb placement database
        """
        ### params for hpwl mode from RePlAce
        ref_hpwl = params.RePlAce_ref_hpwl
        LOWER_PCOF = params.RePlAce_LOWER_PCOF
        UPPER_PCOF = params.RePlAce_UPPER_PCOF
        ### params for overflow mode from elfPlace
        alpha_h = 1.038
        alpha_l = 1.028
        self.density_step_size = alpha_h-1
        assert algo in {"hpwl", "overflow"}, logging.error("density weight update not supports hpwl mode or overflow mode")

        def update_density_weight_op_hpwl(cur_metric, prev_metric, iteration):
            ### based on hpwl
            with torch.no_grad():
                delta_hpwl = cur_metric.hpwl - prev_metric.hpwl
                if delta_hpwl < 0:
                    mu = UPPER_PCOF * np.maximum(
                        np.power(0.9999, float(iteration)), 0.98)
                    #mu = UPPER_PCOF*np.maximum(np.power(0.9999, float(iteration)), 1.03)
                else:
                    mu = UPPER_PCOF * torch.pow(
                        UPPER_PCOF, -delta_hpwl / ref_hpwl).clamp(
                            min=LOWER_PCOF, max=UPPER_PCOF)
                self.density_weight *= mu

        def update_density_weight_op_overflow(cur_metric, prev_metric, iteration):
            ### based on overflow
            with torch.no_grad():
                density_norm = cur_metric.density/self.init_density
                density_weight_grad = density_norm + self.density_quad_coeff/2*density_norm**2
                density_weight_grad /= density_weight_grad.norm(p=2)
                # density_weight_grad = density_norm + self.density_quad_coeff / 2 * density_norm**2
                self.density_weight += self.density_step_size * density_weight_grad * 1e-11
                self.density_step_size *= torch.log(self.density_quad_coeff * density_norm + 1) / (
                    1+torch.log(self.density_quad_coeff * density_norm + 1)) * (alpha_h-alpha_l) + alpha_l

        update_density_weight_op = {"hpwl":update_density_weight_op_hpwl,
                                    "overflow": update_density_weight_op_overflow}[algo]

        return update_density_weight_op

    def base_gamma(self, params, placedb):
        """
        @brief compute base gamma
        @param params parameters
        @param placedb placement database
        """
        return params.gamma * (placedb.bin_size_x + placedb.bin_size_y)

    def update_gamma(self, iteration, overflow, base_gamma):
        """
        @brief update gamma in wirelength model
        @param iteration optimization step
        @param overflow evaluated in current step
        @param base_gamma base gamma
        """
        coef = torch.pow(10, (overflow - 0.1) * 20 / 9 - 1)
        self.gamma.data.fill_(base_gamma * coef)
        return True

    def build_noise(self, params, placedb, data_collections):
        """
        @brief add noise to cell locations
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of data and variables required for constructing ops
        """
        node_size = torch.cat(
            [data_collections.node_size_x, data_collections.node_size_y],
            dim=0).to(data_collections.pos[0].device)

        def noise_op(pos, noise_ratio):
            with torch.no_grad():
                noise = torch.rand_like(pos)
                noise.sub_(0.5).mul_(node_size).mul_(noise_ratio)
                # no noise to fixed cells
                noise[placedb.num_movable_nodes:placedb.num_nodes -
                      placedb.num_filler_nodes].zero_()
                noise[placedb.num_nodes +
                      placedb.num_movable_nodes:2 * placedb.num_nodes -
                      placedb.num_filler_nodes].zero_()
                return pos.add_(noise)

        return noise_op

    def build_precondition(self, params, placedb, data_collections):
        """
        @brief preconditioning to gradient
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of data and variables required for constructing ops
        """

        #def precondition_op(grad):
        #    with torch.no_grad():
        #        # preconditioning
        #        node_areas = data_collections.node_size_x * data_collections.node_size_y
        #        precond = self.density_weight * node_areas
        #        precond[:placedb.num_physical_nodes].add_(data_collections.pin_weights)
        #        precond.clamp_(min=1.0)
        #        grad[0:placedb.num_nodes].div_(precond)
        #        grad[placedb.num_nodes:placedb.num_nodes*2].div_(precond)
        #        #for p in pos:
        #        #    grad_norm = p.grad.norm(p=2)
        #        #    logging.debug("grad_norm = %g" % (grad_norm.data))
        #        #    p.grad.div_(grad_norm.data)
        #        #    logging.debug("grad_norm = %g" % (p.grad.norm(p=2).data))
        #        #grad.data[0:placedb.num_movable_nodes].div_(grad[0:placedb.num_movable_nodes].norm(p=2))
        #        #grad.data[placedb.num_nodes:placedb.num_nodes+placedb.num_movable_nodes].div_(grad[placedb.num_nodes:placedb.num_nodes+placedb.num_movable_nodes].norm(p=2))
        #    return grad

        #return precondition_op

        return PreconditionOp(placedb, data_collections)

    def build_route_utilization_map(self, params, placedb, data_collections):
        """
        @brief routing congestion map based on current cell locations
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of all data and variables required for constructing the ops
        """
        congestion_op = rudy.Rudy(
            netpin_start=data_collections.flat_net2pin_start_map,
            flat_netpin=data_collections.flat_net2pin_map,
            net_weights=data_collections.net_weights,
            xl=placedb.routing_grid_xl,
            yl=placedb.routing_grid_yl,
            xh=placedb.routing_grid_xh,
            yh=placedb.routing_grid_yh,
            num_bins_x=placedb.num_routing_grids_x,
            num_bins_y=placedb.num_routing_grids_y,
            unit_horizontal_capacity=placedb.unit_horizontal_capacity,
            unit_vertical_capacity=placedb.unit_vertical_capacity,
            initial_horizontal_utilization_map=data_collections.
            initial_horizontal_utilization_map,
            initial_vertical_utilization_map=data_collections.
            initial_vertical_utilization_map)

        def route_utilization_map_op(pos):
            pin_pos = self.op_collections.pin_pos_op(pos)
            return congestion_op(pin_pos)

        return route_utilization_map_op

    def build_pin_utilization_map(self, params, placedb, data_collections):
        """
        @brief pin density map based on current cell locations
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of all data and variables required for constructing the ops
        """
        return pin_utilization.PinUtilization(
            pin_weights=data_collections.pin_weights,
            flat_node2pin_start_map=data_collections.flat_node2pin_start_map,
            node_size_x=data_collections.node_size_x,
            node_size_y=data_collections.node_size_y,
            xl=placedb.routing_grid_xl,
            yl=placedb.routing_grid_yl,
            xh=placedb.routing_grid_xh,
            yh=placedb.routing_grid_yh,
            num_movable_nodes=placedb.num_movable_nodes,
            num_filler_nodes=placedb.num_filler_nodes,
            num_bins_x=placedb.num_routing_grids_x,
            num_bins_y=placedb.num_routing_grids_y,
            unit_pin_capacity=data_collections.unit_pin_capacity,
            pin_stretch_ratio=params.pin_stretch_ratio)

    def build_nctugr_congestion_map(self, params, placedb, data_collections):
        """
        @brief call NCTUgr for congestion estimation
        """
        path = "%s/%s" % (params.result_dir, params.design_name())
        return nctugr_binary.NCTUgr(
            aux_input_file=os.path.realpath(params.aux_input),
            param_setting_file="%s/../thirdparty/NCTUgr.ICCAD2012/DAC12.set" %
            (os.path.dirname(os.path.realpath(__file__))),
            tmp_pl_file="%s/%s.NCTUgr.pl" %
            (os.path.realpath(path), params.design_name()),
            tmp_output_file="%s/%s.NCTUgr" %
            (os.path.realpath(path), params.design_name()),
            horizontal_routing_capacities=torch.from_numpy(
                placedb.unit_horizontal_capacities *
                placedb.routing_grid_size_y),
            vertical_routing_capacities=torch.from_numpy(
                placedb.unit_vertical_capacities *
                placedb.routing_grid_size_x),
            params=params,
            placedb=placedb)

    def build_adjust_node_area(self, params, placedb, data_collections):
        """
        @brief adjust cell area according to routing congestion and pin utilization map
        """
        total_movable_area = (
            data_collections.node_size_x[:placedb.num_movable_nodes] *
            data_collections.node_size_y[:placedb.num_movable_nodes]).sum()
        total_filler_area = (
            data_collections.node_size_x[-placedb.num_filler_nodes:] *
            data_collections.node_size_y[-placedb.num_filler_nodes:]).sum()
        total_place_area = (total_movable_area + total_filler_area
                            ) / data_collections.target_density
        adjust_node_area_op = adjust_node_area.AdjustNodeArea(
            flat_node2pin_map=data_collections.flat_node2pin_map,
            flat_node2pin_start_map=data_collections.flat_node2pin_start_map,
            pin_weights=data_collections.pin_weights,
            xl=placedb.routing_grid_xl,
            yl=placedb.routing_grid_yl,
            xh=placedb.routing_grid_xh,
            yh=placedb.routing_grid_yh,
            num_movable_nodes=placedb.num_movable_nodes,
            num_filler_nodes=placedb.num_filler_nodes,
            route_num_bins_x=placedb.num_routing_grids_x,
            route_num_bins_y=placedb.num_routing_grids_y,
            pin_num_bins_x=placedb.num_routing_grids_x,
            pin_num_bins_y=placedb.num_routing_grids_y,
            total_place_area=total_place_area,
            total_whitespace_area=total_place_area - total_movable_area,
            max_route_opt_adjust_rate=params.max_route_opt_adjust_rate,
            route_opt_adjust_exponent=params.route_opt_adjust_exponent,
            max_pin_opt_adjust_rate=params.max_pin_opt_adjust_rate,
            area_adjust_stop_ratio=params.area_adjust_stop_ratio,
            route_area_adjust_stop_ratio=params.route_area_adjust_stop_ratio,
            pin_area_adjust_stop_ratio=params.pin_area_adjust_stop_ratio,
            unit_pin_capacity=data_collections.unit_pin_capacity)

        def build_adjust_node_area_op(pos, route_utilization_map,
                                      pin_utilization_map):
            return adjust_node_area_op(
                pos, data_collections.node_size_x,
                data_collections.node_size_y, data_collections.pin_offset_x,
                data_collections.pin_offset_y, data_collections.target_density,
                route_utilization_map, pin_utilization_map)

        return build_adjust_node_area_op

    def build_fence_region_density_op(self, fence_region_list, node2fence_region_map):
        assert type(fence_region_list) == list and len(fence_region_list) == 2, "Unsupported fence region list"
        self.data_collections.node2fence_region_map = torch.from_numpy(self.placedb.node2fence_region_map[:self.placedb.num_movable_nodes]).to(fence_region_list[0].device)
        self.op_collections.inner_fence_region_density_op = self.build_electric_potential(
            self.params,
            self.placedb,
            self.data_collections,
            self.num_bins_x,
            self.num_bins_y,
            padding=0,
            name=self.name,
            fence_regions=fence_region_list[0],
            fence_region_mask=self.data_collections.node2fence_region_map>1e3) # density penalty for inner cells
        self.op_collections.outer_fence_region_density_op = self.build_electric_potential(
            self.params,
            self.placedb,
            self.data_collections,
            self.num_bins_x,
            self.num_bins_y,
            padding=0,
            name=self.name,
            fence_regions = fence_region_list[1],
            fence_region_mask=self.data_collections.node2fence_region_map<1e3) # density penalty for outer cells

    def build_multi_fence_region_density_op(self):
        import dreamplace.ops.fence_region.fence_region as fence_region

        num_movable_nodes = self.placedb.num_movable_nodes
        num_terminals = self.placedb.num_terminals# + placedb.num_terminal_NIs
        num_filler_nodes = self.placedb.num_filler_nodes
        num_nodes = self.placedb.num_nodes

        non_fence_regions = [fence_region.slice_non_fence_region(region,
            self.placedb.xl, self.placedb.yl, self.placedb.xh, self.placedb.yh, merge=False, device=self.data_collections.pos[0].device,
            macro_pos_x=self.data_collections.pos[0][num_movable_nodes:num_movable_nodes+num_terminals],
            macro_pos_y=self.data_collections.pos[0][num_nodes+num_movable_nodes:num_nodes+num_movable_nodes+num_terminals],
            macro_size_x=self.data_collections.node_size_x[num_movable_nodes:num_movable_nodes+num_terminals],
            macro_size_y=self.data_collections.node_size_y[num_movable_nodes:num_movable_nodes+num_terminals]
            ) for region in self.placedb.regions]

        outer_fence_region = torch.from_numpy(np.concatenate(self.placedb.regions, 0)).to(self.data_collections.pos[0].device)

        fence_region_list = non_fence_regions + [outer_fence_region]


        assert type(fence_region_list) == list and len(fence_region_list) >= 2, "Unsupported fence region list"

        self.data_collections.node2fence_region_map = torch.from_numpy(self.placedb.node2fence_region_map[:self.placedb.num_movable_nodes]).to(fence_region_list[0].device)

        # region 0, ..., region n, non_fence_region
        self.op_collections.fence_region_density_ops = []

        for i, fence_region in enumerate(fence_region_list[:-1]):
            self.op_collections.fence_region_density_ops.append(self.build_electric_potential(
                        self.params,
                        self.placedb,
                        self.data_collections,
                        self.num_bins_x,
                        self.num_bins_y,
                        padding=0,
                        name=self.name,
                        region_id=i,
                        fence_regions=fence_region)
            )

        self.op_collections.fence_region_density_ops.append(self.build_electric_potential(
                        self.params,
                        self.placedb,
                        self.data_collections,
                        self.num_bins_x,
                        self.num_bins_y,
                        padding=0,
                        name=self.name,
                        region_id=len(self.placedb.regions),
                        fence_regions=fence_region_list[-1])
        )
        def merged_op(pos):
            return torch.stack([density_op(pos) for density_op in self.op_collections.fence_region_density_ops])
        self.op_collections.fence_region_density_merged_op = merged_op
        return self.op_collections.fence_region_density_ops, self.op_collections.fence_region_density_merged_op
        ### calculate filler mask for each electric field
        # for i, density_op in enumerate(self.op_collections.fence_region_density_ops):
        #     density_op.compute_fence_region_map(density_op.fence_regions)
        #     num_filler = self.placedb.num_filler_nodes_fence_region[i]#density_op.calc_num_filler()
        #     density_op.num_filler = num_filler

        # self.filler_start_map = torch.from_numpy(self.placedb.filler_start_map).to(fence_region_list[0].device)

        # for i, density_op in enumerate(self.op_collections.fence_region_density_ops):
        #     density_op.filler_start_map = filler_start_map
        #     self.filler_start_map = filler_start_map
        #     density_op.filler_beg = filler_start_map[i]
        #     density_op.filler_end = filler_start_map[i+1]


