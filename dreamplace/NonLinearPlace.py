##
# @file   NonLinearPlace.py
# @author Yibo Lin
# @date   Jul 2018
# @brief  Nonlinear placement engine to be called with parameters and placement database
#

import os
import sys
import time
import pickle
import numpy as np
import logging
import torch
import gzip
import copy
import matplotlib.pyplot as plt
if sys.version_info[0] < 3:
    import cPickle as pickle
else:
    import _pickle as pickle
import BasicPlace
import PlaceObj
import NesterovAcceleratedGradientOptimizer
import EvalMetrics
import pdb
import dreamplace.ops.fence_region.fence_region as fence_region


class NonLinearPlace(BasicPlace.BasicPlace):
    """
    @brief Nonlinear placement engine.
    It takes parameters and placement database and runs placement flow.
    """
    def __init__(self, params, placedb):
        """
        @brief initialization.
        @param params parameters
        @param placedb placement database
        """
        super(NonLinearPlace, self).__init__(params, placedb)

    def __call__(self, params, placedb):
        """
        @brief Top API to solve placement.
        @param params parameters
        @param placedb placement database
        """
        iteration = 0
        all_metrics = []

        # global placement
        if params.global_place_flag:
            # global placement may run in multiple stages according to user specification
            for global_place_params in params.global_place_stages:

                # we formulate each stage as a 3-nested optimization problem
                # f_gamma(g_density(h(x) ; density weight) ; gamma)
                # Lgamma      Llambda        Lsub
                # When optimizing an inner problem, the outer parameters are fixed.
                # This is a generalization to the eplace/RePlAce approach

                # As global placement may easily diverge, we record the position of best overflow
                best_metric = [None]
                best_pos = [None]

                if params.gpu:
                    torch.cuda.synchronize()
                tt = time.time()
                # construct model and optimizer
                density_weight = 0.0
                # construct placement model
                model = PlaceObj.PlaceObj(
                    density_weight, params, placedb, self.data_collections,
                    self.op_collections, global_place_params).to(
                        self.data_collections.pos[0].device)
                optimizer_name = global_place_params["optimizer"]

                # determine optimizer
                if optimizer_name.lower() == "adam":
                    optimizer = torch.optim.Adam(self.parameters(), lr=0)
                elif optimizer_name.lower() == "sgd":
                    optimizer = torch.optim.SGD(self.parameters(), lr=0)
                elif optimizer_name.lower() == "sgd_momentum":
                    optimizer = torch.optim.SGD(self.parameters(),
                                                lr=0,
                                                momentum=0.9,
                                                nesterov=False)
                elif optimizer_name.lower() == "sgd_nesterov":
                    optimizer = torch.optim.SGD(self.parameters(),
                                                lr=0,
                                                momentum=0.9,
                                                nesterov=True)
                elif optimizer_name.lower() == "nesterov":
                    optimizer = NesterovAcceleratedGradientOptimizer.NesterovAcceleratedGradientOptimizer(
                        self.parameters(),
                        lr=0,
                        obj_and_grad_fn=model.obj_and_grad_fn,
                        constraint_fn=self.op_collections.move_boundary_op,
                    )
                else:
                    assert 0, "unknown optimizer %s" % (optimizer_name)

                logging.info("use %s optimizer" % (optimizer_name))
                model.train()
                # defining evaluation ops
                eval_ops = {
                    #"wirelength" : self.op_collections.wirelength_op,
                    #"density" : self.op_collections.density_op,
                    #"objective" : model.obj_fn,
                    "hpwl": self.op_collections.hpwl_op,
                    "overflow": self.op_collections.density_overflow_op
                }
                if params.routability_opt_flag:
                    eval_ops.update({
                        'route_utilization':
                        self.op_collections.route_utilization_map_op,
                        'pin_utilization':
                        self.op_collections.pin_utilization_map_op
                    })

                # a function to initialize learning rate
                def initialize_learning_rate(pos):
                    learning_rate = model.estimate_initial_learning_rate(
                        pos, global_place_params["learning_rate"])
                    # update learning rate
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = learning_rate.data

                if iteration == 0:
                    if params.gp_noise_ratio > 0.0:
                        logging.info("add %g%% noise" %
                                     (params.gp_noise_ratio * 100))
                        model.op_collections.noise_op(
                            model.data_collections.pos[0],
                            params.gp_noise_ratio)
                        initialize_learning_rate(model.data_collections.pos[0])
                # the state must be saved after setting learning rate
                initial_state = copy.deepcopy(optimizer.state_dict())

                if len(placedb.regions) != 0:
                    pos_g = self.data_collections.pos[0].data.clone()
                    admm_multiplier = torch.zeros_like(self.data_collections.pos[0])
                else:
                    pos_g = None
                    admm_multiplier = None

                if params.gpu:
                    torch.cuda.synchronize()
                logging.info("%s initialization takes %g seconds" %
                             (optimizer_name, (time.time() - tt)))

                # as nesterov requires line search, we cannot follow the convention of other solvers
                if optimizer_name.lower() in {
                        "sgd", "adam", "sgd_momentum", "sgd_nesterov"
                }:
                    model.obj_and_grad_fn(model.data_collections.pos[0])
                elif optimizer_name.lower() != "nesterov":
                    assert 0, "unsupported optimizer %s" % (optimizer_name)

                # stopping criteria
                def Lgamma_stop_criterion(Lgamma_step, metrics):
                    with torch.no_grad():
                        if len(metrics) > 1:
                            cur_metric = metrics[-1][-1][-1]
                            prev_metric = metrics[-2][-1][-1]
                            if Lgamma_step > 100 and (
                                (cur_metric.overflow < params.stop_overflow
                                 and cur_metric.hpwl > prev_metric.hpwl)
                                    or cur_metric.max_density <
                                    params.target_density):
                                logging.debug(
                                    "Lgamma stopping criteria: %d > 100 and (( %g < 0.1 and %g > %g ) or %g < 1.0)"
                                    % (Lgamma_step, cur_metric.overflow,
                                       cur_metric.hpwl, prev_metric.hpwl,
                                       cur_metric.max_density))
                                return True
                        return False

                def Llambda_stop_criterion(Lgamma_step,
                                           Llambda_density_weight_step,
                                           metrics):
                    with torch.no_grad():
                        if len(metrics) > 1:
                            cur_metric = metrics[-1][-1]
                            prev_metric = metrics[-2][-1]
                            if (cur_metric.overflow < params.stop_overflow
                                    and cur_metric.hpwl > prev_metric.hpwl
                                ) or cur_metric.max_density < 1.0:
                                logging.debug(
                                    "Llambda stopping criteria: %d and (( %g < 0.1 and %g > %g ) or %g < 1.0)"
                                    %
                                    (Llambda_density_weight_step,
                                     cur_metric.overflow, cur_metric.hpwl,
                                     prev_metric.hpwl, cur_metric.max_density))
                                return True
                    return False

                # use a moving average window for stopping criteria, for an example window of 3
                # 0, 1, 2, 3, 4, 5, 6
                #    window2
                #             window1
                moving_avg_window = max(min(model.Lsub_iteration // 2, 3), 1)

                def Lsub_stop_criterion(Lgamma_step,
                                        Llambda_density_weight_step, Lsub_step,
                                        metrics):
                    with torch.no_grad():
                        if len(metrics) >= moving_avg_window * 2:
                            cur_avg_obj = 0
                            prev_avg_obj = 0
                            for i in range(moving_avg_window):
                                cur_avg_obj += metrics[-1 - i].objective
                                prev_avg_obj += metrics[-1 -
                                                        moving_avg_window -
                                                        i].objective
                            cur_avg_obj /= moving_avg_window
                            prev_avg_obj /= moving_avg_window
                            threshold = 0.999
                            if cur_avg_obj >= prev_avg_obj * threshold:
                                logging.debug(
                                    "Lsub stopping criteria: %d and %g > %g * %g"
                                    % (Lsub_step, cur_avg_obj, prev_avg_obj,
                                       threshold))
                                return True
                    return False

                def one_descent_step(Lgamma_step, Llambda_density_weight_step,
                                     Lsub_step, iteration, metrics):
                    t0 = time.time()

                    # metric for this iteration
                    cur_metric = EvalMetrics.EvalMetrics(
                        iteration,
                        (Lgamma_step, Llambda_density_weight_step, Lsub_step))
                    cur_metric.gamma = model.gamma.data
                    cur_metric.density_weight = model.density_weight.data
                    metrics.append(cur_metric)
                    pos = model.data_collections.pos[0]

                    # move any out-of-bound cell back to placement region
                    self.op_collections.move_boundary_op(pos)

                    if torch.eq(model.density_weight, 0.0):
                        model.initialize_density_weight(params, placedb)
                        logging.info("density_weight = %.6E" %
                                     (model.density_weight.data))

                    optimizer.zero_grad()

                    # t1 = time.time()
                    cur_metric.evaluate(placedb, eval_ops, pos)
                    model.overflow = cur_metric.overflow.data.clone()
                    #logging.debug("evaluation %.3f ms" % ((time.time()-t1)*1000))
                    #t2 = time.time()

                    # as nesterov requires line search, we cannot follow the convention of other solvers
                    if optimizer_name.lower() in [
                            "sgd", "adam", "sgd_momentum", "sgd_nesterov"
                    ]:
                        obj, grad = model.obj_and_grad_fn(pos)
                        cur_metric.objective = obj.data.clone()
                    elif optimizer_name.lower() != "nesterov":
                        assert 0, "unsupported optimizer %s" % (optimizer_name)

                    # plot placement
                    if params.plot_flag and iteration % 100 == 0:
                        cur_pos = self.pos[0].data.clone().cpu().numpy()
                        self.plot(params, placedb, iteration, cur_pos)

                    t3 = time.time()
                    optimizer.step(pos_g, admm_multiplier)
                    logging.info("optimizer step %.3f ms" %
                                 ((time.time() - t3) * 1000))

                    # nesterov has already computed the objective of the next step
                    if optimizer_name.lower() == "nesterov":
                        cur_metric.objective = optimizer.param_groups[0][
                            'obj_k_1'][0].data.clone()

                    # actually reports the metric before step
                    logging.info(cur_metric)
                    # record the best overflow
                    if best_metric[0] is None or best_metric[
                            0].overflow > cur_metric.overflow:
                        best_metric[0] = cur_metric
                        if best_pos[0] is None:
                            best_pos[0] = self.pos[0].data.clone()
                        else:
                            best_pos[0].data.copy_(self.pos[0].data)

                    logging.info("full step %.3f ms" %
                                 ((time.time() - t0) * 1000))

                Lgamma_metrics = all_metrics

                if params.routability_opt_flag:
                    adjust_area_flag = True
                    adjust_route_area_flag = params.adjust_route_area_flag
                    adjust_pin_area_flag = params.adjust_pin_area_flag
                    num_area_adjust = 0

                Llambda_flat_iteration = 0

                if(1 and len(placedb.regions) > 0):
                    model.quad_penalty = True
                    model.start_fence_region_density = True
                    num_movable_nodes = placedb.num_movable_nodes
                    num_terminals = placedb.num_terminals# + placedb.num_terminal_NIs
                    num_filler_nodes = placedb.num_filler_nodes
                    num_nodes = placedb.num_nodes
                    non_fence_regions_ex = fence_region.slice_non_fence_region(placedb.regions, placedb.xl, placedb.yl, placedb.xh, placedb.yh, merge=False, device=self.pos[0].device)
                    non_fence_regions = [fence_region.slice_non_fence_region(region,
                        placedb.xl, placedb.yl, placedb.xh, placedb.yh, merge=False, device=self.pos[0].device,
                        macro_pos_x=self.pos[0][num_movable_nodes:num_movable_nodes+num_terminals],
                        macro_pos_y=self.pos[0][num_nodes+num_movable_nodes:num_nodes+num_movable_nodes+num_terminals],
                        macro_size_x=self.data_collections.node_size_x[num_movable_nodes:num_movable_nodes+num_terminals],
                        macro_size_y=self.data_collections.node_size_y[num_movable_nodes:num_movable_nodes+num_terminals]
                        ) for region in placedb.regions]
                    inner_fence_region = fence_region.slice_non_fence_region(
                        placedb.regions,
                        placedb.xl, placedb.yl, placedb.xh, placedb.yh, merge=False,
                        macro_pos_x=self.pos[0][num_movable_nodes:num_movable_nodes+num_terminals],
                        macro_pos_y=self.pos[0][num_nodes+num_movable_nodes:num_nodes+num_movable_nodes+num_terminals],
                        macro_size_x=self.data_collections.node_size_x[num_movable_nodes:num_movable_nodes+num_terminals],
                        macro_size_y=self.data_collections.node_size_y[num_movable_nodes:num_movable_nodes+num_terminals],
                        device=self.pos[0].device
                        )# macro padded for inner cells
                    outer_fence_region = torch.from_numpy(np.concatenate(placedb.regions, 0)).to(self.pos[0].device)
                    # fence_region_list = [inner_fence_region, outer_fence_region]
                    fence_region_list = non_fence_regions + [outer_fence_region]
                    # model.build_fence_region_density_op(fence_region_list, placedb.node2fence_region_map)
                    model.build_multi_fence_region_density_op(fence_region_list, placedb.node2fence_region_map)


                for Lgamma_step in range(model.Lgamma_iteration):
                    Lgamma_metrics.append([])
                    Llambda_metrics = Lgamma_metrics[-1]
                    for Llambda_density_weight_step in range(
                            model.Llambda_density_weight_iteration):
                        Llambda_metrics.append([])
                        Lsub_metrics = Llambda_metrics[-1]
                        for Lsub_step in range(model.Lsub_iteration):
                            one_descent_step(Lgamma_step,
                                             Llambda_density_weight_step,
                                             Lsub_step, iteration,
                                             Lsub_metrics)
                            iteration += 1
                            # stopping criteria
                            if Lsub_stop_criterion(
                                    Lgamma_step, Llambda_density_weight_step,
                                    Lsub_step, Lsub_metrics):
                                break
                        Llambda_flat_iteration += 1
                        # update density weight
                        if Llambda_flat_iteration > 1:
                            model.op_collections.update_density_weight_op(
                                Llambda_metrics[-1][-1],
                                Llambda_metrics[-2][-1]
                                if len(Llambda_metrics) > 1 else
                                Lgamma_metrics[-2][-1][-1],
                                Llambda_flat_iteration)
                        #logging.debug("update density weight %.3f ms" % ((time.time()-t2)*1000))
                        if Llambda_stop_criterion(Lgamma_step,
                                                  Llambda_density_weight_step,
                                                  Llambda_metrics):
                            break

                        # for routability optimization
                        if params.routability_opt_flag and num_area_adjust < params.max_num_area_adjust and Llambda_metrics[
                                -1][-1].overflow < params.node_area_adjust_overflow:
                            content = "routability optimization round %d: adjust area flags = (%d, %d, %d)" % (
                                num_area_adjust, adjust_area_flag,
                                adjust_route_area_flag, adjust_pin_area_flag)
                            pos = model.data_collections.pos[0]

                            #cur_metric = EvalMetrics.EvalMetrics(iteration)
                            #cur_metric.evaluate(placedb, {
                            #    "hpwl" : self.op_collections.hpwl_op,
                            #    "overflow" : self.op_collections.density_overflow_op,
                            #    "route_utilization" : self.op_collections.route_utilization_map_op,
                            #    "pin_utilization" : self.op_collections.pin_utilization_map_op,
                            #    },
                            #    pos)
                            #logging.info(cur_metric)

                            route_utilization_map = None
                            pin_utilization_map = None
                            if adjust_route_area_flag:
                                #route_utilization_map = model.op_collections.route_utilization_map_op(pos)
                                route_utilization_map = model.op_collections.nctugr_congestion_map_op(
                                    pos)
                                #if params.plot_flag:
                                path = "%s/%s" % (params.result_dir,
                                                  params.design_name())
                                figname = "%s/plot/rudy%d.png" % (
                                    path, num_area_adjust)
                                os.system("mkdir -p %s" %
                                          (os.path.dirname(figname)))
                                plt.imsave(
                                    figname,
                                    route_utilization_map.data.cpu().numpy().T,
                                    origin='lower')
                            if adjust_pin_area_flag:
                                pin_utilization_map = model.op_collections.pin_utilization_map_op(
                                    pos)
                                #if params.plot_flag:
                                path = "%s/%s" % (params.result_dir,
                                                  params.design_name())
                                figname = "%s/plot/pin%d.png" % (
                                    path, num_area_adjust)
                                os.system("mkdir -p %s" %
                                          (os.path.dirname(figname)))
                                plt.imsave(
                                    figname,
                                    pin_utilization_map.data.cpu().numpy().T,
                                    origin='lower')
                            adjust_area_flag, adjust_route_area_flag, adjust_pin_area_flag = model.op_collections.adjust_node_area_op(
                                pos, route_utilization_map,
                                pin_utilization_map)
                            content += " -> (%d, %d, %d)" % (
                                adjust_area_flag, adjust_route_area_flag,
                                adjust_pin_area_flag)
                            logging.info(content)
                            if adjust_area_flag:
                                num_area_adjust += 1
                                # restart Llambda
                                model.op_collections.density_op.reset()
                                model.op_collections.density_overflow_op.reset(
                                )
                                model.op_collections.pin_utilization_map_op.reset(
                                )
                                model.initialize_density_weight(
                                    params, placedb)
                                model.density_weight.mul_(
                                    0.1 / params.density_weight)
                                logging.info("density_weight = %.6E" %
                                             (model.density_weight.data))
                                # load state to restart the optimizer
                                optimizer.load_state_dict(initial_state)
                                # must after loading the state
                                initialize_learning_rate(pos)
                                # increase iterations of the sub problem to slow down the search
                                model.Lsub_iteration = model.routability_Lsub_iteration

                                #cur_metric = EvalMetrics.EvalMetrics(iteration)
                                #cur_metric.evaluate(placedb, {
                                #    "hpwl" : self.op_collections.hpwl_op,
                                #    "overflow" : self.op_collections.density_overflow_op,
                                #    "route_utilization" : self.op_collections.route_utilization_map_op,
                                #    "pin_utilization" : self.op_collections.pin_utilization_map_op,
                                #    },
                                #    pos)
                                #logging.info(cur_metric)

                                break

                    # gradually reduce gamma to tradeoff smoothness and accuracy
                    model.op_collections.update_gamma_op(
                        Lgamma_step, Llambda_metrics[-1][-1].overflow)
                    model.op_collections.precondition_op.set_overflow(
                        Llambda_metrics[-1][-1].overflow)
                    if Lgamma_stop_criterion(Lgamma_step, Lgamma_metrics):
                        break

                    # update learning rate
                    if optimizer_name.lower() in [
                            "sgd", "adam", "sgd_momentum", "sgd_nesterov", "cg"
                    ]:
                        if 'learning_rate_decay' in global_place_params:
                            for param_group in optimizer.param_groups:
                                param_group['lr'] *= global_place_params[
                                    'learning_rate_decay']

                    def solve_problem_2_old(pos_w, admm_multiplier):
                        num_nodes = placedb.num_nodes
                        num_movable_nodes = placedb.num_movable_nodes

                        pos_g = pos_w + admm_multiplier # minimize the L2 norm
                        node2fence_region_map = torch.from_numpy(placedb.node2fence_region_map[:num_movable_nodes]).to(pos_g.device)

                        pos_x, pos_y = pos_g[:num_movable_nodes], pos_g[num_nodes:num_nodes + num_movable_nodes]
                        node_size_x, node_size_y = model.data_collections.node_size_x[:num_movable_nodes], model.data_collections.node_size_y[:num_movable_nodes]
                        num_regions = len(placedb.regions)
                        exclude_mask = (node2fence_region_map > 1e3)
                        pos_x_ex, pos_y_ex = pos_x[exclude_mask], pos_y[exclude_mask]
                        node_size_x_ex, node_size_y_ex = node_size_x[exclude_mask], node_size_y[exclude_mask]
                        regions = placedb.regions
                        for i in range(num_regions):
                            mask = (node2fence_region_map == i)
                            pos_x_i, pos_y_i = pos_x[mask], pos_y[mask]
                            num_movable_nodes_i = pos_x_i.numel()
                            node_size_x_i, node_size_y_i = node_size_x[mask], node_size_y[mask]
                            regions_i = regions[i] # [n_regions, 4]
                            delta_min = torch.empty(num_movable_nodes_i, device=pos_x.device).fill_(((placedb.xh-placedb.xl)**2+(placedb.yh-placedb.yl)**2))
                            delta_x_min = torch.zeros_like(delta_min)
                            delta_y_min = torch.zeros_like(delta_min)
                            margin = 12
                            for sub_region in regions_i:
                                delta_x = torch.zeros_like(delta_min)
                                delta_y = torch.zeros_like(delta_min)
                                xl, yl, xh, yh = sub_region
                                # on the left
                                mask_l = pos_x_i < xl + margin
                                # on the right
                                pos_xh_i = pos_x_i + node_size_x_i
                                mask_r = pos_xh_i > xh - margin
                                # on the top
                                pos_yh_i = pos_y_i + node_size_y_i
                                mask_t = pos_yh_i > yh - margin
                                # on the bottom
                                mask_b = pos_y_i < yl + margin

                                # x replacement for left cell
                                delta_x.masked_scatter_(mask_l, xl + margin - pos_x_i[mask_l])
                                # x replacement for right cell
                                delta_x.masked_scatter_(mask_r, xh - margin - pos_xh_i[mask_r])
                                # delta_x.masked_fill_(~(mask_l | mask_r), 0)
                                # y replacement for top cell
                                delta_y.masked_scatter_(mask_t, yh - margin - pos_yh_i[mask_t])
                                # y replacement for bottom cell
                                delta_y.masked_scatter_(mask_b, yl + margin - pos_y_i[mask_b])
                                # delta_y.masked_fill_(~(mask_t | mask_b), 0)
                                # update minimum replacement
                                delta_i = (delta_x ** 2 + delta_y ** 2)
                                update_mask = delta_i < delta_min

                                delta_x_min.masked_scatter_(update_mask, delta_x[update_mask])
                                delta_y_min.masked_scatter_(update_mask, delta_y[update_mask])
                                delta_min.masked_scatter_(update_mask, delta_i)

                                ##### move excluded cells out of the region
                                pos_x_ex_new = pos_x_ex.clone()
                                pos_y_ex_new = pos_y_ex.clone()
                                move_out_mask = (pos_x_ex < xh+margin) & ((pos_x_ex + node_size_x_ex) > xl-margin) & ((pos_y_ex + node_size_y_ex) > yl-margin) & (pos_y_ex < yh+margin)
                                # print("move out:", move_out_mask.float().sum().data.item(), "out of", move_out_mask.numel())
                                delta_x_ex_l = xl - (pos_x_ex[move_out_mask] + node_size_x_ex[move_out_mask])
                                delta_x_ex_l_abs = delta_x_ex_l.abs()
                                delta_x_ex_r = xh - pos_x_ex[move_out_mask]
                                delta_y_ex_b = yl - (pos_y_ex[move_out_mask] + node_size_y_ex[move_out_mask])
                                delta_y_ex_b_abs = delta_y_ex_b.abs()
                                delta_y_ex_t = yh - pos_y_ex[move_out_mask]

                                move_left_mask = (delta_x_ex_l_abs < delta_x_ex_r) & (delta_x_ex_l_abs < delta_y_ex_b_abs) & (delta_x_ex_l_abs < delta_y_ex_t)
                                move_right_mask = (delta_x_ex_r <= delta_x_ex_l_abs) & (delta_x_ex_r < delta_y_ex_b_abs) & (delta_x_ex_r < delta_y_ex_t)
                                move_top_mask = (delta_y_ex_t < delta_x_ex_l_abs) & (delta_y_ex_t < delta_x_ex_r) & (delta_y_ex_t < delta_y_ex_b_abs)
                                move_btm_mask = (delta_y_ex_b_abs < delta_x_ex_l_abs) & (delta_y_ex_b_abs < delta_x_ex_r) & (delta_y_ex_b_abs <= delta_y_ex_t)

                                pos_x_ex_out = pos_x_ex[move_out_mask]
                                pos_x_ex_out.masked_scatter_(move_left_mask, xl - margin- node_size_x_ex[move_out_mask] )
                                pos_x_ex_out.masked_fill_(move_right_mask, xh + margin)
                                pos_x_ex.masked_scatter_(move_out_mask, pos_x_ex_out)

                                pos_y_ex_out = pos_y_ex[move_out_mask]
                                pos_y_ex_out.masked_scatter_(move_btm_mask, yl - margin - node_size_y_ex[move_out_mask])
                                pos_y_ex_out.masked_fill_(move_top_mask, yh + margin)
                                pos_y_ex.masked_scatter_(move_out_mask, pos_y_ex_out)

                                pos_x.masked_scatter_(exclude_mask, pos_x_ex)
                                pos_y.masked_scatter_(exclude_mask, pos_y_ex)

                                ## check validity
                                # move_out_mask = (pos_x[exclude_mask] < xh) & ((pos_x[exclude_mask] + node_size_x_ex) > xl) & ((pos_y[exclude_mask] + node_size_y_ex) > yl) & (pos_y[exclude_mask] < yh)
                                # error = move_out_mask.float().sum()
                                # print(f"{error} cells are still in region")
                                # assert error < 1e-3
                            # update the minimum replacement for subregions
                            pos_x.masked_scatter_(mask, pos_x_i + delta_x_min)
                            pos_y.masked_scatter_(mask, pos_y_i + delta_y_min)
                        res = pos_g.data.clone()
                        res.data[:num_movable_nodes].copy_(pos_x)
                        res.data[num_nodes:num_nodes + num_movable_nodes].copy_(pos_y)
                        return res


                    def solve_problem_2(pos_w, admm_multiplier, non_fence_regions_ex, non_fence_regions, iteration):
                        def check_valid(regions, pos_x, pos_y, pos_xh, pos_yh, valid_margin_x=0, valid_margin_y=0):
                            if(type(regions) == list):
                                regions = np.concatenate(regions,0)
                            valid_mask = torch.ones_like(pos_x, dtype=torch.bool)
                            for sub_region in regions:
                                xll, yll, xhh, yhh = sub_region
                                valid_margin_x = min((xhh-xll)/2, valid_margin_x)
                                valid_margin_y = min((yhh-yll)/2, valid_margin_y)
                                valid_mask.masked_fill_((pos_x < xhh-valid_margin_x) & (pos_xh > xll+valid_margin_x) & (pos_y < yhh-valid_margin_y) & (pos_yh > yll+valid_margin_y), 0)
                            return valid_mask

                        num_nodes = placedb.num_nodes
                        num_movable_nodes = placedb.num_movable_nodes

                        pos_g = pos_w + admm_multiplier # minimize the L2 norm
                        # node2fence_region_map = torch.from_numpy(placedb.node2fence_region_map[:num_movable_nodes]).to(pos_g.device)
                        node2fence_region_map = self.data_collections.node2fence_region_map

                        pos_x, pos_y = pos_g[:num_movable_nodes], pos_g[num_nodes:num_nodes + num_movable_nodes]
                        node_size_x, node_size_y = model.data_collections.node_size_x[:num_movable_nodes], model.data_collections.node_size_y[:num_movable_nodes]
                        num_regions = len(placedb.regions)

                        regions = placedb.regions
                        # margin = 20 * 0.997**iteration
                        margin_x = placedb.bin_size_x * min(1,4*0.997**iteration)
                        margin_y = placedb.bin_size_y * min(1,4*0.997**iteration)

                        # valid_margin = 1000 * 0.995**iteration
                        valid_margin_x = placedb.bin_size_x * 200*0.996**iteration
                        valid_margin_y = placedb.bin_size_y * 200*0.996**iteration
                        # valid_margin = 0 if valid_margin < 5 else valid_margin
                        ### move cells into fence regions
                        for i in range(num_regions):
                            mask = (node2fence_region_map == i)
                            pos_x_i, pos_y_i = pos_x[mask], pos_y[mask]
                            num_movable_nodes_i = pos_x_i.numel()
                            node_size_x_i, node_size_y_i = node_size_x[mask], node_size_y[mask]
                            pos_xh_i = pos_x_i + node_size_x_i
                            pos_yh_i = pos_y_i + node_size_y_i
                            regions_i = regions[i] # [n_regions, 4]
                            delta_min = torch.empty(num_movable_nodes_i, device=pos_x.device).fill_(((placedb.xh-placedb.xl)**2+(placedb.yh-placedb.yl)**2))
                            delta_x_min = torch.zeros_like(delta_min)
                            delta_y_min = torch.zeros_like(delta_min)

                            valid_mask = check_valid(non_fence_regions[i], pos_x_i, pos_y_i, pos_xh_i, pos_yh_i, valid_margin_x, valid_margin_y)

                            for sub_region in regions_i:
                                delta_x = torch.zeros_like(delta_min)
                                delta_y = torch.zeros_like(delta_min)
                                xl, yl, xh, yh = sub_region

                                # on the left
                                mask_l = (pos_x_i < xl + margin_x).masked_fill_(valid_mask, 0)
                                # on the right
                                mask_r = (pos_xh_i > xh - margin_x).masked_fill_(valid_mask, 0)
                                # on the top
                                mask_t = (pos_yh_i > yh - margin_y).masked_fill_(valid_mask, 0)
                                # on the bottom
                                mask_b = (pos_y_i < yl + margin_y).masked_fill_(valid_mask, 0)

                                # x replacement for left cell
                                delta_x.masked_scatter_(mask_l, xl + margin_x - pos_x_i[mask_l])
                                # x replacement for right cell
                                delta_x.masked_scatter_(mask_r, xh - margin_x - pos_xh_i[mask_r])
                                # delta_x.masked_fill_(~(mask_l | mask_r), 0)
                                # y replacement for top cell
                                delta_y.masked_scatter_(mask_t, yh - margin_y - pos_yh_i[mask_t])
                                # y replacement for bottom cell
                                delta_y.masked_scatter_(mask_b, yl + margin_y - pos_y_i[mask_b])
                                # delta_y.masked_fill_(~(mask_t | mask_b), 0)
                                # update minimum replacement
                                delta_i = (delta_x ** 2 + delta_y ** 2)
                                update_mask = delta_i < delta_min

                                delta_x_min.masked_scatter_(update_mask, delta_x[update_mask])
                                delta_y_min.masked_scatter_(update_mask, delta_y[update_mask])
                                delta_min.masked_scatter_(update_mask, delta_i[update_mask])

                            # update the minimum replacement for subregions
                            pos_x.masked_scatter_(mask, pos_x_i + delta_x_min)
                            pos_y.masked_scatter_(mask, pos_y_i + delta_y_min)

                        ### move cells out of fence regions
                        # margin = 0
                        # valid_margin = 100 * 0.99**iteration
                        exclude_mask = (node2fence_region_map > 1e3)
                        pos_x_ex, pos_y_ex = pos_x[exclude_mask], pos_y[exclude_mask]
                        node_size_x_ex, node_size_y_ex = node_size_x[exclude_mask], node_size_y[exclude_mask]
                        pos_xh_ex = pos_x_ex + node_size_x_ex
                        pos_yh_ex = pos_y_ex + node_size_y_ex

                        delta_min = torch.empty(pos_x_ex.numel(), device=pos_x.device).fill_(((placedb.xh-placedb.xl)**2+(placedb.yh-placedb.yl)**2))
                        delta_x_min = torch.zeros_like(delta_min)
                        delta_y_min = torch.zeros_like(delta_min)
                        ### don't move valid cells
                        valid_mask = check_valid(regions, pos_x_ex, pos_y_ex, pos_xh_ex, pos_yh_ex, valid_margin_x, valid_margin_y)

                        for sub_region in non_fence_regions_ex:
                            delta_x = torch.zeros_like(delta_min)
                            delta_y = torch.zeros_like(delta_min)
                            xl, yl, xh, yh = sub_region

                            # on the left
                            mask_l = (pos_x_ex < xl).masked_fill_(valid_mask, 0)
                            # on the right
                            mask_r = (pos_xh_ex > xh).masked_fill_(valid_mask, 0)
                            # on the top
                            mask_t = (pos_yh_ex > yh).masked_fill_(valid_mask, 0)
                            # on the bottom
                            mask_b = (pos_y_ex < yl).masked_fill_(valid_mask, 0)

                            # x replacement for left cell
                            delta_x.masked_scatter_(mask_l, xl + margin_x - pos_x_ex[mask_l])
                            # x replacement for right cell
                            delta_x.masked_scatter_(mask_r, xh - margin_x - pos_xh_ex[mask_r])
                            # delta_x.masked_fill_(~(mask_l | mask_r), 0)
                            # y replacement for top cell
                            delta_y.masked_scatter_(mask_t, yh - margin_y - pos_yh_ex[mask_t])
                            # y replacement for bottom cell
                            delta_y.masked_scatter_(mask_b, yl + margin_y - pos_y_ex[mask_b])
                            # delta_y.masked_fill_(~(mask_t | mask_b), 0)
                            # update minimum replacement
                            delta_i = (delta_x ** 2 + delta_y ** 2)
                            update_mask = delta_i < delta_min

                            delta_x_min.masked_scatter_(update_mask, delta_x[update_mask])
                            delta_y_min.masked_scatter_(update_mask, delta_y[update_mask])
                            delta_min.masked_scatter_(update_mask, delta_i[update_mask])

                        # update the minimum replacement for subregions
                        pos_x.masked_scatter_(exclude_mask, pos_x_ex + delta_x_min)
                        pos_y.masked_scatter_(exclude_mask, pos_y_ex + delta_y_min)

                        ### write back solution
                        res = pos_g.data.clone()
                        res.data[:num_movable_nodes].copy_(pos_x)
                        res.data[num_nodes:num_nodes + num_movable_nodes].copy_(pos_y)
                        return res


                    if 0 and len(placedb.regions) != 0 and iteration % 1 == 0:
                    # if(1 and iteration == 980):
                        # if(iteration % 10 == 0):
                        #     self.plot(params, placedb, iteration-1,self.pos[0].data.clone().cpu().numpy())
                        if(not model.start_fence_region_density):
                            pos_w = self.pos[0]
                            pos_g = solve_problem_2(pos_w, admm_multiplier, non_fence_regions_ex, non_fence_regions, iteration)
                            admm_multiplier += pos_w - pos_g
                            self.pos[0].data.copy_(pos_g)
                        if(iteration % 20 == 0):
                            self.plot(params, placedb, iteration,self.pos[0].data.clone().cpu().numpy())
                    else:
                        # if(iteration == 2):
                        #     pos_g = solve_problem_2(self.pos[0].data, 0, non_fence_regions_ex, non_fence_regions, iteration)
                        #     self.pos[0].data.copy_(pos_g)
                        if(iteration % 50 == 0):
                            self.plot(params, placedb, iteration,self.pos[0].data.clone().cpu().numpy())



                # in case of divergence, use the best metric
                ### always rollback to best overflow
                self.pos[0].data.copy_(best_pos[0].data)
                logging.error(
                        "possible DIVERGENCE detected, roll back to the best position recorded"
                    )
                last_metric = all_metrics[-1][-1][-1]
                if last_metric.overflow > max(
                        params.stop_overflow, best_metric[0].overflow
                ) and last_metric.hpwl > best_metric[0].hpwl:
                    self.pos[0].data.copy_(best_pos[0].data)
                    logging.error(
                        "possible DIVERGENCE detected, roll back to the best position recorded"
                    )
                    all_metrics.append([best_metric])
                    logging.info(best_metric[0])

                logging.info("optimizer %s takes %.3f seconds" %
                             (optimizer_name, time.time() - tt))

            # recover node size and pin offset for legalization, since node size is adjusted in global placement
            if params.routability_opt_flag:
                with torch.no_grad():
                    # convert lower left to centers
                    self.pos[0][:placedb.num_movable_nodes].add_(
                        self.data_collections.
                        node_size_x[:placedb.num_movable_nodes] / 2)
                    self.pos[0][placedb.num_nodes:placedb.num_nodes +
                                placedb.num_movable_nodes].add_(
                                    self.data_collections.
                                    node_size_y[:placedb.num_movable_nodes] /
                                    2)
                    self.data_collections.node_size_x.copy_(
                        self.data_collections.original_node_size_x)
                    self.data_collections.node_size_y.copy_(
                        self.data_collections.original_node_size_y)
                    # use fixed centers as the anchor
                    self.pos[0][:placedb.num_movable_nodes].sub_(
                        self.data_collections.
                        node_size_x[:placedb.num_movable_nodes] / 2)
                    self.pos[0][placedb.num_nodes:placedb.num_nodes +
                                placedb.num_movable_nodes].sub_(
                                    self.data_collections.
                                    node_size_y[:placedb.num_movable_nodes] /
                                    2)
                    self.data_collections.pin_offset_x.copy_(
                        self.data_collections.original_pin_offset_x)
                    self.data_collections.pin_offset_y.copy_(
                        self.data_collections.original_pin_offset_y)

        else:
            cur_metric = EvalMetrics.EvalMetrics(iteration)
            all_metrics.append(cur_metric)
            cur_metric.evaluate(placedb, {"hpwl": self.op_collections.hpwl_op},
                                self.pos[0])
            logging.info(cur_metric)

        # dump global placement solution for legalization
        if params.dump_global_place_solution_flag:
            self.dump(params, placedb, self.pos[0].cpu(),
                      "%s.lg.pklz" % (params.design_name()))

        # plot placement
        if params.plot_flag:
            self.plot(params, placedb, iteration,
                      self.pos[0].data.clone().cpu().numpy())

        # legalization
        if params.legalize_flag:
            assert len(
                placedb.regions
            ) == 0, "FENCE REGIONS are not supported in legalization yet"
            tt = time.time()
            self.pos[0].data.copy_(self.op_collections.legalize_op(
                self.pos[0]))
            logging.info("legalization takes %.3f seconds" %
                         (time.time() - tt))
            cur_metric = EvalMetrics.EvalMetrics(iteration)
            all_metrics.append(cur_metric)
            cur_metric.evaluate(placedb, {"hpwl": self.op_collections.hpwl_op},
                                self.pos[0])
            logging.info(cur_metric)
            iteration += 1

        # plot placement
        if params.plot_flag:
            self.plot(params, placedb, iteration,
                      self.pos[0].data.clone().cpu().numpy())

        # dump legalization solution for detailed placement
        if params.dump_legalize_solution_flag:
            self.dump(params, placedb, self.pos[0].cpu(),
                      "%s.dp.pklz" % (params.design_name()))

        # detailed placement
        if params.detailed_place_flag:
            tt = time.time()
            self.pos[0].data.copy_(
                self.op_collections.detailed_place_op(self.pos[0]))
            logging.info("detailed placement takes %.3f seconds" %
                         (time.time() - tt))
            cur_metric = EvalMetrics.EvalMetrics(iteration)
            all_metrics.append(cur_metric)
            cur_metric.evaluate(placedb, {"hpwl": self.op_collections.hpwl_op},
                                self.pos[0])
            logging.info(cur_metric)
            iteration += 1

        # save results
        cur_pos = self.pos[0].data.clone().cpu().numpy()
        # apply solution
        placedb.apply(
            params, cur_pos[0:placedb.num_movable_nodes],
            cur_pos[placedb.num_nodes:placedb.num_nodes +
                    placedb.num_movable_nodes])
        # plot placement
        if params.plot_flag:
            self.plot(params, placedb, iteration, cur_pos)
        return all_metrics
