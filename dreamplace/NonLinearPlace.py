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
                    density_weight,
                    params,
                    placedb,
                    self.data_collections,
                    self.op_collections,
                    global_place_params,
                ).to(self.data_collections.pos[0].device)
                optimizer_name = global_place_params["optimizer"]

                # determine optimizer
                if optimizer_name.lower() == "adam":
                    optimizer = torch.optim.Adam(self.parameters(), lr=0)
                elif optimizer_name.lower() == "sgd":
                    optimizer = torch.optim.SGD(self.parameters(), lr=0)
                elif optimizer_name.lower() == "sgd_momentum":
                    optimizer = torch.optim.SGD(self.parameters(), lr=0, momentum=0.9, nesterov=False)
                elif optimizer_name.lower() == "sgd_nesterov":
                    optimizer = torch.optim.SGD(self.parameters(), lr=0, momentum=0.9, nesterov=True)
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
                    # "wirelength" : self.op_collections.wirelength_op,
                    # "density" : self.op_collections.density_op,
                    # "objective" : model.obj_fn,
                    "hpwl": self.op_collections.hpwl_op,
                    "overflow": self.op_collections.density_overflow_op,
                }
                if params.routability_opt_flag:
                    eval_ops.update(
                        {
                            "route_utilization": self.op_collections.route_utilization_map_op,
                            "pin_utilization": self.op_collections.pin_utilization_map_op,
                        }
                    )
                if len(placedb.regions) > 0:
                    eval_ops.update(
                        {
                            "density": self.op_collections.fence_region_density_merged_op,
                            "overflow": self.op_collections.fence_region_density_overflow_merged_op,
                            "goverflow": self.op_collections.density_overflow_op,
                        }
                    )

                # a function to initialize learning rate
                def initialize_learning_rate(pos):
                    learning_rate = model.estimate_initial_learning_rate(
                        pos, global_place_params["learning_rate"]
                    )
                    # update learning rate
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = learning_rate.data

                if iteration == 0:
                    if params.gp_noise_ratio > 0.0:
                        logging.info("add %g%% noise" % (params.gp_noise_ratio * 100))
                        model.op_collections.noise_op(model.data_collections.pos[0], params.gp_noise_ratio)
                        initialize_learning_rate(model.data_collections.pos[0])
                # the state must be saved after setting learning rate
                initial_state = copy.deepcopy(optimizer.state_dict())

                if params.gpu:
                    torch.cuda.synchronize()
                logging.info("%s initialization takes %g seconds" % (optimizer_name, (time.time() - tt)))

                # as nesterov requires line search, we cannot follow the convention of other solvers
                if optimizer_name.lower() in {"sgd", "adam", "sgd_momentum", "sgd_nesterov"}:
                    model.obj_and_grad_fn(model.data_collections.pos[0])
                elif optimizer_name.lower() != "nesterov":
                    assert 0, "unsupported optimizer %s" % (optimizer_name)

                # stopping criteria
                def Lgamma_stop_criterion(Lgamma_step, metrics, stop_mask=None):
                    with torch.no_grad():
                        if len(metrics) > 1:
                            cur_metric = metrics[-1][-1][-1]
                            prev_metric = metrics[-2][-1][-1]
                            ### update stop mask for each fence region
                            # if(stop_mask is not None):
                            #     stop_mask.copy_(cur_metric.overflow < params.stop_overflow)

                            if Lgamma_step > 100 and (
                                ### for fence region, the outer cell overflow decides the stopping of GP
                                (
                                    cur_metric.overflow[-1] < params.stop_overflow
                                    and cur_metric.hpwl > prev_metric.hpwl
                                )
                                or cur_metric.max_density[-1] < params.target_density
                            ):
                                logging.debug(
                                    "Lgamma stopping criteria: %d > 100 and (( %g < 0.1 and %g > %g ) or %g < 1.0)"
                                    % (
                                        Lgamma_step,
                                        cur_metric.overflow[-1],
                                        cur_metric.hpwl,
                                        prev_metric.hpwl,
                                        cur_metric.max_density[-1],
                                    )
                                )
                                return True
                            if len(placedb.regions) > 0 and model.update_mask.sum() == 0:
                                logging.debug("All regions stop updating, finish global placement")
                                return True
                        # a heuristic to detect divergence and stop early
                        if len(metrics) > 50:
                            cur_metric = metrics[-1][-1][-1]
                            prev_metric = metrics[-50][-1][-1]
                            # record HPWL and overflow increase, and check divergence
                            if (
                                cur_metric.overflow[-1] > prev_metric.overflow[-1]
                                and cur_metric.hpwl > best_metric[0].hpwl * 2
                            ):
                                return True
                        return False

                def Llambda_stop_criterion(Lgamma_step, Llambda_density_weight_step, metrics):
                    with torch.no_grad():
                        if len(metrics) > 1:
                            cur_metric = metrics[-1][-1]
                            prev_metric = metrics[-2][-1]
                            ### for fence regions, the outer cell overflow and max_density decides whether to stop
                            if (
                                cur_metric.overflow[-1] < params.stop_overflow
                                and cur_metric.hpwl > prev_metric.hpwl
                            ) or cur_metric.max_density[-1] < 1.0:
                                logging.debug(
                                    "Llambda stopping criteria: %d and (( %g < 0.1 and %g > %g ) or %g < 1.0)"
                                    % (
                                        Llambda_density_weight_step,
                                        cur_metric.overflow[-1],
                                        cur_metric.hpwl,
                                        prev_metric.hpwl,
                                        cur_metric.max_density[-1],
                                    )
                                )
                                return True
                    return False

                # use a moving average window for stopping criteria, for an example window of 3
                # 0, 1, 2, 3, 4, 5, 6
                #    window2
                #             window1
                moving_avg_window = max(min(model.Lsub_iteration // 2, 3), 1)

                def Lsub_stop_criterion(Lgamma_step, Llambda_density_weight_step, Lsub_step, metrics):
                    with torch.no_grad():
                        if len(metrics) >= moving_avg_window * 2:
                            cur_avg_obj = 0
                            prev_avg_obj = 0
                            for i in range(moving_avg_window):
                                cur_avg_obj += metrics[-1 - i].objective
                                prev_avg_obj += metrics[-1 - moving_avg_window - i].objective
                            cur_avg_obj /= moving_avg_window
                            prev_avg_obj /= moving_avg_window
                            threshold = 0.999
                            if cur_avg_obj >= prev_avg_obj * threshold:
                                logging.debug(
                                    "Lsub stopping criteria: %d and %g > %g * %g"
                                    % (Lsub_step, cur_avg_obj, prev_avg_obj, threshold)
                                )
                                return True
                    return False

                def one_descent_step(
                    Lgamma_step, Llambda_density_weight_step, Lsub_step, iteration, metrics, stop_mask=None
                ):
                    t0 = time.time()

                    # metric for this iteration
                    cur_metric = EvalMetrics.EvalMetrics(
                        iteration, (Lgamma_step, Llambda_density_weight_step, Lsub_step)
                    )
                    cur_metric.gamma = model.gamma.data
                    cur_metric.density_weight = model.density_weight.data
                    metrics.append(cur_metric)
                    pos = model.data_collections.pos[0]

                    # move any out-of-bound cell back to placement region
                    self.op_collections.move_boundary_op(pos)

                    # handle multiple density weights for multi-electric field
                    if torch.eq(model.density_weight.mean(), 0.0):
                        model.initialize_density_weight(params, placedb)
                        if model.density_weight.size(0) == 1:
                            logging.info("density_weight = %.6E" % (model.density_weight.data))
                        else:
                            logging.info(
                                "density_weight = [%s]"
                                % ", ".join(["%.3E" % i for i in model.density_weight.cpu().numpy().tolist()])
                            )

                    optimizer.zero_grad()

                    # t1 = time.time()
                    cur_metric.evaluate(placedb, eval_ops, pos, model.data_collections)
                    model.overflow = cur_metric.overflow.data.clone()
                    # logging.debug("evaluation %.3f ms" % ((time.time()-t1)*1000))
                    # t2 = time.time()

                    # as nesterov requires line search, we cannot follow the convention of other solvers
                    if optimizer_name.lower() in ["sgd", "adam", "sgd_momentum", "sgd_nesterov"]:
                        obj, grad = model.obj_and_grad_fn(pos)
                        cur_metric.objective = obj.data.clone()
                    elif optimizer_name.lower() != "nesterov":
                        assert 0, "unsupported optimizer %s" % (optimizer_name)

                    # plot placement
                    if params.plot_flag and (iteration % 100 == 0 or iteration == 999):
                        cur_pos = self.pos[0].data.clone().cpu().numpy()
                        self.plot(params, placedb, iteration, cur_pos)

                    #### stop updating fence regions that are marked stop, exclude the outer cell !
                    t3 = time.time()
                    if model.update_mask is not None:
                        pos_bk = pos.data.clone()
                        optimizer.step()

                        for region_id, fence_region_update_flag in enumerate(model.update_mask):
                            if fence_region_update_flag == 0:
                                ### don't update cell location in that region
                                mask = self.op_collections.fence_region_density_ops[region_id].pos_mask
                                pos.data.masked_scatter_(mask, pos_bk[mask])
                    else:
                        optimizer.step()

                    logging.info("optimizer step %.3f ms" % ((time.time() - t3) * 1000))

                    # nesterov has already computed the objective of the next step
                    if optimizer_name.lower() == "nesterov":
                        cur_metric.objective = optimizer.param_groups[0]["obj_k_1"][0].data.clone()

                    # actually reports the metric before step
                    logging.info(cur_metric)
                    # record the best outer cell overflow
                    if best_metric[0] is None or best_metric[0].overflow[-1] > cur_metric.overflow[-1]:
                        best_metric[0] = cur_metric
                        if best_pos[0] is None:
                            best_pos[0] = self.pos[0].data.clone()
                        else:
                            best_pos[0].data.copy_(self.pos[0].data)

                    logging.info("full step %.3f ms" % ((time.time() - t0) * 1000))

                def check_plateau(x, window=10, threshold=0.001):
                    if len(x) < window:
                        return False
                    x = x[-window:]
                    return (np.max(x) - np.min(x)) / np.mean(x) < threshold

                def check_divergence(x, window=50, threshold=0.05):
                    if len(x) < window or best_metric[0] is None:
                        return False
                    x = np.array(x[-window:])
                    overflow_mean = np.mean(x[:, 1])
                    overflow_diff = np.maximum(0, np.sign(x[1:, 1] - x[:-1, 1])).astype(np.float32)
                    overflow_diff = np.sum(overflow_diff) / overflow_diff.shape[0]
                    overflow_range = np.max(x[:, 1]) - np.min(x[:, 1])
                    wl_mean = np.mean(x[:, 0])
                    wl_ratio, overflow_ratio = (wl_mean - best_metric[0].hpwl.item()) / best_metric[
                        0
                    ].hpwl.item(), (
                        overflow_mean - max(params.stop_overflow, best_metric[0].overflow.item())
                    ) / best_metric[
                        0
                    ].overflow.item()
                    if wl_ratio > threshold * 1.2:
                        # this condition is not suitable for routability-driven opt with cell inflation
                        if (not params.routability_opt_flag) and overflow_ratio > threshold:
                            logging.warning(
                                f"Divergence detected: overflow increases too much than best overflow ({overflow_ratio:.4f} > {threshold:.4f})"
                            )
                            return True
                        elif overflow_range / overflow_mean < threshold:
                            logging.warning(
                                f"Divergence detected: overflow plateau ({overflow_range/overflow_mean:.4f} < {threshold:.4f})"
                            )
                            return True
                        elif overflow_diff > 0.6:
                            logging.warning(
                                f"Divergence detected: overflow fluctuate too frequently ({overflow_diff:.2f} > 0.6)"
                            )
                            return True
                        else:
                            return False
                    else:
                        return False

                def entropy_injection(
                    pos, placedb, shrink_factor=1, noise_intensity=1, mode="random", iteration=1
                ):
                    if mode == "random":
                        # print(pos[: placedb.num_movable_nodes].mean())
                        xc = pos[: placedb.num_movable_nodes].data.mean()
                        yc = pos.data[
                            placedb.num_nodes : placedb.num_nodes + placedb.num_movable_nodes
                        ].mean()
                        num_movable_nodes = placedb.num_movable_nodes
                        num_nodes = placedb.num_nodes
                        num_filler_nodes = placedb.num_filler_nodes
                        num_fixed_nodes = num_nodes - num_movable_nodes - num_filler_nodes

                        fixed_pos_x = pos.data[
                            num_movable_nodes : num_movable_nodes + num_fixed_nodes
                        ].clone()
                        fixed_pos_y = pos.data[
                            num_nodes + num_movable_nodes : num_nodes + num_movable_nodes + num_fixed_nodes
                        ].clone()
                        if shrink_factor != 1:
                            pos.data[:num_nodes] = (pos.data[:num_nodes] - xc) * shrink_factor + xc
                            pos.data[num_nodes:] = (pos.data[num_nodes:] - yc) * shrink_factor + yc
                        if noise_intensity > 0.01:
                            # pos.data.add_(noise_intensity * torch.rand(num_nodes*2, device=pos.device).sub_(0.5))
                            pos.data.add_(noise_intensity * torch.randn(num_nodes * 2, device=pos.device))

                        pos.data[num_movable_nodes : num_movable_nodes + num_fixed_nodes] = fixed_pos_x
                        pos.data[
                            num_nodes + num_movable_nodes : num_nodes + num_movable_nodes + num_fixed_nodes
                        ] = fixed_pos_y
                        # print(pos[: placedb.num_movable_nodes].mean())
                    else:
                        raise NotImplementedError

                Lgamma_metrics = all_metrics

                if params.routability_opt_flag:
                    adjust_area_flag = True
                    adjust_route_area_flag = params.adjust_nctugr_area_flag or params.adjust_rudy_area_flag
                    adjust_pin_area_flag = params.adjust_pin_area_flag
                    num_area_adjust = 0

                Llambda_flat_iteration = 0

                ### preparation for self-adaptive divergence check
                overflow_list = [1]
                divergence_list = []
                min_perturb_interval = 50
                stop_placement = 0
                last_perturb_iter = -min_perturb_interval
                perturb_counter = 0

                for Lgamma_step in range(model.Lgamma_iteration):
                    Lgamma_metrics.append([])
                    Llambda_metrics = Lgamma_metrics[-1]
                    for Llambda_density_weight_step in range(model.Llambda_density_weight_iteration):
                        Llambda_metrics.append([])
                        Lsub_metrics = Llambda_metrics[-1]
                        for Lsub_step in range(model.Lsub_iteration):
                            ## divergence threshold should decrease as overflow decreases
                            ## only detect divergence when overflow is relatively low but not too low
                            if (
                                len(placedb.regions) == 0
                                and params.stop_overflow * 1.1 < overflow_list[-1] < params.stop_overflow * 4
                                and check_divergence(
                                    divergence_list, window=3, threshold=0.01 * overflow_list[-1]
                                )
                            ):
                                self.pos[0].data.copy_(best_pos[0].data)
                                stop_placement = 1

                                logging.error(
                                    "possible DIVERGENCE detected, roll back to the best position recorded"
                                )

                            one_descent_step(
                                Lgamma_step, Llambda_density_weight_step, Lsub_step, iteration, Lsub_metrics
                            )

                            if len(placedb.regions) == 0:
                                overflow_list.append(Llambda_metrics[-1][-1].overflow.data.item())
                                divergence_list.append(
                                    [
                                        Llambda_metrics[-1][-1].hpwl.data.item(),
                                        Llambda_metrics[-1][-1].overflow.data.item(),
                                    ]
                                )

                            ## quadratic penalty and entropy injection
                            if (
                                len(placedb.regions) == 0
                                and iteration - last_perturb_iter > min_perturb_interval
                                and check_plateau(overflow_list, window=15, threshold=0.001)
                            ):
                                if overflow_list[-1] > 0.9:  # stuck at high overflow
                                    model.quad_penalty = True
                                    model.density_factor *= 2
                                    logging.info(
                                        f"Stuck at early stage. Turn on quadratic penalty with double density factor to accelerate convergence"
                                    )
                                    if overflow_list[-1] > 0.95:  # stuck at very high overflow
                                        noise_intensity = min(
                                            max(40 + (120 - 40) * (overflow_list[-1] - 0.95) * 10, 40), 90
                                        )
                                        entropy_injection(
                                            self.pos[0],
                                            placedb,
                                            shrink_factor=0.996,
                                            noise_intensity=noise_intensity,
                                            mode="random",
                                        )
                                        logging.info(
                                            f"Stuck at very early stage. Turn on entropy injection with noise intensity = {noise_intensity} to help convergence"
                                        )
                                    last_perturb_iter = iteration
                                    perturb_counter += 1

                            iteration += 1
                            # stopping criteria
                            if Lsub_stop_criterion(
                                Lgamma_step, Llambda_density_weight_step, Lsub_step, Lsub_metrics
                            ):
                                break
                        Llambda_flat_iteration += 1
                        # update density weight
                        if Llambda_flat_iteration > 1:
                            model.op_collections.update_density_weight_op(
                                Llambda_metrics[-1][-1],
                                Llambda_metrics[-2][-1]
                                if len(Llambda_metrics) > 1
                                else Lgamma_metrics[-2][-1][-1],
                                Llambda_flat_iteration,
                            )
                        # logging.debug("update density weight %.3f ms" % ((time.time()-t2)*1000))
                        if Llambda_stop_criterion(Lgamma_step, Llambda_density_weight_step, Llambda_metrics):
                            break

                        # for routability optimization
                        if (
                            params.routability_opt_flag
                            and num_area_adjust < params.max_num_area_adjust
                            and Llambda_metrics[-1][-1].overflow < params.node_area_adjust_overflow
                        ):
                            content = (
                                "routability optimization round %d: adjust area flags = (%d, %d, %d)"
                                % (
                                    num_area_adjust,
                                    adjust_area_flag,
                                    adjust_route_area_flag,
                                    adjust_pin_area_flag,
                                )
                            )
                            pos = model.data_collections.pos[0]

                            route_utilization_map = None
                            pin_utilization_map = None
                            if adjust_route_area_flag:
                                if params.adjust_nctugr_area_flag:
                                    route_utilization_map = model.op_collections.nctugr_congestion_map_op(pos)
                                else:
                                    route_utilization_map = model.op_collections.route_utilization_map_op(pos)
                                if params.plot_flag:
                                    path = "%s/%s" % (params.result_dir, params.design_name())
                                    figname = "%s/plot/route%d.png" % (path, num_area_adjust)
                                    os.system("mkdir -p %s" % (os.path.dirname(figname)))
                                    plt.imsave(
                                        figname, route_utilization_map.data.cpu().numpy().T, origin="lower"
                                    )
                            if adjust_pin_area_flag:
                                pin_utilization_map = model.op_collections.pin_utilization_map_op(pos)
                                if params.plot_flag:
                                    path = "%s/%s" % (params.result_dir, params.design_name())
                                    figname = "%s/plot/pin%d.png" % (path, num_area_adjust)
                                    os.system("mkdir -p %s" % (os.path.dirname(figname)))
                                    plt.imsave(
                                        figname, pin_utilization_map.data.cpu().numpy().T, origin="lower"
                                    )
                            (
                                adjust_area_flag,
                                adjust_route_area_flag,
                                adjust_pin_area_flag,
                            ) = model.op_collections.adjust_node_area_op(
                                pos, route_utilization_map, pin_utilization_map
                            )
                            content += " -> (%d, %d, %d)" % (
                                adjust_area_flag,
                                adjust_route_area_flag,
                                adjust_pin_area_flag,
                            )
                            logging.info(content)
                            if adjust_area_flag:
                                num_area_adjust += 1
                                # restart Llambda
                                model.op_collections.density_op.reset()
                                model.op_collections.density_overflow_op.reset()
                                model.op_collections.pin_utilization_map_op.reset()
                                model.initialize_density_weight(params, placedb)
                                model.density_weight.mul_(0.1 / params.density_weight)
                                logging.info("density_weight = %.6E" % (model.density_weight.data))
                                # load state to restart the optimizer
                                optimizer.load_state_dict(initial_state)
                                # must after loading the state
                                initialize_learning_rate(pos)
                                # increase iterations of the sub problem to slow down the search
                                model.Lsub_iteration = model.routability_Lsub_iteration

                                # reset best metric
                                best_metric[0] = None
                                best_pos[0] = None

                                break

                    # gradually reduce gamma to tradeoff smoothness and accuracy
                    if len(placedb.regions) > 0 and Llambda_metrics[-1][-1].goverflow is not None:
                        model.op_collections.update_gamma_op(Lgamma_step, Llambda_metrics[-1][-1].goverflow)
                    elif len(placedb.regions) == 0 and Llambda_metrics[-1][-1].overflow is not None:
                        model.op_collections.update_gamma_op(Lgamma_step, Llambda_metrics[-1][-1].overflow)
                    else:
                        model.op_collections.precondition_op.set_overflow(Llambda_metrics[-1][-1].overflow)
                    if Lgamma_stop_criterion(Lgamma_step, Lgamma_metrics) or stop_placement == 1:
                        break

                    # update learning rate
                    if optimizer_name.lower() in ["sgd", "adam", "sgd_momentum", "sgd_nesterov", "cg"]:
                        if "learning_rate_decay" in global_place_params:
                            for param_group in optimizer.param_groups:
                                param_group["lr"] *= global_place_params["learning_rate_decay"]

                # in case of divergence, use the best metric
                # last_metric = all_metrics[-1][-1][-1]
                # if (
                #     last_metric.overflow[-1] > max(params.stop_overflow, best_metric[0].overflow[-1])
                #     and last_metric.hpwl > best_metric[0].hpwl
                # ):
                #     all_metrics.append([best_metric])

                logging.info("optimizer %s takes %.3f seconds" % (optimizer_name, time.time() - tt))

            # recover node size and pin offset for legalization, since node size is adjusted in global placement
            if params.routability_opt_flag:
                with torch.no_grad():
                    # convert lower left to centers
                    self.pos[0][: placedb.num_movable_nodes].add_(
                        self.data_collections.node_size_x[: placedb.num_movable_nodes] / 2
                    )
                    self.pos[0][placedb.num_nodes : placedb.num_nodes + placedb.num_movable_nodes].add_(
                        self.data_collections.node_size_y[: placedb.num_movable_nodes] / 2
                    )
                    self.data_collections.node_size_x.copy_(self.data_collections.original_node_size_x)
                    self.data_collections.node_size_y.copy_(self.data_collections.original_node_size_y)
                    # use fixed centers as the anchor
                    self.pos[0][: placedb.num_movable_nodes].sub_(
                        self.data_collections.node_size_x[: placedb.num_movable_nodes] / 2
                    )
                    self.pos[0][placedb.num_nodes : placedb.num_nodes + placedb.num_movable_nodes].sub_(
                        self.data_collections.node_size_y[: placedb.num_movable_nodes] / 2
                    )
                    self.data_collections.pin_offset_x.copy_(self.data_collections.original_pin_offset_x)
                    self.data_collections.pin_offset_y.copy_(self.data_collections.original_pin_offset_y)

        else:
            cur_metric = EvalMetrics.EvalMetrics(iteration)
            all_metrics.append(cur_metric)
            cur_metric.evaluate(placedb, {"hpwl": self.op_collections.hpwl_op}, self.pos[0])
            logging.info(cur_metric)

        # dump global placement solution for legalization
        if params.dump_global_place_solution_flag:
            self.dump(params, placedb, self.pos[0].cpu(), "%s.lg.pklz" % (params.design_name()))

        # plot placement
        if params.plot_flag:
            self.plot(params, placedb, iteration, self.pos[0].data.clone().cpu().numpy())

        # legalization
        if params.legalize_flag:
            tt = time.time()
            self.pos[0].data.copy_(self.op_collections.legalize_op(self.pos[0]))
            logging.info("legalization takes %.3f seconds" % (time.time() - tt))
            cur_metric = EvalMetrics.EvalMetrics(iteration)
            all_metrics.append(cur_metric)
            cur_metric.evaluate(placedb, {"hpwl": self.op_collections.hpwl_op}, self.pos[0])
            logging.info(cur_metric)
            iteration += 1

        # plot placement
        if params.plot_flag:
            self.plot(params, placedb, iteration, self.pos[0].data.clone().cpu().numpy())

        # dump legalization solution for detailed placement
        if params.dump_legalize_solution_flag:
            self.dump(params, placedb, self.pos[0].cpu(), "%s.dp.pklz" % (params.design_name()))

        # detailed placement
        if params.detailed_place_flag:
            tt = time.time()
            self.pos[0].data.copy_(self.op_collections.detailed_place_op(self.pos[0]))
            logging.info("detailed placement takes %.3f seconds" % (time.time() - tt))
            cur_metric = EvalMetrics.EvalMetrics(iteration)
            all_metrics.append(cur_metric)
            cur_metric.evaluate(placedb, {"hpwl": self.op_collections.hpwl_op}, self.pos[0])
            logging.info(cur_metric)
            iteration += 1

        # save results
        cur_pos = self.pos[0].data.clone().cpu().numpy()
        # apply solution
        placedb.apply(
            params,
            cur_pos[0 : placedb.num_movable_nodes],
            cur_pos[placedb.num_nodes : placedb.num_nodes + placedb.num_movable_nodes],
        )
        # plot placement
        if params.plot_flag:
            self.plot(params, placedb, iteration, cur_pos)
        return all_metrics
