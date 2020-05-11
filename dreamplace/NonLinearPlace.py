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
from optimizer import RAdam, Nesterov_Armijo, ZerothOrderSearch
class NonLinearPlace (BasicPlace.BasicPlace):
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
                model = PlaceObj.PlaceObj(density_weight, params, placedb, self.data_collections, self.op_collections, global_place_params).to(self.data_collections.pos[0].device)
                optimizer_name = global_place_params["optimizer"]

                # determine optimizer
                if optimizer_name.lower() == "adam":
                    optimizer = torch.optim.Adam(self.parameters(), lr=0)
                    # optimizer = RAdam(self.parameters(), lr=0)
                elif optimizer_name.lower() == "sgd":
                    optimizer = torch.optim.SGD(self.parameters(), lr=0)
                elif optimizer_name.lower() == "zoo":
                    optimizer = ZerothOrderSearch(self.parameters(), obj_fn=lambda x: self.op_collections.density_overflow_op(x)[0], placedb=placedb)
                elif optimizer_name.lower() == "sgd_momentum":
                    optimizer = torch.optim.SGD(self.parameters(), lr=0, momentum=0.9, nesterov=False)
                elif optimizer_name.lower() == "sgd_nesterov":
                    optimizer = torch.optim.SGD(self.parameters(), lr=0, momentum=0.9, nesterov=True)
                elif optimizer_name.lower() == "nesterov":
                    optimizer = NesterovAcceleratedGradientOptimizer.NesterovAcceleratedGradientOptimizer(self.parameters(),
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
                        "density" : self.op_collections.density_op,
                        #"objective" : model.obj_fn,
                        "hpwl" : self.op_collections.hpwl_op,
                        "overflow" : self.op_collections.density_overflow_op,
                        "shpwl" : None
                        }
                if params.routability_opt_flag:
                    eval_ops.update({
                        'route_utilization' : self.op_collections.route_utilization_map_op,
                        'pin_utilization' : self.op_collections.pin_utilization_map_op
                        })

                # a function to initialize learning rate
                def initialize_learning_rate(pos):
                    learning_rate = model.estimate_initial_learning_rate(pos, global_place_params["learning_rate"])
                    # update learning rate
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = learning_rate.data

                if iteration == 0:
                    if params.gp_noise_ratio > 0.0:
                        logging.info("add %g%% noise" % (params.gp_noise_ratio*100))
                        model.op_collections.noise_op(model.data_collections.pos[0], params.gp_noise_ratio)
                        initialize_learning_rate(model.data_collections.pos[0])
                # the state must be saved after setting learning rate
                initial_state = copy.deepcopy(optimizer.state_dict())

                if params.gpu:
                    torch.cuda.synchronize()
                logging.info("%s initialization takes %g seconds" % (optimizer_name, (time.time()-tt)))

                # as nesterov requires line search, we cannot follow the convention of other solvers
                if optimizer_name.lower() in {"sgd", "adam", "sgd_momentum", "sgd_nesterov", "zoo"}:
                    model.obj_and_grad_fn(model.data_collections.pos[0])
                elif optimizer_name.lower() != "nesterov":
                    assert 0, "unsupported optimizer %s" % (optimizer_name)

                # stopping criteria
                def Lgamma_stop_criterion(Lgamma_step, metrics):
                    with torch.no_grad():
                        if len(metrics) > 1:
                            cur_metric = metrics[-1][-1][-1]
                            prev_metric = metrics[-2][-1][-1]
                            if Lgamma_step > 100 and ((cur_metric.overflow < params.stop_overflow and cur_metric.hpwl > prev_metric.hpwl) or cur_metric.max_density < params.target_density):
                                logging.debug("Lgamma stopping criteria: %d > 100 and (( %g < 0.1 and %g > %g ) or %g < 1.0)" % (Lgamma_step, cur_metric.overflow, cur_metric.hpwl, prev_metric.hpwl, cur_metric.max_density))
                                return True
                        return False

                def Llambda_stop_criterion(Lgamma_step, Llambda_density_weight_step, metrics):
                    with torch.no_grad():
                        if len(metrics) > 1:
                            cur_metric = metrics[-1][-1]
                            prev_metric = metrics[-2][-1]
                            if (cur_metric.overflow < params.stop_overflow and cur_metric.hpwl > prev_metric.hpwl) or cur_metric.max_density < 1.0:
                                logging.debug("Llambda stopping criteria: %d and (( %g < 0.1 and %g > %g ) or %g < 1.0)" % (Llambda_density_weight_step, cur_metric.overflow, cur_metric.hpwl, prev_metric.hpwl, cur_metric.max_density))
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
                                logging.debug("Lsub stopping criteria: %d and %g > %g * %g" % (Lsub_step, cur_avg_obj, prev_avg_obj, threshold))
                                return True
                    return False

                def inject_perturbation(pos, placedb, shrink_factor=1, noise_intensity=1, mode="random", iteration=1):
                    if(mode == "lg_dp"):
                        pos_lg = self.op_collections.detailed_place_op(self.op_collections.legalize_op(pos))
                        diff = pos_lg - pos.data
                        diff_abs = diff.abs()
                        diff_abs2 = diff_abs[(diff_abs > 0.1) & (diff_abs < 60)]
                        avg, std = diff_abs2.mean(), diff_abs2.std()
                        print(avg, std)
                        mask = (diff_abs < avg + noise_intensity * std) & (diff_abs > avg - noise_intensity * std)
                        n_cell = mask.float().sum().data.item()
                        print(f"{n_cell} cells are moved")
                        pos.data[mask] += diff[mask]
                        # model.density_weight *= 0.9
                        # pos.data[mask] = pos_lg[mask]
                        return
                    elif(mode == "random"):
                        print(pos[:placedb.num_movable_nodes].mean())
                        xc = pos[:placedb.num_movable_nodes].data.mean()
                        yc = pos.data[placedb.num_nodes:placedb.num_nodes+placedb.num_movable_nodes].mean()
                        num_movable_nodes = placedb.num_movable_nodes
                        num_nodes = placedb.num_nodes
                        num_filler_nodes = placedb.num_filler_nodes
                        num_fixed_nodes = num_nodes - num_movable_nodes - num_filler_nodes

                        fixed_pos_x = pos.data[num_movable_nodes:num_movable_nodes+num_fixed_nodes].clone()
                        fixed_pos_y = pos.data[num_nodes+ num_movable_nodes:num_nodes+num_movable_nodes+num_fixed_nodes].clone()
                        if(shrink_factor != 1):
                            pos.data[:num_nodes] = (pos.data[:num_nodes] - xc) * shrink_factor + xc
                            pos.data[num_nodes:] = (pos.data[num_nodes:] - yc) * shrink_factor + yc
                        if(noise_intensity>0.01):
                            # pos.data.add_(noise_intensity * torch.rand(num_nodes*2, device=pos.device).sub_(0.5))
                            pos.data.add_(noise_intensity * torch.randn(num_nodes*2, device=pos.device))

                        pos.data[num_movable_nodes:num_movable_nodes+num_fixed_nodes] = fixed_pos_x
                        pos.data[num_nodes+ num_movable_nodes:num_nodes+num_movable_nodes+num_fixed_nodes] = fixed_pos_y
                        print(pos[:placedb.num_movable_nodes].mean())
                    elif(mode == "search"):
                        # when overflow meets plateau, show search the descent direction of density, not WL
                        # obj_fn = model.op_collections.density_op
                        def obj_fn(x):
                            # return model.op_collections.hpwl_op(self.op_collections.detailed_place_op(self.op_collections.legalize_op(x)))
                            # return model.op_collections.hpwl_op(self.op_collections.legalize_op(x))
                            return self.op_collections.density_overflow_op(x)[0].data + 1e-1 * model.op_collections.hpwl_op(x)
                            return model.op_collections.hpwl_op(x) + model.op_collections.density_op(x) * 2e2
                            return model.op_collections.density_op(x)
                            # return model.obj_fn(x)
                            return model.op_collections.wirelength_op(x)

                        R, r = 8, 1
                        # R /= 2**(iteration//300)
                        K = int(np.log2(R/r)) + 1
                        T = 64
                        num_movable_nodes = placedb.num_movable_nodes
                        num_nodes = placedb.num_nodes
                        num_filler_nodes = placedb.num_filler_nodes
                        num_fixed_nodes = num_nodes - num_movable_nodes - num_filler_nodes
                        obj_min = obj_fn(pos.data).data.item()
                        pos_min = pos.data
                        v_min = 0
                        # print(obj_min)
                        for t in range(T):
                            # obj_fn = [model.op_collections.density_op, model.op_collections.wirelength_op][t > (T//2)]
                            obj_min = obj_fn(pos.data).data.item()
                            obj_start = obj_min
                            # print(f"start obj: {obj_start}")
                            for k in range(K):
                                r_k = 2**(-k) * R
                                for i in range(2):
                                    v_k = torch.randn_like(pos.data)
                                    v_k[num_movable_nodes:num_nodes-num_filler_nodes] = 0
                                    v_k[num_nodes+num_movable_nodes:-num_filler_nodes] = 0
                                    # v_k = v_k / v_k.norm(p=2) * r_k
                                    v_k = v_k / v_k.norm(p=2) * r_k
                                    p1 = pos.data + v_k
                                    obj_k = obj_fn(p1).data.item()
                                    if(obj_k < obj_min):
                                        obj_min = obj_k
                                        v_min = v_k.clone()
                                        r_min = r_k
                                        pos_min = p1.clone()
                                        # print(v_min.sum(), pos_min.mean())
                            # zeroth-order optimization with decaying step size
                            diff = obj_start - obj_min
                            if(diff > 0.001):
                                step_size = max(1, min(1, diff / r_min))
                                # print(f"Search step: {t} stepsize: {step_size:5.2f} r_min: {r_min} obj reduce from {obj_start} to {obj_min}")
                                pos.data.copy_(pos.data + v_min * step_size)

                def one_descent_step(Lgamma_step, Llambda_density_weight_step, Lsub_step, iteration, metrics):
                    t0 = time.time()

                    # metric for this iteration
                    cur_metric = EvalMetrics.EvalMetrics(iteration, (Lgamma_step, Llambda_density_weight_step, Lsub_step))
                    cur_metric.gamma = model.gamma.data
                    cur_metric.density_weight = model.density_weight.data
                    metrics.append(cur_metric)
                    pos = model.data_collections.pos[0]

                    # move any out-of-bound cell back to placement region
                    self.op_collections.move_boundary_op(pos)

                    if torch.eq(model.density_weight, 0.0):
                        model.initialize_density_weight(params, placedb)
                        logging.info("density_weight = %.6E" % (model.density_weight.data))

                    optimizer.zero_grad()

                    # t1 = time.time()
                    cur_metric.evaluate(placedb, eval_ops, pos)
                    model.overflow = cur_metric.overflow.data.clone()
                    #logging.debug("evaluation %.3f ms" % ((time.time()-t1)*1000))
                    #t2 = time.time()

                    # as nesterov requires line search, we cannot follow the convention of other solvers
                    if optimizer_name.lower() in ["sgd", "adam", "sgd_momentum", "sgd_nesterov", "zoo"]:
                        obj, grad = model.obj_and_grad_fn(pos)
                        cur_metric.objective = obj.data.clone()
                    elif optimizer_name.lower() != "nesterov":
                        assert 0, "unsupported optimizer %s" % (optimizer_name)

                    # plot placement
                    if params.plot_flag and iteration % 100 == 0:
                        cur_pos = self.pos[0].data.clone().cpu().numpy()
                        self.plot(params, placedb, iteration, cur_pos)

                    t3 = time.time()
                    optimizer.step()
                    logging.info("optimizer step %.3f ms" % ((time.time()-t3)*1000))

                    # nesterov has already computed the objective of the next step
                    if optimizer_name.lower() == "nesterov":
                        cur_metric.objective = optimizer.param_groups[0]['obj_k_1'][0].data.clone()

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

                    logging.info("full step %.3f ms" % ((time.time()-t0)*1000))

                Lgamma_metrics = all_metrics

                if params.routability_opt_flag:
                    adjust_area_flag = True
                    adjust_route_area_flag = params.adjust_route_area_flag
                    adjust_pin_area_flag = params.adjust_pin_area_flag
                    num_area_adjust = 0

                Llambda_flat_iteration = 0
                noise_number = 0

                def check_plateau(x, window=10, threshold=0.001):
                    if(len(x) < window):
                        return False
                    x = x[-window:]
                    return (np.max(x) - np.min(x)) / np.mean(x) < threshold

                def check_divergence(x, window=50, threshold=0.05):
                    if(len(x) < window):
                        return False
                    x = np.array(x[-window:])
                    smooth = max(1,int(0.1*window))
                    wl_beg, wl_end = np.mean(x[0:smooth,0]), np.mean(x[-smooth:,0])
                    overflow_beg, overflow_end = np.mean(x[0:smooth,1]), np.mean(x[-smooth:,1])
                    # wl_ratio, overflow_ratio = (wl_end - wl_beg)/wl_beg, (overflow_end - overflow_beg)/overflow_beg
                    # wl_ratio, overflow_ratio = (wl_end - wl_beg)/wl_beg, (overflow_end - max(params.stop_overflow, best_metric[0].overflow))/best_metric[0].overflow
                    overflow_mean = np.mean(x[:,1])
                    overflow_diff = np.maximum(0,np.sign(x[1:,1] - x[:-1,1])).astype(np.float32)
                    overflow_diff = np.sum(overflow_diff) / overflow_diff.shape[0]
                    overflow_range = np.max(x[:,1]) - np.min(x[:,1])
                    wl_mean = np.mean(x[:,0])
                    wl_ratio, overflow_ratio = (wl_mean - best_metric[0].hpwl)/best_metric[0].hpwl, (overflow_mean - max(params.stop_overflow, best_metric[0].overflow))/best_metric[0].overflow
                    if(wl_ratio > threshold*1.2):
                        if(overflow_ratio > threshold):
                            print(f"[Warning] Divergence detected: overflow increases too much than best overflow ({overflow_ratio:.4f} > {threshold:.4f})")
                            return True
                        elif(overflow_range/overflow_mean < threshold):
                            print(f"[Warning] Divergence detected: overflow plateau ({overflow_range/overflow_mean:.4f} < {threshold:.4f})")
                            return True
                        elif(overflow_diff > 0.6):
                            print(f"[Warning] Divergence detected: overflow fluctuate too frequently ({overflow_diff:.2f} > 0.6)")
                            return True
                        else:
                            return False
                    else:
                        return False


                try:
                    noise_list = {"adaptec1":[
                        (0.85, 0.996, 200, "random"),(0.75, 0.996, 120, "random")
                                ],
                              "adaptec2":[(0.70, 0.996, 90, "random"),(0.3, 0.996, 0.5, "lg_dp")],
                              "adaptec3":[(0.85, 0.996, 200, "random"),(0.75, 0.996, 120, "random")],
                              "adaptec4":[(0.9, 0.996, 200, "random"),(0.75, 0.996, 150, "random")],
                              "bigblue1": [(0.70, 0.995, 60, "random"),
                            #   (0.5, 0.996,0.5, "lg_dp")
                              ], # b1
                              "bigblue2": [(0.45, 1, 10, "random")], # b2, only 0.45
                            #   "bigblue4": [(0.65, 0.995, 30), (0.55, 0.997, 20)]
                            "bigblue3": [(0.65, 0.995, 20, "random"),(0.55, 0.997, 20, "random"),(0.45, 0.999, 20, "random")],
                              "bigblue4": [(0.95, 0.995, 30, "random")],
                              "ispd19_test1.input": [
                                  (0.99, 0.996, 100, "random"),
                                  (0.6, 0.999, 10, "random"),
                                #   (0.3, 0.996,100,"search"),(0.15, 0.996,100,"search")
                                  ],
                              "ispd19_test2.input": [
                                #   (0.85, 0.996, 100, "random"),(0.75, 0.996, 80, "random"),
                              (0.2, 0.996,100,"search"), (0.15, 0.996,100,"search")],
                              "ispd19_test4.input": [
                                  (0.5, 0.997, 20, "random"),
                                #   (0.75, 0.996, 80, "random"),
                            #   (0.2, 0.996,100,"search"), (0.15, 0.996,100,"search")
                              ],
                              "ispd19_test6.input": [
                                  (0.65, 0.996, 70, "random"),
                              (0.2, 0.996,100,"search"), (0.15, 0.996,100,"search")
                              ],
                              "ispd19_test10.input": [
                                  (0.99, 0.995, 400, "random"),
                                #   (0.96, 0.996, 200, "random"),
                                #   (0.75, 0.996, 120, "random"),
                            #   (0.5, 0.996,100,"search"), (0.3, 0.996,100,"search")
                              ],
                              "leon2": [
                                #   (0.98, 0.999, 400, "random"),
                                  (0.9, 0.996, 200, "random"),
                                  (0.5, 0.997, 20, "random"),
                                #   (0.75, 0.996, 120, "random"),
                            #   (0.5, 0.996,100,"search"), (0.3, 0.996,100,"search")
                              ]
                             }[params.design_name()]
                except:
                    noise_list = []

                overflow_list = [1]
                divergence_list = []
                min_perturb_interval = 50
                stop_placement = 0

                last_perturb_iter = -min_perturb_interval
                pos_trace = []
                noise_injected_flag = 0
                perturb_counter = 0
                search_start = 0
                max_search_step = 10
                allow_update = 1

                # model.quad_penalty = True
                for Lgamma_step in range(model.Lgamma_iteration):
                    Lgamma_metrics.append([])
                    Llambda_metrics = Lgamma_metrics[-1]
                    for Llambda_density_weight_step in range(model.Llambda_density_weight_iteration):
                        Llambda_metrics.append([])
                        Lsub_metrics = Llambda_metrics[-1]
                        for Lsub_step in range(model.Lsub_iteration):
                            flag = 1
                            ## Jiaqi: divergence threshold should decrease as overflow decreases
                            diverge_threshold = 0.01 * overflow_list[-1]

                            ## Jiaqi: only detect divergence when overflow is relatively low but not too low
                            if(flag == 1 and params.stop_overflow * 1.1 < overflow_list[-1] < params.stop_overflow * 4 and search_start == 0 and check_divergence(divergence_list, window=3, threshold=diverge_threshold)):
                                search_start = 1
                                n_step = max(1,500//(global_place_params["iteration"] - iteration))

                                # obj_fn = lambda x: self.op_collections.hpwl_op(x)*(1+self.op_collections.density_overflow_op(x)[0])
                                # optimizer = ZerothOrderSearch(self.parameters(), obj_fn=obj_fn, placedb=placedb, r_max=8, r_min=1, n_step=n_step, n_sample=8)
                                optimizer = Nesterov_Armijo(self.parameters(), lr=1000, momentum=0, obj_and_grad_fn=model.obj_and_grad_fn, obj_fn=model.obj_fn, dampening=0, weight_decay=0, nesterov=False)

                                optimizer_name = "sgd"
                                self.pos[0].data.copy_(best_pos[0].data)
                                stop_placement = 1
                                allow_update = 0

                                logging.error(
                                    "possible DIVERGENCE detected, roll back to the best position recorded and switch to ZerothOrderSearch of overflow and hpwl"
                                )

                            one_descent_step(Lgamma_step, Llambda_density_weight_step, Lsub_step, iteration, Lsub_metrics)
                            iteration += 1

                            overflow_list.append(Llambda_metrics[-1][-1].overflow.data.item())
                            divergence_list.append([Llambda_metrics[-1][-1].hpwl.data.item(), Llambda_metrics[-1][-1].overflow.data.item()])

                            path = "%s/%s" % (params.result_dir, params.design_name())
                            csvname = "%s/%s_ours.csv" % (path, params.design_name())
                            with open(csvname, "a+") as f:
                                f.write(f"{divergence_list[-1][0]},{divergence_list[-1][1]}\n")

                            flag = 1
                            # quadratic penalty and noise perturbation
                            if(flag == 1 and check_plateau(overflow_list, window=20, threshold=0.001) and iteration - last_perturb_iter > min_perturb_interval):
                                # model.density_weight *= max(1, 10*overflow_list[-1])
                                if(overflow_list[-1] > 0.9):

                                    model.quad_penalty = True
                                    # model.init_wl_factor *= 2
                                    # model.density_factor *= max(1, 2 * 0.7**perturb_counter)
                                    model.density_factor *= 2
                                    noise_intensity = min(max(40 + (120-40) * (overflow_list[-1]-0.9)*10, 40), 90)# * 0.5**perturb_counter
                                    # noise_intensity = 5
                                    inject_perturbation(self.pos[0], placedb, shrink_factor=0.996, noise_intensity=noise_intensity, mode="random")
                                    print(f"Adjust Density, noise={noise_intensity}")
                                    last_perturb_iter = iteration
                                    perturb_counter += 1

                            flag = 0
                            ## add manual noise?
                            if(flag and noise_number < len(noise_list) and Llambda_metrics[-1][-1].overflow < noise_list[noise_number][0]):
                                # self.plot(params, placedb, iteration, self.pos[0].data.clone().cpu().numpy())
                                print(iteration)
                                print("Adjust")
                                inject_perturbation(self.pos[0], placedb, shrink_factor=noise_list[noise_number][1], noise_intensity=noise_list[noise_number][2], mode=noise_list[noise_number][3])
                                # self.plot(params, placedb, iteration+1, self.pos[0].data.clone().cpu().numpy())
                                noise_number += 1
                            # stopping criteria
                            if stop_placement == 1 or Lsub_stop_criterion(Lgamma_step, Llambda_density_weight_step, Lsub_step, Lsub_metrics):
                                break
                        Llambda_flat_iteration += 1

                        # update density weight
                        if Llambda_flat_iteration > 1 and allow_update:
                            model.op_collections.update_density_weight_op(Llambda_metrics[-1][-1], Llambda_metrics[-2][-1] if len(Llambda_metrics) > 1 else Lgamma_metrics[-2][-1][-1], Llambda_flat_iteration)
                        #logging.debug("update density weight %.3f ms" % ((time.time()-t2)*1000))
                        if stop_placement == 1 or Llambda_stop_criterion(Lgamma_step, Llambda_density_weight_step, Llambda_metrics):
                            break

                        # for routability optimization
                        if params.routability_opt_flag and num_area_adjust < params.max_num_area_adjust and Llambda_metrics[-1][-1].overflow < params.node_area_adjust_overflow:
                            content = "routability optimization round %d: adjust area flags = (%d, %d, %d)" % (num_area_adjust, adjust_area_flag, adjust_route_area_flag, adjust_pin_area_flag)
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
                                route_utilization_map = model.op_collections.nctugr_congestion_map_op(pos)
                                #if params.plot_flag:
                                path = "%s/%s" % (params.result_dir, params.design_name())
                                figname = "%s/plot/rudy%d.png" % (path, num_area_adjust)
                                os.system("mkdir -p %s" % (os.path.dirname(figname)))
                                plt.imsave(figname, route_utilization_map.data.cpu().numpy().T, origin='lower')
                            if adjust_pin_area_flag:
                                pin_utilization_map = model.op_collections.pin_utilization_map_op(pos)
                                #if params.plot_flag:
                                path = "%s/%s" % (params.result_dir, params.design_name())
                                figname = "%s/plot/pin%d.png" % (path, num_area_adjust)
                                os.system("mkdir -p %s" % (os.path.dirname(figname)))
                                plt.imsave(figname, pin_utilization_map.data.cpu().numpy().T, origin='lower')
                            adjust_area_flag, adjust_route_area_flag, adjust_pin_area_flag = model.op_collections.adjust_node_area_op(
                                    pos,
                                    route_utilization_map,
                                    pin_utilization_map
                                    )
                            content += " -> (%d, %d, %d)" % (adjust_area_flag, adjust_route_area_flag, adjust_pin_area_flag)
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
                    model.op_collections.update_gamma_op(Lgamma_step, Llambda_metrics[-1][-1].overflow)
                    model.op_collections.precondition_op.set_overflow(Llambda_metrics[-1][-1].overflow)
                    if Lgamma_stop_criterion(Lgamma_step, Lgamma_metrics) or stop_placement == 1:
                        break

                    # update learning rate
                    if optimizer_name.lower() in ["sgd", "adam", "sgd_momentum", "sgd_nesterov", "cg"]:
                        if 'learning_rate_decay' in global_place_params:
                            for param_group in optimizer.param_groups:
                                param_group['lr'] *= global_place_params['learning_rate_decay']

                # in case of divergence, use the best metric
                last_metric = all_metrics[-1][-1][-1]
                if last_metric.overflow > max(
                        params.stop_overflow, best_metric[0].overflow
                ) and last_metric.hpwl > best_metric[0].hpwl:
                    self.pos[0].data.copy_(best_pos[0].data)
                    logging.error(
                        "Deprecated: possible DIVERGENCE detected, roll back to the best position recorded"
                    )
                    all_metrics.append([best_metric])
                    logging.info(best_metric[0])

                logging.info("optimizer %s takes %.3f seconds" % (optimizer_name, time.time()-tt))
            # recover node size and pin offset for legalization, since node size is adjusted in global placement
            if params.routability_opt_flag:
                with torch.no_grad():
                    # convert lower left to centers
                    self.pos[0][:placedb.num_movable_nodes].add_(self.data_collections.node_size_x[:placedb.num_movable_nodes] / 2)
                    self.pos[0][placedb.num_nodes: placedb.num_nodes + placedb.num_movable_nodes].add_(self.data_collections.node_size_y[:placedb.num_movable_nodes] / 2)
                    self.data_collections.node_size_x.copy_(self.data_collections.original_node_size_x)
                    self.data_collections.node_size_y.copy_(self.data_collections.original_node_size_y)
                    # use fixed centers as the anchor
                    self.pos[0][:placedb.num_movable_nodes].sub_(self.data_collections.node_size_x[:placedb.num_movable_nodes] / 2)
                    self.pos[0][placedb.num_nodes: placedb.num_nodes + placedb.num_movable_nodes].sub_(self.data_collections.node_size_y[:placedb.num_movable_nodes] / 2)
                    self.data_collections.pin_offset_x.copy_(self.data_collections.original_pin_offset_x)
                    self.data_collections.pin_offset_y.copy_(self.data_collections.original_pin_offset_y)

        else:
            cur_metric = EvalMetrics.EvalMetrics(iteration)
            all_metrics.append(cur_metric)
            cur_metric.evaluate(placedb, {"hpwl" : self.op_collections.hpwl_op}, self.pos[0])
            logging.info(cur_metric)

        # pos_trace = torch.stack(pos_trace, dim=0)
        # wl_trace = torch.tensor([self.op_collections.hpwl_op(self.op_collections.legalize_op(pos_trace[i,:])).data.item() for i in range(pos_trace.size(0))], device=pos_trace.device)
        # top_5 = torch.argsort(wl_trace)[:1]
        # top_5_wl, top_5_pos = wl_trace[top_5], pos_trace[top_5,:]
        # # self.pos[0].data.copy_(pos_trace[np.argmin(wl_trace)])
        # weighted_mean = (top_5_pos * top_5_wl.unsqueeze(1)/top_5_wl.sum()).sum(dim=0)
        # self.pos[0].data.copy_(weighted_mean)
        # del wl_trace
        # del pos_trace
        # inject_perturbation(self.pos[0], placedb, shrink_factor=0.996, noise_intensity=0, mode="search")
        # dump global placement solution for legalization
        if params.dump_global_place_solution_flag:
            self.dump(params, placedb, self.pos[0].cpu(), "%s.lg.pklz" %(params.design_name()))

        # plot placement
        if params.plot_flag:
            self.plot(params, placedb, iteration, self.pos[0].data.clone().cpu().numpy())

        # legalization
        if params.legalize_flag:
            tt = time.time()
            self.pos[0].data.copy_(self.op_collections.legalize_op(self.pos[0]))
            logging.info("legalization takes %.3f seconds" % (time.time()-tt))
            cur_metric = EvalMetrics.EvalMetrics(iteration)
            all_metrics.append(cur_metric)
            cur_metric.evaluate(placedb, {"hpwl" : self.op_collections.hpwl_op, "overflow": self.op_collections.density_overflow_op, "shpwl": None}, self.pos[0])
            logging.info(cur_metric)
            iteration += 1

        # plot placement
        if params.plot_flag:
            self.plot(params, placedb, iteration, self.pos[0].data.clone().cpu().numpy())

        # dump legalization solution for detailed placement
        if params.dump_legalize_solution_flag:
            self.dump(params, placedb, self.pos[0].cpu(), "%s.dp.pklz" %(params.design_name()))

        # detailed placement
        if params.detailed_place_flag:
            tt = time.time()
            self.pos[0].data.copy_(self.op_collections.detailed_place_op(self.pos[0]))
            logging.info("detailed placement takes %.3f seconds" % (time.time()-tt))
            cur_metric = EvalMetrics.EvalMetrics(iteration)
            all_metrics.append(cur_metric)
            cur_metric.evaluate(placedb, {"hpwl" : self.op_collections.hpwl_op, "overflow": self.op_collections.density_overflow_op, "shpwl": None}, self.pos[0])
            logging.info(cur_metric)
            iteration += 1

        # save results
        cur_pos = self.pos[0].data.clone().cpu().numpy()
        # apply solution
        placedb.apply(params, cur_pos[0:placedb.num_movable_nodes], cur_pos[placedb.num_nodes:placedb.num_nodes+placedb.num_movable_nodes])
        # plot placement
        if params.plot_flag:
            self.plot(params, placedb, iteration, cur_pos)
        return all_metrics
