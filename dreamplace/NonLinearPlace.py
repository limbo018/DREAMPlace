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
if sys.version_info[0] < 3: 
    import cPickle as pickle
else:
    import _pickle as pickle
import BasicPlace
import PlaceObj
import NesterovAcceleratedGradientOptimizer
import EvalMetrics
import pdb 

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
                # L1      L2        L3
                # When optimizing an inner problem, the outer parameters are fixed.
                # This is a generalization to the eplace/RePlAce approach 

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
                elif optimizer_name.lower() == "sgd": 
                    optimizer = torch.optim.SGD(self.parameters(), lr=0)
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
                        #"density" : self.op_collections.density_op, 
                        #"objective" : model.obj_fn, 
                        "hpwl" : self.op_collections.hpwl_op, 
                        "overflow" : self.op_collections.density_overflow_op
                        }

                if iteration == 0: 
                    if params.gp_noise_ratio > 0.0: 
                        logging.info("add %g%% noise" % (params.gp_noise_ratio*100))
                        model.op_collections.noise_op(model.data_collections.pos[0], params.gp_noise_ratio)
                        learning_rate = model.estimate_initial_learning_rate(model.data_collections.pos[0], global_place_params["learning_rate"])
                        # update learning rate 
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = learning_rate.data

                if params.gpu: 
                    torch.cuda.synchronize()
                logging.info("%s initialization takes %g seconds" % (optimizer_name, (time.time()-tt)))

                # as nesterov requires line search, we cannot follow the convention of other solvers
                if optimizer_name.lower() in {"sgd", "adam", "sgd_momentum", "sgd_nesterov"}: 
                    model.obj_and_grad_fn(model.data_collections.pos[0])
                elif optimizer_name.lower() != "nesterov":
                    assert 0, "unsupported optimizer %s" % (optimizer_name)

                # stopping criteria 
                def L1_stop_criterion(L1_gamma_step, metrics): 
                    if len(metrics) > 1: 
                        cur_metric = metrics[-1][-1][-1]
                        prev_metric = metrics[-2][-1][-1]
                        if L1_gamma_step > 100 and ((cur_metric.overflow < params.stop_overflow and cur_metric.hpwl > prev_metric.hpwl) or cur_metric.max_density < 1.0):
                            logging.debug("L1 stopping criteria: %d > 100 and (( %g < 0.1 and %g > %g ) or %g < 1.0)" % (L1_gamma_step, cur_metric.overflow, cur_metric.hpwl, prev_metric.hpwl, cur_metric.max_density))
                            return True
                    return False 

                def L2_stop_criterion(L1_gamma_step, L2_density_weight_step, metrics): 
                    if len(metrics) > 1: 
                        cur_metric = metrics[-1][-1]
                        prev_metric = metrics[-2][-1]
                        if (cur_metric.overflow < params.stop_overflow and cur_metric.hpwl > prev_metric.hpwl) or cur_metric.max_density < 1.0:
                            logging.debug("L2 stopping criteria: %d and (( %g < 0.1 and %g > %g ) or %g < 1.0)" % (L2_density_weight_step, cur_metric.overflow, cur_metric.hpwl, prev_metric.hpwl, cur_metric.max_density))
                            return True
                    return False 

                def L3_stop_criterion(L1_gamma_step, L2_density_weight_step, L3_step, metrics):
                    if len(metrics) > 1: 
                        cur_metric = metrics[-1]
                        prev_metric = metrics[-2]
                        if cur_metric.objective >= prev_metric.objective * 0.999:
                            logging.debug("L3 stopping criteria: %d and %g > %g * 0.999" % (L3_step, cur_metric.objective, prev_metric.objective))
                            return True 
                    return False 

                def one_descent_step(L1_gamma_step, L2_density_weight_step, L3_step, iteration, metrics):
                    t0 = time.time()

                    # metric for this iteration 
                    cur_metric = EvalMetrics.EvalMetrics(iteration, (L1_gamma_step, L2_density_weight_step, L3_step))
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
                    #logging.debug("evaluation %.3f ms" % ((time.time()-t1)*1000))
                    #t2 = time.time()

                    # as nesterov requires line search, we cannot follow the convention of other solvers
                    if optimizer_name.lower() in ["sgd", "adam", "sgd_momentum", "sgd_nesterov"]: 
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

                    logging.info("full step %.3f ms" % ((time.time()-t0)*1000))


                L1_metrics = all_metrics
                L2_flat_iteration = 0
                for L1_gamma_step in range(model.L1_gamma_iteration):
                    L1_metrics.append([])
                    L2_metrics = L1_metrics[-1]
                    for L2_density_weight_step in range(model.L2_density_weight_iteration):
                        L2_metrics.append([])
                        L3_metrics = L2_metrics[-1]
                        for L3_step in range(model.L3_iteration):
                            one_descent_step(L1_gamma_step, L2_density_weight_step, L3_step, iteration, L3_metrics)
                            iteration += 1
                            # stopping criteria 
                            if L3_stop_criterion(L1_gamma_step, L2_density_weight_step, L3_step, L3_metrics):
                                break 
                        L2_flat_iteration += 1
                        # update density weight 
                        if L2_flat_iteration > 1: 
                            model.op_collections.update_density_weight_op(L2_metrics[-1][-1], L2_metrics[-2][-1] if len(L2_metrics) > 1 else L1_metrics[-2][-1][-1], L2_flat_iteration)
                        #logging.debug("update density weight %.3f ms" % ((time.time()-t2)*1000))
                        if L2_stop_criterion(L1_gamma_step, L2_density_weight_step, L2_metrics):
                            break 

                    # gradually reduce gamma to tradeoff smoothness and accuracy 
                    model.op_collections.update_gamma_op(L1_gamma_step, L2_metrics[-1][-1].overflow)
                    if L1_stop_criterion(L1_gamma_step, L1_metrics):
                        break 

                    # update learning rate 
                    if optimizer_name.lower() in ["sgd", "adam", "sgd_momentum", "sgd_nesterov", "cg"]: 
                        if 'learning_rate_decay' in global_place_params: 
                            for param_group in optimizer.param_groups:
                                param_group['lr'] *= global_place_params['learning_rate_decay']

                logging.info("optimizer %s takes %.3f seconds" % (optimizer_name, time.time()-tt))
            # recover node size and pin offset for legalization, since node size is adjusted in global placement
            self.data_collections.node_size_x.copy_(self.data_collections.original_node_size_x)
            self.data_collections.node_size_y.copy_(self.data_collections.original_node_size_y)
            self.data_collections.pin_offset_x.copy_(self.data_collections.original_pin_offset_x)
            self.data_collections.pin_offset_y.copy_(self.data_collections.original_pin_offset_y)

        else: 
            cur_metric = EvalMetrics.EvalMetrics(iteration)
            all_metrics.append(cur_metric)
            cur_metric.evaluate(placedb, {"hpwl" : self.op_collections.hpwl_op}, self.pos[0])
            logging.info(cur_metric)

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
            cur_metric.evaluate(placedb, {"hpwl" : self.op_collections.hpwl_op}, self.pos[0])
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
            cur_metric.evaluate(placedb, {"hpwl" : self.op_collections.hpwl_op}, self.pos[0])
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
