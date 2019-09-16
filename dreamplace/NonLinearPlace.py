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
import ConjugateGradientOptimizer
import NesterovAcceleratedGradientOptimizer
import LineSearch
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
        metrics = []

        # global placement 
        if params.global_place_flag: 
            # global placement may run in multiple stages according to user specification 
            for global_place_params in params.global_place_stages:
                if params.gpu: 
                    torch.cuda.synchronize()
                tt = time.time()
                # construct model and optimizer 
                density_weight = metrics[-1].density_weight if metrics else 0.0
                # construct placement model 
                model = PlaceObj.PlaceObj(density_weight, params, placedb, self.data_collections, self.op_collections, global_place_params).to(self.data_collections.pos[0].device)
                optimizer_name = global_place_params["optimizer"]

                # determine optimizer
                if optimizer_name.lower() == "adam": 
                    optimizer = torch.optim.Adam(self.parameters(), lr=model.learning_rate)
                elif optimizer_name.lower() == "sgd": 
                    optimizer = torch.optim.SGD(self.parameters(), lr=model.learning_rate)
                elif optimizer_name.lower() == "sgd_momentum": 
                    optimizer = torch.optim.SGD(self.parameters(), lr=model.learning_rate, momentum=0.9, nesterov=False)
                elif optimizer_name.lower() == "sgd_nesterov": 
                    optimizer = torch.optim.SGD(self.parameters(), lr=model.learning_rate, momentum=0.9, nesterov=True)
                elif optimizer_name.lower() == "cg": 
                    optimizer = ConjugateGradientOptimizer.ConjugateGradientOptimizer(self.parameters(), lr=model.learning_rate)
                elif optimizer_name.lower() == "cgls": 
                    optimizer = ConjugateGradientOptimizer.ConjugateGradientOptimizer(self.parameters(), lr=model.learning_rate, line_search_fn=LineSearch.build_line_search_fn_armijo(model.obj_fn))
                elif optimizer_name.lower() == "nesterov": 
                    optimizer = NesterovAcceleratedGradientOptimizer.NesterovAcceleratedGradientOptimizer(self.parameters(), lr=model.learning_rate, 
                            obj_and_grad_fn=model.obj_and_grad_fn,
                            constraint_fn=self.op_collections.move_boundary_op,
                            )
                else:
                    assert 0, "unknown optimizer %s" % (optimizer_name)

                logging.info("use %s optimizer" % (optimizer_name))
                model.train()
                learning_rate = model.learning_rate
                # defining evaluation ops 
                eval_ops = {
                        #"wirelength" : self.op_collections.wirelength_op, 
                        #"density" : self.op_collections.density_op, 
                        "hpwl" : self.op_collections.hpwl_op, 
                        "overflow" : self.op_collections.density_overflow_op
                        }

                if iteration == 0: 
                    if params.gp_noise_ratio > 0.0: 
                        logging.info("add %g%% noise" % (params.gp_noise_ratio*100))
                        model.op_collections.noise_op(model.data_collections.pos[0], params.gp_noise_ratio)

                if params.gpu: 
                    torch.cuda.synchronize()
                logging.info("%s initialization takes %g seconds" % (optimizer_name, (time.time()-tt)))

                # as nesterov requires line search, we cannot follow the convention of other solvers
                if optimizer_name.lower() in {"sgd", "adam", "sgd_momentum", "sgd_nesterov", "cg"}: 
                    model.obj_and_grad_fn(model.data_collections.pos[0])
                elif optimizer_name.lower() != "nesterov":
                    assert 0, "unsupported optimizer %s" % (optimizer_name)

                for step in range(model.iteration):
                    t0 = time.time()
                    
                    # metric for this iteration 
                    cur_metric = EvalMetrics.EvalMetrics(iteration)
                    metrics.append(cur_metric)

                    # move any out-of-bound cell back to placement region 
                    self.op_collections.move_boundary_op(model.data_collections.pos[0])

                    if torch.eq(model.density_weight, 0.0):
                        model.initialize_density_weight(params, placedb)
                        logging.info("density_weight = %.6E" % (model.density_weight.data))

                    optimizer.zero_grad()
                    
                    # t1 = time.time()
                    cur_metric.evaluate(placedb, eval_ops, model.data_collections.pos[0])
                    #logging.debug("evaluation %.3f ms" % ((time.time()-t1)*1000))
                    #t2 = time.time()
                    # update density weight 
                    # gradually reduce gamma to tradeoff smoothness and accuracy 
                    if len(metrics) > 1:
                        model.op_collections.update_density_weight_op(metrics)
                        model.op_collections.update_gamma_op(step, cur_metric.overflow)
                    cur_metric.density_weight = model.density_weight.data
                    cur_metric.gamma = model.gamma.data
                    #logging.debug("update density weight %.3f ms" % ((time.time()-t2)*1000))

                    # as nesterov requires line search, we cannot follow the convention of other solvers
                    if optimizer_name.lower() in ["sgd", "adam", "sgd_momentum", "sgd_nesterov", "cg"]: 
                        model.obj_and_grad_fn(model.data_collections.pos[0])
                    elif optimizer_name.lower() != "nesterov":
                        assert 0, "unsupported optimizer %s" % (optimizer_name)

                    # stopping criteria 
                    if iteration > 100 and ((cur_metric.overflow < params.stop_overflow and cur_metric.hpwl > metrics[-2].hpwl) or cur_metric.max_density < 1.0):
                        logging.debug("stopping criteria: %d > 100 and (( %g < 0.1 and %g > %g ) or %g < 1.0)" % (iteration, cur_metric.overflow, cur_metric.hpwl, metrics[-2].hpwl, cur_metric.max_density))
                        break 

                    # update learning rate 
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = learning_rate

                    logging.info(cur_metric)
                    # plot placement 
                    if params.plot_flag and iteration % 100 == 0: 
                        cur_pos = self.pos[0].data.clone().cpu().numpy()
                        self.plot(params, placedb, iteration, cur_pos)

                    t3 = time.time()
                    optimizer.step()
                    logging.info("optimizer step %.3f ms" % ((time.time()-t3)*1000))

                    iteration += 1

                    logging.info("full step %.3f ms" % ((time.time()-t0)*1000))

                logging.info("optimizer %s takes %.3f seconds" % (optimizer_name, time.time()-tt))

        # legalization 
        if params.legalize_flag:
            tt = time.time()
            self.pos[0].data.copy_(self.op_collections.greedy_legalize_op(self.pos[0]))
            logging.info("legalization takes %.3f seconds" % (time.time()-tt))

        # detailed placement 
        if params.detailed_place_flag: 
            logging.warning("detailed placement NOT implemented yet, skipped")

        # save results 
        cur_pos = self.pos[0].data.clone().cpu().numpy()
        # apply solution 
        placedb.apply(params, cur_pos[0:placedb.num_movable_nodes], cur_pos[placedb.num_nodes:placedb.num_nodes+placedb.num_movable_nodes])
        # plot placement 
        if params.plot_flag: 
            self.plot(params, placedb, iteration, cur_pos)
        return metrics 
