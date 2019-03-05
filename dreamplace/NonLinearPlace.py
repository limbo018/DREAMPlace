##
# @file   NonLinearPlace.py
# @author Yibo Lin
# @date   Jul 2018
#

import os 
import sys
import time 
import pickle
import numpy as np 
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
    def __init__(self, params, placedb):
        super(NonLinearPlace, self).__init__(params, placedb)

    """
    solve placement 
    """
    def __call__(self, params, placedb, enable_flag=True):
        def evaluate(model, metrics, t): 
            if "iteration" in metrics: 
                content = "[I] iteration %4d" % (metrics["iteration"])

            if "wirelength" not in metrics: 
                #ttt = time.time()
                wirelength = model.op_collections.wirelength_op(model.data_collections.pos[0])
                metrics["wirelength"] = wirelength.data 
                #print("wirelength forward takes %.3f seconds" % (time.time()-ttt))
            content += ", wirelength %.3E" % (metrics["wirelength"])

            if "density" not in metrics: 
                #ttt = time.time()
                density = model.op_collections.density_op(model.data_collections.pos[0])
                metrics["density"] = density.data 
                #print("density forward takes %.3f seconds" % (time.time()-ttt))
            content += ", density %.3E" % (metrics["density"])

            #if "density_weight" not in metrics: 
            #    metrics["density_weight"] = model.density_weight.data 
            #content += ", density_weight %.3E" % (metrics["density_weight"])

            if "hpwl" not in metrics: 
                #ttt = time.time()
                hpwl = self.op_collections.hpwl_op(model.data_collections.pos[0])
                metrics["hpwl"] = hpwl.data 
                #print("hpwl forward takes %.6f seconds" % (time.time()-ttt))
            content += ", HPWL %.6E" % (metrics["hpwl"])

            #if "rmst_wls" not in metrics: 
            #    #ttt = time.time()
            #    rmst_wls = self.op_collections.rmst_wl_op(model.data_collections.pos[0])
            #    rmst_wl = rmst_wls.sum()
            #    metrics["rmst_wl"] = rmst_wl.data 
            #    #print("rmst_wl forward takes %.6f seconds" % (time.time()-ttt))
            #content += ", RMSTWL %.3E" % (metrics["rmst_wl"])

            if "overflow" not in metrics: 
                #ttt = time.time()
                overflow, max_density = self.op_collections.density_overflow_op(model.data_collections.pos[0])
                metrics["overflow"] = overflow.data / placedb.total_movable_node_area
                metrics["max_density"] = max_density.data 
                #print("overflow forward takes %.6f seconds" % (time.time()-ttt))
            content += ", overflow %.6E" % (metrics["overflow"])
            content += ", max density %.3E" % (metrics["max_density"])

            # evaluation 
            #model.op_collections.wirelength_eval_op(model.data_collections.pos[0], rmst_wls.cuda(), write_flag=False)

            content += ", time %.3fms" % ((time.time()-t)*1000)

            print(content)

            return metrics 

        iteration = 0
        metrics = []

        if not enable_flag:
            return metrics 

        if params.global_place_flag: 
            for global_place_params in params.global_place_stages:
                torch.cuda.synchronize()
                tt = time.time()
                # construct model and optimizer 
                density_weight = metrics[-1].density_weight if metrics else 0.0
                model = PlaceObj.PlaceObj(density_weight, params, placedb, self.data_collections, self.op_collections, global_place_params).to(self.data_collections.pos[0].device)
                name = global_place_params["optimizer"]

                if name.lower() == "adam": 
                    optimizer = torch.optim.Adam(self.parameters(), lr=model.learning_rate)
                elif name.lower() == "sgd": 
                    optimizer = torch.optim.SGD(self.parameters(), lr=model.learning_rate)
                elif name.lower() == "sgd_momentum": 
                    optimizer = torch.optim.SGD(self.parameters(), lr=model.learning_rate, momentum=0.9, nesterov=False)
                elif name.lower() == "sgd_nesterov": 
                    optimizer = torch.optim.SGD(self.parameters(), lr=model.learning_rate, momentum=0.9, nesterov=True)
                elif name.lower() == "cg": 
                    optimizer = ConjugateGradientOptimizer.ConjugateGradientOptimizer(self.parameters(), lr=model.learning_rate)
                elif name.lower() == "cgls": 
                    optimizer = ConjugateGradientOptimizer.ConjugateGradientOptimizer(self.parameters(), lr=model.learning_rate, line_search_fn=LineSearch.build_line_search_fn_armijo(model.obj_fn))
                elif name.lower() == "nesterov": 
                    optimizer = NesterovAcceleratedGradientOptimizer.NesterovAcceleratedGradientOptimizer(self.parameters(), lr=model.learning_rate, 
                            obj_and_grad_fn=model.obj_and_grad_fn,
                            constraint_fn=self.op_collections.move_boundary_op,
                            )
                else:
                    assert 0, "unknown optimizer %s" % (name)

                print("[I] use %s optimizer" % (name))
                model.train()
                learning_rate = model.learning_rate
                eval_ops = {
                        #"wirelength" : self.op_collections.wirelength_op, 
                        #"density" : self.op_collections.density_op, 
                        "hpwl" : self.op_collections.hpwl_op, 
                        "overflow" : self.op_collections.density_overflow_op
                        }

                if iteration == 0: 
                    if params.gp_noise_ratio > 0.0: 
                        print("[I] add %g%% noise" % (params.gp_noise_ratio*100))
                        model.op_collections.noise_op(model.data_collections.pos[0], params.gp_noise_ratio)

                torch.cuda.synchronize()
                print("[I] %s initialization takes %g seconds" % (name, (time.time()-tt)))

                for step in range(model.iteration):
                    # metric for this iteration 
                    cur_metric = EvalMetrics.EvalMetrics(iteration)
                    metrics.append(cur_metric)

                    #print("############## iteration start ###################")
                    t0 = time.time()

                    self.op_collections.move_boundary_op(model.data_collections.pos[0])

                    # plot placement 
                    #if iteration == 0 or (iteration > 600 and iteration % 10 == 0): 
                    #    #cur_pos = model.data_collections.pos[0].data.clone().cpu().numpy()
                    #    cur_pos = model.data_collections.pos[0].data.clone().cpu()
                    #    self.plot(params, placedb, iteration, cur_pos)
                    #    exit()

                    if torch.eq(model.density_weight, 0.0):
                        model.initialize_density_weight(params, placedb)
                        print("[I] density_weight = %.6E" % (model.density_weight.data))

                    # I found it is not possible to call backward inside a backward function in autograd 
                    # this means it is difficult to wrap wirelength and density gradients into autograd function 
                    # so the easiest way is to leave it explicitly here 

                    optimizer.zero_grad()
                    #t1 = time.time()
                    cur_metric.evaluate(placedb, eval_ops, model.data_collections.pos[0])
                    #print("evaluation %.3f ms" % ((time.time()-t1)*1000))
                    #t2 = time.time()
                    # update density weight 
                    # gradually reduce gamma to tradeoff smoothness and accuracy 
                    if len(metrics) > 1:
                        model.op_collections.update_density_weight_op(metrics)
                        model.op_collections.update_gamma_op(step, cur_metric.overflow)
                    cur_metric.density_weight = model.density_weight.data
                    cur_metric.gamma = model.gamma.data
                    #print("update density weight %.3f ms" % ((time.time()-t2)*1000))

                    if name.lower() in ["sgd", "adam", "sgd_momentum", "sgd_nesterov", "cg"]: 
                        model.obj_and_grad_fn(model.data_collections.pos[0])
                    elif name.lower() != "nesterov":
                        assert 0, "unsupported optimizer %s" % (name)

                    # exit when overflow goes to zero 
                    if iteration > 100 and ((cur_metric.overflow < params.stop_overflow and cur_metric.hpwl > metrics[-2].hpwl) or cur_metric.max_density < 1.0):
                        print("[D] stoping criteria: %d > 100 and (( %g < 0.1 and %g > %g ) or %g < 1.0)" % (iteration, cur_metric.overflow, cur_metric.hpwl, metrics[-2].hpwl, cur_metric.max_density))
                        break 

                    # update learning rate 
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = learning_rate

                    print(cur_metric)

                    t3 = time.time()
                    optimizer.step()
                    print("[T] optimizer step %.3f ms" % ((time.time()-t3)*1000))

                    #if (model.data_collections.pos[0][:placedb.num_movable_nodes] < placedb.xl-1).any():
                    #    print("out of bounds ", (model.data_collections.pos[0][:placedb.num_movable_nodes] < placedb.xl-1).nonzero().size())
                    #    #pdb.set_trace()
                    #if (model.data_collections.pos[0][placedb.num_physical_nodes:placedb.num_nodes] < placedb.xl-1).any():
                    #    print("out of bounds ", (model.data_collections.pos[0][placedb.num_physical_nodes:placedb.num_nodes] < placedb.xl-1).nonzero().size())
                    #    #pdb.set_trace()
                    #if (model.data_collections.pos[0][placedb.num_nodes:placedb.num_nodes+placedb.num_movable_nodes] < placedb.yl-1).any():
                    #    print("out of bounds ", (model.data_collections.pos[0][placedb.num_nodes:placedb.num_nodes+placedb.num_movable_nodes] < placedb.yl-1).nonzero().size())
                    #    #pdb.set_trace()
                    #if (model.data_collections.pos[0][placedb.num_nodes+placedb.num_physical_nodes:2*placedb.num_nodes] < placedb.yl-1).any():
                    #    print("out of bounds ", (model.data_collections.pos[0][placedb.num_nodes+placedb.num_physical_nodes:2*placedb.num_nodes] < placedb.yl-1).nonzero().size())
                    #    #pdb.set_trace()

                    iteration += 1

                    print("[T] full step %.3f ms" % ((time.time()-t0)*1000))
                    #print("############## iteration end ###################")

                print("[T] optimizer %s takes %.3f seconds" % (name, time.time()-tt))

                #### for debug ### 
                #from ops import density_overflow
                #num_bins_x = 4
                #num_bins_y = 4
                #bin_size_x = (placedb.xh-placedb.xl)/num_bins_x 
                #bin_size_y = (placedb.yh-placedb.yl)/num_bins_y 
                #bin_center_x = placedb.bin_centers(placedb.xl, placedb.xh, bin_size_x)
                #bin_center_y = placedb.bin_centers(placedb.yl, placedb.yh, bin_size_y)
                #density_overflow_op = density_overflow.DensityOverflow(
                #    torch.from_numpy(placedb.node_size_x).to(self.device), torch.from_numpy(placedb.node_size_y).to(self.device), 
                #    torch.from_numpy(bin_center_x).to(self.device), torch.from_numpy(bin_center_y).to(self.device), 
                #    target_density=params.target_density, 
                #    xl=placedb.xl, yl=placedb.yl, xh=placedb.xh, yh=placedb.yh, 
                #    bin_size_x=bin_size_x, bin_size_y=bin_size_y, 
                #    num_movable_nodes=placedb.num_movable_nodes, 
                #    num_terminals=placedb.num_terminals, 
                #    num_filler_nodes=0
                #    )
                #tmp_overflow, tmp_max_density = density_overflow_op(model.data_collections.pos[0])
                #overflow = tmp_overflow.data / placedb.total_movable_node_area
                #max_density = tmp_max_density.data 
                #print("real density overflow for %dx%d bins %g, max_density %g" % (num_bins_x, num_bins_y, overflow, max_density))
                #exit()
                #### for debug ###

        if params.legalize_flag:
            tt = time.time()
            self.pos[0].data.copy_(self.op_collections.greedy_legalize_op(self.pos[0]))
            print("[I] legalization takes %.3f seconds" % (time.time()-tt))

        if params.detailed_place_flag: 
            print("[W] detailed placement NOT implemented yet, skipped")

        # save results 
        cur_pos = self.pos[0].data.clone().cpu().numpy()
        # assign solution 
        placedb.node_x[:placedb.num_movable_nodes] = cur_pos[0:placedb.num_movable_nodes]
        placedb.node_y[:placedb.num_movable_nodes] = cur_pos[placedb.num_nodes:placedb.num_nodes+placedb.num_movable_nodes]
        # plot placement 
        #self.plot(params, placedb, iteration, cur_pos)
        return metrics 
