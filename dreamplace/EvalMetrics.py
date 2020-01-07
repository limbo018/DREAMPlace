##
# @file   EvalMetrics.py
# @author Yibo Lin
# @date   Sep 2018
# @brief  Evaluation metrics 
#

import time
import torch 
import pdb 

class EvalMetrics (object):
    """
    @brief evaluation metrics at one step 
    """
    def __init__(self, iteration=None, detailed_step=None):
        """
        @brief initialization
        @param iteration optimization step 
        """
        self.iteration = iteration 
        self.detailed_step = detailed_step
        self.objective = None 
        self.wirelength = None
        self.density = None 
        self.density_weight = None
        self.hpwl = None 
        self.rmst_wl = None
        self.overflow = None
        self.route_utilization = None 
        self.pin_utilization = None
        self.max_density = None
        self.gamma = None
        self.eval_time = None

    def __str__(self):
        """
        @brief convert to string 
        """
        content = ""
        if self.iteration is not None:
            content = "iteration %4d" % (self.iteration)
        if self.detailed_step is not None:
            content += ", (%4d, %2d, %2d)" % (self.detailed_step[0], self.detailed_step[1], self.detailed_step[2])
        if self.objective is not None:
            content += ", Obj %.6E" % (self.objective)
        if self.wirelength is not None:
            content += ", WL %.3E" % (self.wirelength)
        if self.density is not None: 
            content += ", Density %.3E" % (self.density)
        if self.density_weight is not None: 
            content += ", DensityWeight %.6E" % (self.density_weight)
        if self.hpwl is not None:
            content += ", HPWL %.6E" % (self.hpwl)
        if self.rmst_wl is not None:
            content += ", RMSTWL %.3E" % (self.rmst_wl)
        if self.overflow is not None:
            content += ", Overflow %.6E" % (self.overflow)
        if self.max_density is not None:
            content += ", MaxDensity %.3E" % (self.max_density)
        if self.route_utilization is not None:
            content += ", RouteOverflow %.6E" % (self.route_utilization)
        if self.pin_utilization is not None:
            content += ", PinOverflow %.6E" % (self.pin_utilization)
        if self.gamma is not None: 
            content += ", gamma %.6E" % (self.gamma)
        if self.eval_time is not None: 
            content += ", time %.3fms" % (self.eval_time*1000)

        return content 

    def __repr__(self):
        """
        @brief print 
        """
        return self.__str__()

    def evaluate(self, placedb, ops, var):
        """
        @brief evaluate metrics 
        @param placedb placement database 
        @param ops a list of ops 
        @param var variables 
        """
        tt = time.time()
        with torch.no_grad(): 
            if "objective" in ops: 
                self.objective = ops["objective"](var).data
            if "wirelength" in ops:
                self.wirelength = ops["wirelength"](var).data
            if "density" in ops:
                self.density = ops["density"](var).data
            if "hpwl" in ops:
                self.hpwl = ops["hpwl"](var).data
            if "rmst_wls" in ops:
                rmst_wls = ops["rmst_wls"](var)
                self.rmst_wl = rmst_wls.sum().data
            if "overflow" in ops:
                overflow, max_density = ops["overflow"](var)
                self.overflow = overflow.data / placedb.total_movable_node_area
                self.max_density = max_density.data 
            if "route_utilization" in ops:
                route_utilization_map = ops["route_utilization"](var)
                route_utilization_map_sum = route_utilization_map.sum()
                self.route_utilization = route_utilization_map.sub_(1).clamp_(min=0).sum() / route_utilization_map_sum
            if "pin_utilization" in ops:
                pin_utilization_map = ops["pin_utilization"](var)
                pin_utilization_map_sum = pin_utilization_map.sum()
                self.pin_utilization = pin_utilization_map.sub_(1).clamp_(min=0).sum() / pin_utilization_map_sum
        self.eval_time = time.time()-tt
