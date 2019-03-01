##
# @file   EvalMetrics.py
# @author Yibo Lin
# @date   Sep 2018
#

import time
import pdb 

class EvalMetrics (object):
    def __init__(self, iteration=None):
        self.iteration = iteration 
        self.wirelength = None
        self.density = None 
        self.density_weight = None
        self.hpwl = None 
        self.rmst_wl = None
        self.overflow = None
        self.max_density = None
        self.gamma = None
        self.eval_time = None

    def __str__(self):
        content = ""
        if self.iteration is not None:
            content = "[I] iteration %4d" % (self.iteration)
        if self.wirelength is not None:
            content += ", wirelength %.3E" % (self.wirelength)
        if self.density is not None: 
            content += ", density %.3E" % (self.density)
        if self.density_weight is not None: 
            content += ", density_weight %.6E" % (self.density_weight)
        if self.hpwl is not None:
            content += ", HPWL %.6E" % (self.hpwl)
        if self.rmst_wl is not None:
            content += ", RMSTWL %.3E" % (self.rmst_wl)
        if self.overflow is not None:
            content += ", overflow %.6E" % (self.overflow)
        if self.max_density is not None:
            content += ", max density %.3E" % (self.max_density)
        if self.gamma is not None: 
            content += ", gamma %.6E" % (self.gamma)
        if self.eval_time is not None: 
            content += ", time %.3fms" % (self.eval_time*1000)

        return content 

    def __repr__(self):
        return self.__str__()

    def evaluate(self, placedb, ops, var):
        tt = time.time()
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
        self.eval_time = time.time()-tt
