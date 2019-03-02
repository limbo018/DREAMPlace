##
# @file   Params.py
# @author Yibo Lin
# @date   Apr 2018
#

import json 
import math 

"""
Parameter class 
"""
class Params: 
    """
    initialization 
    """
    def __init__(self):
        self.aux_file = None # directory for .aux file 
        self.gpu = True # enable gpu or not 
        self.num_bins_x = 3 # bin width in number of row height 
        self.num_bins_y = 3 # bin height in number of row height 
        self.opt_num_bins = None # bin information for optimization, a dictionary of {"x", "y", "iteration", "learning_rate"}, learning_rate is relative to bin size 
        self.target_density = 0.8 # target density 
        self.density_weight = 1.0 # weight of density cost
        self.gamma = 0.5 # log-sum-exp coefficient 
        self.random_seed = 1000 # random seed 
        self.summary_dir = "summary" # summary directory
        self.scale_factor = 1e-3 # scale factor to avoid numerical overflow
        self.ignore_net_degree = 100 # ignore net degree larger than some value
        self.gp_noise_ratio = 0.025 # noise to initial positions for global placement 
        self.enable_fillers = True # enable filler cells 
        self.global_place_flag = True # whether use global placement 
        self.legalize_flag = True # whether use internal legalization
        self.detailed_place_flag = True # whether use internal detailed placement
        self.stop_overflow = 0.1 # stopping criteria, consider stop when the overflow reaches to a ratio 
        self.dtype = 'float64' # data type, float32/float64
    """
    convert to json  
    """
    def toJson(self):
        data = dict()
        data['aux_file'] = self.aux_file
        data['gpu'] = self.gpu
        data['num_bins_x'] = self.num_bins_x
        data['num_bins_y'] = self.num_bins_y
        data['opt_num_bins'] = self.opt_num_bins
        data['target_density'] = self.target_density
        data['density_weight'] = self.density_weight
        data['gamma'] = self.gamma
        data['random_seed'] = self.random_seed
        data['summary_dir'] = self.summary_dir
        data['scale_factor'] = self.scale_factor
        data['ignore_net_degree'] = self.ignore_net_degree
        data['gp_noise_ratio'] = self.gp_noise_ratio
        data['enable_fillers'] = self.enable_fillers
        data['global_place_flag'] = self.global_place_flag
        data['legalize_flag'] = self.legalize_flag
        data['detailed_place_flag'] = self.detailed_place_flag
        data['stop_overflow'] = self.stop_overflow
        data['dtype'] = self.dtype
        return data 
    """
    load form json 
    """
    def fromJson(self, data):
        if 'aux_file' in data: self.aux_file = data['aux_file']
        if 'gpu' in data: self.gpu = data['gpu']
        if 'num_bins_x' in data: self.num_bins_x = data['num_bins_x']
        if 'num_bins_y' in data: self.num_bins_y = data['num_bins_y']
        if 'opt_num_bins' in data: self.opt_num_bins = data['opt_num_bins']
        if 'target_density' in data: self.target_density = data['target_density']
        if 'density_weight' in data: self.density_weight = data['density_weight']
        if 'gamma' in data: self.gamma = data['gamma']
        if 'random_seed' in data: self.random_seed = data['random_seed']
        if 'summary_dir' in data: self.summary_dir = data['summary_dir']
        if 'scale_factor' in data: self.scale_factor = data['scale_factor']
        if 'ignore_net_degree' in data: self.ignore_net_degree = data['ignore_net_degree']
        if 'gp_noise_ratio' in data: self.gp_noise_ratio = data['gp_noise_ratio']
        if 'enable_fillers' in data: self.enable_fillers = data['enable_fillers']
        if 'global_place_flag' in data: self.global_place_flag = data['global_place_flag']
        if 'legalize_flag' in data: self.legalize_flag = data['legalize_flag']
        if 'detailed_place_flag' in data: self.detailed_place_flag = data['detailed_place_flag']
        if 'stop_overflow' in data: self.stop_overflow = data['stop_overflow']
        if 'dtype' in data: self.dtype = data['dtype']

    """
    dump to json file 
    """
    def dump(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.toJson(), f)
    """
    load from json file 
    """
    def load(self, filename):
        with open(filename, 'r') as f:
            self.fromJson(json.load(f))
    """
    string 
    """
    def __str__(self):
        return str(self.toJson())
    """
    print 
    """
    def __repr__(self):
        return self.__str__()
