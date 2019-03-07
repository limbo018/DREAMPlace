##
# @file   Params.py
# @author Yibo Lin
# @date   Apr 2018
# @brief  User parameters 
#

import json 
import math 

class Params: 
    """
    @brief Parameter class 
    """
    def __init__(self):
        """
        @brief initialization 
        """
        self.aux_file = None # directory for .aux file 
        self.gpu = True # enable gpu or not 
        self.num_bins_x = 512 # number of bins in horizontal direction 
        self.num_bins_y = 512 # number of bins in vertical direction 
        self.global_place_stages = None # global placement configurations of each stage, a dictionary of {"num_bins_x", "num_bins_y", "iteration", "learning_rate"}, learning_rate is relative to bin size 
        self.target_density = 0.8 # target density 
        self.density_weight = 1.0 # weight of density cost
        self.gamma = 0.5 # log-sum-exp coefficient 
        self.random_seed = 1000 # random seed 
        self.result_dir = "results" # result directory
        self.scale_factor = 1.0 # scale factor to avoid numerical overflow
        self.ignore_net_degree = 100 # ignore net degree larger than some value
        self.gp_noise_ratio = 0.025 # noise to initial positions for global placement 
        self.enable_fillers = True # enable filler cells 
        self.global_place_flag = True # whether use global placement 
        self.legalize_flag = True # whether use internal legalization
        self.detailed_place_flag = True # whether use internal detailed placement
        self.stop_overflow = 0.1 # stopping criteria, consider stop when the overflow reaches to a ratio 
        self.dtype = 'float32' # data type, float32/float64
        self.detailed_place_engine = "" # external detailed placement engine to be called after placement 
        self.plot_flag = False # whether plot solution or not 
        self.RePlAce_ref_hpwl = 3.5e5
        self.RePlAce_LOWER_PCOF = 0.95
        self.RePlAce_UPPER_PCOF = 1.05

    def toJson(self):
        """
        @brief convert to json  
        """
        data = dict()
        data['aux_file'] = self.aux_file
        data['gpu'] = self.gpu
        data['num_bins_x'] = self.num_bins_x
        data['num_bins_y'] = self.num_bins_y
        data['global_place_stages'] = self.global_place_stages
        data['target_density'] = self.target_density
        data['density_weight'] = self.density_weight
        data['gamma'] = self.gamma
        data['random_seed'] = self.random_seed
        data['result_dir'] = self.result_dir
        data['scale_factor'] = self.scale_factor
        data['ignore_net_degree'] = self.ignore_net_degree
        data['gp_noise_ratio'] = self.gp_noise_ratio
        data['enable_fillers'] = self.enable_fillers
        data['global_place_flag'] = self.global_place_flag
        data['legalize_flag'] = self.legalize_flag
        data['detailed_place_flag'] = self.detailed_place_flag
        data['stop_overflow'] = self.stop_overflow
        data['dtype'] = self.dtype
        data['detailed_place_engine'] = self.detailed_place_engine
        data['plot_flag'] = self.plot_flag
        data['RePlAce_ref_hpwl'] = self.RePlAce_ref_hpwl
        data['RePlAce_LOWER_PCOF'] = self.RePlAce_LOWER_PCOF
        data['RePlAce_UPPER_PCOF'] = self.RePlAce_UPPER_PCOF
        return data 

    def fromJson(self, data):
        """
        @brief load form json 
        """
        if 'aux_file' in data: self.aux_file = data['aux_file']
        if 'gpu' in data: self.gpu = data['gpu']
        if 'num_bins_x' in data: self.num_bins_x = data['num_bins_x']
        if 'num_bins_y' in data: self.num_bins_y = data['num_bins_y']
        if 'global_place_stages' in data: self.global_place_stages = data['global_place_stages']
        if 'target_density' in data: self.target_density = data['target_density']
        if 'density_weight' in data: self.density_weight = data['density_weight']
        if 'gamma' in data: self.gamma = data['gamma']
        if 'random_seed' in data: self.random_seed = data['random_seed']
        if 'result_dir' in data: self.result_dir = data['result_dir']
        if 'scale_factor' in data: self.scale_factor = data['scale_factor']
        if 'ignore_net_degree' in data: self.ignore_net_degree = data['ignore_net_degree']
        if 'gp_noise_ratio' in data: self.gp_noise_ratio = data['gp_noise_ratio']
        if 'enable_fillers' in data: self.enable_fillers = data['enable_fillers']
        if 'global_place_flag' in data: self.global_place_flag = data['global_place_flag']
        if 'legalize_flag' in data: self.legalize_flag = data['legalize_flag']
        if 'detailed_place_flag' in data: self.detailed_place_flag = data['detailed_place_flag']
        if 'stop_overflow' in data: self.stop_overflow = data['stop_overflow']
        if 'dtype' in data: self.dtype = data['dtype']
        if 'detailed_place_engine' in data: self.detailed_place_engine = data['detailed_place_engine']
        if 'plot_flag' in data: self.plot_flag = data['plot_flag']
        if 'RePlAce_ref_hpwl' in data: self.RePlAce_ref_hpwl = data['RePlAce_ref_hpwl']
        if 'RePlAce_LOWER_PCOF' in data: self.RePlAce_LOWER_PCOF = data['RePlAce_LOWER_PCOF']
        if 'RePlAce_UPPER_PCOF' in data: self.RePlAce_UPPER_PCOF = data['RePlAce_UPPER_PCOF']

    def dump(self, filename):
        """
        @brief dump to json file 
        """
        with open(filename, 'w') as f:
            json.dump(self.toJson(), f)

    def load(self, filename):
        """
        @brief load from json file 
        """
        with open(filename, 'r') as f:
            self.fromJson(json.load(f))

    def __str__(self):
        """
        @brief string 
        """
        return str(self.toJson())

    def __repr__(self):
        """
        @brief print 
        """
        return self.__str__()
