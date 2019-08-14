##
# @file   Params.py
# @author Yibo Lin
# @date   Apr 2018
# @brief  User parameters
#

import os
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
        # for Bookshelf format 
        self.aux_input = None # directory for .aux file 
        # for LEF/DEF format 
        self.lef_input = None # input LEF file 
        self.def_input = None # input DEF file 
        self.verilog_input = None # input VERILOG file 

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
        self.detailed_place_command = "-nolegal -nodetail" # commands for external detailed placement 
        self.plot_flag = False # whether plot solution or not 
        self.RePlAce_ref_hpwl = 3.5e5 # reference HPWL used in RePlAce for updating density weight 
        self.RePlAce_LOWER_PCOF = 0.95 # lower bound ratio used in RePlAce for updating density weight 
        self.RePlAce_UPPER_PCOF = 1.05 # upper bound ratio used in RePlAce for updating density weight 
        self.random_center_init_flag = True # whether perform random initialization around the center for global placement 
        self.sort_nets_by_degree = False # whether sort net and pins such that pins belonging to the same net are together 
        self.num_threads = 8 # number of CPU threads
        self.dump_global_place_solution_flag = False # whether dump intermediate global placement solution as a compressed pickle file 
        self.dump_legalize_solution_flag = False # whether dump intermediate legalization solution as a compressed pickle file 

    def printWelcome(self):
        """
        @brief print welcome message
        """
        content = """\
========================================================
                       DREAMPlace
            Yibo Lin (http://yibolin.com)
   David Z. Pan (http://users.ece.utexas.edu/~dpan)
========================================================"""
        print(content)

    def printHelp(self):
        """
        @brief print help message for JSON parameters
        """
        content = """\
                    JSON Parameters
========================================================
aux_input [required for Bookshelf]    | input .aux file 
lef_input [required for LEF/DEF]      | input LEF file  
def_input [required for LEF/DEF]      | input DEF file  
verilog_input [optional for LEF/DEF]  | input VERILOG file, provide circuit netlist information if it is not included in DEF file 
gpu [default %d]                       | enable gpu or not 
num_bins_x [default %d]              | number of bins in horizontal direction 
num_bins_y [default %d]              | number of bins in vertical direction 
global_place_stages [required]        | global placement configurations of each stage, a dictionary of {"num_bins_x", "num_bins_y", "iteration", "learning_rate"}, learning_rate is relative to bin size
target_density [default %g]          | target density
density_weight [default %.1f]          | weight of density cost
gamma [default %g]                   | coefficient for log-sum-exp and weighted-average wirelength
random_seed [default %d]            | random seed
result_dir [default %s]          | result directory
scale_factor [default %.1f]            | scale factor to avoid numerical overflow
ignore_net_degree [default %d]       | ignore net degree larger than some value
gp_noise_ratio [default %g]        | noise to initial positions for global placement
enable_fillers [default %d]            | enable filler cells
global_place_flag [default %d]         | whether use global placement
legalize_flag [default %d]             | whether use internal legalization
detailed_place_flag [default %d]       | whether use internal detailed placement
stop_overflow [default %g]           | stopping criteria, consider stop when the overflow reaches to a ratio
dtype [default %s]               | data type, float32 | float64
detailed_place_engine [default %s]      | external detailed placement engine to be called after placement 
detailed_place_command [default %s]     | commands for external detailed placement engine 
plot_flag [default %d]                 | whether plot solution or not 
RePlAce_ref_hpwl [default %g]     | reference HPWL used in RePlAce for updating density weight 
RePlAce_LOWER_PCOF [default %g]     | lower bound ratio used in RePlAce for updating density weight 
RePlAce_UPPER_PCOF [default %g]     | upper bound ratio used in RePlAce for updating density weight 
random_center_init_flag [default %d] | whether perform random initialization around the center for global placement 
sort_nets_by_degree [default %d]    | whether sort nets by degree or not
num_threads [default %d]            | number of CPU threads
dump_global_place_solution_flag [default %d] | whether dump intermediate global placement solution 
dump_legalize_solution_flag [default %d] | whether dump intermediate legalization solution 
        """ % (self.gpu, 
                self.num_bins_x, 
                self.num_bins_y, 
                self.target_density, 
                self.density_weight, 
                self.gamma, 
                self.random_seed, 
                self.result_dir, 
                self.scale_factor, 
                self.ignore_net_degree, 
                self.gp_noise_ratio, 
                self.enable_fillers, 
                self.global_place_flag, 
                self.legalize_flag, 
                self.detailed_place_flag, 
                self.stop_overflow, 
                self.dtype, 
                self.detailed_place_engine, 
                self.detailed_place_command, 
                self.plot_flag, 
                self.RePlAce_ref_hpwl, 
                self.RePlAce_LOWER_PCOF, 
                self.RePlAce_UPPER_PCOF, 
                self.random_center_init_flag, 
                self.sort_nets_by_degree,
                self.num_threads,
                self.dump_global_place_solution_flag,
                self.dump_legalize_solution_flag
                )
        print(content)

    def toJson(self):
        """
        @brief convert to json
        """
        data = dict()
        data['aux_input'] = self.aux_input
        data['lef_input'] = self.lef_input
        data['def_input'] = self.def_input
        data['verilog_input'] = self.verilog_input
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
        data['detailed_place_command'] = self.detailed_place_command
        data['plot_flag'] = self.plot_flag
        data['RePlAce_ref_hpwl'] = self.RePlAce_ref_hpwl
        data['RePlAce_LOWER_PCOF'] = self.RePlAce_LOWER_PCOF
        data['RePlAce_UPPER_PCOF'] = self.RePlAce_UPPER_PCOF
        data['random_center_init_flag'] = self.random_center_init_flag
        data['sort_nets_by_degree'] = self.sort_nets_by_degree
        data['num_threads'] = self.num_threads
        data['dump_global_place_solution_flag'] = self.dump_global_place_solution_flag
        data['dump_legalize_solution_flag'] = self.dump_legalize_solution_flag
        return data 

    def fromJson(self, data):
        """
        @brief load form json
        """
        if 'aux_input' in data: self.aux_input = data['aux_input']
        if 'lef_input' in data: self.lef_input = data['lef_input']
        if 'def_input' in data: self.def_input = data['def_input']
        if 'verilog_input' in data: self.verilog_input = data['verilog_input']
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
        if 'detailed_place_command' in data: self.detailed_place_command = data['detailed_place_command']
        if 'plot_flag' in data: self.plot_flag = data['plot_flag']
        if 'RePlAce_ref_hpwl' in data: self.RePlAce_ref_hpwl = data['RePlAce_ref_hpwl']
        if 'RePlAce_LOWER_PCOF' in data: self.RePlAce_LOWER_PCOF = data['RePlAce_LOWER_PCOF']
        if 'RePlAce_UPPER_PCOF' in data: self.RePlAce_UPPER_PCOF = data['RePlAce_UPPER_PCOF']
        if 'random_center_init_flag' in data: self.random_center_init_flag = data['random_center_init_flag']
        if 'sort_nets_by_degree' in data: self.sort_nets_by_degree = data['sort_nets_by_degree']
        if 'num_threads' in data: self.num_threads = data['num_threads']
        if 'dump_global_place_solution_flag' in data: self.dump_global_place_solution_flag = data['dump_global_place_solution_flag']
        if 'dump_legalize_solution_flag' in data: self.dump_legalize_solution_flag = data['dump_legalize_solution_flag']

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

    def design_name(self):
        """
        @brief speculate the design name for dumping out intermediate solutions 
        """
        if self.aux_input: 
            design_name = os.path.basename(self.aux_input).replace(".aux", "").replace(".AUX", "")
        elif self.verilog_input:
            design_name = os.path.basename(self.verilog_input).replace(".v", "").replace(".V", "")
        elif self.def_input: 
            design_name = os.path.basename(self.def_input).replace(".def", "").replace(".DEF", "")
        return design_name 

    def solution_file_suffix(self): 
        """
        @brief speculate placement solution file suffix 
        """
        if self.def_input is not None: # LEF/DEF 
            return "def"
        else: # Bookshelf
            return "pl"
