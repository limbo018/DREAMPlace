##
# @file   Params.py
# @author Yibo Lin
# @date   Apr 2018
# @brief  User parameters
#

import os
import sys
import json 
import math 
from collections import OrderedDict
import pdb

class Params:
    """
    @brief Parameter class
    """
    def __init__(self):
        """
        @brief initialization
        """
        filename = os.path.join(os.path.dirname(__file__), 'params.json')
        self.__dict__ = {}
        params_dict = {}
        with open(filename, "r") as f:
            params_dict = json.load(f, object_pairs_hook=OrderedDict)
        for key, value in params_dict.items():
            if 'default' in value: 
                self.__dict__[key] = value['default']
            else:
                self.__dict__[key] = None
        self.__dict__['params_dict'] = params_dict

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
        content = self.toMarkdownTable()
        print(content)

    def toMarkdownTable(self):
        """
        @brief convert to markdown table 
        """
        key_length = len('JSON Parameter')
        key_length_map = []
        default_length = len('Default')
        default_length_map = []
        description_length = len('Description')
        description_length_map = []

        def getDefaultColumn(key, value):
            if sys.version_info.major < 3: # python 2
                flag = isinstance(value['default'], unicode)
            else: #python 3
                flag = isinstance(value['default'], str)
            if flag and not value['default'] and 'required' in value: 
                return value['required']
            else:
                return value['default']

        for key, value in self.params_dict.items():
            key_length_map.append(len(key))
            default_length_map.append(len(str(getDefaultColumn(key, value))))
            description_length_map.append(len(value['descripton']))
            key_length = max(key_length, key_length_map[-1])
            default_length = max(default_length, default_length_map[-1])
            description_length = max(description_length, description_length_map[-1])

        content = "| %s %s| %s %s| %s %s|\n" % (
                'JSON Parameter', 
                " " * (key_length - len('JSON Parameter') + 1), 
                'Default', 
                " " * (default_length - len('Default') + 1), 
                'Description', 
                " " * (description_length - len('Description') + 1)
                )
        content += "| %s | %s | %s |\n" % (
                "-" * (key_length + 1), 
                "-" * (default_length + 1), 
                "-" * (description_length + 1)
                )
        count = 0
        for key, value in self.params_dict.items():
            content += "| %s %s| %s %s| %s %s|\n" % (
                    key, 
                    " " * (key_length - key_length_map[count] + 1), 
                    str(getDefaultColumn(key, value)), 
                    " " * (default_length - default_length_map[count] + 1), 
                    value['descripton'], 
                    " " * (description_length - description_length_map[count] + 1)
                    )
            count += 1
        return content 

    def toJson(self):
        """
        @brief convert to json
        """
        data = {}
        for key, value in self.__dict__.items():
            if key != 'params_dict': 
                data[key] = value
        return data

    def fromJson(self, data):
        """
        @brief load form json
        """
        for key, value in data.items(): 
            self.__dict__[key] = value

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
        if self.def_input is not None and os.path.exists(self.def_input): # LEF/DEF 
            return "def"
        else: # Bookshelf
            return "pl"

def get_bin_size(width, height, num_bins=128*128):
    """
    @brief find the power two bin size closest to the canvas ratio.
    """
    num_bins_x = math.sqrt(num_bins * width / height)
    num_bins_x = int(math.pow(2, round(math.log(num_bins_x) / math.log(2))))
    num_bins_x = max(min(num_bins_x, num_bins), 1) # constrain num_bins_x between 1 and num_bins
    num_bins_y = int(num_bins / num_bins_x)
    
    return num_bins_x, num_bins_y

def get_dreamplace_params(
    iteration, 
    target_density,
    learning_rate,
    canvas_width=None,
    canvas_height=None,
    num_bins_x=None,
    num_bins_y=None,
    gpu=False,
    result_dir='results',
    legalize_flag=False,
    stop_overflow=0.1,
    routability_opt_flag=False):
    """"
    @brief return the parameters to config Dreamplace in circuit training.
    """
    params = Params()
    params.circuit_training_mode = True
    
    # set number of bins
    if num_bins_x and num_bins_y:
        params.num_bins_x = num_bins_x
        params.num_bins_y = num_bins_y
    elif canvas_width and canvas_height:
        # extract #bins from canvas info
        params.num_bins_x, params.num_bins_y = get_bin_size(canvas_width,
                                                            canvas_height)
    else:
        num_bins_x = 128
        num_bins_y = 128
    
    params.global_place_stages = [{
        'num_bins_x': params.num_bins_x,
        'num_bins_y': params.num_bins_y,
        'iteration': iteration,
        'learning_rate': learning_rate,
        'wirelength': 'weighted_average',
        'optimizer': 'nesterov',
    }]
    params.legalize_flag = legalize_flag
    params.detailed_place_flag = False
    params.target_density = target_density
    params.density_weight = 8e-5
    params.gpu = gpu
    params.result_dir = result_dir
    params.stop_overflow = stop_overflow
    
    # disable regioning
    params.regioning = False
    
    # routability related flag
    params.routability_opt_flag = routability_opt_flag
    params.adjust_nctugr_area_flag = False
    
    return params
