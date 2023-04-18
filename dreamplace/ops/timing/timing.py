import torch
from torch.autograd import Function
from torch import nn
import numpy as np
import dreamplace.ops.timing.timing_cpp as timing_cpp
import logging
import pdb

class TimingIO(Function):
    """
    @brief The timer we use will read some external files like celllibs,
     sdc, etc. Obviously we do not expect the timer to read them every time
     it is called, so the file reading and parsing will be done only once
     exactly after the initialization of placement database.
    """
    @staticmethod
    def read(params):
        """
        @brief read design and store in placement database
        @param params the parameters defined in json.
        """
        # This string is required! When fed into boost program option
        # parsers, the first argument should be non-empty.
        args = "DREAMPLACE"
        if "early_lib_input" in params.__dict__ and params.early_lib_input:
            args += " --early_lib_input %s" % (params.early_lib_input)
        if "late_lib_input" in params.__dict__ and params.late_lib_input:
            args += " --late_lib_input %s" % (params.late_lib_input)
        if "lib_input" in params.__dict__ and params.lib_input:
            args += " --lib_input %s" % (params.lib_input)
        if "sdc_input" in params.__dict__ and params.sdc_input:
            args += " --sdc_input %s" % (params.sdc_input)
        if "verilog_input" in params.__dict__ and params.verilog_input:
            args += " --verilog_input %s" % (params.verilog_input)

        return timing_cpp.io_forward(args.split(' '))

class TimingOptFunction(Function):
    @staticmethod
    def forward(ctx, timer, pos, net_names, pin_names, flat_netpin,
                netpin_start, pin2node_map, pin_offset_x, pin_offset_y,
                wire_resistance_per_micron,
                wire_capacitance_per_micron,
                scale_factor, lef_unit, def_unit,
                ignore_net_degree):
        """
        @brief compute Elmore delay using Flute.
        @param timer the timer used when timing_cpp-driven mode is opened
        @param pos pin location (x array, y array), not cell location 
        @param net_names the name of each net
        @param pin_names the name of each pin
        @param flat_netpin flat netpin map, length of #pins 
        @param netpin_start starting index in netpin map for each net,
         length of #nets + 1, the last entry is #pins
        @param pin2node the 1d array pin2node map.
        @param pin_offset_x pin offset x to its node.
        @param pin_offset_y pin offset y to its node.
        @param wire_resistance_per_micron unit-length resistance value
        @param wire_capacitance_per_micron unit-length capacitance value
        @param scale_factor the scaling factor to be applied to the design
        @param lef_unit the unit distance microns defined in the LEF file
        @param def_unit the unit distance microns defined in the DEF file 
        @param ignore_net_degree the degree threshold
        """
        num_pins = netpin_start[-1].item()
        # Construct some new position arrays.
        # They will be updated after the cpp core function call.
        if pos.is_cuda:
            assert 0, "CUDA version is NOT IMPLEMENTED!"
        else:
            # Call the cpp version function timing_forward defined in
            # the source file timing_cpp.cpp
            timing_cpp.forward(
                timer,
                pos.view(pos.numel()),
                net_names, pin_names,
                torch.from_numpy(flat_netpin),
                torch.from_numpy(netpin_start),
                torch.from_numpy(pin2node_map),
                torch.from_numpy(pin_offset_x),
                torch.from_numpy(pin_offset_y),
                wire_resistance_per_micron,
                wire_capacitance_per_micron,
                scale_factor, lef_unit, def_unit,
                ignore_net_degree)
        return torch.zeros(num_pins);

class TimingOpt(nn.Module):
    def __init__(self, timer, net_names, pin_names, flat_netpin,
                 netpin_start, net_name2id_map, pin_name2id_map,
                 pin2node_map, pin_offset_x, pin_offset_y,
                 net_criticality, net_criticality_deltas,
                 net_weights, net_weight_deltas,
                 wire_resistance_per_micron,
                 wire_capacitance_per_micron,
                 net_weighting_scheme,
                 momentum_decay_factor,
                 scale_factor, lef_unit, def_unit,
                 ignore_net_degree):
        """
        @brief Initialize the feedback module that inherits from the
         base neural network module in torch framework.
        @param timer the OpenTimer python object
        @param net_names the name of each net
        @param pin_names the name of each pin
        @param flat_netpin the net2pin map logic (1d flatten array)
        @param netpin_start the start indices in the @flat_netpin
        @param net_name2id_map the net name to id map
        @param pin_name2id_map the pin name to id map
        @param pin2node the 1d array pin2node map.
        @param pin_offset_x pin offset x to its node.
        @param pin_offset_y pin offset y to its node.
        @param net_criticality net criticality value.
        @param net_criticality_deltas net criticality delta value.
        @param net_weights net weights of placedb.
        @param net_weight_deltas the increment of net weights.
        @param wire_resistance_per_micron unit-length resistance value
        @param wire_capacitance_per_micron unit-length capacitance value
        @param net_weighting_scheme the net-weighting scheme
        @param momentum_decay_factor the decay factor in momentum iteration
        @param scale_factor the scaling factor to be applied to the design
        @param lef_unit the unit distance microns defined in the LEF file
        @param def_unit the unit distance microns defined in the DEF file 
        @param ignore_net_degree the degree threshold
        """
        super(TimingOpt, self).__init__()
        self.timer = timer
        self.net_names = net_names
        self.pin_names = pin_names
        self.flat_netpin = flat_netpin
        self.netpin_start = netpin_start
        self.net_name2id_map = net_name2id_map
        self.pin_name2id_map = pin_name2id_map
        self.pin2node_map = pin2node_map
        self.pin_offset_x = pin_offset_x
        self.pin_offset_y = pin_offset_y
        self.net_criticality = net_criticality
        self.net_criticality_deltas = net_criticality_deltas
        self.net_weights = net_weights
        self.net_weight_deltas = net_weight_deltas
        self.wire_resistance_per_micron = wire_resistance_per_micron
        self.wire_capacitance_per_micron = wire_capacitance_per_micron
        self.net_weighting_scheme = net_weighting_scheme
        self.momentum_decay_factor = momentum_decay_factor

        # The scale factor is important, together with the lef/def unit.
        # Since we require the actual wire-length evaluation (microns) to
        # enable the timing_cpp analysis, the parameters should be passed into
        # the cpp core functions.
        self.scale_factor = scale_factor
        self.lef_unit = lef_unit
        self.def_unit = def_unit
        self.ignore_net_degree = ignore_net_degree
        self.degree_map = self.netpin_start[1:] - self.netpin_start[:-1]

    def forward(self, pos):
        """
        @brief call timing_forward function defined in the c++ operator.
        @pos the tensor determining a sketch placement.
        """
        return TimingOptFunction.apply(
            self.timer.raw_timer, # Pass the raw object!!
            pos, # The coordinates
            self.net_names, self.pin_names,
            self.flat_netpin,
            self.netpin_start,
            self.pin2node_map, self.pin_offset_x, self.pin_offset_y,
            self.wire_resistance_per_micron,
            self.wire_capacitance_per_micron,
            self.scale_factor, self.lef_unit, self.def_unit,
            self.ignore_net_degree)
    
    def report_timing(self, n=1):
        """
        @brief call the underlying cpp core of report_timing function.
        @param n the maximum number of paths to be reported.
        """
        return timing_cpp.report_timing(
            self.timer.raw_timer, n,
            self.net_name2id_map)
    
    def update_net_weights(self, max_net_weight=np.inf, n=1):
        """
        @brief update net weights of placedb
        @param max_net_weight the maximum net weight in timing opt
        @param n the maximum number of paths to be reported.
        """
        if self.net_weighting_scheme == "adams": scm = 0
        elif self.net_weighting_scheme == "lilith": scm = 1
        else:
            logging.warning("unsupported net-weighting scheme %r" % \
                (self.net_weighting_scheme))
            scm = -1 # Unsupported scheme.
        return timing_cpp.update_net_weights(
            self.timer.raw_timer, n,
            self.net_name2id_map,
            torch.from_numpy(self.net_criticality),
            torch.from_numpy(self.net_criticality_deltas),
            torch.from_numpy(self.net_weights),
            torch.from_numpy(self.net_weight_deltas),
            torch.from_numpy(self.degree_map),
            scm, # Pass integers instead of strings.
            self.momentum_decay_factor,
            max_net_weight, # -1 indicates infinity upper bound
            self.ignore_net_degree
            )

    def evaluate_slack(self):
        """
        @brief evaluate the slack array of pins
        """
        num_pins = self.pin_names.shape[0]
        slack = np.zeros(num_pins, dtype=np.float32)
        timing_cpp.evaluate_slack(
            self.timer.raw_timer,
            self.pin_name2id_map,
            torch.from_numpy(slack))
        return slack
