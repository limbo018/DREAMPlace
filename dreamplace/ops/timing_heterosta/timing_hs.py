import torch
from torch.autograd import Function
from torch import nn
import numpy as np
import dreamplace.ops.timing_heterosta.timing_heterosta_cpp as timing_hs_cpp
import logging
import pdb

def _convert_pin_direction_to_numeric(pin_direct_strings):
    """
    @brief Convert pin direction strings to numeric encoding
    @param pin_direct_strings numpy array of byte strings (e.g., b'INPUT', b'OUTPUT')
    @return numpy array of uint8 values (0=INPUT, 1=OUTPUT, 2=INOUT)
    """
    direction_map = {
        b'INPUT': 0,
        b'OUTPUT': 1, 
        b'INOUT': 2,
        b'OUTPUT_TRISTATE': 1,  # Treat as OUTPUT
        b'UNKNOWN': 2 # Heterosta treats any number other than 0 and 1 as unknown.
    }
    
    # Handle both string and byte string inputs
    result = np.zeros(len(pin_direct_strings), dtype=np.uint8)
    for i, direction in enumerate(pin_direct_strings):
        # Convert string to bytes if needed
        if isinstance(direction, str):
            direction = direction.encode('utf-8')
        result[i] = direction_map.get(direction, 0)  # Default to INPUT if unknown
    
    return result

def _package_dreamplace_mappings(placedb):
    """
    @brief Package DREAMPlace mappings into a dictionary for C++ interface
    @param placedb the placement database containing the mappings
    @return dictionary containing only the necessary mappings as torch tensors
    """
    return {
        # Only include mappings that are actually used by the C++ code
        'pin2net_map': torch.from_numpy(placedb.pin2net_map),
        'pin2node_map': torch.from_numpy(placedb.pin2node_map),
        'pin_direct': torch.from_numpy(_convert_pin_direction_to_numeric(placedb.pin_direct)),
        'num_terminal_NIs': torch.tensor(placedb.num_terminal_NIs, dtype=torch.int32),
    }


class TimingIO(Function):
    """
    @brief The timer IO class for HeteroSTA integration
    HeteroSTA reads some external files like liberty libraries, SDC, etc. 
    The file reading and parsing will be done only once exactly after 
    the initialization of placement database.
    """
    @staticmethod
    def read(params, placedb):
        """
        @brief read design and store in placement database
        @param params the parameters defined in json
        @param placedb the placement database for netlist integration
        """
        # Build argument string for HeteroSTA
        args = "DREAMPLACE"  # First argument should be non-empty
        
        if "early_lib_input" in params.__dict__ and params.early_lib_input:
            args += " --early_lib_input %s" % (params.early_lib_input)
        if "late_lib_input" in params.__dict__ and params.late_lib_input:
            args += " --late_lib_input %s" % (params.late_lib_input)
        if "lib_input" in params.__dict__ and params.lib_input:
            # Use same library for both early and late if only one specified
            args += " --lib_input %s" % (params.lib_input)
        if "sdc_input" in params.__dict__ and params.sdc_input:
            args += " --sdc_input %s" % (params.sdc_input)
        # Note: verilog_input is not used since we reuse netlist from PlaceDB
        
        # Package DREAMPlace mappings to ensure data consistency
        dreamplace_mappings = _package_dreamplace_mappings(placedb)
        
        return timing_hs_cpp.io_forward(
            args.split(' '), 
            placedb.rawdb,
            dreamplace_mappings
        )


class TimingOptFunction(Function):
    @staticmethod
    def forward(ctx, timer, pos, 
                num_pins,
                wire_resistance_per_micron,
                wire_capacitance_per_micron,
                scale_factor, lef_unit, def_unit,
                pin_pos_op,
                slacks_rf=None,
                ignore_net_degree=np.iinfo(np.int32).max, use_cuda=False):
        """
        @brief compute timing analysis using HeteroSTA
        @param timer the HeteroSTA timer object
        @param pos node/cell locations (x array, y array), NOT pin locations
        @param num_pins total number of pins in the design
        @param wire_resistance_per_micron unit-length resistance value
        @param wire_capacitance_per_micron unit-length capacitance value
        @param scale_factor the scaling factor to be applied to the design
        @param lef_unit the unit distance microns defined in the LEF file
        @param def_unit the unit distance microns defined in the DEF file 
        @param pin_pos_op the pin position operator to compute pin locations from cell locations
        @param slacks_rf optional output tensor for pin slacks (rise/fall)
        @param ignore_net_degree the degree threshold
        @param use_cuda whether to use CUDA for computation
        """
        
        # Handle CUDA/CPU mode consistency
        pos_is_cuda = pos.is_cuda
        if use_cuda and not pos_is_cuda:
            logging.info("HeteroSTA: Converting pos to CUDA for GPU timing analysis")
            pos = pos.cuda()
        elif not use_cuda and pos_is_cuda:
            logging.info("HeteroSTA: Converting pos to CPU for CPU timing analysis")
            pos = pos.cpu()
        
        # Calculate pin positions using pin_pos_op
        pin_pos = pin_pos_op(pos)
        
        # Create slack output tensor if requested
        if slacks_rf is None:
            slacks_rf = torch.Tensor()
        
        timing_hs_cpp.forward(
            timer,
            pin_pos,
            num_pins,
            wire_resistance_per_micron,
            wire_capacitance_per_micron,
            scale_factor, lef_unit, def_unit,
            slacks_rf,
            ignore_net_degree, bool(use_cuda))
        
        return torch.zeros(num_pins)

class TimingOpt(nn.Module):
    def __init__(self, timer, net_names, pin_names, flat_netpin,
                 netpin_start, net_name2id_map, pin_name2id_map,
                 pin2node_map, pin_offset_x, pin_offset_y,
                 pin2net_map, net_criticality, net_criticality_deltas,
                 net_weights, net_weight_deltas,
                 wire_resistance_per_micron,
                 wire_capacitance_per_micron,
                 momentum_decay_factor,
                 scale_factor, lef_unit, def_unit,
                 pin_pos_op,
                 ignore_net_degree, use_cuda=False):
        """
        @brief Initialize the feedback module for HeteroSTA timing analysis
        @param timer the HeteroSTA timer object
        @param net_names the name of each net
        @param pin_names the name of each pin
        @param flat_netpin the net2pin map logic (1d flatten array)
        @param netpin_start the start indices in the flat_netpin
        @param net_name2id_map the net name to id map
        @param pin_name2id_map the pin name to id map
        @param pin2node_map the 1d array pin2node map
        @param pin_offset_x pin offset x to its node
        @param pin_offset_y pin offset y to its node
        @param pin2net_map the pin to net mapping array
        @param net_criticality net criticality value
        @param net_criticality_deltas net criticality delta value
        @param net_weights net weights of placedb
        @param net_weight_deltas the increment of net weights
        @param wire_resistance_per_micron unit-length resistance value
        @param wire_capacitance_per_micron unit-length capacitance value
        @param momentum_decay_factor the decay factor in momentum iteration
        @param scale_factor the scaling factor to be applied to the design
        @param lef_unit the unit distance microns defined in the LEF file
        @param def_unit the unit distance microns defined in the DEF file 
        @param pin_pos_op the pin position operator to compute pin locations from cell locations
        @param ignore_net_degree the degree threshold
        @param use_cuda whether to use CUDA for computation
        """
        super(TimingOpt, self).__init__()
        self.timer = timer
        # Only keep parameters actually needed for timing analysis
        self.flat_netpin = flat_netpin  # numpy array - will be converted when needed
        self.netpin_start = netpin_start  # numpy array - will be converted when needed
        self.net_name2id_map = net_name2id_map
        self.pin_name2id_map = pin_name2id_map
        self.pin2net_map = pin2net_map  # Store the provided pin2net_map (can be numpy or tensor)

        # Store tensor references directly (already on correct device - CPU or GPU)
        self.net_criticality = net_criticality  # torch.Tensor
        self.net_criticality_deltas = net_criticality_deltas  # torch.Tensor
        self.net_weights = net_weights  # torch.Tensor (shared with data_collections)
        self.net_weight_deltas = net_weight_deltas  # torch.Tensor
        self.wire_resistance_per_micron = wire_resistance_per_micron
        self.wire_capacitance_per_micron = wire_capacitance_per_micron
        self.momentum_decay_factor = momentum_decay_factor
        
        # The scale factor is important, together with the lef/def unit.
        # Since we require the actual wire-length evaluation (microns) to
        # enable timing analysis, the parameters should be passed into
        # the cpp core functions.
        self.scale_factor = scale_factor
        self.lef_unit = lef_unit
        self.def_unit = def_unit
        self.ignore_net_degree = ignore_net_degree
        self.use_cuda = use_cuda
        
        # Calculate pin and net counts directly from data
        self.num_pins = len(pin_names)
        self.num_nets = len(net_names)
        
        self.degree_map = self.netpin_start[1:] - self.netpin_start[:-1]
        
        
        # Store the pin_pos_op passed from outside
        self.pin_pos_op = pin_pos_op

    def forward(self, pos):
        """
        @brief call HeteroSTA timing forward function
        @param pos node/cell coordinates (x and y arrays), pin coordinates will be calculated internally
        """
        
        result = TimingOptFunction.apply(
            self.timer.raw_timer,  # Pass the raw C++ timer object
            pos,  # The coordinates
            self.num_pins,  # Pass the pin count directly
            self.wire_resistance_per_micron,
            self.wire_capacitance_per_micron,
            self.scale_factor, self.lef_unit, self.def_unit,
            self.pin_pos_op,  # Pass the pin_pos_op
            None,  # slacks_rf (optional output tensor)
            self.ignore_net_degree, self.use_cuda)
        
        return result
    

    def update_net_weights(self, max_net_weight=np.inf, n=1):
        """
        @brief update net weights of placedb using HeteroSTA simplified algorithm
        @param max_net_weight the maximum net weight in timing opt
        @param n the maximum number of paths to be reported
        """
        try:
            # Convert flat_netpin and netpin_start to tensors
            flat_netpin_t = torch.from_numpy(self.flat_netpin)
            netpin_start_t = torch.from_numpy(self.netpin_start)

            # Handle pin2net_map (can be numpy or tensor)
            if hasattr(self.pin2net_map, 'cpu'):
                pin2net_map_t = self.pin2net_map
            else:
                pin2net_map_t = torch.from_numpy(np.array(self.pin2net_map, dtype=np.int32))

            # Convert degree_map to tensor
            degree_map_t = torch.from_numpy(self.degree_map)

            # Move to CUDA if needed
            if self.use_cuda:
                flat_netpin_t = flat_netpin_t.cuda()
                netpin_start_t = netpin_start_t.cuda()
                if not pin2net_map_t.is_cuda:
                    pin2net_map_t = pin2net_map_t.cuda()
                degree_map_t = degree_map_t.cuda()

            # Call C++ update_net_weights with num_nets and num_pins instead of maps
            # Note: self.net_criticality, self.net_weights are already torch.Tensors on correct device
            # They will be modified in-place by C++
            timing_hs_cpp.update_net_weights(
                self.timer.raw_timer, n,
                self.num_nets, self.num_pins,  # Pass counts instead of maps
                flat_netpin_t,
                netpin_start_t,
                pin2net_map_t,
                self.net_criticality,  # GPU tensor - modified in-place
                self.net_criticality_deltas,  # GPU tensor
                self.net_weights,  # GPU tensor - modified in-place (shared with data_collections)
                self.net_weight_deltas,  # GPU tensor
                degree_map_t,
                max_net_weight, self.momentum_decay_factor,
                self.ignore_net_degree, bool(self.use_cuda))

            # No need to copy back - tensors were modified in-place!

        except Exception as e:
            logging.error(f"HeteroSTA net weight update failed: {e}")
            raise  # Re-raise the exception so caller knows it failed
    
    def write_spef(self,file_path):
        return self.timer.raw_timer.write_spef(file_path)
    
    
    def report_wns_tns(self):
        """
        @brief report WNS and TNS in the design
        """
        return timing_hs_cpp.report_wns_tns(
            self.timer.raw_timer, bool(self.use_cuda))
    
    def report_wns(self):
        """
        @brief report WNS in the design
        """
        return timing_hs_cpp.report_wns(
            self.timer.raw_timer, bool(self.use_cuda))
    
    def report_tns(self):
        """
        @brief report TNS in the design
        """
        return timing_hs_cpp.report_tns(
            self.timer.raw_timer, bool(self.use_cuda))

    def update_timing(self):
        """
        This is a no-op since timing is updated in forward()
        """
        pass

    def time_unit(self):
        """
        @brief report time unit in the design
        """
        return 1e-12  # HeteroSTA uses picosecond as unit



    def dump_paths_setup_to_file(self, num_paths, nworst, file_path, use_cuda=False):
        """
        @brief dump timing paths to file using HeteroSTA
        @param num_paths number of paths to dump
        @param nworst number of worst paths per endpoint
        @param file_path output file path
        @param use_cuda whether to use CUDA
        """
        return self.timer.raw_timer.dump_paths_setup_to_file(num_paths, nworst, file_path, use_cuda)
    
import torch
from torch.autograd import Function
from torch import nn
import numpy as np
import dreamplace.ops.timing_heterosta.timing_heterosta_cpp as timing_hs_cpp
import logging
import pdb

def _convert_pin_direction_to_numeric(pin_direct_strings):
    """
    @brief Convert pin direction strings to numeric encoding
    @param pin_direct_strings numpy array of byte strings (e.g., b'INPUT', b'OUTPUT')
    @return numpy array of uint8 values (0=INPUT, 1=OUTPUT, 2=INOUT)
    """
    direction_map = {
        b'INPUT': 0,
        b'OUTPUT': 1, 
        b'INOUT': 2,
        b'OUTPUT_TRISTATE': 1,  # Treat as OUTPUT
        b'UNKNOWN': 2 # Heterosta treats any number other than 0 and 1 as unknown.
    }
    
    # Handle both string and byte string inputs
    result = np.zeros(len(pin_direct_strings), dtype=np.uint8)
    for i, direction in enumerate(pin_direct_strings):
        # Convert string to bytes if needed
        if isinstance(direction, str):
            direction = direction.encode('utf-8')
        result[i] = direction_map.get(direction, 0)  # Default to INPUT if unknown
    
    return result

def _package_dreamplace_mappings(placedb):
    """
    @brief Package DREAMPlace mappings into a dictionary for C++ interface
    @param placedb the placement database containing the mappings
    @return dictionary containing only the necessary mappings as torch tensors
    """
    return {
        # Only include mappings that are actually used by the C++ code
        'pin2net_map': torch.from_numpy(placedb.pin2net_map),
        'pin2node_map': torch.from_numpy(placedb.pin2node_map),
        'pin_direct': torch.from_numpy(_convert_pin_direction_to_numeric(placedb.pin_direct)),
        'num_terminal_NIs': torch.tensor(placedb.num_terminal_NIs, dtype=torch.int32),
    }


class TimingIO(Function):
    """
    @brief The timer IO class for HeteroSTA integration
    HeteroSTA reads some external files like liberty libraries, SDC, etc. 
    The file reading and parsing will be done only once exactly after 
    the initialization of placement database.
    """
    @staticmethod
    def read(params, placedb):
        """
        @brief read design and store in placement database
        @param params the parameters defined in json
        @param placedb the placement database for netlist integration
        """
        # Build argument string for HeteroSTA
        args = "DREAMPLACE"  # First argument should be non-empty
        
        if "early_lib_input" in params.__dict__ and params.early_lib_input:
            args += " --early_lib_input %s" % (params.early_lib_input)
        if "late_lib_input" in params.__dict__ and params.late_lib_input:
            args += " --late_lib_input %s" % (params.late_lib_input)
        if "lib_input" in params.__dict__ and params.lib_input:
            # Use same library for both early and late if only one specified
            args += " --lib_input %s" % (params.lib_input)
        if "sdc_input" in params.__dict__ and params.sdc_input:
            args += " --sdc_input %s" % (params.sdc_input)
        # Note: verilog_input is not used since we reuse netlist from PlaceDB
        
        # Package DREAMPlace mappings to ensure data consistency
        dreamplace_mappings = _package_dreamplace_mappings(placedb)
        
        return timing_hs_cpp.io_forward(
            args.split(' '), 
            placedb.rawdb,
            dreamplace_mappings
        )


class TimingOptFunction(Function):
    @staticmethod
    def forward(ctx, timer, pos, 
                num_pins,
                wire_resistance_per_micron,
                wire_capacitance_per_micron,
                scale_factor, lef_unit, def_unit,
                pin_pos_op,
                slacks_rf=None,
                ignore_net_degree=np.iinfo(np.int32).max, use_cuda=False):
        """
        @brief compute timing analysis using HeteroSTA
        @param timer the HeteroSTA timer object
        @param pos node/cell locations (x array, y array), NOT pin locations
        @param num_pins total number of pins in the design
        @param wire_resistance_per_micron unit-length resistance value
        @param wire_capacitance_per_micron unit-length capacitance value
        @param scale_factor the scaling factor to be applied to the design
        @param lef_unit the unit distance microns defined in the LEF file
        @param def_unit the unit distance microns defined in the DEF file 
        @param pin_pos_op the pin position operator to compute pin locations from cell locations
        @param slacks_rf optional output tensor for pin slacks (rise/fall)
        @param ignore_net_degree the degree threshold
        @param use_cuda whether to use CUDA for computation
        """
        
        # Handle CUDA/CPU mode consistency
        pos_is_cuda = pos.is_cuda
        if use_cuda and not pos_is_cuda:
            logging.info("HeteroSTA: Converting pos to CUDA for GPU timing analysis")
            pos = pos.cuda()
        elif not use_cuda and pos_is_cuda:
            logging.info("HeteroSTA: Converting pos to CPU for CPU timing analysis")
            pos = pos.cpu()
        
        # Calculate pin positions using pin_pos_op
        pin_pos = pin_pos_op(pos)
        
        # Create slack output tensor if requested
        if slacks_rf is None:
            slacks_rf = torch.Tensor()
        
        timing_hs_cpp.forward(
            timer,
            pin_pos,
            num_pins,
            wire_resistance_per_micron,
            wire_capacitance_per_micron,
            scale_factor, lef_unit, def_unit,
            slacks_rf,
            ignore_net_degree, bool(use_cuda))
        
        return torch.zeros(num_pins)

class TimingOpt(nn.Module):
    def __init__(self, timer, net_names, pin_names, flat_netpin,
                 netpin_start, net_name2id_map, pin_name2id_map,
                 pin2node_map, pin_offset_x, pin_offset_y,
                 pin2net_map, net_criticality, net_criticality_deltas,
                 net_weights, net_weight_deltas,
                 wire_resistance_per_micron,
                 wire_capacitance_per_micron,
                 momentum_decay_factor,
                 scale_factor, lef_unit, def_unit,
                 pin_pos_op,
                 ignore_net_degree, use_cuda=False):
        """
        @brief Initialize the feedback module for HeteroSTA timing analysis
        @param timer the HeteroSTA timer object
        @param net_names the name of each net
        @param pin_names the name of each pin
        @param flat_netpin the net2pin map logic (1d flatten array)
        @param netpin_start the start indices in the flat_netpin
        @param net_name2id_map the net name to id map
        @param pin_name2id_map the pin name to id map
        @param pin2node_map the 1d array pin2node map
        @param pin_offset_x pin offset x to its node
        @param pin_offset_y pin offset y to its node
        @param pin2net_map the pin to net mapping array
        @param net_criticality net criticality value
        @param net_criticality_deltas net criticality delta value
        @param net_weights net weights of placedb
        @param net_weight_deltas the increment of net weights
        @param wire_resistance_per_micron unit-length resistance value
        @param wire_capacitance_per_micron unit-length capacitance value
        @param momentum_decay_factor the decay factor in momentum iteration
        @param scale_factor the scaling factor to be applied to the design
        @param lef_unit the unit distance microns defined in the LEF file
        @param def_unit the unit distance microns defined in the DEF file 
        @param pin_pos_op the pin position operator to compute pin locations from cell locations
        @param ignore_net_degree the degree threshold
        @param use_cuda whether to use CUDA for computation
        """
        super(TimingOpt, self).__init__()
        self.timer = timer
        # Only keep parameters actually needed for timing analysis
        self.flat_netpin = flat_netpin  # numpy array - will be converted when needed
        self.netpin_start = netpin_start  # numpy array - will be converted when needed
        self.net_name2id_map = net_name2id_map
        self.pin_name2id_map = pin_name2id_map
        self.pin2net_map = pin2net_map  # Store the provided pin2net_map (can be numpy or tensor)

        # Store tensor references directly (already on correct device - CPU or GPU)
        self.net_criticality = net_criticality  # torch.Tensor
        self.net_criticality_deltas = net_criticality_deltas  # torch.Tensor
        self.net_weights = net_weights  # torch.Tensor (shared with data_collections)
        self.net_weight_deltas = net_weight_deltas  # torch.Tensor
        self.wire_resistance_per_micron = wire_resistance_per_micron
        self.wire_capacitance_per_micron = wire_capacitance_per_micron
        self.momentum_decay_factor = momentum_decay_factor
        
        # The scale factor is important, together with the lef/def unit.
        # Since we require the actual wire-length evaluation (microns) to
        # enable timing analysis, the parameters should be passed into
        # the cpp core functions.
        self.scale_factor = scale_factor
        self.lef_unit = lef_unit
        self.def_unit = def_unit
        self.ignore_net_degree = ignore_net_degree
        self.use_cuda = use_cuda
        
        # Calculate pin and net counts directly from data
        self.num_pins = len(pin_names)
        self.num_nets = len(net_names)
        
        self.degree_map = self.netpin_start[1:] - self.netpin_start[:-1]
        
        
        # Store the pin_pos_op passed from outside
        self.pin_pos_op = pin_pos_op

    def forward(self, pos):
        """
        @brief call HeteroSTA timing forward function
        @param pos node/cell coordinates (x and y arrays), pin coordinates will be calculated internally
        """
        
        result = TimingOptFunction.apply(
            self.timer.raw_timer,  # Pass the raw C++ timer object
            pos,  # The coordinates
            self.num_pins,  # Pass the pin count directly
            self.wire_resistance_per_micron,
            self.wire_capacitance_per_micron,
            self.scale_factor, self.lef_unit, self.def_unit,
            self.pin_pos_op,  # Pass the pin_pos_op
            None,  # slacks_rf (optional output tensor)
            self.ignore_net_degree, self.use_cuda)
        
        return result
    

    def update_net_weights(self, max_net_weight=np.inf, n=1):
        """
        @brief update net weights of placedb using HeteroSTA simplified algorithm
        @param max_net_weight the maximum net weight in timing opt
        @param n the maximum number of paths to be reported
        """
        try:
            # Convert flat_netpin and netpin_start to tensors
            flat_netpin_t = torch.from_numpy(self.flat_netpin)
            netpin_start_t = torch.from_numpy(self.netpin_start)

            # Handle pin2net_map (can be numpy or tensor)
            if hasattr(self.pin2net_map, 'cpu'):
                pin2net_map_t = self.pin2net_map
            else:
                pin2net_map_t = torch.from_numpy(np.array(self.pin2net_map, dtype=np.int32))

            # Convert degree_map to tensor
            degree_map_t = torch.from_numpy(self.degree_map)

            # Move to CUDA if needed
            if self.use_cuda:
                flat_netpin_t = flat_netpin_t.cuda()
                netpin_start_t = netpin_start_t.cuda()
                if not pin2net_map_t.is_cuda:
                    pin2net_map_t = pin2net_map_t.cuda()
                degree_map_t = degree_map_t.cuda()

            # Call C++ update_net_weights with num_nets and num_pins instead of maps
            # Note: self.net_criticality, self.net_weights are already torch.Tensors on correct device
            # They will be modified in-place by C++
            timing_hs_cpp.update_net_weights(
                self.timer.raw_timer, n,
                self.num_nets, self.num_pins,  # Pass counts instead of maps
                flat_netpin_t,
                netpin_start_t,
                pin2net_map_t,
                self.net_criticality,  # GPU tensor - modified in-place
                self.net_criticality_deltas,  # GPU tensor
                self.net_weights,  # GPU tensor - modified in-place (shared with data_collections)
                self.net_weight_deltas,  # GPU tensor
                degree_map_t,
                max_net_weight, self.momentum_decay_factor,
                self.ignore_net_degree, bool(self.use_cuda))

            # No need to copy back - tensors were modified in-place!

        except Exception as e:
            logging.error(f"HeteroSTA net weight update failed: {e}")
            raise  # Re-raise the exception so caller knows it failed
    
    def write_spef(self,file_path):
        return self.timer.raw_timer.write_spef(file_path)
    
    
    def report_wns_tns(self):
        """
        @brief report WNS and TNS in the design
        """
        return timing_hs_cpp.report_wns_tns(
            self.timer.raw_timer, bool(self.use_cuda))
    
    def report_wns(self):
        """
        @brief report WNS in the design
        """
        return timing_hs_cpp.report_wns(
            self.timer.raw_timer, bool(self.use_cuda))
    
    def report_tns(self):
        """
        @brief report TNS in the design
        """
        return timing_hs_cpp.report_tns(
            self.timer.raw_timer, bool(self.use_cuda))

    def update_timing(self):
        """
        This is a no-op since timing is updated in forward()
        """
        pass

    def time_unit(self):
        """
        @brief report time unit in the design
        """
        return 1e-12  # HeteroSTA uses picosecond as unit



    def dump_paths_setup_to_file(self, num_paths, nworst, file_path, use_cuda=False):
        """
        @brief dump timing paths to file using HeteroSTA
        @param num_paths number of paths to dump
        @param nworst number of worst paths per endpoint
        @param file_path output file path
        @param use_cuda whether to use CUDA
        """
        return self.timer.raw_timer.dump_paths_setup_to_file(num_paths, nworst, file_path, use_cuda)
    
