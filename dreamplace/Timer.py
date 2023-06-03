##
# @file   Placer.py
# @author Enzo
# @date   Apr 2021
# @brief  The timer required in the timing-driven mode
#

import time
import logging
import dreamplace.ops.timing.timing as timing
import dreamplace.ops.timing.timing_cpp as timing_cpp

class Timer(object):
    """
    @brief The timer python class. The timer we are going to use is
     actually OpenTimer written in c++.
    """
    def __init__(self):
        """The initialization of the timer python class.
        In fact we do not prefer using timer directly in python, even in
        the timing-driven mode. Ideally, everything should smoothly goes
        on at the underlying level.
        """
        self.raw_timer = None

    def read(self, params): 
        """
        @brief read using c++ and save the pybind object.
        @param params the parameters specified in json.
        """
        self.raw_timer = timing.TimingIO.read(params)

    def __call__(self, params, placedb):
        """
        @brief top API to read placement files 
        @param params parameters
        @param placedb the placement database.
        """
        tt = time.time()
        self.read(params)
        self.placedb = placedb
        logging.info("reading timer constraints takes %g seconds" % \
            (time.time() - tt))

    def update_timing(self):
        """@brief update timing.
        Note that the parsers only build tasks in cpp-taskflow and will
        not be executed before the update_timing call.
        """        
        return self.raw_timer.update_timing()
    
    def report_timing(self, n):
        """@brief report timing paths and related nets.
        The underlying implementation is completed in the c++ core.
        The return value is in fact a list of net indices. The net
        indices can be very useful when aiming at adjusting the net
        weight of specific nets.
        """
        # The extraction should be called after the timing update every
        # time. The two functions are separated.
        return timing_cpp.report_timing(
            self.raw_timer, n, self.placedb.net_name2id_map)

    # ----------------------    
    # In the following, we define some accessors. Note that they are
    # not properties so call them in a function way.
    # All these accessors directly call the instance variable of raw
    # timer binded from c++ operators inside their implementations.
    def num_primary_inputs(self):
        """@brief aquire the number of primary inputs in the design.
        """
        return self.raw_timer.num_primary_inputs()

    def num_primary_outputs(self):
        """@brief aquire the number of primary outputs in the design.
        """
        return self.raw_timer.num_primary_outputs()

    def num_nets(self):
        """@brief aquire the total number of nets in the design.
        """
        return self.raw_timer.num_nets()

    def num_pins(self):
        """@brief aquire the total number of pins in the design.
        """
        return self.raw_timer.num_pins()

    def num_arcs(self):
        """@brief aquire the total number of arcs in the design.
        """
        return self.raw_timer.num_arcs()

    def num_gates(self):
        """@brief aquire the total number of gates in the design.
        """
        return self.raw_timer.num_gates()

    def num_tests(self):
        """@brief aquire the total number of tests in the design.
        """
        return self.raw_timer.num_tests()

    def num_sccs(self):
        """@brief aquire the total number of sccs in the design.
        """
        return self.raw_timer.num_sccs()

    def num_worst_endpoints(self):
        """@brief aquire the maximum size of end points for EL/RF
        """
        return self.raw_timer.num_worst_endpoints()

    # ----------------------    
    # In the following, we define some dump functions.
    def dump_graph(self, fout=None):
        """@brief dump timing graph
        """
        if fout is None: return self.raw_timer.dump_graph()
        return self.raw_timer.dump_graph_file(fout)

    def dump_taskflow(self, fout=None):
        """@brief dump taskflow graph
        """
        if fout is None: return self.raw_timer.dump_taskflow()
        return self.raw_timer.dump_taskflow_file(fout)

    def dump_netload(self, fout=None):
        """@brief dump netload
        """
        if fout is None: return self.raw_timer.dump_netload()
        return self.raw_timer.dump_netload_file(fout)

    def dump_pin_cap(self, fout=None):
        """@brief dump pin capacitance values
        """
        if fout is None: return self.raw_timer.dump_pin_cap()
        return self.raw_timer.dump_pin_cap_file(fout)

    def dump_at(self, fout=None):
        """@brief dump at
        """
        if fout is None: return self.raw_timer.dump_at()
        return self.raw_timer.dump_at_file(fout)

    def dump_rat(self, fout=None):
        """@brief dump rat
        """
        if fout is None: return self.raw_timer.dump_rat()
        return self.raw_timer.dump_rat_file(fout)

    def dump_slew(self, fout=None):
        """@brief dump slews
        """
        if fout is None: return self.raw_timer.dump_slew()
        return self.raw_timer.dump_slew_file(fout)

    def dump_slack(self, fout=None):
        """@brief dump slacks
        """
        if fout is None: return self.raw_timer.dump_slack()
        return self.raw_timer.dump_slack_file(fout)

    def dump_timer(self, fout=None):
        """@brief dump the current timer
        """
        if fout is None: return self.raw_timer.dump_timer()
        return self.raw_timer.dump_timer_file(fout)

    def dump_spef(self, fout=None):
        """@brief dump slews
        """
        if fout is None: return self.raw_timer.dump_spef()
        return self.raw_timer.dump_spef_file(fout)

    def dump_rctree(self, fout=None):
        """@brief dump slews
        """
        if fout is None: return self.raw_timer.dump_rctree()
        return self.raw_timer.dump_rctree_file(fout)
    
    # ----------------------    
    # In the following, we define some report functions.
    def report_tns(self, split=None, tran=None):
        """@brief report tns value
        """
        if split is None and tran is None:
            return self.raw_timer.report_tns_all()
        elif split is not None and tran is None:
            return self.raw_timer.report_tns_el(split)
        elif split is None and tran is not None:
            return self.raw_timer.report_tns_rl(tran)
        else:
            return self.raw_timer.report_tns_el_rf(split, tran)
    
    def report_wns(self, split=None, tran=None):
        """@brief report wns value
        """
        if split is None and tran is None:
            return self.raw_timer.report_wns_all()
        elif split is not None and tran is None:
            return self.raw_timer.report_wns_el(split)
        elif split is None and tran is not None:
            return self.raw_timer.report_wns_rl(tran)
        else:
            return self.raw_timer.report_wns_el_rf(split, tran)

    def report_tns_elw(self, split=None):
        """@brief report tns value with only the worst between rise
        and fall considered
        """
        if split is None:
            return self.raw_timer.report_tns_elw()
        else:
            return self.raw_timer.report_tns_elw(split)

    # ----------------------
    # In the following, we define some functions related to units.
    def cap_unit(self):
        """
        @brief return the capacitance unit value (farah)
        @return the cap unit value (farah)
        """
        return self.raw_timer.cap_unit()
    
    def res_unit(self):
        """
        @brief return the resistance unit value (ohm)
        @return the res unit value (ohm)
        """
        return self.raw_timer.res_unit()

    def time_unit(self):
        """
        @brief return the current time unit value (s)
        @return the time unit value (s)
        """
        return self.raw_timer.time_unit()
