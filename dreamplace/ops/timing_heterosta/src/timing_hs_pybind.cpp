#include "timing_hs_cpp.h"
#include "timing_hs_io_cpp.h"
#include "place_io/src/PlaceDB.h"   
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <algorithm>

DREAMPLACE_BEGIN_NAMESPACE

using namespace _timing_heterosta_impl;

// Wrapper class to hold the unique_ptr for Python binding
class TimerWrapper {
public:
    TimingHeterostaIO::STAHoldingsPtr timer_;
    
    TimerWrapper(TimingHeterostaIO::STAHoldingsPtr timer) : timer_(std::move(timer)) {}
    
    STAHoldings* get_raw_timer() const { return timer_.get(); }
    STAHoldings& get_raw_timer_ref() const { return *timer_; }
};

///
/// @brief Parse arguments and create HeteroSTA timer with PlaceDB integration and DREAMPlace mappings
/// @param args the arguments including files required to construct a timer
/// @param placedb the placement database for netlist reuse
/// @param dreamplace_mappings packaged DREAMPlace mappings to ensure data consistency
/// @return wrapped timer object
///
std::shared_ptr<TimerWrapper> _timing_io_forward_heterosta(const pybind11::list& args, PlaceDB& placedb, 
                                                            const pybind11::dict& dreamplace_mappings) {
    // Convert Python list to C arguments
    int argc = pybind11::len(args);
    char** argv = new char*[argc];
    
    for (int i = 0; i < argc; ++i) {
        std::string token = pybind11::str(args[i]);
        argv[i] = new char[token.size() + 1];
        std::copy(token.begin(), token.end(), argv[i]);
        argv[i][token.size()] = '\0';
    }
    
    // Initialize HeteroSTA
    auto timer_ptr = TimingHeterostaIO::initialize_heterosta();
    if (!timer_ptr) {
        dreamplacePrint(kERROR, "CRITICAL ERROR: HeteroSTA initialization failed!\n");
        // Clean up arguments
        for (int i = 0; i < argc; ++i) {
            delete[] argv[i];
        }
        delete[] argv;
        return nullptr;
    }
    
    // Use the new unified setupTiming function with DREAMPlace mappings
    if (!TimingHeterostaIO::setupTiming(*timer_ptr, placedb, argc, argv, dreamplace_mappings)) {
        dreamplacePrint(kERROR, "CRITICAL ERROR: Complete timing setup with DREAMPlace mappings failed!\n");
        // Clean up arguments
        for (int i = 0; i < argc; ++i) {
            delete[] argv[i];
        }
        delete[] argv;
        return nullptr;
    }
    
    // Clean up arguments
    for (int i = 0; i < argc; ++i) {
        delete[] argv[i];
    }
    delete[] argv;
    
    auto wrapper = std::make_shared<TimerWrapper>(std::move(timer_ptr));
    return wrapper;
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Bind TimerWrapper class instead of STAHoldings directly
    pybind11::class_<DREAMPLACE_NAMESPACE::TimerWrapper, std::shared_ptr<DREAMPLACE_NAMESPACE::TimerWrapper>>(m, "Timer")
        // Core HeteroSTA functions 
        .def("reset", [](DREAMPLACE_NAMESPACE::TimerWrapper& wrapper) {
            heterosta_reset(wrapper.get_raw_timer());
        }, "Keep liberty and clear all other data")
        
        .def("update_delay", [](DREAMPLACE_NAMESPACE::TimerWrapper& wrapper, bool use_cuda) {
            heterosta_update_delay(wrapper.get_raw_timer(), use_cuda);
        }, "Update arcs delay in parallel", pybind11::arg("use_cuda") = false)
        
        .def("update_arrivals", [](DREAMPLACE_NAMESPACE::TimerWrapper& wrapper, bool use_cuda) {
            heterosta_update_arrivals(wrapper.get_raw_timer(), use_cuda);
        }, "Update pin arrival times and required arrival times", pybind11::arg("use_cuda") = false)
        
        
        // Add endpoint-related bindings
        .def("get_is_endpoint", [](DREAMPLACE_NAMESPACE::TimerWrapper& wrapper, pybind11::array_t<bool> is_endpoint) {
            auto buf = is_endpoint.request();
            if (buf.ndim != 1) {
                throw std::runtime_error("is_endpoint array must be 1-dimensional");
            }
            heterosta_get_is_endpoint(wrapper.get_raw_timer(), static_cast<bool*>(buf.ptr));
        }, "Get endpoint status for all pins", pybind11::arg("is_endpoint"))
        
        .def("count_endpoints", [](DREAMPLACE_NAMESPACE::TimerWrapper& wrapper, size_t num_pins) -> size_t {
            std::vector<uint8_t> is_endpoint(num_pins, 0);
            heterosta_get_is_endpoint(wrapper.get_raw_timer(), reinterpret_cast<bool*>(is_endpoint.data()));
            return std::count(is_endpoint.begin(), is_endpoint.end(), 1);
        }, "Count total number of endpoints", pybind11::arg("num_pins"))
        
        
        .def("dump_paths_max_to_file", [](DREAMPLACE_NAMESPACE::TimerWrapper& wrapper, uintptr_t num_paths, 
                                            uintptr_t nworst, const std::string& file_path,
                                            bool use_cuda) {
            heterosta_dump_paths_max_to_file(wrapper.get_raw_timer(), num_paths, nworst, file_path.c_str(), use_cuda);
        }, "Dump timing report to file", 
           pybind11::arg("num_paths"), pybind11::arg("nworst"), 
           pybind11::arg("file_path"), pybind11::arg("use_cuda") = false)
        
        
        // Add method to access raw timer pointer for compatibility 
        .def("get_raw_timer_ptr", [](DREAMPLACE_NAMESPACE::TimerWrapper& wrapper) -> uintptr_t {
            return reinterpret_cast<uintptr_t>(wrapper.get_raw_timer());
        }, "Get raw STAHoldings pointer as integer for C++ functions")
        
        
        // SPEF file operations
        .def("write_spef", [](DREAMPLACE_NAMESPACE::TimerWrapper& wrapper, const std::string& spef_path) {
            return heterosta_write_spef(wrapper.get_raw_timer(), spef_path.c_str());
        }, "Write SPEF parasitics file", pybind11::arg("spef_path"))
        
        .def("read_spef", [](DREAMPLACE_NAMESPACE::TimerWrapper& wrapper, const std::string& spef_path) {
            return heterosta_read_spef(wrapper.get_raw_timer(), spef_path.c_str());
        }, "Read SPEF parasitics file", pybind11::arg("spef_path"));

    // Static function bindings - core functionality with TimerWrapper approach
    m.def("forward", [](DREAMPLACE_NAMESPACE::TimerWrapper& wrapper, torch::Tensor pos,
                       size_t num_pins,
                       double wire_resistance_per_micron, double wire_capacitance_per_micron,
                       double scale_factor, int lef_unit, int def_unit,
                       torch::Tensor slacks_rf, int ignore_net_degree, bool use_cuda) {
        DREAMPLACE_NAMESPACE::TimingHeterostaCpp::forward(
            wrapper.get_raw_timer_ref(), pos,
            num_pins,
            wire_resistance_per_micron, wire_capacitance_per_micron,
            scale_factor, lef_unit, def_unit, slacks_rf, ignore_net_degree, use_cuda);
    }, "HeteroSTA timing forward analysis",
       pybind11::arg("wrapper"), pybind11::arg("pos"), pybind11::arg("num_pins"),
       pybind11::arg("wire_resistance_per_micron"), pybind11::arg("wire_capacitance_per_micron"),
       pybind11::arg("scale_factor"), pybind11::arg("lef_unit"), pybind11::arg("def_unit"),
       pybind11::arg("slacks_rf") = torch::Tensor(),
       pybind11::arg("ignore_net_degree") = std::numeric_limits<int>::max(),
       pybind11::arg("use_cuda") = false);
    
    m.def("update_net_weights", [](DREAMPLACE_NAMESPACE::TimerWrapper& wrapper, int n,
                                    int num_nets, int num_pins,
                                    torch::Tensor flat_netpin, torch::Tensor netpin_start,
                                    torch::Tensor pin2net_map, torch::Tensor net_criticality,
                                    torch::Tensor net_criticality_deltas, torch::Tensor net_weights,
                                    torch::Tensor net_weight_deltas, torch::Tensor degree_map,
                                    double max_net_weight, double momentum_decay_factor,
                                    int ignore_net_degree, bool use_cuda) {
        DREAMPLACE_NAMESPACE::TimingHeterostaCpp::update_net_weights(
            wrapper.get_raw_timer_ref(), n, num_nets, num_pins,
            flat_netpin, netpin_start, pin2net_map, net_criticality, net_criticality_deltas,
            net_weights, net_weight_deltas, degree_map, max_net_weight, momentum_decay_factor,
            ignore_net_degree, use_cuda);
    }, "Update net weights using HeteroSTA",
       pybind11::arg("wrapper"), pybind11::arg("n"),
       pybind11::arg("num_nets"), pybind11::arg("num_pins"),
       pybind11::arg("flat_netpin"), pybind11::arg("netpin_start"), pybind11::arg("pin2net_map"),
       pybind11::arg("net_criticality"), pybind11::arg("net_criticality_deltas"),
       pybind11::arg("net_weights"), pybind11::arg("net_weight_deltas"), pybind11::arg("degree_map"),
       pybind11::arg("max_net_weight"), pybind11::arg("momentum_decay_factor"),
       pybind11::arg("ignore_net_degree"), pybind11::arg("use_cuda"));
    
    
    // HeteroSTA IO function with PlaceDB integration and DREAMPlace mappings
    m.def("io_forward", &DREAMPLACE_NAMESPACE::_timing_io_forward_heterosta, 
            "HeteroSTA IO function with PlaceDB integration and DREAMPlace mappings",
            pybind11::arg("args"), pybind11::arg("placedb"), pybind11::arg("dreamplace_mappings"));
    
    m.def("report_wns_tns", [](DREAMPLACE_NAMESPACE::TimerWrapper& wrapper, bool use_cuda) {
            float wns, tns;
            
            // Add debug information about timing state
            STAHoldings* sta = wrapper.get_raw_timer();
            
            bool success = heterosta_report_wns_tns_max(sta, &wns, &tns, use_cuda);  // setup mode
            
            if (success) {
                DREAMPLACE_NAMESPACE::dreamplacePrint(DREAMPLACE_NAMESPACE::kDEBUG, "WNS/TNS Report: WNS=%.3f, TNS=%.3f (gpu=%s)\n", 
                               wns, tns, use_cuda ? "true" : "false");
                
                if (wns == 0.0f && tns == 0.0f) {
                    DREAMPLACE_NAMESPACE::dreamplacePrint(DREAMPLACE_NAMESPACE::kWARN, "WNS/TNS=0 - check SDC constraints\n");
                }
                
                return std::make_tuple(wns, tns);
            } else {
                DREAMPLACE_NAMESPACE::dreamplacePrint(DREAMPLACE_NAMESPACE::kERROR, "Failed to query WNS/TNS from HeteroSTA\n");
                return std::make_tuple(std::nanf(""), std::nanf(""));
            }
        }, "Report WNS and TNS", pybind11::arg("wrapper"), pybind11::arg("use_cuda") = false);

    m.def("report_wns", [](DREAMPLACE_NAMESPACE::TimerWrapper& wrapper, bool use_cuda) {
            float wns, tns; 
            if (heterosta_report_wns_tns_max(wrapper.get_raw_timer(), &wns, &tns,use_cuda)) {
                return wns;  
            }
            return std::nanf("");
        }, "Report WNS");

    m.def("report_tns", [](DREAMPLACE_NAMESPACE::TimerWrapper& wrapper, bool use_cuda) {
            float wns, tns; 
            if (heterosta_report_wns_tns_max(wrapper.get_raw_timer(), &wns, &tns, use_cuda)) {
                return tns;
            }
            return std::nanf("");
        }, "Report TNS");
}
