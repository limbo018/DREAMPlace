#ifndef DREAMPLACE_TIMING_HS_CPP_H_
#define DREAMPLACE_TIMING_HS_CPP_H_

#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include "heterosta.h"
#include "timing_hs_io_cpp.h"
#include "utility/src/torch.h"
#include "utility/src/utils.h"

DREAMPLACE_BEGIN_NAMESPACE

namespace _timing_heterosta_impl {
    using string2index_map_type = std::unordered_map<std::string, int>;
}

class TimingHeterostaCpp {
public:
    // Use the same type alias as IO class for consistency
    using STAHoldingsPtr = TimingHeterostaIO::STAHoldingsPtr;

    ///
    /// @brief The forward function for timing analysis
    /// @param sta the HeteroSTA holdings object
    /// @param pos the cell locations tensor
    /// @param num_pins total number of pins in the design
    /// @param wire_resistance_per_micron unit-length resistance value
    /// @param wire_capacitance_per_micron unit-length capacitance value
    /// @param scale_factor the scaling factor to be applied to the design
    /// @param lef_unit the unit distance microns defined in the LEF file
    /// @param def_unit the unit distance microns defined in the DEF file
    /// @param slacks_rf output tensor for pin slacks (rise/fall)
    /// @param ignore_net_degree the degree threshold
    /// @param use_cuda whether to use CUDA for computation

    ///
    static void forward(
        STAHoldings& sta, torch::Tensor pos,
        size_t num_pins,
        double wire_resistance_per_micron,
        double wire_capacitance_per_micron,
        double scale_factor, int lef_unit, int def_unit,
        torch::Tensor slacks_rf = torch::Tensor(),
        int ignore_net_degree = std::numeric_limits<int>::max(),
        bool use_cuda = false);
    
     ///
    /// @brief Update net weights based on timing criticality
    /// @param sta the HeteroSTA holdings object
    /// @param n the maximum number of paths
    /// @param num_nets the number of nets in the design
    /// @param num_pins the number of pins in the design
    /// @param flat_netpin flatten version of net2pin_map
    /// @param netpin_start starting index in netpin map for each net
    /// @param pin2net_map the pin to net mapping array
    /// @param net_criticality the criticality values of nets
    /// @param net_criticality_deltas the criticality deltas of nets
    /// @param net_weights the weights of nets
    /// @param net_weight_deltas the increment of net weights
    /// @param degree_map the degree map of nets
    /// @param max_net_weight maximum net weight in timing opt
    /// @param momentum_decay_factor momentum decay factor
    /// @param ignore_net_degree the degree threshold
    /// @param use_cuda whether to use CUDA for computation
    ///
    static void update_net_weights(
        STAHoldings& sta, int n,
        int num_nets,
        int num_pins,
        torch::Tensor flat_netpin, torch::Tensor netpin_start,
        torch::Tensor pin2net_map,
        torch::Tensor net_criticality, torch::Tensor net_criticality_deltas,
        torch::Tensor net_weights, torch::Tensor net_weight_deltas,
        torch::Tensor degree_map,
        double max_net_weight, double momentum_decay_factor,
        int ignore_net_degree, bool use_cuda);

    private:
    // Helper functions for internal use - no longer need pin mapping helpers
};

DREAMPLACE_END_NAMESPACE

#endif // DREAMPLACE_TIMING_HS_CPP_H_