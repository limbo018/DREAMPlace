#ifndef DREAMPLACE_TIMING_CPP_H_
#define DREAMPLACE_TIMING_CPP_H_

#include <unordered_map>
#include "net_weighting_scheme.h"
#include "utility/src/torch.h"
#include "utility/src/utils.h"
#include "place_io/src/Point.h"

DREAMPLACE_BEGIN_NAMESPACE

class TimingCpp {
public:
  // Default constructor.
  constexpr TimingCpp() = default;

  ///
  /// \brief the forward function of timing.
  /// \param timer the OpenTimer object.
  /// \param pos the cell locations in each iteration.
  /// \param net_names the vector of strings indicating the names of nets.
  /// \param pin_names the vector of strings indicating the names of pins.
  /// \param flat_netpin flatten version of net2pin_map which stores 
  ///  pins contained in specific nets.
  /// \param net2pin_start the 1d array with each entry specifying the
  ///  starting index of a specific net in flat_net2pin_map.
  /// \param pin2node the 1d array pin2node map.
  /// \param pin_offset_x the 1d array indicating pin offset x to its node.
  /// \param pin_offset_y the 1d array indicating pin offset y to its node.
  /// \param wire_resistance_per_micron unit-length resistance value.
  /// \param wire_capacitance_per_micron unit-length capacitance value.
  /// \param scale_factor the scaling factor to be applied to the design.
  /// \param lef_unit the unit distance microns defined in the LEF file.
  /// \param def_unit the unit distance microns defined in the DEF file.
  /// \param ignore_net_degree the degree threshold.
  ///
  static void forward(
    ot::Timer& timer, torch::Tensor pos,
    const std::vector<std::string>& net_names, /* The net names. */
    const std::vector<std::string>& pin_names, /* The pin names. */
    torch::Tensor flat_netpin, torch::Tensor netpin_start,
    torch::Tensor pin2node, torch::Tensor pin_offset_x, torch::Tensor pin_offset_y,
    double wire_resistance_per_micron,
    double wire_capacitance_per_micron,
    double scale_factor, int lef_unit, int def_unit,
    int ignore_net_degree = std::numeric_limits<int>::max());
  
  ///
  /// \brief The forward function of net-weighting
  /// \param timer the OpenTimer object.
  /// \param n the maximum number of paths.
  /// \param net_name2id_map the net name to id map.
  /// \param net_criticality the criticality values of nets (torch tensor).
  /// \param net_criticality_deltas the criticality deltas of nets (array).
  /// \param net_weights the weights of nets (torch tensor).
  /// \param net_weight_deltas the increment of net weights.
  /// \param degree_map the degree map of nets.
  /// \param net_weighting_scheme the net-weighting scheme.
  /// \param max_net_weight maximum net weight in timing opt.
  /// \param num_threads number of threads for parallel computing.
  ///
  static void update_net_weights(
    ot::Timer& timer, int n,
    const _timing_impl::string2index_map_type& net_name2id_map,
    torch::Tensor net_criticality, torch::Tensor net_criticality_deltas,
    torch::Tensor net_weights, torch::Tensor net_weight_deltas,
    torch::Tensor degree_map,
    int net_weighting_scheme, double max_net_weight,
    double momentum_decay_factor, int ignore_net_degree);

  ///
  /// \brief Compute pin slack at once.
  /// \param timer the timer object.
  /// \param pin_name2id_map the pin name to id map.
  /// \param slack the result array.
  ///
  static void evaluate_slack(
    ot::Timer& timer,
    const _timing_impl::string2index_map_type& pin_name2id_map,
    torch::Tensor slack);

  ///
  /// \brief read constraint files to timer.
  /// \param timer the OpenTimer timer object.
  /// \param pydb the python placement database.
  /// \param argc the total number of program options.
  /// \param argv the strings of program options.
  /// \return the status of parser tasks.
  ///
  static bool read_constraints(ot::Timer& timer, int argc, char** argv);
};

DREAMPLACE_END_NAMESPACE

#endif // DREAMPLACE_TIMING_CPP_H_
