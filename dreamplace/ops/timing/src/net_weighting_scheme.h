#ifndef DREAMPLACE_NET_WEIGHTING_SCHEME_H_
#define DREAMPLACE_NET_WEIGHTING_SCHEME_H_

#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <ot/timer/timer.hpp>
#include "utility/src/torch.h"
#include "utility/src/utils.h"
#include "place_io/src/Util.h"

namespace _timing_impl {
template<typename T>
using index_type = typename DREAMPLACE_NAMESPACE::coordinate_traits<T>::index_type;
using string2index_map_type = std::unordered_map<std::string, index_type<int> >;
}

DREAMPLACE_BEGIN_NAMESPACE

// The net-weighting scheme enum class.
// We try to implement different net-weighting schemes.
// For different schemes, we implement different algorithms to update net
// weights in each timing iteration.
enum class NetWeightingScheme {
  ADAMS, LILITH
};

///
/// \brief Implementation of net-weighting scheme.
/// \param timer the OpenTimer object.
/// \param n the maximum number of paths.
/// \param net_name2id_map the net name to id map.
/// \param net_criticality the criticality values of nets (array).
/// \param net_criticality_deltas the criticality delta values of nets (array).
/// \param net_weights the weights of nets (array).
/// \param net_weight_deltas the increment of net weights.
/// \param degree_map the degree map of nets.
/// \param decay the decay factor in momemtum iteration.
/// \param max_net_weight the maximum net weight in timing opt.
/// \param ignore_net_degree the net degree threshold.
/// \param num_threads number of threads for parallel computing.
///
#define DEFINE_APPLY_SCHEME                                        \
  static void apply(                                               \
      ot::Timer& timer, int n,                                     \
      const _timing_impl::string2index_map_type& net_name2id_map,  \
      T* net_criticality, T* net_criticality_deltas,               \
      T* net_weights, T* net_weight_deltas, const int* degree_map, \
      T decay, T max_net_weight, int ignore_net_degree,            \
      int num_threads)

///
/// \brief The implementation of net-weighting algorithms.
/// \tparam T the array data type (usually float).
/// \tparam scm the enum net-weighting scheme.
/// Partial specialization of full class should be implemented to correctly
/// enable compile-time polymorphism.
///
template <typename T, NetWeightingScheme scm>
struct NetWeighting {
  DEFINE_APPLY_SCHEME;
};

///
/// \brief Report the slack of a specific pin (given the name of this pin).
//    The report_slack method will be invoked. Note that we extract the worst
//    one of [MIN, MAX] * [FALL, RISE] (4 slacks).
/// \param timer the OpenTimer object.
/// \param name the specific pin name.
///
inline float report_pin_slack(ot::Timer& timer, const std::string& name) {
  using namespace ot;
  // The pin slack defaults to be the largest float number.
  // Use a valid float number instead of the infinity.
  float ps = std::numeric_limits<float>::max();
  FOR_EACH_EL_RF (el, rf) {
    auto s = timer.report_slack(name, el, rf);
    // Check whether the std::optional<float> value indeed has a value or not.
    // The comparison is enabled only when @s has a value.
    if (s) ps = std::min(ps, *s);
  }
  return ps;
}

///
/// \brief Report the slack of a specific net.
/// \param timer the OpenTimer object.
/// \param net the specific net structure in the OpenTimer object.
///
inline float report_net_slack(ot::Timer& timer, const ot::Net& net) {
  // The net slack defaults to the worst one of sinks.
  float slack = std::numeric_limits<float>::max();
  const ot::Pin* root = net.root();
  for (const auto ptr : net.pins()) {
    // Skip the driver in the traversal.
    if (ptr == root) continue;
    float ps = report_pin_slack(timer, ptr->name());
      slack = std::min(slack, ps);
  }
  return slack;
}

////////////////////////////////////////////////////////////////////////////
// Partial specialization of naive net-weighting schemes.
template <typename T>
struct NetWeighting<T, NetWeightingScheme::ADAMS> {
  DEFINE_APPLY_SCHEME {
    // Apply net-weighting scheme.
    dreamplacePrint(kINFO, "apply adams net-weighting scheme...\n");
    
    // Report the first several paths of the critical ones.
    // Note that a path is actually a derived class of std::list<ot::Point>.
    // A Point object contains the corresponding pin.
    // Report timing using the timer object.
    auto beg = std::chrono::steady_clock::now();
    const auto& paths = timer.report_timing(n);
    auto end = std::chrono::steady_clock::now();
    dreamplacePrint(kINFO, "finish report-timing (%f s)\n",
      std::chrono::duration_cast<std::chrono::milliseconds>(
        end - beg).count() * 0.001);

    // Check paths returned by timer.
    if (paths.empty()) {
      dreamplacePrint(kWARN, "report_timing: no critical path found\n");
      return;
    }
    size_t num_nets = timer.num_nets();
    std::vector<bool> net_critical_flag(num_nets, 0);
    beg = std::chrono::steady_clock::now();
    for (auto& path : paths) {
      for (auto& point : path) {
        auto name = point.pin.net()->name();
        int net_id = net_name2id_map.at(name);
        net_critical_flag.at(net_id) = 1;
      }
    }
    // Update the net weights accordingly.
#pragma omp parallel for num_threads(num_threads)
    for (size_t i = 0; i < num_nets; ++i) {
      if (degree_map[i] > ignore_net_degree) continue;
      net_criticality[i] *= 0.5;
      if (net_critical_flag[i]) net_criticality[i] += 0.5;
      net_weights[i] *= (1 + net_criticality[i]);
    }
    end = std::chrono::steady_clock::now();
    dreamplacePrint(kINFO, "finish net-weighting (%f s)\n",
      std::chrono::duration_cast<std::chrono::milliseconds>(
        end - beg).count() * 0.001);
  }
};

// Partial specialization of lilith net-weighting.
template <typename T>
struct NetWeighting<T, NetWeightingScheme::LILITH> {
  DEFINE_APPLY_SCHEME {
    // Apply net-weighting scheme.
    dreamplacePrint(kINFO, "apply lilith net-weighting scheme...\n");
    dreamplacePrint(kINFO, "lilith mode momentum decay factor: %f\n", decay);
    
    // Calculate run-time of net-weighting update.
    auto beg = std::chrono::steady_clock::now();
    float wns = timer.report_wns().value();
    double max_nw = 0;
    for (const auto& [name, net] : timer.nets()) {
      // The net id in the dreamplace database.
      int net_id = net_name2id_map.at(name);
      float slack = report_net_slack(timer, net);
      if (wns < 0) {
        float nc = (slack < 0)? std::max(0.f, slack / wns) : 0;
        // Decay the criticality value of the current net.
        net_criticality[net_id] = std::pow(1 + net_criticality[net_id], decay) *
          std::pow(1 + nc, 1 - decay) - 1;
      }
      // Update the net weights accordingly.
      // Ignore the clock net.
      if (degree_map[net_id] > ignore_net_degree)
        continue;
      net_weights[net_id] *= (1 + net_criticality[net_id]);

      // Manually limit the upper bound of the net weights, as it may
      // introduce illegality or divergence for some cases.
      if (net_weights[net_id] > max_net_weight)
        net_weights[net_id] = max_net_weight;
      if (max_nw < net_weights[net_id]) max_nw = net_weights[net_id];
    }
    auto end = std::chrono::steady_clock::now();
    dreamplacePrint(kINFO, "finish net-weighting (%f s)\n",
      std::chrono::duration_cast<std::chrono::milliseconds>(
        end - beg).count() * 0.001);
  }
};

#undef DEFINE_APPLY_SCHEME

DREAMPLACE_END_NAMESPACE

#endif // DREAMPLACE_NET_WEIGHTING_SCHEME_H_
