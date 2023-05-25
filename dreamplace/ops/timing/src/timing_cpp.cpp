#include "timing_cpp.h"

#include <flute.hpp>

DREAMPLACE_BEGIN_NAMESPACE

///
/// \brief perform timing analysis.
/// \param x the horizontal coordinates of cell locations.
/// \param y the vertical coordinates of cell locations.
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
template <typename T>
int timingCppLauncher(
    /* Pybind does NOT allow unique_ptr<> as function argument type,
     * so we use raw pointers instead.
     * The reference type ot::Timer& is also available here, but again
     * pay attention that unique_ptr<ot::Timer> is not allowed to be
     * function argument as python needs to give up ownership of an
     * object passed to this function, which is generally impossible.
     */
    ot::Timer& timer,
    const T* x, const T* y,
    const std::vector<std::string>& net_names, /* The net names. */
    const std::vector<std::string>& pin_names, /* The pin names. */
    const int* flat_netpin, const int* netpin_start,
    const int* pin2node, const T* pin_offset_x, const T* pin_offset_y,
    T wire_resistance_per_micron,
    T wire_capacitance_per_micron,
    double scale_factor, int lef_unit, int def_unit,
    int ignore_net_degree);

// Implementation of a static class method.
void TimingCpp::forward(
    ot::Timer& timer, torch::Tensor pos,
    const std::vector<std::string>& net_names, /* The net names. */
    const std::vector<std::string>& pin_names, /* The pin names. */
    torch::Tensor flat_netpin, torch::Tensor netpin_start,
    torch::Tensor pin2node, torch::Tensor pin_offset_x, torch::Tensor pin_offset_y,
    double wire_resistance_per_micron,
    double wire_capacitance_per_micron,
    double scale_factor, int lef_unit, int def_unit,
    int ignore_net_degree) {
  // Check configuity and cpu info.
  CHECK_EVEN(pos);
  CHECK_CONTIGUOUS(pos);
  CHECK_FLAT_CPU(flat_netpin);
  CHECK_CONTIGUOUS(flat_netpin);
  CHECK_FLAT_CPU(netpin_start);
  CHECK_CONTIGUOUS(netpin_start);
  CHECK_FLAT_CPU(pin2node);
  CHECK_CONTIGUOUS(pin2node);
  CHECK_FLAT_CPU(pin_offset_x);
  CHECK_CONTIGUOUS(pin_offset_x);
  CHECK_FLAT_CPU(pin_offset_y);
  CHECK_CONTIGUOUS(pin_offset_y);

  DREAMPLACE_DISPATCH_FLOATING_TYPES(
      pos, "timingCppLauncher",
      [&] {
        timingCppLauncher<scalar_t>(
            timer,
            DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + pos.numel() / 2,
            net_names,  // Net names array (required by OpenTimer).
            pin_names,  // Pin names array (required by OpenTimer).
            DREAMPLACE_TENSOR_DATA_PTR(flat_netpin, int),
            DREAMPLACE_TENSOR_DATA_PTR(netpin_start, int),
            DREAMPLACE_TENSOR_DATA_PTR(pin2node, int),
            DREAMPLACE_TENSOR_DATA_PTR(pin_offset_x, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(pin_offset_y, scalar_t),
            wire_resistance_per_micron,
            wire_capacitance_per_micron,
            scale_factor, lef_unit, def_unit,
            ignore_net_degree);
      });
}

///
/// \brief This function takes a point coordinate as input and
///   find its corresponding node id.
/// \param pos2pins_map the point-indices map.
/// \param point the point object as the key value.
/// \param index if the point cannot be found in the given map,
///    the new point should be inserted as this new index.
///
auto& retrieve_pins_from_pos(
    std::map<Point<int>, std::set<int> >& pos2pins_map,
    const Point<int>& point,
    int& index) {
  if (pos2pins_map.find(point) != pos2pins_map.end())
    return pos2pins_map[point];
  // If a Steiner point is not taken by a pin, then it should be
  // added as a virtual pin with a new id.
  pos2pins_map.emplace(point, std::set<int>{index++});
  return pos2pins_map[point];
}

// The implementations of timing analysis using OpenTimer.
// Note that we have to generate RCTrees using FLUTE given a sketch
// placement result.
template <typename T>
int timingCppLauncher(
    ot::Timer& timer, /* The timer containing RCTrees. */
    const T* x, const T* y,
    const std::vector<std::string>& net_names,
    const std::vector<std::string>& pin_names,
    const int* flat_netpin, const int* netpin_start,
    const int* pin2node, const T* pin_offset_x, const T* pin_offset_y,
    T wire_resistance_per_micron,
    T wire_capacitance_per_micron,
    double scale_factor, int lef_unit, int def_unit,
    int ignore_net_degree) {
  // Before generating the trees, calculate some important factors required.
  // The units of dreamplace coordinates and timer analysis are different.
  double unit_to_micron = scale_factor * def_unit;
  auto res_unit = timer.resistance_unit()->value();
  auto cap_unit = timer.capacitance_unit()->value();
  T rf = static_cast<T>(wire_resistance_per_micron) / res_unit;
  T cf = static_cast<T>(wire_capacitance_per_micron) / cap_unit;

  // TODO: this LUT read may be done only once.
  // We must read LUT first before calling FLUTE, otherwise you will face
  // a tedious segmentation fault.
  using Point2i = ::DreamPlace::Point<int>;
  dreamplacePrint(kINFO, "launch rc tree construction...\n");
  flute::readLUT(
      "thirdparty/flute/lut.ICCAD2015/POWV9.dat",
      "thirdparty/flute/lut.ICCAD2015/POST9.dat");
  auto beg = std::chrono::steady_clock::now();

  // TODO: check the template argument, integers or not?
  // The vector to store temporary x and y positions. This operation must be
  // done because FLUTE only supports integer inputs, so we have to perform
  // a scaling to convert the input data types into integers.
  constexpr const int scale = 1000;  // flute only supports integers.
  int num_nets = net_names.size();

  // We traverse all the nets in the netlist.
  omp_lock_t lock;
  omp_init_lock(&lock);
#pragma omp parallel for
  for (int i = 0; i < num_nets; ++i) {
    // The reference to the net instance in the timer.
    // We are goint to add RC nodes and RC edges into the RC tree stored
    // as the instance variable in this net.
    auto net_iter = timer.nets().find(net_names[i]);
    dreamplaceAssertMsg(
        net_iter != timer.nets().end(),
        "could not find net name %s in timer\n", net_names[i].c_str());
    auto& net = net_iter->second;
    auto& tree = net.emplace_rct();
    const int degree = netpin_start[i + 1] - netpin_start[i];
    const int root = flat_netpin[netpin_start[i]];

    // This is a lambda expression takes a local pin index in this net as input
    // and insert node into the rc tree if the node has not been inserted yet.
    // The name of rc node will be returned.
    auto emplace_rc_node =
        [&](int index) -> std::string {
      auto name = net_names[i] + ":" + std::to_string(index - degree + 1);
      if (index < degree) {
        name = pin_names[flat_netpin[index + netpin_start[i]]];
        if (!tree.node(name)) tree.insert_node(name, 0);
        // Set the inner pin pointer of this rc node.
        auto pin_iter = timer.pins().find(name);
        dreamplaceAssertMsg(
            pin_iter != timer.pins().end(),
            "could not find pin name %s in timer\n", name.c_str());
        tree.node(name)->pin(pin_iter->second);
      } else {  // This rc node is a Steiner point.
        if (!tree.node(name)) tree.insert_node(name, 0);
      }
      return name;
    };

    std::map<Point2i, std::set<int> > pos2pins_map;
    // Note that some pins may have duplicate coordinates, so we have to
    // do a filtering and feed clean data without superposition into FLUTE
    // to generate Steiner trees.
    std::vector<int> vx, vy;
    vx.reserve(degree);
    vy.reserve(degree);

    // We have a inner id for each pin and Steiner node, which should be
    // distinguished from the global id.
    std::map<int, int> global2inner_map;

    // Define an empty edge vector.
    for (int j = 0; j < degree; ++j) {
      int pin = flat_netpin[j + netpin_start[i]];
      int node = pin2node[pin];
      T offset_x = pin_offset_x[pin], offset_y = pin_offset_y[pin];
      // Find the correct pin locations given cell locations.
      auto x_ = static_cast<int>((x[node] + offset_x) * scale);
      auto y_ = static_cast<int>((y[node] + offset_y) * scale);
      global2inner_map[pin] = j;

      // Add a new key-value pair into pos-pin map.
      // Here we always assume that a position can only be taken by
      // exactly one pin.
      if (pos2pins_map.find(Point2i(x_, y_)) != pos2pins_map.end())
        pos2pins_map[Point2i(x_, y_)].insert(j);
      else {  // Emplace a new key-value pair.
        pos2pins_map.emplace(Point2i(x_, y_), std::set<int>{j});
        vx.emplace_back(x_);
        vy.emplace_back(y_);
      }
    }
    // A temporary store of degree value of the current net.
    const int valid_size = static_cast<int>(vx.size());
    int num_pins = degree;

    // This variable stores all points that is taken by multiple pins.
    std::set<Point2i> multipin_pos;
    std::map<Point2i, Point2i> pos2neighbor_map;

    // Call FLUTE to generate a steiner tree. Note that the degree must be
    // larger than 2 so that the tree is not degraded.
    if (valid_size > 1) {
      flute::Tree flutetree = flute::flute(
          valid_size, vx.data(), vy.data(), ACCURACY);
      for (int bid = 0, ub = 2 * flutetree.deg - 2; bid < ub; ++bid) {
        flute::Branch& branch1 = flutetree.branch[bid];
        flute::Branch& branch2 = flutetree.branch[branch1.n];

        // Note that the coordinates fed into flute has been scaled by
        // a hyperparameter so that they can handle float-point data.
        Point2i p1(branch1.x, branch1.y), p2(branch2.x, branch2.y);

        // We only care about 2d steiner tree construction without any info
        // about layer assignment or routing. Therefore, simply skip the
        // current iteration if two points are the same.
        if (p1 == p2) continue;

        // Extract the Steiner point and retrieve corresponding pin id.
        // It is possible that the two points have exactly the same
        // coordinates.
        pos2neighbor_map.emplace(p2, p1);
        auto& id1 = retrieve_pins_from_pos(pos2pins_map, p1, num_pins);
        auto& id2 = retrieve_pins_from_pos(pos2pins_map, p2, num_pins);

        // Calculate the Manhattan distance (l1) between the two pins and
        // add an edge to the vector.
        auto distance = manhattanDistance(p1, p2);
        T wl = static_cast<T>(distance * 1.0) / scale / unit_to_micron;
        if (degree > ignore_net_degree) {
          // We manage to ignore the clock net.
          // If the net degree is larger than a threshold, we directly
          // ignore it by treating any line segment here as degraded.
          wl = 0;
        }

        // params @id1 and @id2 should both be non-empty sets.
        if (!id1.empty() && !id2.empty()) {
          auto base1 = id1.begin(), base2 = id2.begin();
          if (*base1 != *base2) {
            auto from = emplace_rc_node(*base1);
            auto to = emplace_rc_node(*base2);
            tree.insert_segment(from, to, rf * wl);
            tree.increase_cap(from, cf * wl * 0.5);
            tree.increase_cap(to, cf * wl * 0.5);
          }
          // Record if a pin set contains multiple pins.
          if (id1.size() > 1) multipin_pos.insert(p1);
          if (id2.size() > 1) multipin_pos.insert(p2);
        }
      }
      free(flutetree.branch);
    } else if (valid_size == 1 && degree > 1) {
      // All pins (at least 2 pins) in this net are superposed at point
      // (vx[0], vy[0]), so we add this pos to multipin_pos.
      multipin_pos.emplace(vx[0], vy[0]);
    }

    // Revert all pins filtered in the previous stage.
    for (const auto& pos : multipin_pos) {
      const auto& pins = pos2pins_map[pos];
      int adj_pin = global2inner_map[root];
      const auto& _ppos = pos2neighbor_map[pos];
      if (auto itr = pos2pins_map.find(_ppos); itr != pos2pins_map.end()) {
        // Only take the first one. In fact an adjacent rc node must be either a
        // Steiner node or the root.
        adj_pin = *itr->second.cbegin();
      }
      auto from = emplace_rc_node(adj_pin);
      auto distance = manhattanDistance(pos, _ppos);
      T wl = static_cast<T>(distance * 1.0) / scale / unit_to_micron;
      for (auto it = std::next(pins.cbegin()); it != pins.cend(); ++it)
        tree.insert_segment(from, emplace_rc_node(*it), wl * rf);
    }
    if (degree == 1) {
      // Special handling: the net contains only one pin!
      // In this case, the edge vector is actually empty, so no node
      // will be inserted into the RC tree. We have to create one.
      // The RC tree only contains a single root node.
      emplace_rc_node(global2inner_map[root]);
    }
    // Assign root node to this RC tree.
    tree.assign_root(pin_names[root]);
    tree.update_rc_timing();

    // Dump some useful information for debugging.
    // timer.dump_spef(std::cout);
    // timer.dump_rctree(std::cout);
    // --------------------------------------------
    // The frontier insertion is very IMPORTANT!
    // Otherwise you will not get updated timing report each time.
    omp_set_lock(&lock);
    timer.insert_frontier(*net.root());
    omp_unset_lock(&lock);
  }
  omp_destroy_lock(&lock);
  dreamplacePrint(kINFO, "rc tree construction is done.\n");
  auto end = std::chrono::steady_clock::now();
  auto trc = std::chrono::duration_cast<std::chrono::milliseconds>(end - beg);
  dreamplacePrint(kINFO, "finish rc tree construction (%f s)\n",
                  trc.count() * 0.001);

  // Remember to call update_states!
  // The normal update_timing will check _lineage first, so if we don't
  // explicitly speficy a new .spef file via read_spef, the _lineage will not
  // be updated add the update_timing will definitely do nothing.
  beg = std::chrono::steady_clock::now();
  timer.update_states();
  end = std::chrono::steady_clock::now();
  auto usc = std::chrono::duration_cast<std::chrono::milliseconds>(end - beg);
  dreamplacePrint(kINFO, "finish state updates (%f s)\n",
                  usc.count() * 0.001);

  // Probably you may consider to dump the timer or spef into files.
  // File IO will cost a lot of time. It will be useful sometimes because
  // file io will always work for timing.
  // -----------------------------------
  // dreamplacePrint(kINFO, "begin to dump spef (temporary solution)...\n");
  // std::ofstream fout("rc.spef");
  // timer.dump_spef(fout);
  // fout.close();
  return 0;
}

///
/// \brief Implementation of net-weighting scheme.
/// \param timer the OpenTimer object.
/// \param n the maximum number of paths.
/// \param net_name2id_map the net name to id map.
/// \param net_criticality the criticality values of nets (array).
/// \param net_criticality_deltas the criticality deltas of nets (array).
/// \param net_weights the weights of nets (array).
/// \param net_weight_deltas the increment of net weights.
/// \param degree_map the degree map of nets.
/// \param momentum_decay_factor the decay factor in momemtum iteration.
/// \param max_net_weight the maximum net weight in timing opt.
/// \param ignore_net_degree the net degree threshold.
/// \param net_weighting_scheme the net-weighting scheme.
/// \param num_threads number of threads for parallel computing.
///
template <typename T>
void updateNetWeightCppLauncher(
    ot::Timer& timer, int n,
    const _timing_impl::string2index_map_type& net_name2id_map,
    T* net_criticality, T* net_criticality_deltas,
    T* net_weights, T* net_weight_deltas, const int* degree_map,
    int net_weighting_scheme, T momentum_decay_factor,
    T max_net_weight, int ignore_net_degree, int num_threads) {
#define SELECT_SCHEME(angel)                         \
  NetWeighting<T, NetWeightingScheme::angel>::apply( \
      timer, n, net_name2id_map,                     \
      net_criticality, net_criticality_deltas,       \
      net_weights, net_weight_deltas, degree_map,    \
      momentum_decay_factor, max_net_weight,         \
      ignore_net_degree, num_threads)
  // Apply the net-weighting algorithm.
  switch (net_weighting_scheme) {
    case 0:
      SELECT_SCHEME(ADAMS);
      break;
    case 1:
      SELECT_SCHEME(LILITH);
      break;
    default:
      // WARNING: unsupported net-weighting scheme. Do nothing.
      // Do not report a warning since it has been done in python.
      // dreamplacePrint(kWARN, "unsupported net-weighting scheme!\n");
      break;
  }
#undef SELECT_SCHEME
}

// Implementation of a static class method.
void TimingCpp::update_net_weights(
    ot::Timer& timer, int n,
    const _timing_impl::string2index_map_type& net_name2id_map,
    torch::Tensor net_criticality, torch::Tensor net_criticality_deltas,
    torch::Tensor net_weights, torch::Tensor net_weight_deltas,
    torch::Tensor degree_map,
    int net_weighting_scheme, double momentum_decay_factor,
    double max_net_weight, int ignore_net_degree) {
  // Check torch tensors.
  CHECK_FLAT_CPU(net_criticality);
  CHECK_CONTIGUOUS(net_criticality);
  CHECK_FLAT_CPU(net_weights);
  CHECK_CONTIGUOUS(net_weights);
  CHECK_FLAT_CPU(net_weight_deltas);
  CHECK_CONTIGUOUS(net_weight_deltas);
  CHECK_FLAT_CPU(degree_map);
  CHECK_CONTIGUOUS(degree_map);

  DREAMPLACE_DISPATCH_FLOATING_TYPES(
      net_weights, "updateNetWeightCppLauncher",
      [&] {
        updateNetWeightCppLauncher<scalar_t>(
            timer, n,
            net_name2id_map,
            DREAMPLACE_TENSOR_DATA_PTR(net_criticality, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(net_criticality_deltas, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(net_weights, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(net_weight_deltas, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(degree_map, int),
            net_weighting_scheme,
            static_cast<scalar_t>(momentum_decay_factor),
            static_cast<scalar_t>(max_net_weight),
            ignore_net_degree,
            at::get_num_threads());
      });
}

///
/// \brief Compute pin slack at once.
/// \param timer the timer object.
/// \param pin_name2id_map the pin name to id map.
/// \param slack the result array.
///
template <typename T>
void pinSlackCppLauncher(
    ot::Timer& timer,
    const _timing_impl::string2index_map_type& pin_name2id_map,
    T* slack) {
  auto report_pin_slack = /* Calculate the pin slack. */
      [&](const std::string& name) -> float {
    using namespace ot;
    float ps = std::numeric_limits<float>::max();
    FOR_EACH_EL_RF(el, rf) {
      auto s = timer.report_slack(name, el, rf);
      if (s) ps = std::min(ps, *s);
    }
    return ps;
  };
  for (const auto& [name, pin] : timer.pins()) {
    if (pin_name2id_map.find(name) == pin_name2id_map.end()) continue;
    auto id = pin_name2id_map.at(name);
    float slk = report_pin_slack(name);
    slack[id] = slk;
  }
}

// Implementation of a static class method.
void TimingCpp::evaluate_slack(
    ot::Timer& timer,
    const _timing_impl::string2index_map_type& pin_name2id_map,
    torch::Tensor slack) {
  // Check torch tensors.
  CHECK_FLAT_CPU(slack);
  CHECK_CONTIGUOUS(slack);

  DREAMPLACE_DISPATCH_FLOATING_TYPES(
      slack, "pinSlackCppLauncher",
      [&] {
        pinSlackCppLauncher<scalar_t>(
            timer, pin_name2id_map,
            DREAMPLACE_TENSOR_DATA_PTR(slack, scalar_t));
      });
}

DREAMPLACE_END_NAMESPACE
