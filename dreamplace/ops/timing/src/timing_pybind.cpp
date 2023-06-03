#include "timing/src/timing_cpp.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

DREAMPLACE_BEGIN_NAMESPACE

///
/// Implementation of a static class method.
/// This function is implemented only for the python api.
///
/// \brief report the nets (indices) on the critical paths.
/// \param timer the OpenTimer object.
/// \param n the maximum number of paths.
/// \param net_name2id_map the net name to id map.
/// \return the indices of nets on the critical paths.
///
std::vector<std::vector<int> > _report_timing(
    ot::Timer& timer, int n,
    const _timing_impl::string2index_map_type& net_name2id_map) {
  // The return object is the list of net indices.
  std::vector<std::vector<int> > result;

  // Report the first several paths of the critical ones.
  // Note that a path is actually a derived class of std::list<ot::Point>.
  // A Point object contains the corresponding pin.
  // Report timing using the timer object.
  const auto& paths = timer.report_timing(n);
  if (paths.empty()) {
    dreamplacePrint(kWARN, "report_timing: no critical path found\n");
    return result;
  }
  for (auto& path : paths) {
    result.emplace_back();
    auto& pl = result.back();
    int prev = -1; // Remove duplication.
    for (auto& point : path) {
      auto name = point.pin.net()->name();
      int net_id = net_name2id_map.at(name);
      if (net_id != prev)
        pl.push_back(prev = net_id);
    }
  }
  return result;
}

///
/// \brief Parse a list of arguments and return a timer.
/// \param args the arguments including files required to construct a timer.
/// \return the unique pointer of timer.
///
std::unique_ptr<ot::Timer> _timing_io_forward(const pybind11::list& args) {
  // Call the default constructor of timer in namespace ot.
  // Note that only when all external files specifying timing constraints
  // are loaded into @timer can it perform timing analysis.
  std::unique_ptr<ot::Timer> ptr(new ot::Timer());

  // Preprocess the input arguments.
  int argc = pybind11::len(args); 
  char** argv = new char*[argc]; 
  for (int i = 0; i < argc; ++i) {
    std::string token = pybind11::str(args[i]); 
    argv[i] = new char[token.size() + 1];
    std::copy(token.begin(), token.end(), argv[i]); 
    argv[i][token.size()] = '\0';
  }

  // Successfully convert the arguments into C type and feed them into
  // the program option parsers.
  TimingCpp::read_constraints(*ptr, argc, argv); 
  for (int i = 0; i < argc; ++i) delete[] argv[i];
  delete[] argv;

  // NOTE:!! the return value is a unique_ptr.
  // Class ot::Timer does not have a copy constructor. In fact, it is
  // explicitly deleted, so we have to return a unique pointer in this
  // case, and bind it into a python interface.
  return std::move(ptr);
}

///
/// \brief report the arrival time.
/// \param timer the OpenTimer object.
/// \param pin_name the specific pin name.
/// \param split binary number determining Split::MIN, Splot::MAX.
/// \param tran binary number determining Tran::RISE, Tran::FALL.
///
float _report_at(ot::Timer& timer, const std::string& pin_name, bool split, bool tran) {
  auto at = timer.report_at(
    pin_name, static_cast<ot::Split>(split), static_cast<ot::Tran>(tran));
  return at.value_or(std::nanf(""));
}

///
/// \brief report the pin slack.
/// \param timer the OpenTimer object.
/// \param pin_name the specific pin name.
/// \param split binary number determining Split::MIN, Splot::MAX.
/// \param tran binary number determining Tran::RISE, Tran::FALL.
///
float _report_slack(ot::Timer& timer, const std::string& pin_name, bool split, bool tran) {
  auto slack = timer.report_slack(
    pin_name, static_cast<ot::Split>(split), static_cast<ot::Tran>(tran));
  return slack.value_or(std::nanf(""));
}

///
/// \brief report tns with early/late or rise/fall
/// \param split binary number determining Split::MIN, Splot::MAX.
/// \param tran binary number determining Tran::RISE, Tran::FALL.
/// \return the corresponding tns value
///
auto _report_tns_elw(ot::Timer& timer, bool split) {
  return timer.report_tns_elw(static_cast<ot::Split>(split)).value_or(std::nanf(""));
}
auto _report_tns_all(ot::Timer& timer) {
  return timer.report_tns().value_or(std::nanf(""));
}
auto _report_tns_el(ot::Timer& timer, bool split) {
  return timer.report_tns(static_cast<ot::Split>(split), std::nullopt).value_or(std::nanf(""));
}
auto _report_tns_rf(ot::Timer& timer, bool tran) {
  return timer.report_tns(std::nullopt, static_cast<ot::Tran>(tran)).value_or(std::nanf(""));
}
auto _report_tns_el_rf(ot::Timer& timer, bool split, bool tran) {
  return timer.report_tns(static_cast<ot::Split>(split), static_cast<ot::Tran>(tran)).value_or(std::nanf(""));
}

///
/// \brief report wns with early/late or rise/fall
/// \param split binary number determining Split::MIN, Splot::MAX.
/// \param tran binary number determining Tran::RISE, Tran::FALL.
/// \return the corresponding wns value
///
auto _report_wns_all(ot::Timer& timer) {
  return timer.report_wns().value_or(std::nanf(""));
}
auto _report_wns_el(ot::Timer& timer, bool split) {
  return timer.report_wns(static_cast<ot::Split>(split), std::nullopt).value_or(std::nanf(""));
}
auto _report_wns_rf(ot::Timer& timer, bool tran) {
  return timer.report_wns(std::nullopt, static_cast<ot::Tran>(tran)).value_or(std::nanf(""));
}
auto _report_wns_el_rf(ot::Timer& timer, bool split, bool tran) {
  return timer.report_wns(static_cast<ot::Split>(split), static_cast<ot::Tran>(tran)).value_or(std::nanf(""));
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  pybind11::class_< // This binding converts a unique pointer.
    ot::Timer, std::unique_ptr<ot::Timer> // <- A holder type!
    >(m, "Timer")
      .def(pybind11::init<>())
      .def("num_primary_inputs", &ot::Timer::num_primary_inputs)
      .def("num_primary_outputs", &ot::Timer::num_primary_outputs)
      .def("num_pins", &ot::Timer::num_pins)
      .def("num_nets", &ot::Timer::num_nets)
      .def("num_arcs", &ot::Timer::num_arcs)
      .def("num_gates", &ot::Timer::num_gates)
      .def("num_tests", &ot::Timer::num_tests)
      .def("num_sccs", &ot::Timer::num_sccs)
      .def("update_timing", &ot::Timer::update_timing)
      .def("num_worst_endpoints", &ot::Timer::num_worst_endpoints)
      .def("dump_graph",    [](ot::Timer& timer) { timer.dump_graph(std::cout); })
      .def("dump_taskflow", [](ot::Timer& timer) { timer.dump_taskflow(std::cout); })
      .def("dump_net_load", [](ot::Timer& timer) { timer.dump_net_load(std::cout); })
      .def("dump_pin_cap",  [](ot::Timer& timer) { timer.dump_pin_cap(std::cout); })
      .def("dump_at",       [](ot::Timer& timer) { timer.dump_at(std::cout); })
      .def("dump_rat",      [](ot::Timer& timer) { timer.dump_rat(std::cout); })
      .def("dump_slew",     [](ot::Timer& timer) { timer.dump_slew(std::cout); })
      .def("dump_slack",    [](ot::Timer& timer) { timer.dump_slack(std::cout); })
      .def("dump_timer",    [](ot::Timer& timer) { timer.dump_timer(std::cout); })
      .def("dump_spef",     [](ot::Timer& timer) { timer.dump_spef(std::cout); })
      .def("dump_rctree",   [](ot::Timer& timer) { timer.dump_rctree(std::cout); })
      .def("dump_graph_file",    [](ot::Timer& timer, const std::string& out) { std::ofstream f(out); timer.dump_graph(f); f.close(); })
      .def("dump_taskflow_file", [](ot::Timer& timer, const std::string& out) { std::ofstream f(out); timer.dump_taskflow(f); f.close(); })
      .def("dump_net_load_file", [](ot::Timer& timer, const std::string& out) { std::ofstream f(out); timer.dump_net_load(f); f.close(); })
      .def("dump_pin_cap_file",  [](ot::Timer& timer, const std::string& out) { std::ofstream f(out); timer.dump_pin_cap(f); f.close(); })
      .def("dump_at_file",       [](ot::Timer& timer, const std::string& out) { std::ofstream f(out); timer.dump_at(f); f.close(); })
      .def("dump_rat_file",      [](ot::Timer& timer, const std::string& out) { std::ofstream f(out); timer.dump_rat(f); f.close(); })
      .def("dump_slew_file",     [](ot::Timer& timer, const std::string& out) { std::ofstream f(out); timer.dump_slew(f); f.close(); })
      .def("dump_slack_file",    [](ot::Timer& timer, const std::string& out) { std::ofstream f(out); timer.dump_slack(f); f.close(); })
      .def("dump_timer_file",    [](ot::Timer& timer, const std::string& out) { std::ofstream f(out); timer.dump_timer(f); f.close(); })
      .def("dump_spef_file",     [](ot::Timer& timer, const std::string& out) { std::ofstream f(out); timer.dump_spef(f); f.close(); })
      .def("dump_rctree_file",   [](ot::Timer& timer, const std::string& out) { std::ofstream f(out); timer.dump_rctree(f); f.close(); })
      .def("cap_unit", [](ot::Timer& timer) { return timer.capacitance_unit()->value(); })
      .def("res_unit", [](ot::Timer& timer) { return timer.resistance_unit()->value(); })
      .def("time_unit", [](ot::Timer& timer) { return timer.time_unit()->value(); })
      .def("report_tns_elw", &DREAMPLACE_NAMESPACE::_report_tns_elw)
      .def("report_tns_all", &DREAMPLACE_NAMESPACE::_report_tns_all)
      .def("report_tns_el", &DREAMPLACE_NAMESPACE::_report_tns_el)
      .def("report_tns_rf", &DREAMPLACE_NAMESPACE::_report_tns_rf)
      .def("report_tns_el_rf", &DREAMPLACE_NAMESPACE::_report_tns_el_rf)
      .def("report_wns_all", &DREAMPLACE_NAMESPACE::_report_wns_all)
      .def("report_wns_el", &DREAMPLACE_NAMESPACE::_report_wns_el)
      .def("report_wns_rf", &DREAMPLACE_NAMESPACE::_report_wns_rf)
      .def("report_wns_el_rf", &DREAMPLACE_NAMESPACE::_report_wns_el_rf)
      .def("report_at", &DREAMPLACE_NAMESPACE::_report_at)
      .def("report_slack", &DREAMPLACE_NAMESPACE::_report_slack)
      ;

  m.def("forward", &DREAMPLACE_NAMESPACE::TimingCpp::forward, "Report timing forward");
  m.def("update_net_weights", &DREAMPLACE_NAMESPACE::TimingCpp::update_net_weights, "Update net weights");
  m.def("evaluate_slack", &DREAMPLACE_NAMESPACE::TimingCpp::evaluate_slack, "Evaluate nets hpwl");
  m.def("report_timing", &DREAMPLACE_NAMESPACE::_report_timing, "Report timing paths");
  m.def("io_forward", &DREAMPLACE_NAMESPACE::_timing_io_forward, "Timing IO function parsing constraint files");
}
