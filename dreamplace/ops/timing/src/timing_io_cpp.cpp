#include "timing_cpp.h"
#include <limbo/programoptions/ProgramOptions.h>
#include <fstream>

DREAMPLACE_BEGIN_NAMESPACE

// Implementation of a static class method.
bool TimingCpp::read_constraints(ot::Timer& timer, int argc, char** argv) {
  // The strings specifying required program options.
  std::string earlyLibInput, lateLibInput, libInput, sdcInput, verilogInput;
  typedef limbo::programoptions::ProgramOptions po_type;
  using limbo::programoptions::Value;
  po_type desc(std::string("Available options"));

  // Add program options into boost parser.
  auto options = std::array<Value<std::string>, 5> { {
      { "--early_lib_input", &earlyLibInput, "input celllib file (early)" },
      { "--late_lib_input",  &lateLibInput,  "input celllib file (late)"  },
      { "--lib_input",       &libInput,      "input celllib file"         },
      { "--sdc_input",       &sdcInput,      "input sdc file"             },
      { "--verilog_input",   &verilogInput,  "input verilog file"         }
    } };
  for (auto val : options)
    desc.add_option(val);

  try { // Now parse the program options!
    desc.parse(argc, argv);
    if (!libInput.empty())        timer.read_celllib(libInput);
    else { // Single library file or separate library files.
      if (!earlyLibInput.empty()) timer.read_celllib(earlyLibInput, ot::MIN);
      if (!lateLibInput.empty())  timer.read_celllib(lateLibInput,  ot::MAX);
    }
    if (!verilogInput.empty())    timer.read_verilog(verilogInput);
    if (!sdcInput.empty())        timer.read_sdc(sdcInput);

    // --------------
    // End of file reading and parsing.
    // Note that we do not read spef explicitly because the related info
    // should be parsed from the RCTree structure generated customly.
  } catch (std::exception& e) {
    // The parser failed to parse the program options! Print error messages.
    std::cout << "program option parsing failed: " << desc << "\n";
    dreamplacePrint(kERROR, "%s\n", e.what());
    return false;
  }
  return true;
}

DREAMPLACE_END_NAMESPACE
