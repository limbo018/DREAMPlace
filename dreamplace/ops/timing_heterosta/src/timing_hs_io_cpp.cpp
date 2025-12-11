#include "timing_hs_io_cpp.h"
#include <cstring>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cstdlib>
#include <limbo/programoptions/ProgramOptions.h>
#include "utility/src/utils.h"
#include "place_io/src/PlaceDB.h"

DREAMPLACE_BEGIN_NAMESPACE

// Global storage for NetlistDB data 
TimingHeterostaIO::NetlistDataStorage TimingHeterostaIO::g_netlist_data;

// Callback function for HeteroSTA logging integration
void dreamplace_heterosta_print_callback(uint8_t level, const char* message) {
	MessageType dreamplace_level;
	switch (level) {
		case 1: dreamplace_level = kERROR; break;
		case 2: dreamplace_level = kWARN; break;
		case 3: dreamplace_level = kINFO; break;
		case 4: dreamplace_level = kDEBUG; break;
		case 5: dreamplace_level = kDEBUG; break;
		default: dreamplace_level = kINFO; break;
	}
	dreamplacePrint(dreamplace_level, "%s\n", message);
}

void TimingHeterostaIO::NetlistDataStorage::clear() {
	cell_names.clear();
	cell_types.clear();
	pin_names.clear();
	net_names.clear();
	cell_name_ptrs.clear();
	cell_type_ptrs.clear();
	pin_name_ptrs.clear();
	net_name_ptrs.clear();
	pin_directions.clear();
	pin2cell_map.clear();
	pin2net_map.clear();
	nets_zero.clear();
	nets_one.clear();
	design_name.clear();
}

/// @brief Parse all timing options from command line arguments
/// @param argc 
/// @param argv 
/// @param early_lib_path  early timing library path
/// @param late_lib_path late timing library path
/// @param single_lib_path  single timing library path
/// @param sdc_path  SDC file path
/// @return 
bool TimingHeterostaIO::parse_all_timing_options(int argc, char** argv, 
		std::string& early_lib_path, std::string& late_lib_path,
		std::string& single_lib_path, std::string& sdc_path) {
	typedef limbo::programoptions::ProgramOptions po_type;
	using limbo::programoptions::Value;
	po_type desc(std::string("All HeteroSTA timing options"));

	auto options = std::array<Value<std::string>, 4> { {
		{ "--early_lib_input", &early_lib_path, "input celllib file (early)" },
			{ "--late_lib_input", &late_lib_path, "input celllib file (late)" },
			{ "--lib_input",      &single_lib_path, "input celllib file"        },
			{ "--sdc_input",      &sdc_path,        "input sdc file"            }
	} };
	for (auto val : options)
		desc.add_option(val);

	try {
		desc.parse(argc, argv);
		return true;
	} catch (std::exception& e) {
		dreamplacePrint(kERROR, "Error parsing timing arguments: %s\n", e.what());
		return false;
	}
}

/// @brief Read timing libraries from specified paths
/// @param sta 
/// @param early_lib_path 
/// @param late_lib_path 
/// @param single_lib_path 
/// @return 
bool TimingHeterostaIO::read_liberty_libraries_with_paths(STAHoldings& sta, 
		const std::string& early_lib_path,
		const std::string& late_lib_path, 
		const std::string& single_lib_path) {
	// Single library file for both early and late
	if (!single_lib_path.empty()) {
		if (heterosta_read_liberty(&sta, EARLY, single_lib_path.c_str()) &&
				heterosta_read_liberty(&sta, LATE, single_lib_path.c_str())) {
			return true;
		} else {
			dreamplacePrint(kERROR, "Failed to load Liberty library: %s\n", single_lib_path.c_str());
			return false;
		}
	} 

	// Separate early/late library files
	bool success = true;
	bool has_any_lib = false;

	if (!early_lib_path.empty()) {
		has_any_lib = true;
		if (!heterosta_read_liberty(&sta, EARLY, early_lib_path.c_str())) {
			dreamplacePrint(kERROR, "Failed to load early Liberty library: %s\n", early_lib_path.c_str());
			success = false;
		}
		else {
			dreamplacePrint(kDEBUG, "Early Liberty library loaded: %s\n", early_lib_path.c_str());
		}
	}

	if (!late_lib_path.empty()) {
		has_any_lib = true;
		if (!heterosta_read_liberty(&sta, LATE, late_lib_path.c_str())) {
			dreamplacePrint(kERROR, "Failed to load late Liberty library: %s\n", late_lib_path.c_str());
			success = false;
		}
		else {
			dreamplacePrint(kDEBUG, "Late Liberty library loaded: %s\n", late_lib_path.c_str());
		}
	}

	if (!has_any_lib) {
		dreamplacePrint(kERROR, "No Liberty library specified\n");
		return false;
	}

	return success;
}

/// 
/// @brief Read SDC constraints from specified path
/// @param sta 
/// @param sdc_path 
/// @return 
bool TimingHeterostaIO::read_sdc_constraints_with_path(STAHoldings& sta, const std::string& sdc_path) {
	if (sdc_path.empty()) {
		dreamplacePrint(kWARN, "No SDC file specified - timing analysis will have no constraints\n");
		return true; 
	}

	dreamplacePrint(kINFO, "Reading SDC constraints from: %s\n", sdc_path.c_str());
	if (heterosta_read_sdc(&sta, sdc_path.c_str())) {
		dreamplacePrint(kDEBUG, "SDC file parsing completed successfully\n");
		return true;
	} else {
		dreamplacePrint(kERROR, "SDC parsing failed for file: %s\n", sdc_path.c_str());
		return false;
	}
}

/// 
/// @brief Setup timing engine with PlaceDB and DREAMPlace mappings
/// @param sta 
/// @param placedb 
/// @param argc 
/// @param argv 
/// @param dreamplace_mappings 
/// @return 
bool TimingHeterostaIO::setupTiming(STAHoldings& sta, PlaceDB& placedb, int argc, char** argv, 
		const pybind11::dict& dreamplace_mappings) {
	// Parse all timing-related arguments
	std::string early_lib_path, late_lib_path, single_lib_path, sdc_path;
	if (!parse_all_timing_options(argc, argv, early_lib_path, late_lib_path, single_lib_path, sdc_path)) {
		dreamplacePrint(kERROR, "Failed to parse timing arguments\n");
		return false;
	}

	// Read Liberty libraries
	if (!read_liberty_libraries_with_paths(sta, early_lib_path, late_lib_path, single_lib_path)) {
		dreamplacePrint(kERROR, "Failed to read Liberty libraries\n");
		return false;
	}

	// Build NetlistDB from DREAMPlace mappings instead of rebuilding from PlaceDB
	dreamplacePrint(kINFO, "Using DREAMPlace mappings to build NetlistDB for data consistency\n");
	NetlistDB* netlistdb = build_netlistdb_from_dreamplace(placedb, dreamplace_mappings);
	if (!netlistdb) {
		dreamplacePrint(kERROR, "Failed to create NetlistDB from DREAMPlace mappings\n");
		return false;
	}

	// Setup timer database 
	if (!buildTimerDB(sta, netlistdb)) {
		dreamplacePrint(kERROR, "Failed to build timer database\n");
		return false;
	}

	// Read SDC constraints with enhanced error handling
	if (!read_sdc_constraints_with_path(sta, sdc_path)) {
		dreamplacePrint(kERROR, "SDC constraints reading failed for: %s\n", sdc_path.c_str());
	} else {
		dreamplacePrint(kINFO, "SDC constraints loaded successfully from: %s\n", sdc_path.c_str());
	}

	return true;
}
NetlistDB* TimingHeterostaIO::build_netlistdb_from_dreamplace(PlaceDB& placedb, 
		const pybind11::dict& dreamplace_mappings) {
	dreamplacePrint(kINFO, "Building NetlistDB from DREAMPlace mappings for data consistency\n");

	// Clear previous data
	g_netlist_data.clear();

	try {
		// Extract only the DREAMPlace mappings that are actually used - must match _package_dreamplace_mappings() keys
		auto pin2net_map = dreamplace_mappings["pin2net_map"].cast<torch::Tensor>();
		auto pin2node_map = dreamplace_mappings["pin2node_map"].cast<torch::Tensor>();
		auto pin_direct = dreamplace_mappings["pin_direct"].cast<torch::Tensor>();


		// Convert torch tensors to CPU and get data pointers
		auto pin2net_cpu = pin2net_map.to(torch::kCPU).to(torch::kInt32);
		auto pin2node_cpu = pin2node_map.to(torch::kCPU).to(torch::kInt32);
		auto pin_direct_cpu = pin_direct.to(torch::kCPU).to(torch::kUInt8);

		const int32_t* pin2net_data = DREAMPLACE_TENSOR_DATA_PTR(pin2net_cpu, int32_t);
		const int32_t* pin2node_data = DREAMPLACE_TENSOR_DATA_PTR(pin2node_cpu, int32_t);
		const uint8_t* pin_direct_data = DREAMPLACE_TENSOR_DATA_PTR(pin_direct_cpu, uint8_t);

		size_t num_pins = pin2net_cpu.numel();
		size_t num_nodes = placedb.nodes().size();
		size_t num_nets = placedb.nets().size();

		// Extract num_terminal_NIs from dreamplace_mappings
		auto num_terminal_NIs_tensor = dreamplace_mappings["num_terminal_NIs"].cast<torch::Tensor>();
		size_t num_terminal_NIs = num_terminal_NIs_tensor.item<int32_t>();

		// Analyze PlaceDB structure for IO pin identification
		size_t numMovable, numFixed, numPlaceBlockages, iopinNodeStart;
		numMovable = placedb.numMovable();
		numFixed = placedb.numFixed(); 
		numPlaceBlockages = placedb.numPlaceBlockages();
		if (!analyze_placedb_structure(placedb, numMovable, numFixed, numPlaceBlockages, 
					iopinNodeStart, num_terminal_NIs, num_pins)) {
			dreamplacePrint(kERROR, "Failed to analyze PlaceDB structure\n");
			return nullptr;
		}

		// Setup cell data: Cell 0 = top module
		size_t totalcells = 1 + numMovable + numFixed;
		std::string real_design_name = placedb.designName();
		g_netlist_data.design_name = real_design_name;

		if (!setup_cell_data(placedb, totalcells, real_design_name)) {
			dreamplacePrint(kERROR, "Failed to setup cell data\n");
			return nullptr;
		}

		// Setup pin data using DREAMPlace mappings
		if (!setup_pin_data(placedb, iopinNodeStart, pin2node_data, pin2net_data, pin_direct_data, num_pins)) {
			dreamplacePrint(kERROR, "Failed to setup pin data using DREAMPlace mappings\n");
			return nullptr;
		}

		// Setup net data
		if (!setup_net_data(placedb)) {
			dreamplacePrint(kERROR, "Failed to setup net data\n");
			return nullptr;
		}

		// Create interface arrays
		if (!create_interface_arrays()) {
			dreamplacePrint(kERROR, "Failed to create interface arrays\n");
			return nullptr;
		}


		// Create NetlistDB interface
		NetlistDBCppInterface interface;
		interface.top_design_name = g_netlist_data.design_name.c_str();
		interface.num_cells = totalcells;
		interface.num_pins = g_netlist_data.pin_names.size();
		interface.num_ports = 0; // Will be calculated below
		interface.num_nets = g_netlist_data.net_names.size();
		interface.num_nets_zero = g_netlist_data.nets_zero.size();
		interface.num_nets_one = g_netlist_data.nets_one.size();

		interface.cellname_array = g_netlist_data.cell_name_ptrs.data();
		interface.celltype_array = g_netlist_data.cell_type_ptrs.data();
		interface.pinname_array = g_netlist_data.pin_name_ptrs.data();
		interface.netname_array = g_netlist_data.net_name_ptrs.data();
		interface.pindirection_array = g_netlist_data.pin_directions.data();
		interface.pin2cell_array = g_netlist_data.pin2cell_map.data();
		interface.pin2net_array = g_netlist_data.pin2net_map.data();
		interface.nets_zero_array = g_netlist_data.nets_zero.data();
		interface.nets_one_array = g_netlist_data.nets_one.data();

		// Calculate number of top-level ports
		for (size_t i = 0; i < g_netlist_data.pin2cell_map.size(); ++i) {
			if (g_netlist_data.pin2cell_map[i] == 0) {
				interface.num_ports++;
			}
		}

		NetlistDB* netlistdb = netlistdb_new(&interface);

		if (!netlistdb) {
			dreamplacePrint(kERROR, "Failed to create NetlistDB\n");
			return nullptr;
		}

		dreamplacePrint(kINFO, "NetlistDB created successfully with %zu cells, %zu pins, %zu nets\n",
				totalcells, g_netlist_data.pin_names.size(), g_netlist_data.net_names.size());
		dreamplacePrint(kINFO, "Using DREAMPlace mappings ensures data consistency with placement engine\n");

		return netlistdb;

	} catch (const std::exception& e) {
		dreamplacePrint(kERROR, "Failed to extract DREAMPlace mappings: %s\n", e.what());
		return nullptr;
	}
}
/// @brief Build the timing database for HeteroSTA
/// @param sta The STA holdings object
/// @param netlistdb The NetlistDB object
/// @return True on success, false on failure
bool TimingHeterostaIO::buildTimerDB(STAHoldings& sta, NetlistDB* netlistdb) {
	if (!netlistdb) {
		dreamplacePrint(kERROR, "NetlistDB is null\n");
		return false;
	}

	dreamplacePrint(kINFO, "This is where the flat_parasitics panic may occur...\n");

	// Step 1: Set NetlistDB in HeteroSTA
	heterosta_set_netlistdb(&sta, netlistdb);

	// Step 2: Flatten all hierarchical designs
	heterosta_flatten_all(&sta);

	// Step 3: Set delay calculator
	//heterosta_set_delay_calculator_elmore_scaled(&sta);
	heterosta_set_delay_calculator_elmore(&sta);

	// Step 4: Build timing graph
	heterosta_build_graph(&sta);

	// Step 5: Initialize slew values
	heterosta_zero_slew(&sta);

	// Initialize pin mapping
	size_t num_pins = g_netlist_data.pin_names.size();
	dreamplacePrint(kINFO, " Pin count: %zu\n", num_pins);

	// Step 6: Initialize and identify timing endpoints
	static_assert(sizeof(bool) == 1);
	std::vector<uint8_t> timingpin_is_endpoint;
	timingpin_is_endpoint.resize(num_pins);
	heterosta_get_is_endpoint(&sta, (bool *) timingpin_is_endpoint.data());

	dreamplacePrint(kINFO, "Timer database built successfully\n");
	return true;
}
/// @brief analyze PlaceDB structure to identify IO pin starting index
/// @param placedb 
/// @param numMovable 
/// @param numFixed 
/// @param numPlaceBlockages 
/// @param iopinNodeStart 
/// @param num_terminal_NIs 
/// @param num_pins 
/// @return True on success, false on failure
bool TimingHeterostaIO::analyze_placedb_structure(PlaceDB& placedb, size_t& numMovable, 
		size_t& numFixed, size_t& numPlaceBlockages, 
		size_t& iopinNodeStart, size_t num_terminal_NIs, 
		size_t num_pins) {

	// Validate num_terminal_NIs
	if (num_terminal_NIs > num_pins) {
		dreamplacePrint(kERROR, "Invalid num_terminal_NIs: %zu > num_pins: %zu\n", num_terminal_NIs, num_pins);
		return false;
	}

	// Calculate starting index of IO pins using num_terminal_NIs
	// Last num_terminal_NIs pins are IO pins
	iopinNodeStart = numMovable + numFixed + numPlaceBlockages;

	dreamplacePrint(kINFO, "PlaceDB structure analysis:\n");
	dreamplacePrint(kINFO, " numMovable: %zu\n", numMovable);
	dreamplacePrint(kINFO, " numFixed: %zu\n", numFixed);
	dreamplacePrint(kINFO, " numPlaceBlockages: %zu\n", numPlaceBlockages);
	dreamplacePrint(kINFO, " num_pins: %zu\n", num_pins);
	dreamplacePrint(kINFO, " num_terminal_NIs: %zu\n", num_terminal_NIs);
	dreamplacePrint(kINFO, " iopinNodeStart: %zu (calculated from num_terminal_NIs)\n", iopinNodeStart);

	return true;
}
/// @brief Setup cell data with correct HeteroSTA indexing
/// @param placedb 
/// @param totalcells 
/// @param designName 
/// @return True on success, false on failure

bool TimingHeterostaIO::setup_cell_data(PlaceDB& placedb, size_t totalcells, const std::string& designName) {
	dreamplacePrint(kINFO, "Setting up cell data with correct HeteroSTA indexing...\n");

	g_netlist_data.cell_names.reserve(totalcells);
	g_netlist_data.cell_types.reserve(totalcells);

	// Cell 0: Top module 
	g_netlist_data.cell_names.push_back(""); // Cell 0 name is empty 
	g_netlist_data.cell_types.push_back(designName); // Cell 0 type is design name

	// Cells 1 to N: DREAMPlace nodes (with +1 offset)
	for (size_t i = 0; i < totalcells; ++i) {
		const auto& node = placedb.nodes()[i];
		const auto& node_property = placedb.nodeProperty(i);

		// Cell name: use node name or generate one
		std::string cell_name = node_property.name().empty() ? 
			("cell_" + std::to_string(i)) : node_property.name();
		g_netlist_data.cell_names.push_back(cell_name);

		// Cell type: get macro name
		std::string cell_type = "UNKNOWN";
		if (node_property.macroId() < placedb.macros().size()) {
			cell_type = placedb.macros()[node_property.macroId()].name();
		}
		g_netlist_data.cell_types.push_back(cell_type);
	}

	dreamplacePrint(kINFO, "Cell data setup complete: %zu cells (including top module)\n", totalcells);
	return true;
}

/// @brief Setup pin data using DREAMPlace mappings for consistency
/// @param placedb 
/// @param iopinNodeStart 
/// @param pin2node_data    
/// @param pin2net_data
/// @param pin_direct_data
/// @param num_pins
/// @return True on success, false on failure

bool TimingHeterostaIO::setup_pin_data(PlaceDB& placedb, size_t iopinNodeStart, 
		const int32_t* pin2node_data, const int32_t* pin2net_data, 
		const uint8_t* pin_direct_data, size_t num_pins) {
	dreamplacePrint(kINFO, "Setting up pin data using DREAMPlace mappings\n");

	const auto& pins = placedb.pins();
	const auto& nodes = placedb.nodes();
	// Verify consistency between DREAMPlace and PlaceDB
	if (num_pins != pins.size()) {
		dreamplacePrint(kERROR, "Pin count mismatch: DREAMPlace=%zu, PlaceDB=%zu\n", num_pins, pins.size());
		return false;
	}

	g_netlist_data.pin_names.reserve(num_pins);
	g_netlist_data.pin_directions.reserve(num_pins);
	g_netlist_data.pin2cell_map.reserve(num_pins);
	g_netlist_data.pin2net_map.reserve(num_pins);

	size_t valid_port_count = 0;
	size_t instance_pin_count = 0;

	// Process each pin using DREAMPlace mappings
	for (size_t pin_idx = 0; pin_idx < num_pins; ++pin_idx) {
		const auto& pin = pins[pin_idx];

		// Get DREAMPlace node ID, net ID, and pin direction for this pin
		int32_t dreamplace_node_id = pin2node_data[pin_idx];
		int32_t dreamplace_net_id = pin2net_data[pin_idx];
		uint8_t dreamplace_pin_direct = pin_direct_data[pin_idx];

		// Pin name 
		std::string pin_name;
		bool is_io_pin = (dreamplace_node_id >= 0 && 
				static_cast<size_t>(dreamplace_node_id) >= iopinNodeStart &&
				static_cast<size_t>(dreamplace_node_id) < nodes.size());

		if (is_io_pin && pin.name().length() > 0) {
			pin_name = pin.name();
			valid_port_count++;
			//dreamplacePrint(kDEBUG, "Top-level port detected: pin_idx=%zu, name=%s\n",    
			//              pin_idx, pin_name.c_str());
		} else {
			// This is an instance pin - change ':' to '/'
			pin_name = pin.name();
			std::replace(pin_name.begin(), pin_name.end(), ':', '/');
			//dreamplacePrint(kDEBUG, "Instance pin detected: pin_idx=%zu, name=%s\n",    
			//              pin_idx, pin_name.c_str());
			instance_pin_count++;
		}

		// Pin direction - use DREAMPlace data directly
		// DREAMPlace pin_direct encoding: 0=INPUT, 1=OUTPUT, 2=INOUT
		// HeteroSTA encoding: 0=INPUT, 1=OUTPUT, 2=INOUT (same as DREAMPlace)
		uint8_t direction = dreamplace_pin_direct;

		g_netlist_data.pin_names.push_back(pin_name);
		g_netlist_data.pin_directions.push_back(direction);

		// Convert DREAMPlace node ID to HeteroSTA cell ID
		size_t heterosta_cell_id;
		if (dreamplace_node_id >= 0 && dreamplace_node_id < static_cast<int32_t>(nodes.size())) {
			// Check if this is an IO pin that should map to top module cell 0
			size_t node_id = static_cast<size_t>(dreamplace_node_id);
			if (node_id >= iopinNodeStart && pin.name().length() > 0) {
				heterosta_cell_id = 0; // Top module
			} else {
				heterosta_cell_id = dreamplace_node_id + 1; // +1 offset for HeteroSTA
			}
		} else {
			dreamplacePrint(kWARN, "Invalid node ID %d for pin %zu, using cell ID 0\n", dreamplace_node_id, pin_idx);
			heterosta_cell_id = 0;
		}

		g_netlist_data.pin2cell_map.push_back(heterosta_cell_id);
		g_netlist_data.pin2net_map.push_back(static_cast<size_t>(dreamplace_net_id));
	}
	return true;
}

/// @brief Setup net data
/// @param placedb 
/// @return True on success, false on failure
bool TimingHeterostaIO::setup_net_data(PlaceDB& placedb) {
	dreamplacePrint(kINFO, "Setting up net data...\n");

	const auto& nets = placedb.nets();
	g_netlist_data.net_names.reserve(nets.size());
	g_netlist_data.nets_zero.reserve(0); //zero for temporary use
	g_netlist_data.nets_one.reserve(0); //zero for temporary use

	for (size_t net_idx = 0; net_idx < nets.size(); ++net_idx) {
		const auto& net = nets[net_idx];
		const auto& net_property = placedb.netProperty(net_idx);

		// Net name
		std::string net_name = net_property.name();
		g_netlist_data.net_names.push_back(net_name);

		// Nets zero/one (power/ground nets, typically 0 for normal nets)
		//g_netlist_data.nets_zero.push_back(0);
		//g_netlist_data.nets_one.push_back(0);
	}

	dreamplacePrint(kINFO, "Net data setup complete: %zu nets\n", nets.size());
	return true;
}

/// Create interface arrays for NetlistDB
bool TimingHeterostaIO::create_interface_arrays() {
	// Create const char* arrays for NetlistDB interface
	g_netlist_data.cell_name_ptrs.reserve(g_netlist_data.cell_names.size());
	g_netlist_data.cell_type_ptrs.reserve(g_netlist_data.cell_types.size());
	g_netlist_data.pin_name_ptrs.reserve(g_netlist_data.pin_names.size());
	g_netlist_data.net_name_ptrs.reserve(g_netlist_data.net_names.size());

	for (const auto& name : g_netlist_data.cell_names) {
		g_netlist_data.cell_name_ptrs.push_back(name.c_str());
	}
	for (const auto& type : g_netlist_data.cell_types) {
		g_netlist_data.cell_type_ptrs.push_back(type.c_str());
	}
	for (const auto& name : g_netlist_data.pin_names) {
		g_netlist_data.pin_name_ptrs.push_back(name.c_str());
	}
	for (const auto& name : g_netlist_data.net_names) {
		g_netlist_data.net_name_ptrs.push_back(name.c_str());
	}

	return true;
}

// A hardcoded fallback license..
const char* hardcode_lic = "";

/// @brief initialize HeteroSTA instance with logging callback
TimingHeterostaIO::STAHoldingsPtr TimingHeterostaIO::initialize_heterosta() {

	dreamplacePrint(kINFO, "HeteroSTA instance created successfully\n");
	heterosta_init_logger(dreamplace_heterosta_print_callback);
    const char* lic = std::getenv("HeteroSTA_Lic");
    bool have_env = (lic != nullptr) && (lic[0] != '\0');
    if (have_env) {
        dreamplacePrint(kINFO, "Successfully loaded license from 'HeteroSTA_Lic' environment variable.\n");
    } else if (hardcode_lic != nullptr && hardcode_lic[0] != '\0') {
        lic = hardcode_lic;
        dreamplacePrint(kWARN, "'HeteroSTA_Lic' environment variable not found. Using hardcoded license.\n");
    } else {
        dreamplacePrint(kERROR, "License not found.\n");
        dreamplacePrint(kERROR, "Neither 'HeteroSTA_Lic' environment variable nor hardcoded license is provided.\n");
        dreamplacePrint(kINFO,  "Please set one of the following before running, e.g.:\n");
        dreamplacePrint(kINFO,  "  - export HeteroSTA_Lic=\"<your-license-string>\"\n");
        dreamplacePrint(kINFO,  "  - or set 'hardcode_lic' in timing_hs_io_cpp.cpp to your license string.\n");
        std::exit(EXIT_FAILURE);
    }

	heterosta_init_license(lic);

	auto sta = STAHoldingsPtr(heterosta_new(), heterosta_free);
	if (!sta) {
		dreamplacePrint(kERROR, "Failed to create HeteroSTA instance\n");
		return STAHoldingsPtr(nullptr, heterosta_free);
	}
	
	dreamplacePrint(kINFO, "HeteroSTA initialization completed\n");

	return sta;
}



// Implementation of pin name accessor functions
const char* TimingHeterostaIO::getPinName(size_t pin_index) {
	if (pin_index < g_netlist_data.pin_names.size()) {
		return g_netlist_data.pin_names[pin_index].c_str();
	}
	return "unknown";
}

size_t TimingHeterostaIO::getPinCount() {
	return g_netlist_data.pin_names.size();
}

DREAMPLACE_END_NAMESPACE
