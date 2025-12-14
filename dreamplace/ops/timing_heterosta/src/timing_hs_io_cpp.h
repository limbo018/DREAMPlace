#ifndef DREAMPLACE_TIMING_HS_IO_CPP_H_
#define DREAMPLACE_TIMING_HS_IO_CPP_H_

#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <pybind11/pybind11.h>
#include "utility/src/namespace.h"
#include "utility/src/torch.h"
#include "netlistdb.h"
#include "heterosta.h"
#include "place_io/src/PlaceDB.h"

DREAMPLACE_BEGIN_NAMESPACE

class TimingHeterostaIO {
public:
    // RAII wrapper for STAHoldings with custom deleter
    using STAHoldingsPtr = std::unique_ptr<STAHoldings, void(*)(STAHoldings*)>;

    ///
    /// @brief Main timing setup function 
    /// @param sta reference to STAHoldings instance 
    /// @param placedb the placement database containing netlist data
    /// @param argc the total number of program options
    /// @param argv the strings of program options
    /// @param dreamplace_mappings packaged DREAMPlace mappings to ensure data consistency
    /// @return true if successful, false otherwise
    ///
    static bool setupTiming(STAHoldings& sta, PlaceDB& placedb, int argc, char** argv, 
                           const pybind11::dict& dreamplace_mappings);

    ///
    /// @brief Create NetlistDB from DREAMPlace mappings instead of rebuilding from PlaceDB
    /// @param placedb the placement database containing netlist data
    /// @param dreamplace_mappings packaged DREAMPlace mappings
    /// @return pointer to NetlistDB or nullptr if failed
    ///
    static NetlistDB* build_netlistdb_from_dreamplace(PlaceDB& placedb, 
                                                             const pybind11::dict& dreamplace_mappings);

    ///
    /// @brief Initialize HeteroSTA timer instance
    /// @return unique pointer to STAHoldings with RAII management
    ///
    static STAHoldingsPtr initialize_heterosta();

    ///
    /// @brief Read Liberty library files
    /// @param sta reference to STAHoldings instance
    /// @param argc the total number of program options
    /// @param argv the strings of program options
    /// @return true if successful, false otherwise
    ///
    static bool read_liberty_libraries(STAHoldings& sta, int argc, char** argv);

    ///
    /// @brief Read SDC constraint file with specific path
    /// @param sta reference to STAHoldings instance
    /// @param sdc_path path to SDC file
    /// @return true if successful, false otherwise
    ///
    static bool read_sdc_constraints_with_path(STAHoldings& sta, const std::string& sdc_path);

    ///
    /// @brief Read Liberty library files with specific paths
    /// @param sta reference to STAHoldings instance
    /// @param early_lib_path path to early Liberty file
    /// @param late_lib_path path to late Liberty file 
    /// @param single_lib_path path to single Liberty file for both early/late
    /// @return true if successful, false otherwise
    ///
    static bool read_liberty_libraries_with_paths(STAHoldings& sta, 
                                                 const std::string& early_lib_path,
                                                 const std::string& late_lib_path, 
                                                 const std::string& single_lib_path);

    ///
    /// @brief Get pin name by index (safe accessor for g_netlist_data.pin_names)
    /// @param pin_index the pin index
    /// @return pin name or "unknown" if index out of bounds
    ///
    static const char* getPinName(size_t pin_index);

    ///
    /// @brief Get number of pins (safe accessor for g_netlist_data.pin_names.size())
    /// @return number of pins in the netlist data
    ///
    static size_t getPinCount();

private:
    ///
    /// @brief Parse all timing-related command line arguments
    /// @param argc the total number of program options
    /// @param argv the strings of program options
    /// @param early_lib_path output early Liberty library path
    /// @param late_lib_path output late Liberty library path
    /// @param single_lib_path output single Liberty library path
    /// @param sdc_path output SDC file path
    /// @return true if successful, false otherwise
    ///
    static bool parse_all_timing_options(int argc, char** argv, 
                                        std::string& early_lib_path, std::string& late_lib_path,
                                        std::string& single_lib_path, std::string& sdc_path);


    ///
    /// @brief Build timer database 
    /// @param sta reference to STAHoldings instance
    /// @param netlistdb the NetlistDB to be set in HeteroSTA
    /// @return true if successful, false otherwise
    ///
    static bool buildTimerDB(STAHoldings& sta, NetlistDB* netlistdb);

    // Endpoint information storage 
    static std::vector<uint8_t> timingpin_is_endpoint;

private:
    // Data structure for persistent NetlistDB interface data
    struct NetlistDataStorage {
        std::vector<std::string> cell_names;
        std::vector<std::string> cell_types; 
        std::vector<std::string> pin_names;
        std::vector<std::string> net_names;
        std::vector<const char*> cell_name_ptrs;
        std::vector<const char*> cell_type_ptrs;
        std::vector<const char*> pin_name_ptrs;
        std::vector<const char*> net_name_ptrs;
        std::vector<uint8_t> pin_directions;
        std::vector<uintptr_t> pin2cell_map;
        std::vector<uintptr_t> pin2net_map;
        std::vector<uintptr_t> nets_zero;
        std::vector<uintptr_t> nets_one;
        std::string design_name;
        
        void clear();
    };

    // Global storage to ensure data persistence during NetlistDB lifetime
    static NetlistDataStorage g_netlist_data;

    ///
    /// @brief Extract and validate PlaceDB structure
    /// @param placedb placement database to analyze
    /// @param numMovable output: number of movable cells
    /// @param numFixed output: number of fixed cells 
    /// @param numPlaceBlockages output: number of placement blockages
    /// @param iopinNodeStart output: starting index of IO pin nodes
    /// @return true if valid structure, false otherwise
    ///
    static bool analyze_placedb_structure(PlaceDB& placedb, size_t& numMovable, 
                                        size_t& numFixed, size_t& numPlaceBlockages, 
                                        size_t& iopinNodeStart, size_t num_terminal_NIs, 
                                        size_t num_pins);

    ///
    /// @brief Setup cell data in NetlistDB interface (with correct indexing)
    /// @param placedb placement database 
    /// @param totalNodes total number of nodes
    /// @return true if successful, false otherwise
    ///
    static bool setup_cell_data(PlaceDB& placedb, size_t totalNodes, const std::string& designName);

    ///
    /// @brief Setup pin data and mappings using DREAMPlace mappings
    /// @param placedb placement database
    /// @param iopinNodeStart starting index of IO pin nodes
    /// @param pin2node_data pointer to DREAMPlace pin2node mapping data
    /// @param pin2net_data pointer to DREAMPlace pin2net mapping data
    /// @param pin_direct_data pointer to DREAMPlace pin direction data
    /// @param num_pins number of pins to process
    /// @return true if successful, false otherwise
    ///
    static bool setup_pin_data(PlaceDB& placedb, size_t iopinNodeStart, 
                             const int32_t* pin2node_data, const int32_t* pin2net_data, 
                             const uint8_t* pin_direct_data, size_t num_pins);

    ///
    /// @brief Setup net data
    /// @param placedb placement database
    /// @return true if successful, false otherwise 
    ///
    static bool setup_net_data(PlaceDB& placedb);

    ///
    /// @brief Create interface pointer arrays for NetlistDB
    /// @return true if successful, false otherwise
    ///
    static bool create_interface_arrays();


};

DREAMPLACE_END_NAMESPACE

#endif // DREAMPLACE_TIMING_HS_IO_CPP_H_
