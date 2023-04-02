/**
 * @file   PyPlaceDB.h
 * @author Yibo Lin
 * @date   Apr 2020
 * @brief  Placement database for python 
 */

#ifndef _DREAMPLACE_PLACE_IO_PYPLACEDB_H
#define _DREAMPLACE_PLACE_IO_PYPLACEDB_H

//#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#include <sstream>
//#include <boost/timer/timer.hpp>
#include "PlaceDB.h"
#include "Iterators.h"
#include "utility/src/torch.h"

DREAMPLACE_BEGIN_NAMESPACE

bool readLef(PlaceDB& db);
bool readDef(PlaceDB& db);
void prereadDef(PlaceDB& db, std::string const& filename);
bool readVerilog(PlaceDB& db);
bool readBookshelf(PlaceDB& db);

/// database for python 
struct PyPlaceDB
{
    typedef PlaceDB::coordinate_type coordinate_type; 
    typedef PlaceDB::index_type index_type; 

    unsigned int num_nodes; ///< number of nodes, including terminals and terminal_NIs 
    unsigned int num_terminals; ///< number of terminals, essentially fixed macros  
    unsigned int num_terminal_NIs; ///< number of terminal_NIs, essentially IO pins 
    pybind11::dict node_name2id_map; ///< node name to id map, cell name 
    pybind11::list node_names; ///< 1D array, cell name 
    pybind11::list node_x; ///< 1D array, cell position x 
    pybind11::list node_y; ///< 1D array, cell position y 
    pybind11::list node_orient; ///< 1D array, cell orientation 
    pybind11::list node_size_x; ///< 1D array, cell width  
    pybind11::list node_size_y; ///< 1D array, cell height

    pybind11::list node2orig_node_map; ///< due to some fixed nodes may have non-rectangular shapes, we flat the node list; 
                                        ///< this map maps the new indices back to the original ones 

    pybind11::list pin_direct; ///< 1D array, pin direction IO 
    pybind11::list pin_offset_x; ///< 1D array, pin offset x to its node 
    pybind11::list pin_offset_y; ///< 1D array, pin offset y to its node 
    pybind11::list pin_names; ///< pin name

    pybind11::dict net_name2id_map; ///< net name to id map
    pybind11::dict pin_name2id_map; ///< pin name to id map
    pybind11::list net_names; ///< net name 
    pybind11::list net2pin_map; ///< array of 1D array, each row stores pin id
    pybind11::list flat_net2pin_map; ///< flatten version of net2pin_map 
    pybind11::list flat_net2pin_start_map; ///< starting index of each net in flat_net2pin_map
    pybind11::list net_weights; ///< net weight 
    pybind11::list net_weight_deltas; ///< net weight deltas
    pybind11::list net_criticality; ///< net criticality
    pybind11::list net_criticality_deltas; ///< net criticality deltas

    pybind11::list node2pin_map; ///< array of 1D array, contains pin id of each node 
    pybind11::list flat_node2pin_map; ///< flatten version of node2pin_map 
    pybind11::list flat_node2pin_start_map; ///< starting index of each node in flat_node2pin_map

    pybind11::list pin2node_map; ///< 1D array, contain parent node id of each pin 
    pybind11::list pin2net_map; ///< 1D array, contain parent net id of each pin 

    pybind11::list rows; ///< NumRows x 4 array, stores xl, yl, xh, yh of each row 

    pybind11::list regions; ///< array of 1D array, each region contains rectangles 
    pybind11::list flat_region_boxes; ///< flatten version of regions 
    pybind11::list flat_region_boxes_start; ///< starting index of each region in flat_region_boxes

    pybind11::list node2fence_region_map; ///< only record fence regions for each cell 

    unsigned int num_routing_grids_x; ///< number of routing grids in x 
    unsigned int num_routing_grids_y; ///< number of routing grids in y 
    int routing_grid_xl; ///< routing grid region may be different from placement region 
    int routing_grid_yl; 
    int routing_grid_xh; 
    int routing_grid_yh;
    pybind11::list unit_horizontal_capacities; ///< number of horizontal tracks of layers per unit distance 
    pybind11::list unit_vertical_capacities; /// number of vertical tracks of layers per unit distance 
    pybind11::list initial_horizontal_demand_map; ///< initial routing demand from fixed cells, indexed by (layer, grid x, grid y) 
    pybind11::list initial_vertical_demand_map; ///< initial routing demand from fixed cells, indexed by (layer, grid x, grid y)   

    int xl; 
    int yl; 
    int xh; 
    int yh; 

    int row_height;
    int site_width;
    double total_space_area; ///< total placeable space area excluding fixed cells. 
                            ///< This is not the exact area, because we cannot exclude the overlapping fixed cells within a bin. 

    int num_movable_pins; 

    PyPlaceDB()
    {
    }

    PyPlaceDB(PlaceDB const& db)
    {
        set(db); 
    }

    void set(PlaceDB const& db);

    /// @brief Convert orientation to (degree, flip) pair. 
    /// N, S, W, E correspond to degree = 0, 180, 90, 270, flip = 0; 
    /// FN, FS, FW, FE correspond to degree = 0, 180, 90, 270, flip = 1. 
    /// The operation is rotation and then flipping. 
    /// Note flip means flipping about Y axis.  
    std::pair<int32_t, int32_t> getOrientDegreeFlip(std::string const& orient) const; 

    /// @brief Get rotated width and height. 
    std::pair<coordinate_type, coordinate_type> getRotatedSizes(int32_t rot_degree, coordinate_type src_width, coordinate_type src_height) const; 
    /// @brief Get rotated pin offsets. 
    std::pair<coordinate_type, coordinate_type> getRotatedPinOffsets(int32_t rot_degree, 
        coordinate_type src_width, coordinate_type src_height, 
        coordinate_type src_pin_offset_x, coordinate_type src_pin_offset_y) const; 
    /// @brief Get flipped pin offsets about Y axis. 
    std::pair<coordinate_type, coordinate_type> getFlipYPinOffsets(coordinate_type src_width, coordinate_type src_height, 
        coordinate_type src_pin_offset_x, coordinate_type src_pin_offset_y) const; 

    /// @brief Top function to convert orientations of all nodes to N 
    /// and change the width, height, and pin offsets accordingly. 
    /// Note lower left corner dost noe change.  
    void convertOrient(); 

    void computeAreaStatistics();
};

DREAMPLACE_END_NAMESPACE

#endif
