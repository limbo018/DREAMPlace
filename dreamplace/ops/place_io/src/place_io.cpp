/*************************************************************************
    > File Name: place_io.cpp
    > Author: Yibo Lin
    > Mail: yibolin@utexas.edu
    > Created Time: Thu Jun 18 23:08:28 2015
 ************************************************************************/

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

bool readLef(PlaceDB& db)
{
	// read lef 
    std::vector<std::string> const& vLefInput = db.userParam().vLefInput;
    for (std::vector<std::string>::const_iterator it = vLefInput.begin(), ite = vLefInput.end(); it != ite; ++it)
    {
        std::string const& filename = *it;
        dreamplacePrint(kINFO, "reading %s\n", filename.c_str());
        bool flag = LefParser::read(db, filename);
        if (!flag) 
        {
            dreamplacePrint(kERROR, "LEF file parsing failed: %s\n", filename.c_str());
            return false;
        }
    }

	return true;
}

bool readDef(PlaceDB& db)
{
	// read def 
    std::string const& defInput = db.userParam().defInput;
    if (!defInput.empty())
    {
        std::string const& filename = defInput;
        dreamplacePrint(kINFO, "reading %s\n", filename.c_str());
        // a pre-reading phase to grep number of components, nets, and pins 
        prereadDef(db, filename);
        bool flag = DefParser::read(db, filename);
        if (!flag) 
        {
            dreamplacePrint(kERROR, "DEF file parsing failed: %s\n", filename.c_str());
            return false;
        }
    }
	else dreamplacePrint(kWARN, "no DEF file specified\n");

	return true;
}

void prereadDef(PlaceDB& db, std::string const& filename)
{
    std::ifstream inFile (filename.c_str());
    if (!inFile.good())
        return;

    // need to extract following information 
    unsigned numRows = 0;
    unsigned numNodes = 0;
    unsigned numIOPin = 0;
    unsigned numNets = 0;
    unsigned numBlockages = 0;

    std::string line;
    std::string token;
    while (!inFile.eof())
    {
        std::getline(inFile, line);
        if (line.compare(0, 3, "ROW") == 0) // a line starts with keyword "ROW"
            ++numRows;
        else if (line.compare(0, 10, "COMPONENTS") == 0)
        {
            std::istringstream iss (line);
            iss >> token >> numNodes >> token;
        }
        else if (line.compare(0, 4, "PINS") == 0)
        {
            std::istringstream iss (line);
            iss >> token >> numIOPin >> token;
        }
        else if (line.compare(0, 4, "NETS") == 0)
        {
            std::istringstream iss (line);
            iss >> token >> numNets >> token;
        }
        else if (line.compare(0, 9, "BLOCKAGES") == 0)
        {
            std::istringstream iss (line);
            iss >> token >> numBlockages >> token;
        }
    }

    dreamplacePrint(kINFO, "detect %u rows, %u components, %u IO pins, %u nets, %u blockages\n", numRows, numNodes, numIOPin, numNets, numBlockages);
    db.prepare(numRows, numNodes, numIOPin, numNets, numBlockages);

    inFile.close();
}

bool readVerilog(PlaceDB& db)
{
    // read verilog 
    std::string const& verilogInput = db.userParam().verilogInput;
    if (!verilogInput.empty())
    {
        std::string const& filename = verilogInput;
        dreamplacePrint(kINFO, "reading %s\n", filename.c_str());
        bool flag = VerilogParser::read(db, filename);
        if (!flag)
        {
            dreamplacePrint(kERROR, "Verilog file parsing failed: %s\n", filename.c_str());
            return false;
        }
    }
    else dreamplacePrint(kWARN, "no Verilog file specified\n");

    return true;
}

bool readBookshelf(PlaceDB& db)
{
    // read bookshelf 
    std::string const& bookshelfAuxInput = db.userParam().bookshelfAuxInput;
    if (!bookshelfAuxInput.empty())
    {
        std::string const& filename = bookshelfAuxInput;
        dreamplacePrint(kINFO, "reading %s\n", filename.c_str());
        bool flag = BookshelfParser::read(db, filename);
        if (!flag)
        {
            dreamplacePrint(kERROR, "Bookshelf file parsing failed: %s\n", filename.c_str());
            return false;
        }
    }
    else dreamplacePrint(kWARN, "no Bookshelf file specified\n");

    // read additional .pl file 
    std::string const& bookshelfPlInput = db.userParam().bookshelfPlInput;
    if (!bookshelfPlInput.empty())
    {
        std::string const& filename = bookshelfPlInput;
        dreamplacePrint(kINFO, "reading %s\n", filename.c_str());
        bool flag = BookshelfParser::readPl(db, filename);
        if (!flag)
        {
            dreamplacePrint(kERROR, "Bookshelf additional .pl file parsing failed: %s\n", filename.c_str());
            return false;
        }
    }
    else dreamplacePrint(kWARN, "no additional Bookshelf .pl file specified\n");

    return true;
}

/// take numpy array 
template <typename T>
bool write(PlaceDB const& db, 
        std::string const& filename, SolutionFileFormat ff, 
        pybind11::array_t<T, pybind11::array::c_style | pybind11::array::forcecast> const& x, 
        pybind11::array_t<T, pybind11::array::c_style | pybind11::array::forcecast> const& y 
        )
{
    PlaceDB::coordinate_type* vx = NULL; 
    PlaceDB::coordinate_type* vy = NULL; 

    // assume all the movable nodes are in front of fixed nodes 
    // this is ensured by PlaceDB::sortNodeByPlaceStatus()
    PlaceDB::index_type lenx = x.size(); 
    if (lenx >= db.numMovable())
    {
        vx = new PlaceDB::coordinate_type [lenx];
        for (PlaceDB::index_type i = 0; i < lenx; ++i)
        {
            vx[i] = x.at(i); 
        }
    }
    PlaceDB::index_type leny = y.size(); 
    if (leny >= db.numMovable())
    {
        vy = new PlaceDB::coordinate_type [leny];
        for (PlaceDB::index_type i = 0; i < leny; ++i)
        {
            vy[i] = y.at(i); 
        }
    }

    bool flag = db.write(filename, ff, vx, vy);

    if (vx)
    {
        delete [] vx; 
    }
    if (vy)
    {
        delete [] vy; 
    }

    return flag; 
}

/// take numpy array 
template <typename T>
void apply(PlaceDB& db, 
        pybind11::array_t<T, pybind11::array::c_style | pybind11::array::forcecast> const& x, 
        pybind11::array_t<T, pybind11::array::c_style | pybind11::array::forcecast> const& y 
        )
{
    // assume all the movable nodes are in front of fixed nodes 
    // this is ensured by PlaceDB::sortNodeByPlaceStatus()
    for (auto& node : db.nodes())
    {
        if (node.status() != PlaceStatusEnum::FIXED)
        {
            PlaceDB::coordinate_type xx = x.at(node.id()); 
            PlaceDB::coordinate_type yy = y.at(node.id()); 
            moveTo(node, xx, yy);

            // update place status 
            node.setStatus(PlaceStatusEnum::PLACED); 
            // update orient 
            auto rowId = db.getRowIndex(node.yl());
            auto const& row = db.row(rowId); 
            if (node.orient() == OrientEnum::UNKNOWN)
            {
                node.setOrient(row.orient()); 
            }
            else 
            {
                if (row.orient() == Orient::vflip(node.orient())) // only vertically flipped
                {
                    node.setOrient(row.orient()); 
                }
                else if (row.orient() == Orient::hflip(Orient::vflip(node.orient()))) // both vertically and horizontally flipped
                {
                    // flip vertically 
                    node.setOrient(Orient::vflip(node.orient()));
                }
                // other cases, no need to change 
            }
        }
    }
}

/// database for python 
struct PyPlaceDB
{
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

    pybind11::dict net_name2id_map; ///< net name to id map
    pybind11::list net_names; ///< net name 
    pybind11::list net2pin_map; ///< array of 1D array, each row stores pin id
    pybind11::list flat_net2pin_map; ///< flatten version of net2pin_map 
    pybind11::list flat_net2pin_start_map; ///< starting index of each net in flat_net2pin_map
    pybind11::list net_weights; ///< net weight 

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

    int num_movable_pins; 

    PyPlaceDB()
    {
    }

    PyPlaceDB(PlaceDB const& db)
    {
        set(db); 
    }

    void set(PlaceDB const& db)
    {
        num_terminal_NIs = db.numIOPin(); // IO pins 
        // record original node to new node mapping 
        std::vector<std::vector<PlaceDB::index_type> > mNode2NewNodes (db.nodes().size()); 

        // general add a node 
        auto addNode = [&](Node const& node, 
                std::string const& name, Orient orient, 
                Box<PlaceDB::coordinate_type> const& box) {
            // this id may be different from node id 
            int id = node_names.size(); 
            node_name2id_map[pybind11::str(name)] = id; 
            node_names.append(pybind11::str(name)); 
            node_x.append(box.xl()); 
            node_y.append(box.yl()); 
            node_orient.append(pybind11::str(std::string(orient))); 
            node_size_x.append(box.width()); 
            node_size_y.append(box.height()); 
            // map new node to original index 
            node2orig_node_map.append(node.id()); 
            // record original node to new node mapping 
            mNode2NewNodes.at(node.id()).push_back(id); 
        };
        // add obstruction boxes for fixed nodes 
        // initialize node shapes from obstruction 
        // I do not differentiate obstruction boxes at different layers
        // At least, this is true for DAC/ICCAD 2012 benchmarks 
        auto addObsBoxes = [&](Node const& node, std::vector<Box<PlaceDB::coordinate_type> > const& vBox){
            for (PlaceDB::index_type i = 0; i < vBox.size(); ++i)
            {
                auto box = vBox[i]; 
                box.set(box.xl() + node.xl(), 
                        box.yl() + node.yl(), 
                        box.xh() + node.xl(), 
                        box.yh() + node.yl()
                       );
                char buf[128]; 
                dreamplaceSPrint(kNONE, buf, "%s.DREAMPlace.Shape%u", db.nodeName(node).c_str(), i); 
                addNode(node, std::string(buf), Orient(node.orient()), box);
            }
        };

        num_terminals = 0; // regard both fixed macros and placement blockages as macros 
        for (unsigned int i = 0; i < db.nodes().size(); ++i)
        {
            Node const& node = db.node(i); 
            if (node.status() != PlaceStatusEnum::FIXED || i >= db.nodes().size() - num_terminal_NIs)
            {
                addNode(node, db.nodeName(node), Orient(node.orient()), node); 
            }
            else // fixed cells are special cases 
            {
                Macro const& macro = db.macro(db.macroId(node));

                if (!macro.obs().empty())
                {
                    MacroObs::ObsConstIterator foundObs = macro.obs().obsMap().find("Bookshelf.Shape"); 
                    if (foundObs != macro.obs().end()) // for BOOKSHELF
                    {
                        addObsBoxes(node, foundObs->second); 
                        num_terminals += foundObs->second.size(); 
                    }
                    else 
                    {
                        for (auto it = macro.obs().begin(), ite = macro.obs().end(); it != ite; ++it)
                        {
                            addObsBoxes(node, it->second); 
                            num_terminals += it->second.size(); 
                        }
                    }
                }
                else 
                {
                    addNode(node, db.nodeName(node), Orient(node.orient()), node); 
                    num_terminals += 1; 
                }
            }
        }
        dreamplacePrint(kINFO, "convert %u fixed cells into %u ones due to some non-rectangular cells\n", db.numFixed(), num_terminals); 
        // we only know num_nodes when all fixed cells with shapes are expanded 
        num_nodes = db.nodes().size() + num_terminals - db.numFixed(); 
        dreamplaceAssertMsg(num_nodes == node_x.size(), "%u != %lu", num_nodes, node_x.size());

        // construct node2pin_map and flat_node2pin_map
        int count = 0; 
        for (unsigned int i = 0; i < mNode2NewNodes.size(); ++i)
        {
            Node const& node = db.node(i);
            for (unsigned int j = 0; j < mNode2NewNodes.at(i).size(); ++j)
            {
                pybind11::list pins; 
                if (j == 0) // for fixed macros with multiple boxes, put all pins to the first one 
                {
                    for (auto pin_id : node.pins())
                    {
                        pins.append(pin_id); 
                        flat_node2pin_map.append(pin_id); 
                    }
                }
                node2pin_map.append(pins); 
                flat_node2pin_start_map.append(count); 
                if (j == 0) // for fixed macros with multiple boxes, put all pins to the first one 
                {
                    count += node.pins().size(); 
                }
            }
        }
        flat_node2pin_start_map.append(count); 

        num_movable_pins = 0; 
        for (unsigned int i = 0, ie = db.pins().size(); i < ie; ++i)
        {
            Pin const& pin = db.pin(i); 
            Node const& node = db.getNode(pin); 
            pin_direct.append(std::string(pin.direct())); 
            // for fixed macros with multiple boxes, put all pins to the first one 
            PlaceDB::index_type new_node_id = mNode2NewNodes.at(node.id()).at(0); 
            Pin::point_type pin_pos (node.pinPos(pin));
            pin_offset_x.append(pin_pos.x() - node_x[new_node_id].cast<PlaceDB::coordinate_type>()); 
            pin_offset_y.append(pin_pos.y() - node_y[new_node_id].cast<PlaceDB::coordinate_type>()); 
            pin2node_map.append(new_node_id); 
            pin2net_map.append(db.getNet(pin).id()); 

            if (node.status() != PlaceStatusEnum::FIXED /*&& node.status() != PlaceStatusEnum::DUMMY_FIXED*/)
            {
                num_movable_pins += 1; 
            }
        }
        count = 0; 
        for (unsigned int i = 0, ie = db.nets().size(); i < ie; ++i)
        {
            Net const& net = db.net(i); 
            net_weights.append(net.weight());
            net_name2id_map[pybind11::str(db.netName(net))] = net.id(); 
            net_names.append(pybind11::str(db.netName(net))); 
            pybind11::list pins; 
            for (std::vector<Net::index_type>::const_iterator it = net.pins().begin(), ite = net.pins().end(); it != ite; ++it)
            {
                pins.append(*it);
            }
            net2pin_map.append(pins); 

            for (std::vector<Net::index_type>::const_iterator it = net.pins().begin(), ite = net.pins().end(); it != ite; ++it)
            {
                flat_net2pin_map.append(*it); 
            }
            flat_net2pin_start_map.append(count); 
            count += net.pins().size(); 
        }
        flat_net2pin_start_map.append(count); 

        for (std::vector<Row>::const_iterator it = db.rows().begin(), ite = db.rows().end(); it != ite; ++it)
        {
            pybind11::tuple row = pybind11::make_tuple(it->xl(), it->yl(), it->xh(), it->yh()); 
            rows.append(row); 
        }

        // initialize regions 
        count = 0; 
        for (std::vector<Region>::const_iterator it = db.regions().begin(), ite = db.regions().end(); it != ite; ++it)
        {
            Region const& region = *it; 
            pybind11::list boxes; 
            for (std::vector<Region::box_type>::const_iterator itb = region.boxes().begin(), itbe = region.boxes().end(); itb != itbe; ++itb)
            {
                pybind11::tuple box = pybind11::make_tuple(itb->xl(), itb->yl(), itb->xh(), itb->yh());
                boxes.append(box);
                flat_region_boxes.append(box);
            }
            regions.append(boxes);
            flat_region_boxes_start.append(count); 
            count += region.boxes().size();
        }
        flat_region_boxes_start.append(count); 

        // I assume one cell only belongs to one FENCE region 
        std::vector<int> vNode2FenceRegion (db.numMovable() + db.numFixed(), std::numeric_limits<int>::max()); 
        for (std::vector<Group>::const_iterator it = db.groups().begin(), ite = db.groups().end(); it != ite; ++it)
        {
            Group const& group = *it; 
            Region const& region = db.region(group.region());
            if (region.type() == RegionTypeEnum::FENCE)
            {
                for (std::vector<Group::index_type>::const_iterator itn = group.nodes().begin(), itne = group.nodes().end(); itn != itne; ++itn)
                {
                    Group::index_type node_id = *itn; 
                    if (db.node(node_id).status() != PlaceStatusEnum::FIXED) // ignore fixed cells 
                    {
                        vNode2FenceRegion.at(node_id) = region.id();
                    }
                }
            }
        }
        for (std::vector<int>::const_iterator it = vNode2FenceRegion.begin(), ite = vNode2FenceRegion.end(); it != ite; ++it)
        {
            node2fence_region_map.append(*it); 
        }

        xl = db.rowXL(); 
        yl = db.rowYL(); 
        xh = db.rowXH(); 
        yh = db.rowYH(); 

        row_height = db.rowHeight(); 
        site_width = db.siteWidth(); 

        // routing information initialized 
        num_routing_grids_x = 0; 
        num_routing_grids_y = 0; 
        routing_grid_xl = xl; 
        routing_grid_yl = yl; 
        routing_grid_xh = xh; 
        routing_grid_yh = yh; 
        if (!db.routingCapacity(PlanarDirectEnum::HORIZONTAL).empty())
        {
            num_routing_grids_x = db.numRoutingGrids(kX); 
            num_routing_grids_y = db.numRoutingGrids(kY);
            routing_grid_xl = db.routingGridOrigin(kX);
            routing_grid_yl = db.routingGridOrigin(kY); 
            routing_grid_xh = routing_grid_xl + num_routing_grids_x * db.routingTileSize(kX); 
            routing_grid_yh = routing_grid_yl + num_routing_grids_y * db.routingTileSize(kY); 
            for (PlaceDB::index_type layer = 0; layer < db.numRoutingLayers(); ++layer)
            {
                unit_horizontal_capacities.append((double)db.numRoutingTracks(PlanarDirectEnum::HORIZONTAL, layer) / db.routingTileSize(kY)); 
                unit_vertical_capacities.append((double)db.numRoutingTracks(PlanarDirectEnum::VERTICAL, layer) / db.routingTileSize(kX));
            }
            // this is slightly different from db.routingGridOrigin 
            // to be consistent with global placement 
            double routing_grid_size_x = db.routingTileSize(kX); 
            double routing_grid_size_y = db.routingTileSize(kY);
            double routing_grid_area = routing_grid_size_x * routing_grid_size_y;
            std::vector<int> initial_horizontal_routing_map (db.numRoutingLayers() * num_routing_grids_x * num_routing_grids_y, 0); 
            std::vector<int> initial_vertical_routing_map (initial_horizontal_routing_map.size(), 0); 
            for (FixedNodeConstIterator it = db.fixedNodeBegin(); it.inRange(); ++it)
            {
                Node const& node = *it; 
                Macro const& macro = db.macro(db.macroId(node)); 

                for (MacroObs::ObsConstIterator ito = macro.obs().begin(); ito != macro.obs().end(); ++ito)
                {
                    if (ito->first != "Bookshelf.Shape") // skip dummy layer for BOOKSHELF 
                    {
                        std::string const& layerName = ito->first; 
                        PlaceDB::index_type layer = db.getLayer(layerName); 
                        for (auto const& obs_box : ito->second)
                        {
                            // convert to absolute box 
                            MacroObs::box_type box (node.xl() + obs_box.xl(), node.yl() + obs_box.yl(), 
                                    node.xl() + obs_box.xh(), node.yl() + obs_box.yh());
                            PlaceDB::index_type grid_index_xl = std::max(int((box.xl() - db.routingGridOrigin(kX)) / routing_grid_size_x), 0); 
                            PlaceDB::index_type grid_index_yl = std::max(int((box.yl() - db.routingGridOrigin(kY)) / routing_grid_size_y), 0);
                            PlaceDB::index_type grid_index_xh = std::min(unsigned((box.xh() - db.routingGridOrigin(kX)) / routing_grid_size_x) + 1, num_routing_grids_x); 
                            PlaceDB::index_type grid_index_yh = std::min(unsigned((box.yh() - db.routingGridOrigin(kY)) / routing_grid_size_y) + 1, num_routing_grids_y); 
                            for (PlaceDB::index_type k = grid_index_xl; k < grid_index_xh; ++k)
                            {
                                PlaceDB::coordinate_type grid_xl = db.routingGridOrigin(kX) + k * routing_grid_size_x; 
                                PlaceDB::coordinate_type grid_xh = grid_xl + routing_grid_size_x;
                                for (PlaceDB::index_type h = grid_index_yl; h < grid_index_yh; ++h)
                                {
                                    PlaceDB::coordinate_type grid_yl = db.routingGridOrigin(kY) + h * routing_grid_size_y; 
                                    PlaceDB::coordinate_type grid_yh = grid_yl + routing_grid_size_y; 
                                    MacroObs::box_type grid_box (grid_xl, grid_yl, grid_xh, grid_yh); 
                                    PlaceDB::index_type index = layer * num_routing_grids_x * num_routing_grids_y + (k * num_routing_grids_y + h);
                                    double intersect_ratio = intersectArea(box, grid_box) / routing_grid_area;
                                    dreamplaceAssert(intersect_ratio <= 1);
                                    initial_horizontal_routing_map[index] += ceil(intersect_ratio * db.numRoutingTracks(PlanarDirectEnum::HORIZONTAL, layer)); 
                                    initial_vertical_routing_map[index] += ceil(intersect_ratio * db.numRoutingTracks(PlanarDirectEnum::VERTICAL, layer)); 
                                    if (layer == 2)
                                    {
                                        dreamplaceAssert(db.numRoutingTracks(PlanarDirectEnum::VERTICAL, layer) == 0);
                                        dreamplaceAssertMsg(initial_vertical_routing_map[index] == 0, "intersect_ratio %g, initial_vertical_routing_map[%u] = %d, capacity %u, product %g", 
                                                intersect_ratio, index, initial_vertical_routing_map[index], 
                                                db.numRoutingTracks(PlanarDirectEnum::VERTICAL, layer), 
                                                intersect_ratio * db.numRoutingTracks(PlanarDirectEnum::VERTICAL, layer)
                                                );
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // clamp maximum for overlapping fixed cells 
            for (PlaceDB::index_type layer = 0; layer < db.numRoutingLayers(); ++layer)
            {
                for (int i = 0, ie = num_routing_grids_x * num_routing_grids_y; i < ie; ++i)
                {
                    auto& hvalue = initial_horizontal_routing_map[layer * ie + i];
                    hvalue = std::min(hvalue, (int)db.numRoutingTracks(PlanarDirectEnum::HORIZONTAL, layer));

                    auto& vvalue = initial_vertical_routing_map[layer * ie + i];
                    vvalue = std::min(vvalue, (int)db.numRoutingTracks(PlanarDirectEnum::VERTICAL, layer));
                }
            }
            for (auto item : initial_horizontal_routing_map)
            {
                initial_horizontal_demand_map.append(item); 
            }
            for (auto item : initial_vertical_routing_map)
            {
                initial_vertical_demand_map.append(item); 
            }
        }
    }
};

DREAMPLACE_END_NAMESPACE

DREAMPLACE_NAMESPACE::PlaceDB place_io_forward(pybind11::list const& args)
{
    //char buf[256];
    //DREAMPLACE_NAMESPACE::dreamplaceSPrint(DREAMPLACE_NAMESPACE::kINFO, buf, "reading input files takes %%t seconds CPU, %%w seconds real\n");
	//boost::timer::auto_cpu_timer timer (buf);

    DREAMPLACE_NAMESPACE::PlaceDB db; 

    int argc = pybind11::len(args); 
    char** argv = new char* [argc]; 
    for (int i = 0; i < argc; ++i)
    {
        std::string token = pybind11::str(args[i]); 
        argv[i] = new char [token.size()+1];
        std::copy(token.begin(), token.end(), argv[i]); 
        argv[i][token.size()] = '\0';
    }
    db.userParam().read(argc, argv); 

    for (int i = 0; i < argc; ++i)
    {
        delete [] argv[i];
    }
    delete [] argv; 
	
	// order for reading files 
	// 1. lef files 
	// 2. def files 
	bool flag; 

	// read lef 
	flag = DREAMPLACE_NAMESPACE::readLef(db);
    dreamplaceAssertMsg(flag, "failed to read input LEF files");

	// read def 
	flag = DREAMPLACE_NAMESPACE::readDef(db);
    dreamplaceAssertMsg(flag, "failed to read input DEF files");

    // if netlist is not set by DEF, read verilog 
    if (db.nets().empty()) 
    {
        // read verilog 
        flag = DREAMPLACE_NAMESPACE::readVerilog(db);
        dreamplaceAssertMsg(flag, "failed to read input Verilog files");
    }

    // read bookshelf 
    flag = DREAMPLACE_NAMESPACE::readBookshelf(db);
    dreamplaceAssertMsg(flag, "failed to read input Bookshelf files");

    // adjust input parameters 
    db.adjustParams();

    //return DREAMPLACE_NAMESPACE::PyPlaceDB(db); 
    return db; 
}

// create Python binding 

PYBIND11_MAKE_OPAQUE(std::vector<bool>);
//PYBIND11_MAKE_OPAQUE(std::vector<char>);
//PYBIND11_MAKE_OPAQUE(std::vector<unsigned char>);
PYBIND11_MAKE_OPAQUE(std::vector<DREAMPLACE_NAMESPACE::PlaceDB::coordinate_type>);
PYBIND11_MAKE_OPAQUE(std::vector<DREAMPLACE_NAMESPACE::PlaceDB::index_type>);
//PYBIND11_MAKE_OPAQUE(std::vector<long>);
//PYBIND11_MAKE_OPAQUE(std::vector<unsigned long>);
//PYBIND11_MAKE_OPAQUE(std::vector<float>);
//PYBIND11_MAKE_OPAQUE(std::vector<double>);
PYBIND11_MAKE_OPAQUE(std::vector<std::string>);

PYBIND11_MAKE_OPAQUE(DREAMPLACE_NAMESPACE::PlaceDB::string2index_map_type);

PYBIND11_MAKE_OPAQUE(std::vector<DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::index_type>>);
PYBIND11_MAKE_OPAQUE(std::vector<DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::coordinate_type>>);
PYBIND11_MAKE_OPAQUE(std::vector<DREAMPLACE_NAMESPACE::Pin>);
PYBIND11_MAKE_OPAQUE(std::vector<DREAMPLACE_NAMESPACE::Node>);
PYBIND11_MAKE_OPAQUE(std::vector<DREAMPLACE_NAMESPACE::NodeProperty>);
PYBIND11_MAKE_OPAQUE(std::vector<DREAMPLACE_NAMESPACE::Net>);
PYBIND11_MAKE_OPAQUE(std::vector<DREAMPLACE_NAMESPACE::NetProperty>);
PYBIND11_MAKE_OPAQUE(std::vector<DREAMPLACE_NAMESPACE::MacroPort>);
PYBIND11_MAKE_OPAQUE(std::vector<DREAMPLACE_NAMESPACE::MacroPin>);
PYBIND11_MAKE_OPAQUE(std::vector<DREAMPLACE_NAMESPACE::MacroObs>);
PYBIND11_MAKE_OPAQUE(std::vector<DREAMPLACE_NAMESPACE::Macro>);
PYBIND11_MAKE_OPAQUE(std::vector<DREAMPLACE_NAMESPACE::Row>);
PYBIND11_MAKE_OPAQUE(std::vector<DREAMPLACE_NAMESPACE::SubRow>);
PYBIND11_MAKE_OPAQUE(std::vector<DREAMPLACE_NAMESPACE::BinRow>);
PYBIND11_MAKE_OPAQUE(std::vector<DREAMPLACE_NAMESPACE::Region>);
PYBIND11_MAKE_OPAQUE(std::vector<DREAMPLACE_NAMESPACE::Group>);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    pybind11::bind_vector<std::vector<bool> >(m, "VectorBool");
    //pybind11::bind_vector<std::vector<char> >(m, "VectorChar", pybind11::buffer_protocol());
    //pybind11::bind_vector<std::vector<unsigned char> >(m, "VectorUChar", pybind11::buffer_protocol());
    pybind11::bind_vector<std::vector<DREAMPLACE_NAMESPACE::PlaceDB::coordinate_type> >(m, "VectorCoordinate", pybind11::buffer_protocol());
    pybind11::bind_vector<std::vector<DREAMPLACE_NAMESPACE::PlaceDB::index_type> >(m, "VectorIndex", pybind11::buffer_protocol());
    //pybind11::bind_vector<std::vector<long> >(m, "VectorLong", pybind11::buffer_protocol());
    //pybind11::bind_vector<std::vector<unsigned long> >(m, "VectorULong", pybind11::buffer_protocol());
    //pybind11::bind_vector<std::vector<float> >(m, "VectorFloat", pybind11::buffer_protocol());
    //pybind11::bind_vector<std::vector<double> >(m, "VectorDouble", pybind11::buffer_protocol());
    pybind11::bind_vector<std::vector<std::string> >(m, "VectorString");

    pybind11::bind_map<DREAMPLACE_NAMESPACE::PlaceDB::string2index_map_type>(m, "MapString2Index");

    // Params.h
    pybind11::enum_<DREAMPLACE_NAMESPACE::SolutionFileFormat>(m, "SolutionFileFormat")
        .value("DEF", DREAMPLACE_NAMESPACE::DEF)
        .value("DEFSIMPLE", DREAMPLACE_NAMESPACE::DEFSIMPLE)
        .value("BOOKSHELF", DREAMPLACE_NAMESPACE::BOOKSHELF)
        .value("BOOKSHELFALL", DREAMPLACE_NAMESPACE::BOOKSHELFALL)
        .export_values()
        ;

    // Util.h
    pybind11::enum_<DREAMPLACE_NAMESPACE::Direction1DType>(m, "Direction1DType")
        .value("kLOW", DREAMPLACE_NAMESPACE::kLOW)
        .value("kHIGH", DREAMPLACE_NAMESPACE::kHIGH)
        .value("kX", DREAMPLACE_NAMESPACE::kX)
        .value("kY", DREAMPLACE_NAMESPACE::kY)
        .value("kLEFT", DREAMPLACE_NAMESPACE::kLEFT)
        .value("kRIGHT", DREAMPLACE_NAMESPACE::kRIGHT)
        .value("kBOTTOM", DREAMPLACE_NAMESPACE::kBOTTOM)
        .value("kTOP", DREAMPLACE_NAMESPACE::kTOP)
        .export_values()
        ;

    pybind11::enum_<DREAMPLACE_NAMESPACE::Direction2DType>(m, "Direction2DType")
        .value("kXLOW", DREAMPLACE_NAMESPACE::kXLOW)
        .value("kXHIGH", DREAMPLACE_NAMESPACE::kXHIGH)
        .value("kYLOW", DREAMPLACE_NAMESPACE::kYLOW)
        .value("kYHIGH", DREAMPLACE_NAMESPACE::kYHIGH)
        .export_values()
        ;

    // Enums.h
    pybind11::class_<DREAMPLACE_NAMESPACE::OrientEnum>orientenum (m, "OrientEnum")
        ;
    pybind11::enum_<DREAMPLACE_NAMESPACE::OrientEnum::OrientType>(orientenum, "OrientType")
        .value("N", DREAMPLACE_NAMESPACE::OrientEnum::N)
        .value("S", DREAMPLACE_NAMESPACE::OrientEnum::S)
        .value("W", DREAMPLACE_NAMESPACE::OrientEnum::W)
        .value("E", DREAMPLACE_NAMESPACE::OrientEnum::E)
        .value("FN", DREAMPLACE_NAMESPACE::OrientEnum::FN)
        .value("FS", DREAMPLACE_NAMESPACE::OrientEnum::FS)
        .value("FW", DREAMPLACE_NAMESPACE::OrientEnum::FW)
        .value("FE", DREAMPLACE_NAMESPACE::OrientEnum::FE)
        .value("UNKNOWN", DREAMPLACE_NAMESPACE::OrientEnum::UNKNOWN)
        .export_values()
        ;
    pybind11::class_<DREAMPLACE_NAMESPACE::Orient>(m, "Orient")
        .def(pybind11::init<>())
        .def("value", &DREAMPLACE_NAMESPACE::Orient::value)
        ;

    pybind11::class_<DREAMPLACE_NAMESPACE::PlaceStatusEnum> placestatusenum (m, "PlaceStatusEnum")
        ;
    pybind11::enum_<DREAMPLACE_NAMESPACE::PlaceStatusEnum::PlaceStatusType>(placestatusenum, "PlaceStatusType")
        .value("UNPLACED", DREAMPLACE_NAMESPACE::PlaceStatusEnum::UNPLACED)
        .value("PLACED", DREAMPLACE_NAMESPACE::PlaceStatusEnum::PLACED)
        .value("FIXED", DREAMPLACE_NAMESPACE::PlaceStatusEnum::FIXED)
        .value("DUMMY_FIXED", DREAMPLACE_NAMESPACE::PlaceStatusEnum::DUMMY_FIXED)
        .value("UNKNOWN", DREAMPLACE_NAMESPACE::PlaceStatusEnum::UNKNOWN)
        .export_values()
        ;
    pybind11::class_<DREAMPLACE_NAMESPACE::PlaceStatus> (m, "PlaceStatus")
        .def(pybind11::init<>())
        .def("value", &DREAMPLACE_NAMESPACE::PlaceStatus::value)
        ;

    pybind11::class_<DREAMPLACE_NAMESPACE::MultiRowAttrEnum> multirowattrenum (m, "MultiRowAttrEnum")
        ;
    pybind11::enum_<DREAMPLACE_NAMESPACE::MultiRowAttrEnum::MultiRowAttrType>(multirowattrenum, "MultiRowAttrType")
        .value("SINGLE_ROW", DREAMPLACE_NAMESPACE::MultiRowAttrEnum::SINGLE_ROW)
        .value("MULTI_ROW_ANY", DREAMPLACE_NAMESPACE::MultiRowAttrEnum::MULTI_ROW_ANY)
        .value("MULTI_ROW_N", DREAMPLACE_NAMESPACE::MultiRowAttrEnum::MULTI_ROW_N)
        .value("MULTI_ROW_S", DREAMPLACE_NAMESPACE::MultiRowAttrEnum::MULTI_ROW_S)
        .value("UNKNOWN", DREAMPLACE_NAMESPACE::MultiRowAttrEnum::UNKNOWN)
        .export_values()
        ;
    pybind11::class_<DREAMPLACE_NAMESPACE::MultiRowAttr> (m, "MultiRowAttr")
        .def(pybind11::init<>())
        .def("value", &DREAMPLACE_NAMESPACE::MultiRowAttr::value)
        ;

    pybind11::class_<DREAMPLACE_NAMESPACE::SignalDirectEnum> signaldirectenum (m, "SignalDirectEnum")
        ;
    pybind11::enum_<DREAMPLACE_NAMESPACE::SignalDirectEnum::SignalDirectType>(signaldirectenum, "SignalDirectType")
        .value("INPUT", DREAMPLACE_NAMESPACE::SignalDirectEnum::INPUT)
        .value("OUTPUT", DREAMPLACE_NAMESPACE::SignalDirectEnum::OUTPUT)
        .value("INOUT", DREAMPLACE_NAMESPACE::SignalDirectEnum::INOUT)
        .value("UNKNOWN", DREAMPLACE_NAMESPACE::SignalDirectEnum::UNKNOWN)
        .export_values()
        ;
    pybind11::class_<DREAMPLACE_NAMESPACE::SignalDirect> (m, "SignalDirect")
        .def(pybind11::init<>())
        .def("value", &DREAMPLACE_NAMESPACE::SignalDirect::value)
        ;

    pybind11::class_<DREAMPLACE_NAMESPACE::PlanarDirectEnum> planardirectenum (m, "PlanarDirectEnum")
        ;
    pybind11::enum_<DREAMPLACE_NAMESPACE::PlanarDirectEnum::PlanarDirectType>(planardirectenum, "PlanarDirectType")
        .value("HORIZONTAL", DREAMPLACE_NAMESPACE::PlanarDirectEnum::HORIZONTAL)
        .value("VERTICAL", DREAMPLACE_NAMESPACE::PlanarDirectEnum::VERTICAL)
        .value("UNKNOWN", DREAMPLACE_NAMESPACE::PlanarDirectEnum::UNKNOWN)
        .export_values()
        ;
    pybind11::class_<DREAMPLACE_NAMESPACE::PlanarDirect> (m, "PlanarDirect")
        .def(pybind11::init<>())
        .def("value", &DREAMPLACE_NAMESPACE::PlanarDirect::value)
        ;

    pybind11::class_<DREAMPLACE_NAMESPACE::RegionTypeEnum> regiontypeenum (m, "RegionTypeEnum")
        ;
    pybind11::enum_<DREAMPLACE_NAMESPACE::RegionTypeEnum::RegionEnumType>(regiontypeenum, "RegionEnumType")
        .value("FENCE", DREAMPLACE_NAMESPACE::RegionTypeEnum::FENCE)
        .value("GUIDE", DREAMPLACE_NAMESPACE::RegionTypeEnum::GUIDE)
        .value("UNKNOWN", DREAMPLACE_NAMESPACE::RegionTypeEnum::UNKNOWN)
        .export_values()
        ;
    pybind11::class_<DREAMPLACE_NAMESPACE::RegionType> (m, "RegionType")
        .def(pybind11::init<>())
        .def("value", &DREAMPLACE_NAMESPACE::RegionType::value)
        ;

    // DREAMPLACE_NAMESPACE::Object.h
    pybind11::class_<DREAMPLACE_NAMESPACE::Object> (m, "Object")
        .def(pybind11::init<>())
        //.def("setId", &DREAMPLACE_NAMESPACE::Object::setId)
        .def("id", &DREAMPLACE_NAMESPACE::Object::id)
        .def("__str__", &DREAMPLACE_NAMESPACE::Object::toString)
        ;

    // Box.h
    pybind11::class_<DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::coordinate_type>> (m, "BoxCoordinate")
        .def(pybind11::init<>())
        .def(pybind11::init<DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::coordinate_type>::coordinate_type, DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::coordinate_type>::coordinate_type, DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::coordinate_type>::coordinate_type, DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::coordinate_type>::coordinate_type>())
        //.def("unset", &DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::coordinate_type>::unset, "set to uninitialized status")
        //.def("set", (DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::coordinate_type>& (DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::coordinate_type>::*)(DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::coordinate_type>::coordinate_type, DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::coordinate_type>::coordinate_type, DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::coordinate_type>::coordinate_type, DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::coordinate_type>::coordinate_type)) &DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::coordinate_type>::set, "set xl, yl, xh, yh of the box")
        .def("xl", &DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::coordinate_type>::xl)
        .def("yl", &DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::coordinate_type>::yl)
        .def("xh", &DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::coordinate_type>::xh)
        .def("yh", &DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::coordinate_type>::yh)
        .def("width", &DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::coordinate_type>::width)
        .def("height", &DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::coordinate_type>::height)
        .def("area", &DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::coordinate_type>::area)
        .def("__str__", &DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::coordinate_type>::toString)
        ;
    pybind11::class_<DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::index_type>> (m, "BoxIndex")
        .def(pybind11::init<>())
        .def(pybind11::init<DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::index_type>::coordinate_type, DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::index_type>::coordinate_type, DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::index_type>::coordinate_type, DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::index_type>::coordinate_type>())
        //.def("unset", &DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::index_type>::unset, "set to uninitialized status")
        //.def("set", (DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::index_type>& (DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::index_type>::*)(DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::index_type>::coordinate_type, DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::index_type>::coordinate_type, DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::index_type>::coordinate_type, DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::index_type>::coordinate_type)) &DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::index_type>::set, "set xl, yl, xh, yh of the box")
        .def("xl", &DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::index_type>::xl)
        .def("yl", &DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::index_type>::yl)
        .def("xh", &DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::index_type>::xh)
        .def("yh", &DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::index_type>::yh)
        .def("width", &DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::index_type>::width)
        .def("height", &DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::index_type>::height)
        .def("area", &DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::index_type>::area)
        .def("__str__", &DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::index_type>::toString)
        ;
    pybind11::bind_vector<std::vector<DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::coordinate_type>> >(m, "VectorBoxCoordinate");
    pybind11::bind_vector<std::vector<DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::index_type>> >(m, "VectorBoxIndex");

    // DREAMPLACE_NAMESPACE::Site.h
    pybind11::class_<DREAMPLACE_NAMESPACE::Site> (m, "Site")
        .def(pybind11::init<>())
        .def("name", &DREAMPLACE_NAMESPACE::Site::name)
        .def("className", &DREAMPLACE_NAMESPACE::Site::className)
        .def("symmetry", &DREAMPLACE_NAMESPACE::Site::symmetry)
        .def("size", &DREAMPLACE_NAMESPACE::Site::size)
        .def("width", &DREAMPLACE_NAMESPACE::Site::width)
        .def("height", &DREAMPLACE_NAMESPACE::Site::height)
        ;

    // DREAMPLACE_NAMESPACE::Pin.h
    pybind11::class_<DREAMPLACE_NAMESPACE::Pin, DREAMPLACE_NAMESPACE::Object> (m, "Pin")
        .def(pybind11::init<>())
        .def("macroPinId", &DREAMPLACE_NAMESPACE::Pin::macroPinId)
        .def("nodeId", &DREAMPLACE_NAMESPACE::Pin::nodeId)
        .def("netId", &DREAMPLACE_NAMESPACE::Pin::netId)
        .def("offset", &DREAMPLACE_NAMESPACE::Pin::offset)
        .def("direct", &DREAMPLACE_NAMESPACE::Pin::direct)
        ;
    pybind11::bind_vector<std::vector<DREAMPLACE_NAMESPACE::Pin> >(m, "VectorPin");

    // DREAMPLACE_NAMESPACE::Node.h
    pybind11::class_<DREAMPLACE_NAMESPACE::Node, DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::coordinate_type>, DREAMPLACE_NAMESPACE::Object> (m, "Node")
        .def(pybind11::init<>())
        .def("status", &DREAMPLACE_NAMESPACE::Node::status)
        .def("multiRowAttr", &DREAMPLACE_NAMESPACE::Node::multiRowAttr)
        .def("orient", &DREAMPLACE_NAMESPACE::Node::orient)
        .def("initPos", &DREAMPLACE_NAMESPACE::Node::initPos)
        .def("pins", (std::vector<DREAMPLACE_NAMESPACE::Node::index_type> const& (DREAMPLACE_NAMESPACE::Node::*)() const) &DREAMPLACE_NAMESPACE::Node::pins)
        .def("pinPos", (DREAMPLACE_NAMESPACE::Node::point_type (DREAMPLACE_NAMESPACE::Node::*)(DREAMPLACE_NAMESPACE::Pin const&, DREAMPLACE_NAMESPACE::Node::point_type const&) const) &DREAMPLACE_NAMESPACE::Node::pinPos)
        .def("pinPos", (DREAMPLACE_NAMESPACE::Node::point_type (DREAMPLACE_NAMESPACE::Node::*)(DREAMPLACE_NAMESPACE::Pin const&, DREAMPLACE_NAMESPACE::Node::coordinate_type, DREAMPLACE_NAMESPACE::Node::coordinate_type) const) &DREAMPLACE_NAMESPACE::Node::pinPos)
        .def("pinPos", (DREAMPLACE_NAMESPACE::Node::point_type (DREAMPLACE_NAMESPACE::Node::*)(DREAMPLACE_NAMESPACE::Pin const&) const) &DREAMPLACE_NAMESPACE::Node::pinPos)
        .def("pinPos", (DREAMPLACE_NAMESPACE::Node::coordinate_type (DREAMPLACE_NAMESPACE::Node::*)(DREAMPLACE_NAMESPACE::Pin const&, DREAMPLACE_NAMESPACE::Direction1DType) const) &DREAMPLACE_NAMESPACE::Node::pinPos)
        .def("pinX", &DREAMPLACE_NAMESPACE::Node::pinX)
        .def("pinY", &DREAMPLACE_NAMESPACE::Node::pinY)
        .def("siteArea", &DREAMPLACE_NAMESPACE::Node::siteArea)
        .def("setStatus", (DREAMPLACE_NAMESPACE::Node& (DREAMPLACE_NAMESPACE::Node::*)(DREAMPLACE_NAMESPACE::PlaceStatusEnum::PlaceStatusType s)) &DREAMPLACE_NAMESPACE::Node::setStatus)
        ;
    pybind11::bind_vector<std::vector<DREAMPLACE_NAMESPACE::Node> >(m, "VectorNode");

    pybind11::class_<DREAMPLACE_NAMESPACE::NodeProperty> (m, "NodeProperty")
        .def(pybind11::init<>())
        .def("name", &DREAMPLACE_NAMESPACE::NodeProperty::name)
        .def("macroId", &DREAMPLACE_NAMESPACE::NodeProperty::macroId)
        ;
    pybind11::bind_vector<std::vector<DREAMPLACE_NAMESPACE::NodeProperty> >(m, "VectorNodeProperty");

    // DREAMPLACE_NAMESPACE::Net.h
    pybind11::class_<DREAMPLACE_NAMESPACE::Net, DREAMPLACE_NAMESPACE::Object> (m, "Net")
        .def(pybind11::init<>())
        .def("bbox", (DREAMPLACE_NAMESPACE::Net::box_type const& (DREAMPLACE_NAMESPACE::Net::*)() const) &DREAMPLACE_NAMESPACE::Net::bbox)
        .def("weight", &DREAMPLACE_NAMESPACE::Net::weight)
        .def("pins", (std::vector<DREAMPLACE_NAMESPACE::Net::index_type> const& (DREAMPLACE_NAMESPACE::Net::*)() const) &DREAMPLACE_NAMESPACE::Net::pins)
        ;
    pybind11::bind_vector<std::vector<DREAMPLACE_NAMESPACE::Net> >(m, "VectorNet");

    pybind11::class_<DREAMPLACE_NAMESPACE::NetProperty> (m, "NetProperty")
        .def(pybind11::init<>())
        .def("name", &DREAMPLACE_NAMESPACE::NetProperty::name)
        ;
    pybind11::bind_vector<std::vector<DREAMPLACE_NAMESPACE::NetProperty> >(m, "VectorNetProperty");

    // DREAMPLACE_NAMESPACE::MacroPin.h
    pybind11::class_<DREAMPLACE_NAMESPACE::MacroPort, DREAMPLACE_NAMESPACE::Object> (m, "MacroPort")
        .def(pybind11::init<>())
        .def("bbox", &DREAMPLACE_NAMESPACE::MacroPort::bbox)
        .def("boxes", (std::vector<DREAMPLACE_NAMESPACE::MacroPort::box_type> const& (DREAMPLACE_NAMESPACE::MacroPort::*)() const) &DREAMPLACE_NAMESPACE::MacroPort::boxes)
        .def("layers", (std::vector<std::string> const& (DREAMPLACE_NAMESPACE::MacroPort::*)() const) &DREAMPLACE_NAMESPACE::MacroPort::layers)
        ;
    pybind11::bind_vector<std::vector<DREAMPLACE_NAMESPACE::MacroPort> >(m, "VectorMacroPort");

    pybind11::class_<DREAMPLACE_NAMESPACE::MacroPin, DREAMPLACE_NAMESPACE::Object> (m, "MacroPin")
        .def(pybind11::init<>())
        .def("name", &DREAMPLACE_NAMESPACE::MacroPin::name)
        .def("direct", &DREAMPLACE_NAMESPACE::MacroPin::direct)
        .def("bbox", &DREAMPLACE_NAMESPACE::MacroPin::bbox)
        .def("macroPorts", (std::vector<DREAMPLACE_NAMESPACE::MacroPort> const& (DREAMPLACE_NAMESPACE::MacroPin::*)() const) &DREAMPLACE_NAMESPACE::MacroPin::macroPorts)
        .def("macroPort", (DREAMPLACE_NAMESPACE::MacroPort const& (DREAMPLACE_NAMESPACE::MacroPin::*)(DREAMPLACE_NAMESPACE::MacroPin::index_type) const) &DREAMPLACE_NAMESPACE::MacroPin::macroPort)
        ;
    pybind11::bind_vector<std::vector<DREAMPLACE_NAMESPACE::MacroPin> >(m, "VectorMacroPin");

    // DREAMPLACE_NAMESPACE::MacroObs.h
    pybind11::class_<DREAMPLACE_NAMESPACE::MacroObs, DREAMPLACE_NAMESPACE::Object> (m, "MacroObs")
        .def(pybind11::init<>())
        .def("obsMap", (DREAMPLACE_NAMESPACE::MacroObs::obs_map_type const& (DREAMPLACE_NAMESPACE::MacroObs::*)() const) &DREAMPLACE_NAMESPACE::MacroObs::obsMap)
        ;
    pybind11::bind_vector<std::vector<DREAMPLACE_NAMESPACE::MacroObs> >(m, "VectorMacroObs");

    // DREAMPLACE_NAMESPACE::Macro.h 
    pybind11::class_<DREAMPLACE_NAMESPACE::Macro, DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::coordinate_type>, DREAMPLACE_NAMESPACE::Object> (m, "Macro")
        .def(pybind11::init<>())
        .def("name", &DREAMPLACE_NAMESPACE::Macro::name)
        .def("className", &DREAMPLACE_NAMESPACE::Macro::className)
        .def("siteName", &DREAMPLACE_NAMESPACE::Macro::siteName)
        .def("edgeName", &DREAMPLACE_NAMESPACE::Macro::edgeName)
        .def("symmetry", &DREAMPLACE_NAMESPACE::Macro::symmetry)
        .def("initOrigin", &DREAMPLACE_NAMESPACE::Macro::initOrigin)
        .def("obs", (DREAMPLACE_NAMESPACE::MacroObs const& (DREAMPLACE_NAMESPACE::Macro::*)() const) &DREAMPLACE_NAMESPACE::Macro::obs)
        .def("macroPins", (std::vector<DREAMPLACE_NAMESPACE::MacroPin> const& (DREAMPLACE_NAMESPACE::Macro::*)() const) &DREAMPLACE_NAMESPACE::Macro::macroPins)
        .def("macroPinName2Index", (DREAMPLACE_NAMESPACE::Macro::string2index_map_type const& (DREAMPLACE_NAMESPACE::Macro::*)() const) &DREAMPLACE_NAMESPACE::Macro::macroPinName2Index)
        .def("macroPin", (DREAMPLACE_NAMESPACE::MacroPin const& (DREAMPLACE_NAMESPACE::Macro::*)(DREAMPLACE_NAMESPACE::Macro::index_type) const) &DREAMPLACE_NAMESPACE::Macro::macroPin)
        ;
    pybind11::bind_vector<std::vector<DREAMPLACE_NAMESPACE::Macro> >(m, "VectorMacro");

    // DREAMPLACE_NAMESPACE::Row.h
    pybind11::class_<DREAMPLACE_NAMESPACE::Row, DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::coordinate_type>, DREAMPLACE_NAMESPACE::Object> (m, "Row")
        .def(pybind11::init<>())
        .def("name", &DREAMPLACE_NAMESPACE::Row::name)
        .def("macroName", &DREAMPLACE_NAMESPACE::Row::macroName)
        .def("orient", &DREAMPLACE_NAMESPACE::Row::orient)
        .def("step", &DREAMPLACE_NAMESPACE::Row::step)
        .def("numSites", &DREAMPLACE_NAMESPACE::Row::numSites)
        ;
    pybind11::bind_vector<std::vector<DREAMPLACE_NAMESPACE::Row> >(m, "VectorRow");

    pybind11::class_<DREAMPLACE_NAMESPACE::SubRow, DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::coordinate_type> > (m, "SubRow")
        .def(pybind11::init<>())
        .def("index1D", &DREAMPLACE_NAMESPACE::SubRow::index1D)
        .def("rowId", &DREAMPLACE_NAMESPACE::SubRow::rowId)
        .def("subRowId", &DREAMPLACE_NAMESPACE::SubRow::subRowId)
        .def("binRows", (std::vector<DREAMPLACE_NAMESPACE::SubRow::index_type> const& (DREAMPLACE_NAMESPACE::SubRow::*)() const) &DREAMPLACE_NAMESPACE::SubRow::binRows)
        ;
    pybind11::bind_vector<std::vector<DREAMPLACE_NAMESPACE::SubRow> >(m, "VectorSubRow");

    pybind11::class_<DREAMPLACE_NAMESPACE::BinRow, DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::coordinate_type>> (m, "BinRow")
        .def(pybind11::init<>())
        .def("index1D", &DREAMPLACE_NAMESPACE::BinRow::index1D)
        .def("binId", &DREAMPLACE_NAMESPACE::BinRow::binId)
        .def("subRowId", &DREAMPLACE_NAMESPACE::BinRow::subRowId)
        ;
    pybind11::bind_vector<std::vector<DREAMPLACE_NAMESPACE::BinRow> >(m, "VectorBinRow");

    // DREAMPLACE_NAMESPACE::Region.h
    pybind11::class_<DREAMPLACE_NAMESPACE::Region, DREAMPLACE_NAMESPACE::Object> (m, "Region")
        .def(pybind11::init<>())
        .def("name", &DREAMPLACE_NAMESPACE::Region::name)
        .def("boxes", (std::vector<DREAMPLACE_NAMESPACE::Region::box_type> const& (DREAMPLACE_NAMESPACE::Region::*)() const) &DREAMPLACE_NAMESPACE::Region::boxes)
        .def("type", &DREAMPLACE_NAMESPACE::Region::type)
        ;
    pybind11::bind_vector<std::vector<DREAMPLACE_NAMESPACE::Region> >(m, "VectorRegion");

    // DREAMPLACE_NAMESPACE::Group.h
    pybind11::class_<DREAMPLACE_NAMESPACE::Group, DREAMPLACE_NAMESPACE::Object> (m, "Group")
        .def(pybind11::init<>())
        .def("name", &DREAMPLACE_NAMESPACE::Group::name)
        .def("nodeNames", (std::vector<std::string> const& (DREAMPLACE_NAMESPACE::Group::*)() const) &DREAMPLACE_NAMESPACE::Group::nodeNames)
        .def("nodes", (std::vector<DREAMPLACE_NAMESPACE::Group::index_type> const& (DREAMPLACE_NAMESPACE::Group::*)() const) &DREAMPLACE_NAMESPACE::Group::nodes)
        .def("region", &DREAMPLACE_NAMESPACE::Group::region)
        ;
    pybind11::bind_vector<std::vector<DREAMPLACE_NAMESPACE::Group> >(m, "VectorGroup");

    // DREAMPLACE_NAMESPACE::PlaceDB.h
    pybind11::class_<DREAMPLACE_NAMESPACE::PlaceDB> (m, "PlaceDB")
        .def(pybind11::init<>())
        .def("nodes", (std::vector<DREAMPLACE_NAMESPACE::Node> const& (DREAMPLACE_NAMESPACE::PlaceDB::*)() const) &DREAMPLACE_NAMESPACE::PlaceDB::nodes)
        .def("node", (DREAMPLACE_NAMESPACE::Node const& (DREAMPLACE_NAMESPACE::PlaceDB::*)(DREAMPLACE_NAMESPACE::PlaceDB::index_type) const) &DREAMPLACE_NAMESPACE::PlaceDB::node)
        .def("nodeProperty", (DREAMPLACE_NAMESPACE::NodeProperty const& (DREAMPLACE_NAMESPACE::PlaceDB::*)(DREAMPLACE_NAMESPACE::PlaceDB::index_type) const) &DREAMPLACE_NAMESPACE::PlaceDB::nodeProperty)
        .def("nodeProperty", (DREAMPLACE_NAMESPACE::NodeProperty const& (DREAMPLACE_NAMESPACE::PlaceDB::*)(DREAMPLACE_NAMESPACE::Node const&) const) &DREAMPLACE_NAMESPACE::PlaceDB::nodeProperty)
        .def("setNodeStatus", (DREAMPLACE_NAMESPACE::Node const& (DREAMPLACE_NAMESPACE::PlaceDB::*)(DREAMPLACE_NAMESPACE::PlaceDB::index_type, DREAMPLACE_NAMESPACE::PlaceStatusEnum::PlaceStatusType)) &DREAMPLACE_NAMESPACE::PlaceDB::setNodeStatus)
        .def("setNodeMultiRowAttr", (DREAMPLACE_NAMESPACE::Node const& (DREAMPLACE_NAMESPACE::PlaceDB::*)(DREAMPLACE_NAMESPACE::PlaceDB::index_type, DREAMPLACE_NAMESPACE::MultiRowAttrEnum::MultiRowAttrType)) &DREAMPLACE_NAMESPACE::PlaceDB::setNodeMultiRowAttr)
        .def("setNodeOrient", (DREAMPLACE_NAMESPACE::Node const& (DREAMPLACE_NAMESPACE::PlaceDB::*)(DREAMPLACE_NAMESPACE::PlaceDB::index_type, DREAMPLACE_NAMESPACE::OrientEnum::OrientType s)) &DREAMPLACE_NAMESPACE::PlaceDB::setNodeOrient)
        .def("nets", (std::vector<DREAMPLACE_NAMESPACE::Net> const& (DREAMPLACE_NAMESPACE::PlaceDB::*)() const) &DREAMPLACE_NAMESPACE::PlaceDB::nets)
        .def("net", (DREAMPLACE_NAMESPACE::Net const& (DREAMPLACE_NAMESPACE::PlaceDB::*)(DREAMPLACE_NAMESPACE::PlaceDB::index_type) const) &DREAMPLACE_NAMESPACE::PlaceDB::net)
        .def("netProperty", (DREAMPLACE_NAMESPACE::NetProperty const& (DREAMPLACE_NAMESPACE::PlaceDB::*)(DREAMPLACE_NAMESPACE::PlaceDB::index_type) const) &DREAMPLACE_NAMESPACE::PlaceDB::netProperty)
        .def("netProperty", (DREAMPLACE_NAMESPACE::NetProperty const& (DREAMPLACE_NAMESPACE::PlaceDB::*)(DREAMPLACE_NAMESPACE::Net const&) const) &DREAMPLACE_NAMESPACE::PlaceDB::netProperty)
        .def("setNetWeight", (DREAMPLACE_NAMESPACE::Net const& (DREAMPLACE_NAMESPACE::PlaceDB::*)(DREAMPLACE_NAMESPACE::PlaceDB::index_type, DREAMPLACE_NAMESPACE::Net::weight_type)) &DREAMPLACE_NAMESPACE::PlaceDB::setNetWeight)
        .def("pins", (std::vector<DREAMPLACE_NAMESPACE::Pin> const& (DREAMPLACE_NAMESPACE::PlaceDB::*)() const) &DREAMPLACE_NAMESPACE::PlaceDB::pins)
        .def("pin", (DREAMPLACE_NAMESPACE::Pin const& (DREAMPLACE_NAMESPACE::PlaceDB::*)(DREAMPLACE_NAMESPACE::PlaceDB::index_type) const) &DREAMPLACE_NAMESPACE::PlaceDB::pin)
        .def("macros", (std::vector<DREAMPLACE_NAMESPACE::Macro> const& (DREAMPLACE_NAMESPACE::PlaceDB::*)() const) &DREAMPLACE_NAMESPACE::PlaceDB::macros)
        .def("macro", (DREAMPLACE_NAMESPACE::Macro const& (DREAMPLACE_NAMESPACE::PlaceDB::*)(DREAMPLACE_NAMESPACE::PlaceDB::index_type) const) &DREAMPLACE_NAMESPACE::PlaceDB::macro)
        .def("rows", (std::vector<DREAMPLACE_NAMESPACE::Row> const& (DREAMPLACE_NAMESPACE::PlaceDB::*)() const) &DREAMPLACE_NAMESPACE::PlaceDB::rows)
        .def("row", (DREAMPLACE_NAMESPACE::Row const& (DREAMPLACE_NAMESPACE::PlaceDB::*)(DREAMPLACE_NAMESPACE::PlaceDB::index_type) const) &DREAMPLACE_NAMESPACE::PlaceDB::row)
        .def("site", &DREAMPLACE_NAMESPACE::PlaceDB::site)
        .def("siteArea", &DREAMPLACE_NAMESPACE::PlaceDB::siteArea)
        .def("dieArea", &DREAMPLACE_NAMESPACE::PlaceDB::dieArea)
        .def("macroName2Index", (DREAMPLACE_NAMESPACE::PlaceDB::string2index_map_type const& (DREAMPLACE_NAMESPACE::PlaceDB::*)() const) &DREAMPLACE_NAMESPACE::PlaceDB::macroName2Index)
        .def("nodeName2Index", (DREAMPLACE_NAMESPACE::PlaceDB::string2index_map_type const& (DREAMPLACE_NAMESPACE::PlaceDB::*)() const) &DREAMPLACE_NAMESPACE::PlaceDB::nodeName2Index)
        .def("numMovable", &DREAMPLACE_NAMESPACE::PlaceDB::numMovable)
        .def("numFixed", &DREAMPLACE_NAMESPACE::PlaceDB::numFixed)
        .def("numMacro", &DREAMPLACE_NAMESPACE::PlaceDB::numMacro)
        .def("numIOPin", &DREAMPLACE_NAMESPACE::PlaceDB::numIOPin)
        .def("numPlaceBlockages", &DREAMPLACE_NAMESPACE::PlaceDB::numPlaceBlockages)
        .def("numIgnoredNet", &DREAMPLACE_NAMESPACE::PlaceDB::numIgnoredNet)
        .def("movableNodeIndices", (std::vector<DREAMPLACE_NAMESPACE::PlaceDB::index_type> const& (DREAMPLACE_NAMESPACE::PlaceDB::*)() const) &DREAMPLACE_NAMESPACE::PlaceDB::movableNodeIndices)
        .def("fixedNodeIndices", (std::vector<DREAMPLACE_NAMESPACE::PlaceDB::index_type> const& (DREAMPLACE_NAMESPACE::PlaceDB::*)() const) &DREAMPLACE_NAMESPACE::PlaceDB::fixedNodeIndices)
        .def("placeBlockageIndices", (std::vector<DREAMPLACE_NAMESPACE::PlaceDB::index_type> const& (DREAMPLACE_NAMESPACE::PlaceDB::*)() const) &DREAMPLACE_NAMESPACE::PlaceDB::placeBlockageIndices)
        .def("regions", (std::vector<DREAMPLACE_NAMESPACE::Region> const& (DREAMPLACE_NAMESPACE::PlaceDB::*)() const) &DREAMPLACE_NAMESPACE::PlaceDB::regions)
        .def("region", (DREAMPLACE_NAMESPACE::Region const& (DREAMPLACE_NAMESPACE::PlaceDB::*)(DREAMPLACE_NAMESPACE::PlaceDB::index_type) const) &DREAMPLACE_NAMESPACE::PlaceDB::region)
        .def("groups", (std::vector<DREAMPLACE_NAMESPACE::Group> const& (DREAMPLACE_NAMESPACE::PlaceDB::*)() const) &DREAMPLACE_NAMESPACE::PlaceDB::groups)
        .def("group", (DREAMPLACE_NAMESPACE::Group const& (DREAMPLACE_NAMESPACE::PlaceDB::*)(DREAMPLACE_NAMESPACE::PlaceDB::index_type) const) &DREAMPLACE_NAMESPACE::PlaceDB::group)
        .def("lefUnit", &DREAMPLACE_NAMESPACE::PlaceDB::lefUnit)
        .def("lefVersion", &DREAMPLACE_NAMESPACE::PlaceDB::lefVersion)
        .def("defUnit", &DREAMPLACE_NAMESPACE::PlaceDB::defUnit)
        .def("defVersion", &DREAMPLACE_NAMESPACE::PlaceDB::defVersion)
        .def("designName", &DREAMPLACE_NAMESPACE::PlaceDB::designName)
        .def("userParam", (DREAMPLACE_NAMESPACE::UserParam const& (DREAMPLACE_NAMESPACE::PlaceDB::*)() const) &DREAMPLACE_NAMESPACE::PlaceDB::userParam)
        .def("benchMetrics", (DREAMPLACE_NAMESPACE::BenchMetrics const& (DREAMPLACE_NAMESPACE::PlaceDB::*)() const) &DREAMPLACE_NAMESPACE::PlaceDB::benchMetrics)
        .def("getNode", (DREAMPLACE_NAMESPACE::Node const& (DREAMPLACE_NAMESPACE::PlaceDB::*)(DREAMPLACE_NAMESPACE::PlaceDB::index_type) const) &DREAMPLACE_NAMESPACE::PlaceDB::getNode)
        .def("getNode", (DREAMPLACE_NAMESPACE::Node const& (DREAMPLACE_NAMESPACE::PlaceDB::*)(DREAMPLACE_NAMESPACE::Pin const&) const) &DREAMPLACE_NAMESPACE::PlaceDB::getNode)
        .def("getNet", (DREAMPLACE_NAMESPACE::Net const& (DREAMPLACE_NAMESPACE::PlaceDB::*)(DREAMPLACE_NAMESPACE::PlaceDB::index_type) const) &DREAMPLACE_NAMESPACE::PlaceDB::getNet)
        .def("getNet", (DREAMPLACE_NAMESPACE::Net const& (DREAMPLACE_NAMESPACE::PlaceDB::*)(DREAMPLACE_NAMESPACE::Pin const&) const) &DREAMPLACE_NAMESPACE::PlaceDB::getNet)
        .def("pinPos", (DREAMPLACE_NAMESPACE::PlaceDB::coordinate_type (DREAMPLACE_NAMESPACE::PlaceDB::*)(DREAMPLACE_NAMESPACE::PlaceDB::index_type, DREAMPLACE_NAMESPACE::Direction1DType) const) &DREAMPLACE_NAMESPACE::PlaceDB::pinPos)
        .def("pinPos", (DREAMPLACE_NAMESPACE::PlaceDB::coordinate_type (DREAMPLACE_NAMESPACE::PlaceDB::*)(DREAMPLACE_NAMESPACE::Pin const&, DREAMPLACE_NAMESPACE::Direction1DType) const) &DREAMPLACE_NAMESPACE::PlaceDB::pinPos)
        .def("pinBbox", (DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::coordinate_type> (DREAMPLACE_NAMESPACE::PlaceDB::*)(DREAMPLACE_NAMESPACE::PlaceDB::index_type) const) &DREAMPLACE_NAMESPACE::PlaceDB::pinBbox)
        .def("pinBbox", (DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::coordinate_type> (DREAMPLACE_NAMESPACE::PlaceDB::*)(DREAMPLACE_NAMESPACE::Pin const&) const) &DREAMPLACE_NAMESPACE::PlaceDB::pinBbox)
        .def("macroPin", (DREAMPLACE_NAMESPACE::MacroPin const& (DREAMPLACE_NAMESPACE::PlaceDB::*)(DREAMPLACE_NAMESPACE::PlaceDB::index_type) const) &DREAMPLACE_NAMESPACE::PlaceDB::macroPin)
        .def("macroPin", (DREAMPLACE_NAMESPACE::MacroPin const& (DREAMPLACE_NAMESPACE::PlaceDB::*)(DREAMPLACE_NAMESPACE::Pin const&) const) &DREAMPLACE_NAMESPACE::PlaceDB::macroPin)
        .def("nodeName", (std::string const& (DREAMPLACE_NAMESPACE::PlaceDB::*)(DREAMPLACE_NAMESPACE::PlaceDB::index_type) const) &DREAMPLACE_NAMESPACE::PlaceDB::nodeName)
        .def("nodeName", (std::string const& (DREAMPLACE_NAMESPACE::PlaceDB::*)(DREAMPLACE_NAMESPACE::Node const&) const) &DREAMPLACE_NAMESPACE::PlaceDB::nodeName)
        .def("macroId", (DREAMPLACE_NAMESPACE::PlaceDB::index_type (DREAMPLACE_NAMESPACE::PlaceDB::*)(DREAMPLACE_NAMESPACE::PlaceDB::index_type) const) &DREAMPLACE_NAMESPACE::PlaceDB::macroId)
        .def("macroId", (DREAMPLACE_NAMESPACE::PlaceDB::index_type (DREAMPLACE_NAMESPACE::PlaceDB::*)(DREAMPLACE_NAMESPACE::Node const&) const) &DREAMPLACE_NAMESPACE::PlaceDB::macroId)
        .def("netName", (std::string const& (DREAMPLACE_NAMESPACE::PlaceDB::*)(DREAMPLACE_NAMESPACE::Net const&) const) &DREAMPLACE_NAMESPACE::PlaceDB::netName)
        .def("macroName", (std::string const& (DREAMPLACE_NAMESPACE::PlaceDB::*)(DREAMPLACE_NAMESPACE::Node const&) const) &DREAMPLACE_NAMESPACE::PlaceDB::macroName)
        .def("macroObs", (DREAMPLACE_NAMESPACE::MacroObs const& (DREAMPLACE_NAMESPACE::PlaceDB::*)(DREAMPLACE_NAMESPACE::Node const&) const) &DREAMPLACE_NAMESPACE::PlaceDB::macroObs)
        .def("xl", &DREAMPLACE_NAMESPACE::PlaceDB::xl)
        .def("yl", &DREAMPLACE_NAMESPACE::PlaceDB::yl)
        .def("xh", &DREAMPLACE_NAMESPACE::PlaceDB::xh)
        .def("yh", &DREAMPLACE_NAMESPACE::PlaceDB::yh)
        .def("width", &DREAMPLACE_NAMESPACE::PlaceDB::width)
        .def("height", &DREAMPLACE_NAMESPACE::PlaceDB::height)
        .def("rowHeight", &DREAMPLACE_NAMESPACE::PlaceDB::rowHeight)
        .def("rowBbox", &DREAMPLACE_NAMESPACE::PlaceDB::rowBbox)
        .def("rowXL", &DREAMPLACE_NAMESPACE::PlaceDB::rowXL)
        .def("rowYL", &DREAMPLACE_NAMESPACE::PlaceDB::rowYL)
        .def("rowXH", &DREAMPLACE_NAMESPACE::PlaceDB::rowXH)
        .def("rowYH", &DREAMPLACE_NAMESPACE::PlaceDB::rowYH)
        .def("siteWidth", &DREAMPLACE_NAMESPACE::PlaceDB::siteWidth)
        .def("siteHeight", &DREAMPLACE_NAMESPACE::PlaceDB::siteHeight)
        .def("maxDisplace", &DREAMPLACE_NAMESPACE::PlaceDB::maxDisplace)
        .def("minMovableNodeWidth", &DREAMPLACE_NAMESPACE::PlaceDB::minMovableNodeWidth)
        .def("maxMovableNodeWidth", &DREAMPLACE_NAMESPACE::PlaceDB::maxMovableNodeWidth)
        .def("avgMovableNodeWidth", &DREAMPLACE_NAMESPACE::PlaceDB::avgMovableNodeWidth)
        .def("totalMovableNodeArea", &DREAMPLACE_NAMESPACE::PlaceDB::totalMovableNodeArea)
        .def("totalFixedNodeArea", &DREAMPLACE_NAMESPACE::PlaceDB::totalFixedNodeArea)
        .def("totalRowArea", &DREAMPLACE_NAMESPACE::PlaceDB::totalRowArea)
        .def("computeMovableUtil", &DREAMPLACE_NAMESPACE::PlaceDB::computeMovableUtil)
        .def("computePinUtil", &DREAMPLACE_NAMESPACE::PlaceDB::computePinUtil)
        .def("numMultiRowMovable", &DREAMPLACE_NAMESPACE::PlaceDB::numMultiRowMovable)
        .def("numKRowMovable", &DREAMPLACE_NAMESPACE::PlaceDB::numKRowMovable)
        .def("isMultiRowMovable", (bool (DREAMPLACE_NAMESPACE::PlaceDB::*)(DREAMPLACE_NAMESPACE::PlaceDB::index_type) const) &DREAMPLACE_NAMESPACE::PlaceDB::isMultiRowMovable)
        .def("isMultiRowMovable", (bool (DREAMPLACE_NAMESPACE::PlaceDB::*)(DREAMPLACE_NAMESPACE::Node const&) const) &DREAMPLACE_NAMESPACE::PlaceDB::isMultiRowMovable)
        .def("isIgnoredNet", (bool (DREAMPLACE_NAMESPACE::PlaceDB::*)(DREAMPLACE_NAMESPACE::PlaceDB::index_type) const) &DREAMPLACE_NAMESPACE::PlaceDB::isIgnoredNet)
        .def("isIgnoredNet", (bool (DREAMPLACE_NAMESPACE::PlaceDB::*)(DREAMPLACE_NAMESPACE::Net const&) const) &DREAMPLACE_NAMESPACE::PlaceDB::isIgnoredNet)
        .def("netIgnoreFlag", &DREAMPLACE_NAMESPACE::PlaceDB::netIgnoreFlag)
        .def("numRoutingGrids", (DREAMPLACE_NAMESPACE::PlaceDB::index_type (DREAMPLACE_NAMESPACE::PlaceDB::*)(DREAMPLACE_NAMESPACE::Direction1DType) const) &DREAMPLACE_NAMESPACE::PlaceDB::numRoutingGrids)
        .def("numRoutingLayers", &DREAMPLACE_NAMESPACE::PlaceDB::numRoutingLayers)
        .def("routingGridOrigin", (DREAMPLACE_NAMESPACE::PlaceDB::coordinate_type (DREAMPLACE_NAMESPACE::PlaceDB::*)(DREAMPLACE_NAMESPACE::Direction1DType) const) &DREAMPLACE_NAMESPACE::PlaceDB::routingGridOrigin)
        .def("routingTileSize", (DREAMPLACE_NAMESPACE::PlaceDB::coordinate_type (DREAMPLACE_NAMESPACE::PlaceDB::*)(DREAMPLACE_NAMESPACE::Direction1DType) const) &DREAMPLACE_NAMESPACE::PlaceDB::routingTileSize)
        .def("routingBlockagePorosity", &DREAMPLACE_NAMESPACE::PlaceDB::routingBlockagePorosity)
        .def("numRoutingTracks", (DREAMPLACE_NAMESPACE::PlaceDB::index_type (DREAMPLACE_NAMESPACE::PlaceDB::*)(DREAMPLACE_NAMESPACE::Direction1DType, DREAMPLACE_NAMESPACE::PlaceDB::index_type) const) &DREAMPLACE_NAMESPACE::PlaceDB::numRoutingTracks)
        .def("adjustParams", &DREAMPLACE_NAMESPACE::PlaceDB::adjustParams)
        ;

    pybind11::class_<DREAMPLACE_NAMESPACE::PyPlaceDB>(m, "PyPlaceDB")
        .def(pybind11::init<>())
        .def_readwrite("num_nodes", &DREAMPLACE_NAMESPACE::PyPlaceDB::num_nodes)
        .def_readwrite("num_terminals", &DREAMPLACE_NAMESPACE::PyPlaceDB::num_terminals)
        .def_readwrite("num_terminal_NIs", &DREAMPLACE_NAMESPACE::PyPlaceDB::num_terminal_NIs)
        .def_readwrite("node_name2id_map", &DREAMPLACE_NAMESPACE::PyPlaceDB::node_name2id_map)
        .def_readwrite("node_names", &DREAMPLACE_NAMESPACE::PyPlaceDB::node_names)
        .def_readwrite("node_x", &DREAMPLACE_NAMESPACE::PyPlaceDB::node_x)
        .def_readwrite("node_y", &DREAMPLACE_NAMESPACE::PyPlaceDB::node_y)
        .def_readwrite("node_orient", &DREAMPLACE_NAMESPACE::PyPlaceDB::node_orient)
        .def_readwrite("node_size_x", &DREAMPLACE_NAMESPACE::PyPlaceDB::node_size_x)
        .def_readwrite("node_size_y", &DREAMPLACE_NAMESPACE::PyPlaceDB::node_size_y)
        .def_readwrite("node2orig_node_map", &DREAMPLACE_NAMESPACE::PyPlaceDB::node2orig_node_map)
        .def_readwrite("pin_direct", &DREAMPLACE_NAMESPACE::PyPlaceDB::pin_direct)
        .def_readwrite("pin_offset_x", &DREAMPLACE_NAMESPACE::PyPlaceDB::pin_offset_x)
        .def_readwrite("pin_offset_y", &DREAMPLACE_NAMESPACE::PyPlaceDB::pin_offset_y)
        .def_readwrite("net_name2id_map", &DREAMPLACE_NAMESPACE::PyPlaceDB::net_name2id_map)
        .def_readwrite("net_names", &DREAMPLACE_NAMESPACE::PyPlaceDB::net_names)
        .def_readwrite("net2pin_map", &DREAMPLACE_NAMESPACE::PyPlaceDB::net2pin_map)
        .def_readwrite("flat_net2pin_map", &DREAMPLACE_NAMESPACE::PyPlaceDB::flat_net2pin_map)
        .def_readwrite("flat_net2pin_start_map", &DREAMPLACE_NAMESPACE::PyPlaceDB::flat_net2pin_start_map)
        .def_readwrite("net_weights", &DREAMPLACE_NAMESPACE::PyPlaceDB::net_weights)
        .def_readwrite("node2pin_map", &DREAMPLACE_NAMESPACE::PyPlaceDB::node2pin_map)
        .def_readwrite("flat_node2pin_map", &DREAMPLACE_NAMESPACE::PyPlaceDB::flat_node2pin_map)
        .def_readwrite("flat_node2pin_start_map", &DREAMPLACE_NAMESPACE::PyPlaceDB::flat_node2pin_start_map)
        .def_readwrite("regions", &DREAMPLACE_NAMESPACE::PyPlaceDB::regions)
        .def_readwrite("flat_region_boxes", &DREAMPLACE_NAMESPACE::PyPlaceDB::flat_region_boxes)
        .def_readwrite("flat_region_boxes_start", &DREAMPLACE_NAMESPACE::PyPlaceDB::flat_region_boxes_start)
        .def_readwrite("node2fence_region_map", &DREAMPLACE_NAMESPACE::PyPlaceDB::node2fence_region_map)
        .def_readwrite("pin2node_map", &DREAMPLACE_NAMESPACE::PyPlaceDB::pin2node_map)
        .def_readwrite("pin2net_map", &DREAMPLACE_NAMESPACE::PyPlaceDB::pin2net_map)
        .def_readwrite("rows", &DREAMPLACE_NAMESPACE::PyPlaceDB::rows)
        .def_readwrite("xl", &DREAMPLACE_NAMESPACE::PyPlaceDB::xl)
        .def_readwrite("yl", &DREAMPLACE_NAMESPACE::PyPlaceDB::yl)
        .def_readwrite("xh", &DREAMPLACE_NAMESPACE::PyPlaceDB::xh)
        .def_readwrite("yh", &DREAMPLACE_NAMESPACE::PyPlaceDB::yh)
        .def_readwrite("row_height", &DREAMPLACE_NAMESPACE::PyPlaceDB::row_height)
        .def_readwrite("site_width", &DREAMPLACE_NAMESPACE::PyPlaceDB::site_width)
        .def_readwrite("num_movable_pins", &DREAMPLACE_NAMESPACE::PyPlaceDB::num_movable_pins)
        .def_readwrite("num_routing_grids_x", &DREAMPLACE_NAMESPACE::PyPlaceDB::num_routing_grids_x)
        .def_readwrite("num_routing_grids_y", &DREAMPLACE_NAMESPACE::PyPlaceDB::num_routing_grids_y)
        .def_readwrite("routing_grid_xl", &DREAMPLACE_NAMESPACE::PyPlaceDB::routing_grid_xl)
        .def_readwrite("routing_grid_yl", &DREAMPLACE_NAMESPACE::PyPlaceDB::routing_grid_yl)
        .def_readwrite("routing_grid_xh", &DREAMPLACE_NAMESPACE::PyPlaceDB::routing_grid_xh)
        .def_readwrite("routing_grid_yh", &DREAMPLACE_NAMESPACE::PyPlaceDB::routing_grid_yh)
        .def_readwrite("unit_horizontal_capacities", &DREAMPLACE_NAMESPACE::PyPlaceDB::unit_horizontal_capacities)
        .def_readwrite("unit_vertical_capacities", &DREAMPLACE_NAMESPACE::PyPlaceDB::unit_vertical_capacities)
        .def_readwrite("initial_horizontal_demand_map", &DREAMPLACE_NAMESPACE::PyPlaceDB::initial_horizontal_demand_map)
        .def_readwrite("initial_vertical_demand_map", &DREAMPLACE_NAMESPACE::PyPlaceDB::initial_vertical_demand_map)
        ;

    m.def("forward", &place_io_forward, "PlaceDB IO Read");
    m.def("pydb", [](DREAMPLACE_NAMESPACE::PlaceDB const& db){return DREAMPLACE_NAMESPACE::PyPlaceDB(db);}, "Convert PlaceDB to PyPlaceDB");
    m.def("write", [](DREAMPLACE_NAMESPACE::PlaceDB const& db, 
                std::string const& filename, DREAMPLACE_NAMESPACE::SolutionFileFormat ff, 
                pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast> const& x, 
                pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast> const& y) {return write(db, filename, ff, x, y);}, 
            "Write Placement Solution (float)");
    m.def("write", [](DREAMPLACE_NAMESPACE::PlaceDB const& db, 
                std::string const& filename, DREAMPLACE_NAMESPACE::SolutionFileFormat ff, 
                pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> const& x, 
                pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> const& y) {return write(db, filename, ff, x, y);}, 
            "Write Placement Solution (double)");
    m.def("apply", [](DREAMPLACE_NAMESPACE::PlaceDB& db, 
                pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast> const& x, 
                pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast> const& y) {apply(db, x, y);}, 
             "Apply Placement Solution (float)");
    m.def("apply", [](DREAMPLACE_NAMESPACE::PlaceDB& db, 
                pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> const& x, 
                pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> const& y) {apply(db, x, y);},
             "Apply Placement Solution (double)");
}

