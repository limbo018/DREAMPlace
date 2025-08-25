/**
 * @file   PyPlaceDB.cpp
 * @author Yibo Lin
 * @date   Apr 2020
 * @brief  Placement database for python
 */

#include "PyPlaceDB.h"
#include <boost/polygon/polygon.hpp>

DREAMPLACE_BEGIN_NAMESPACE

namespace gtl = boost::polygon;
using namespace gtl::operators;
typedef gtl::polygon_90_set_data<PlaceDB::coordinate_type> PolygonSet;

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

void PyPlaceDB::set(PlaceDB const& db)
{
    num_terminal_NIs = db.numIOPin();  // IO pins
    //double total_fixed_node_area = 0; // compute total area of fixed cells, which is an upper bound
    // collect boxes for fixed cells and put in a polygon set to remove overlap later
    //std::vector<gtl::rectangle_data<PlaceDB::coordinate_type>> fixed_boxes;
    // record original node to new node mapping
    std::vector<std::vector<PlaceDB::index_type> > mNode2NewNodes (db.nodes().size());

    //// add a node to a bin
    //auto addNode2Bin = [&](Box<PlaceDB::coordinate_type> const& box) {
    //    fixed_boxes.emplace_back(box.xl(), box.yl(), box.xh(), box.yh());
    //};
    // general add a node
    auto addNode = [&](Node const& node,
            std::string const& name, Orient orient,
            Box<PlaceDB::coordinate_type> const& box, bool /*dist2map*/) {
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
        //if (dist2map)
        //{
        //    //dreamplacePrint(kDEBUG, "node %s\n", db.nodeName(node).c_str());
        //    addNode2Bin(box);
        //}
    };
    // add obstruction boxes for fixed nodes
    // initialize node shapes from obstruction
    // I do not differentiate obstruction boxes at different layers
    // At least, this is true for DAC/ICCAD 2012 benchmarks
    auto addObsBoxes = [&](Node const& node, std::vector<Box<PlaceDB::coordinate_type> > const& vBox, bool dist2map){
        Box<PlaceDB::coordinate_type> bbox;
        for (PlaceDB::index_type i = 0; i < vBox.size(); ++i)
        {
            auto box = vBox[i];
            box.set(box.xl() + node.xl(),
                    box.yl() + node.yl(),
                    box.xh() + node.xl(),
                    box.yh() + node.yl()
                   );
            std::ostringstream oss; 
            oss << db.nodeName(node) << ".DREAMPlace.Shape" << i; 
            addNode(node, oss.str(), Orient(node.orient()), box, dist2map);
            bbox.encompass(box);
        }
        // compute the upper bound of fixed cell area
        //if (dist2map)
        //{
        //    total_fixed_node_area += bbox.area();
        //}
    };

    num_terminals = 0; // regard only fixed macros as macros, placement blockages are ignored
    for (unsigned int i = 0; i < db.nodes().size(); ++i)
    {
        Node const& node = db.node(i);
        Macro const& macro = db.macro(db.macroId(node));
        if (node.status() != PlaceStatusEnum::FIXED || i >= db.nodes().size() - num_terminal_NIs)
        {
            addNode(node, db.nodeName(node), Orient(node.orient()), node, false);
        }
        // else if (macro.className() != "DREAMPlace.PlaceBlockage") // fixed cells are special cases, skip placement blockages (looks like ISPD2015 benchmarks do not process placement blockages)
        else //Jiaqi: To compare with NTUPlace4dr, we have to consider blockages in ISPD2015 benchmarks
        {
            Macro const& macro = db.macro(db.macroId(node));

            if (!macro.obs().empty())
            {
                MacroObs::ObsConstIterator foundObs = macro.obs().obsMap().find("Bookshelf.Shape");
                // add obstruction boxes for fixed nodes
                // initialize node shapes from obstruction
                // I do not differentiate obstruction boxes at different layers
                // At least, this is true for DAC/ICCAD 2012 benchmarks

                // put all boxes into a polygon set to remove overlaps
                // this can make the placement engine more robust
                PolygonSet ps;
                if (foundObs != macro.obs().end()) // for BOOKSHELF
                {
                    for (auto const& box : foundObs->second)
                    {
                        ps.insert(gtl::rectangle_data<PlaceDB::coordinate_type>(box.xl(), box.yl(), box.xh(), box.yh()));
                    }
                }
                else
                {
                    for (auto it = macro.obs().begin(), ite = macro.obs().end(); it != ite; ++it)
                    {
                        for (auto const& box : it->second)
                        {
                            ps.insert(gtl::rectangle_data<PlaceDB::coordinate_type>(box.xl(), box.yl(), box.xh(), box.yh()));
                        }
                    }

                    // I do not know whether we should add the bounding box of this fixed cell as well
                    ps.insert(gtl::rectangle_data<PlaceDB::coordinate_type>(0, 0, node.width(), node.height()));
                }

                // Get unique boxes without overlap for each fixed cell
                // However, there may still be overlapping between fixed cells.
                // We cannot eliminate these because we want to keep the mapping from boxes to cells.
                std::vector<gtl::rectangle_data<PlaceDB::coordinate_type>> vRect;
                ps.get_rectangles(vRect);
                std::vector<Box<PlaceDB::coordinate_type>> vBox;
                vBox.reserve(vRect.size());
                for (auto const& rect : vRect)
                {
                    vBox.emplace_back(gtl::xl(rect), gtl::yl(rect), gtl::xh(rect), gtl::yh(rect));
                }
                addObsBoxes(node, vBox, true);
                num_terminals += vBox.size();
            }
            else
            {
                addNode(node, db.nodeName(node), Orient(node.orient()), node, true);
                num_terminals += 1;
                // compute upper bound of total fixed cell area
                //total_fixed_node_area += node.area();
            }
        }
    }
    // we only know num_nodes when all fixed cells with shapes are expanded 
    dreamplacePrint(kDEBUG, "num_terminals %d, numFixed %u, numPlaceBlockages %u, num_terminal_NIs %d\n", 
        num_terminals, db.numFixed(), db.numPlaceBlockages(), num_terminal_NIs);
    num_nodes = db.nodes().size() + num_terminals - db.numFixed() - db.numPlaceBlockages(); 
    dreamplaceAssertMsg(num_nodes == node_x.size(), 
        "%u != %lu, db.nodes().size = %lu, num_terminals = %d, numFixed = %u, numPlaceBlockages = %u, num_terminal_NIs = %d", 
        num_nodes, node_x.size(), db.nodes().size(), num_terminals, db.numFixed(), db.numPlaceBlockages(), num_terminal_NIs);

    //// this is different from simply summing up the area of all fixed nodes
    //double total_fixed_node_overlap_area = 0;
    //// compute total area uniquely
    //{
    //    PolygonSet ps (gtl::HORIZONTAL, fixed_boxes.begin(), fixed_boxes.end());
    //    // critical to make sure only overlap with the die area is computed
    //    ps &= gtl::rectangle_data<PlaceDB::coordinate_type>(db.rowXL(), db.rowYL(), db.rowXH(), db.rowYH());
    //    total_fixed_node_overlap_area = gtl::area(ps);
    //}
    //// the total overlap area should not exceed the upper bound; 
    //// current estimation may exceed if there are many overlapping fixed cells or boxes 
    //total_space_area = db.rowBbox().area() - std::min(total_fixed_node_overlap_area, total_fixed_node_area); 
    //dreamplacePrint(kDEBUG, "fixed area overlap: %g fixed area total: %g, space area = %g\n", total_fixed_node_overlap_area, total_fixed_node_area, total_space_area);

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
        pin_names.append(std::string(pin.name()));
        pin_name2id_map[pybind11::str(pin.name())] = i;
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
        net_weight_deltas.append(0.);
        net_criticality.append(0.);
        net_criticality_deltas.append(0.);
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

    convertOrient(); 

    // must be called after conversion of orientations
    computeAreaStatistics();
}

std::pair<int32_t, int32_t> PyPlaceDB::getOrientDegreeFlip(std::string const& orient) const
{
  int32_t degree = 0; 
  int32_t flip = 0; 
  if (orient == "N") 
  {
    degree = 0; 
    flip = 0; 
  }
  else if (orient == "S")
  {
    degree = 180; 
    flip = 0; 
  }
  else if (orient == "W")
  {
    degree = 90; 
    flip = 0; 
  }
  else if (orient == "E")
  {
    degree = 270; 
    flip = 0; 
  } 
  else if (orient == "FN") 
  {
    degree = 0; 
    flip = 1; 
  }
  else if (orient == "FS")
  {
    degree = 180; 
    flip = 1; 
  }
  else if (orient == "FW")
  {
    degree = 90; 
    flip = 1; 
  }
  else if (orient == "FE")
  {
    degree = 270; 
    flip = 1; 
  } 
  else // asssume UNKNOWN is N 
  {
    degree = 0; 
    flip = 0; 
  }
  return std::make_pair(degree, flip); 
}


std::pair<PyPlaceDB::coordinate_type, PyPlaceDB::coordinate_type> PyPlaceDB::getRotatedSizes(int32_t rot_degree, 
    PyPlaceDB::coordinate_type src_width, PyPlaceDB::coordinate_type src_height) const
{
  coordinate_type dst_width = std::numeric_limits<coordinate_type>::max(); 
  coordinate_type dst_height = std::numeric_limits<coordinate_type>::max(); 
  // apply rotation 
  // compute width and height 
  switch (rot_degree) 
  {
    default:
      dreamplacePrint(kWARN, "Unknown rotation degree %d, regarded as 0\n", rot_degree); 
    case 0: 
    case 180:
      dst_width = src_width; 
      dst_height = src_height; 
      break; 
    case 90:
    case 270:
      dst_width = src_height; 
      dst_height = src_width; 
      break; 
  }

  return std::make_pair(dst_width, dst_height); 
}
/// @param rot_degree assume positive degree means clock-wise rotation
std::pair<PyPlaceDB::coordinate_type, PyPlaceDB::coordinate_type> PyPlaceDB::getRotatedPinOffsets(int32_t rot_degree, 
    PyPlaceDB::coordinate_type src_width, PyPlaceDB::coordinate_type src_height, 
    PyPlaceDB::coordinate_type src_pin_offset_x, PyPlaceDB::coordinate_type src_pin_offset_y) const 
{
  coordinate_type dst_pin_offset_x = std::numeric_limits<coordinate_type>::max(); 
  coordinate_type dst_pin_offset_y = std::numeric_limits<coordinate_type>::max(); 
  // compute pin offsets 
  switch (rot_degree) 
  {
    default:
      dreamplacePrint(kWARN, "Unknown rotation degree %d, regarded as 0\n", rot_degree); 
    case 0: 
      dst_pin_offset_x = src_pin_offset_x; 
      dst_pin_offset_y = src_pin_offset_y; 
      break; 
    case 180:
      dst_pin_offset_x = src_width - src_pin_offset_x; 
      dst_pin_offset_y = src_height - src_pin_offset_y; 
      break; 
    case 270:
      dst_pin_offset_x = src_height - src_pin_offset_y; 
      dst_pin_offset_y = src_pin_offset_x; 
      break; 
    case 90:
      dst_pin_offset_x = src_pin_offset_y; 
      dst_pin_offset_y = src_width - src_pin_offset_x; 
      break; 
  }

  return std::make_pair(dst_pin_offset_x, dst_pin_offset_y); 
}

std::pair<PyPlaceDB::coordinate_type, PyPlaceDB::coordinate_type> PyPlaceDB::getFlipYPinOffsets(
    PyPlaceDB::coordinate_type src_width, PyPlaceDB::coordinate_type src_height, 
    PyPlaceDB::coordinate_type src_pin_offset_x, PyPlaceDB::coordinate_type src_pin_offset_y) const
{
  // apply flipping about Y axis 
  // assume the src values here are after rotation 
  coordinate_type dst_pin_offset_x = src_width - src_pin_offset_x;
  coordinate_type dst_pin_offset_y = src_pin_offset_y; 

  return std::make_pair(dst_pin_offset_x, dst_pin_offset_y); 
}

void PyPlaceDB::convertOrient() 
{
  dreamplacePrint(kINFO, "-- Converting Node Orientation --\n"); 
  std::map<std::string, int32_t> orient_count; 
  // ignore initial orientations for movable nodes 
  for (unsigned int node_id = num_nodes - num_terminals - num_terminal_NIs; node_id < node_orient.size(); ++node_id)
  {
    std::string src_orient = node_orient[node_id].cast<std::string>(); 
    if (src_orient != "N" && src_orient != "UNKNOWN")
    {
      // count how many nodes converted 
      if (orient_count.find(src_orient) == orient_count.end())
      {
        orient_count[src_orient] = 0; 
      }
      orient_count[src_orient]++; 

      // convert orientation to rotation degree and flipping Y
      std::pair<int32_t, int32_t> dst_degree_flip = getOrientDegreeFlip("N"); 
      std::pair<int32_t, int32_t> src_degree_flip = getOrientDegreeFlip(src_orient); 

      // compute rotation degree and flipping Y
      int32_t rot_degree = (dst_degree_flip.first - src_degree_flip.first + 360) % 360; 
      bool flip = (dst_degree_flip.second != src_degree_flip.second); 

      // apply rotation to get new width and height 
      std::pair<coordinate_type, coordinate_type> sizes = getRotatedSizes(rot_degree, node_size_x[node_id].cast<coordinate_type>(), node_size_y[node_id].cast<coordinate_type>()); 

      pybind11::list pins = node2pin_map[node_id].cast<pybind11::list>();
      for (unsigned int j = 0; j < pins.size(); ++j)
      {
        unsigned int pin_id = pins[j].cast<index_type>(); 

        // apply rotations to get new pin offsets 
        std::pair<coordinate_type, coordinate_type> pin_offsets = getRotatedPinOffsets(rot_degree, 
            node_size_x[node_id].cast<coordinate_type>(), node_size_y[node_id].cast<coordinate_type>(), pin_offset_x[pin_id].cast<coordinate_type>(), pin_offset_y[pin_id].cast<coordinate_type>()); 

        // apply changes 
        pin_offset_x[pin_id] = pin_offsets.first; 
        pin_offset_y[pin_id] = pin_offsets.second; 
      }

      // apply changes 
      node_size_x[node_id] = sizes.first; 
      node_size_y[node_id] = sizes.second; 

      // apply flipping 
      if (flip) 
      {
        pins = node2pin_map[node_id].cast<pybind11::list>();
        for (unsigned int j = 0; j < pins.size(); ++j)
        {
          unsigned int pin_id = pins[j].cast<index_type>(); 

          // apply rotations to get new pin offsets 
          std::pair<coordinate_type, coordinate_type> pin_offsets = getFlipYPinOffsets(
              node_size_x[node_id].cast<coordinate_type>(), node_size_y[node_id].cast<coordinate_type>(), pin_offset_x[pin_id].cast<coordinate_type>(), pin_offset_y[pin_id].cast<coordinate_type>()); 

          // apply changes 
          pin_offset_x[pin_id] = pin_offsets.first; 
          pin_offset_y[pin_id] = pin_offsets.second; 
        }
      }
    }
  }

  for (std::map<std::string, int32_t>::const_iterator it = orient_count.begin(); it != orient_count.end(); ++it)
  {
    dreamplacePrint(kINFO, "%s -> N: %d nodes\n", it->first.c_str(), it->second);
  }
  dreamplacePrint(kINFO, "---------------------------------\n"); 
}

void PyPlaceDB::computeAreaStatistics() 
{
  double total_fixed_node_area = 0; // compute total area of fixed cells, which is an upper bound
  std::vector<gtl::rectangle_data<PlaceDB::coordinate_type>> fixed_boxes;

  for (index_type i = num_nodes - num_terminals - num_terminal_NIs; i < num_nodes - num_terminal_NIs; ++i)
  {
    total_fixed_node_area += node_size_x[i].cast<double>() * node_size_y[i].cast<double>(); 
    fixed_boxes.emplace_back(node_x[i].cast<coordinate_type>(), node_y[i].cast<coordinate_type>(), node_x[i].cast<coordinate_type>() + node_size_x[i].cast<coordinate_type>(), node_y[i].cast<coordinate_type>() + node_size_y[i].cast<coordinate_type>());
  }

  // this is different from simply summing up the area of all fixed nodes
  double total_fixed_node_overlap_area = 0;
  // compute total area uniquely
  {
    PolygonSet ps (gtl::HORIZONTAL, fixed_boxes.begin(), fixed_boxes.end());
    // critical to make sure only overlap with the die area is computed
    ps &= gtl::rectangle_data<PlaceDB::coordinate_type>(xl, yl, xh, yh);
    total_fixed_node_overlap_area = gtl::area(ps);
  }
  // the total overlap area should not exceed the upper bound; 
  // current estimation may exceed if there are many overlapping fixed cells or boxes 
  total_space_area = (double)(xh - xl) * (yh - yl) - std::min(total_fixed_node_overlap_area, total_fixed_node_area); 
  dreamplacePrint(kINFO, "-------- Area Statistics --------\n"); 
  dreamplacePrint(kINFO, "fixed area overlap = %g\n", total_fixed_node_overlap_area);
  dreamplacePrint(kINFO, "fixed area total = %g\n", total_fixed_node_area);
  dreamplacePrint(kINFO, "space area = %g\n", total_space_area);
  dreamplacePrint(kINFO, "---------------------------------\n"); 
}

DREAMPLACE_END_NAMESPACE
