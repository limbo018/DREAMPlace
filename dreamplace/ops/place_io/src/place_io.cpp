/*************************************************************************
    > File Name: place_io.cpp
    > Author: Yibo Lin
    > Mail: yibolin@utexas.edu
    > Created Time: Thu Jun 18 23:08:28 2015
 ************************************************************************/

#include "PlaceDB.h"
#include <sstream>
//#include <boost/timer/timer.hpp>
//#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
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
        }
    }
}

/// database for python 
struct PyPlaceDB
{
    unsigned int num_nodes; // number of nodes, including terminals  
    unsigned int num_terminals; // number of terminals 
    pybind11::dict node_name2id_map; // node name to id map, cell name 
    pybind11::list node_names; // 1D array, cell name 
    pybind11::list node_x; // 1D array, cell position x 
    pybind11::list node_y; // 1D array, cell position y 
    pybind11::list node_orient; // 1D array, cell orientation 
    pybind11::list node_size_x; // 1D array, cell width  
    pybind11::list node_size_y; // 1D array, cell height

    pybind11::list pin_direct; // 1D array, pin direction IO 
    pybind11::list pin_offset_x; // 1D array, pin offset x to its node 
    pybind11::list pin_offset_y; // 1D array, pin offset y to its node 

    pybind11::dict net_name2id_map; // net name to id map
    pybind11::list net_names; // net name 
    pybind11::list net2pin_map; // array of 1D array, each row stores pin id
    pybind11::list flat_net2pin_map; // flatten version of net2pin_map 
    pybind11::list flat_net2pin_start_map; // starting index of each net in flat_net2pin_map
    pybind11::list net_weights; // net weight 

    pybind11::list node2pin_map; // array of 1D array, contains pin id of each node 
    pybind11::list flat_node2pin_map; // flatten version of node2pin_map 
    pybind11::list flat_node2pin_start_map; // starting index of each node in flat_node2pin_map

    pybind11::list pin2node_map; // 1D array, contain parent node id of each pin 
    pybind11::list pin2net_map; // 1D array, contain parent net id of each pin 

    pybind11::list rows; // NumRows x 4 array, stores xl, yl, xh, yh of each row 

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
        num_nodes = db.nodes().size(); 
        num_terminals = db.numFixed()+db.numIOPin(); // Bookshelf does not differentiate fixed macros and IO pins 

        for (PlaceDB::string2index_map_type::const_iterator it = db.nodeName2Index().begin(), ite = db.nodeName2Index().end(); it != ite; ++it)
        {
            node_name2id_map[pybind11::str(it->first)] = it->second; 
        }
        int count = 0; 
        for (unsigned int i = 0; i < num_nodes; ++i)
        {
            Node const& node = db.node(i); 
            node_names.append(pybind11::str(db.nodeName(i))); 
            node_x.append(node.xl());
            node_y.append(node.yl());
            node_orient.append(pybind11::str(std::string(Orient(node.orient())))); 
            node_size_x.append(node.width()); 
            node_size_y.append(node.height());

            pybind11::list pins;
            for (std::vector<Node::index_type>::const_iterator it = node.pins().begin(), ite = node.pins().end(); it != ite; ++it)
            {
                pins.append(*it);
            }
            node2pin_map.append(pins); 

            for (std::vector<Node::index_type>::const_iterator it = node.pins().begin(), ite = node.pins().end(); it != ite; ++it)
            {
                flat_node2pin_map.append(*it); 
            }
            flat_node2pin_start_map.append(count); 
            count += node.pins().size(); 
        }
        flat_node2pin_start_map.append(count); 

        num_movable_pins = 0; 
        for (unsigned int i = 0, ie = db.pins().size(); i < ie; ++i)
        {
            Pin const& pin = db.pin(i); 
            Node const& node = db.getNode(pin); 
            pin_direct.append(std::string(pin.direct())); 
            pin_offset_x.append(pin.offset().x()); 
            pin_offset_y.append(pin.offset().y()); 
            pin2node_map.append(node.id()); 
            pin2net_map.append(db.getNet(pin).id()); 

            if (node.status() != PlaceStatusEnum::FIXED && node.status() != PlaceStatusEnum::DUMMY_FIXED)
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

        xl = db.rowXL(); 
        yl = db.rowYL(); 
        xh = db.rowXH(); 
        yh = db.rowYH(); 

        row_height = db.rowHeight(); 
        site_width = db.siteWidth(); 
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
using SolutionFileFormat = DREAMPLACE_NAMESPACE::SolutionFileFormat;
using Direction1DType = DREAMPLACE_NAMESPACE::Direction1DType; 
using Direction2DType = DREAMPLACE_NAMESPACE::Direction2DType;
using OrientEnum = DREAMPLACE_NAMESPACE::OrientEnum;
using Orient = DREAMPLACE_NAMESPACE::Orient;
using PlaceStatusEnum = DREAMPLACE_NAMESPACE::PlaceStatusEnum;
using PlaceStatus = DREAMPLACE_NAMESPACE::PlaceStatus;
using MultiRowAttrEnum = DREAMPLACE_NAMESPACE::MultiRowAttrEnum;
using MultiRowAttr = DREAMPLACE_NAMESPACE::MultiRowAttr;
using SignalDirectEnum = DREAMPLACE_NAMESPACE::SignalDirectEnum;
using SignalDirect = DREAMPLACE_NAMESPACE::SignalDirect;
using PlanarDirectEnum = DREAMPLACE_NAMESPACE::PlanarDirectEnum;
using PlanarDirect = DREAMPLACE_NAMESPACE::PlanarDirect;
using Object = DREAMPLACE_NAMESPACE::Object;
using BoxCoordinate = DREAMPLACE_NAMESPACE::Box<Object::coordinate_type>;
using BoxIndex = DREAMPLACE_NAMESPACE::Box<Object::index_type>;
using Site = DREAMPLACE_NAMESPACE::Site;
using Pin = DREAMPLACE_NAMESPACE::Pin;
using Node = DREAMPLACE_NAMESPACE::Node;
using NodeProperty = DREAMPLACE_NAMESPACE::NodeProperty;
using Net = DREAMPLACE_NAMESPACE::Net;
using NetProperty = DREAMPLACE_NAMESPACE::NetProperty;
using MacroPort = DREAMPLACE_NAMESPACE::MacroPort;
using MacroPin = DREAMPLACE_NAMESPACE::MacroPin;
using MacroObs = DREAMPLACE_NAMESPACE::MacroObs;
using Macro = DREAMPLACE_NAMESPACE::Macro;
using Row = DREAMPLACE_NAMESPACE::Row;
using SubRow = DREAMPLACE_NAMESPACE::SubRow;
using BinRow = DREAMPLACE_NAMESPACE::BinRow;
using PlaceDB = DREAMPLACE_NAMESPACE::PlaceDB;

PYBIND11_MAKE_OPAQUE(std::vector<bool>);
PYBIND11_MAKE_OPAQUE(std::vector<char>);
PYBIND11_MAKE_OPAQUE(std::vector<unsigned char>);
PYBIND11_MAKE_OPAQUE(std::vector<int>);
PYBIND11_MAKE_OPAQUE(std::vector<unsigned int>);
PYBIND11_MAKE_OPAQUE(std::vector<long>);
PYBIND11_MAKE_OPAQUE(std::vector<unsigned long>);
PYBIND11_MAKE_OPAQUE(std::vector<float>);
PYBIND11_MAKE_OPAQUE(std::vector<double>);
PYBIND11_MAKE_OPAQUE(std::vector<std::string>);

PYBIND11_MAKE_OPAQUE(PlaceDB::string2index_map_type);

PYBIND11_MAKE_OPAQUE(std::vector<BoxIndex>);
PYBIND11_MAKE_OPAQUE(std::vector<BoxCoordinate>);
PYBIND11_MAKE_OPAQUE(std::vector<Pin>);
PYBIND11_MAKE_OPAQUE(std::vector<Node>);
PYBIND11_MAKE_OPAQUE(std::vector<NodeProperty>);
PYBIND11_MAKE_OPAQUE(std::vector<Net>);
PYBIND11_MAKE_OPAQUE(std::vector<NetProperty>);
PYBIND11_MAKE_OPAQUE(std::vector<MacroPort>);
PYBIND11_MAKE_OPAQUE(std::vector<MacroPin>);
PYBIND11_MAKE_OPAQUE(std::vector<MacroObs>);
PYBIND11_MAKE_OPAQUE(std::vector<Macro>);
PYBIND11_MAKE_OPAQUE(std::vector<Row>);
PYBIND11_MAKE_OPAQUE(std::vector<SubRow>);
PYBIND11_MAKE_OPAQUE(std::vector<BinRow>);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    pybind11::bind_vector<std::vector<bool> >(m, "VectorBool");
    pybind11::bind_vector<std::vector<char> >(m, "VectorChar", pybind11::buffer_protocol());
    pybind11::bind_vector<std::vector<unsigned char> >(m, "VectorUChar", pybind11::buffer_protocol());
    pybind11::bind_vector<std::vector<int> >(m, "VectorInt", pybind11::buffer_protocol());
    pybind11::bind_vector<std::vector<unsigned int> >(m, "VectorUInt", pybind11::buffer_protocol());
    pybind11::bind_vector<std::vector<long> >(m, "VectorLong", pybind11::buffer_protocol());
    pybind11::bind_vector<std::vector<unsigned long> >(m, "VectorULong", pybind11::buffer_protocol());
    pybind11::bind_vector<std::vector<float> >(m, "VectorFloat", pybind11::buffer_protocol());
    pybind11::bind_vector<std::vector<double> >(m, "VectorDouble", pybind11::buffer_protocol());
    pybind11::bind_vector<std::vector<std::string> >(m, "VectorString");

    pybind11::bind_map<DREAMPLACE_NAMESPACE::PlaceDB::string2index_map_type>(m, "MapString2Index");

    // Params.h
    pybind11::enum_<SolutionFileFormat>(m, "SolutionFileFormat")
        .value("DEF", DREAMPLACE_NAMESPACE::DEF)
        .value("DEFSIMPLE", DREAMPLACE_NAMESPACE::DEFSIMPLE)
        .value("BOOKSHELF", DREAMPLACE_NAMESPACE::BOOKSHELF)
        .value("BOOKSHELFALL", DREAMPLACE_NAMESPACE::BOOKSHELFALL)
        .export_values()
        ;

    // Util.h
    pybind11::enum_<Direction1DType>(m, "Direction1DType")
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

    pybind11::enum_<Direction2DType>(m, "Direction2DType")
        .value("kXLOW", DREAMPLACE_NAMESPACE::kXLOW)
        .value("kXHIGH", DREAMPLACE_NAMESPACE::kXHIGH)
        .value("kYLOW", DREAMPLACE_NAMESPACE::kYLOW)
        .value("kYHIGH", DREAMPLACE_NAMESPACE::kYHIGH)
        .export_values()
        ;

    // Enums.h
    pybind11::class_<OrientEnum>orientenum (m, "OrientEnum")
        ;
    pybind11::enum_<OrientEnum::OrientType>(orientenum, "OrientType")
        .value("N", OrientEnum::N)
        .value("S", OrientEnum::S)
        .value("W", OrientEnum::W)
        .value("E", OrientEnum::E)
        .value("FN", OrientEnum::FN)
        .value("FS", OrientEnum::FS)
        .value("FW", OrientEnum::FW)
        .value("FE", OrientEnum::FE)
        .value("UNKNOWN", OrientEnum::UNKNOWN)
        .export_values()
        ;
    pybind11::class_<Orient>(m, "Orient")
        .def(pybind11::init<>())
        .def("value", &Orient::value)
        ;

    pybind11::class_<PlaceStatusEnum> placestatusenum (m, "PlaceStatusEnum")
        ;
    pybind11::enum_<PlaceStatusEnum::PlaceStatusType>(placestatusenum, "PlaceStatusType")
        .value("UNPLACED", PlaceStatusEnum::UNPLACED)
        .value("PLACED", PlaceStatusEnum::PLACED)
        .value("FIXED", PlaceStatusEnum::FIXED)
        .value("DUMMY_FIXED", PlaceStatusEnum::DUMMY_FIXED)
        .value("UNKNOWN", PlaceStatusEnum::UNKNOWN)
        .export_values()
        ;
    pybind11::class_<PlaceStatus> (m, "PlaceStatus")
        .def(pybind11::init<>())
        .def("value", &PlaceStatus::value)
        ;

    pybind11::class_<MultiRowAttrEnum> multirowattrenum (m, "MultiRowAttrEnum")
        ;
    pybind11::enum_<MultiRowAttrEnum::MultiRowAttrType>(multirowattrenum, "MultiRowAttrType")
        .value("SINGLE_ROW", MultiRowAttrEnum::SINGLE_ROW)
        .value("MULTI_ROW_ANY", MultiRowAttrEnum::MULTI_ROW_ANY)
        .value("MULTI_ROW_N", MultiRowAttrEnum::MULTI_ROW_N)
        .value("MULTI_ROW_S", MultiRowAttrEnum::MULTI_ROW_S)
        .value("UNKNOWN", MultiRowAttrEnum::UNKNOWN)
        .export_values()
        ;
    pybind11::class_<MultiRowAttr> (m, "MultiRowAttr")
        .def(pybind11::init<>())
        .def("value", &MultiRowAttr::value)
        ;

    pybind11::class_<SignalDirectEnum> signaldirectenum (m, "SignalDirectEnum")
        ;
    pybind11::enum_<SignalDirectEnum::SignalDirectType>(signaldirectenum, "SignalDirectType")
        .value("INPUT", SignalDirectEnum::INPUT)
        .value("OUTPUT", SignalDirectEnum::OUTPUT)
        .value("INOUT", SignalDirectEnum::INOUT)
        .value("UNKNOWN", SignalDirectEnum::UNKNOWN)
        .export_values()
        ;
    pybind11::class_<SignalDirect> (m, "SignalDirect")
        .def(pybind11::init<>())
        .def("value", &SignalDirect::value)
        ;

    pybind11::class_<PlanarDirectEnum> planardirectenum (m, "PlanarDirectEnum")
        ;
    pybind11::enum_<PlanarDirectEnum::PlanarDirectType>(planardirectenum, "PlanarDirectType")
        .value("HORIZONTAL", PlanarDirectEnum::HORIZONTAL)
        .value("VERTICAL", PlanarDirectEnum::VERTICAL)
        .value("UNKNOWN", PlanarDirectEnum::UNKNOWN)
        .export_values()
        ;
    pybind11::class_<PlanarDirect> (m, "PlanarDirect")
        .def(pybind11::init<>())
        .def("value", &PlanarDirect::value)
        ;

    // Object.h
    pybind11::class_<Object> (m, "Object")
        .def(pybind11::init<>())
        //.def("setId", &Object::setId)
        .def("id", &Object::id)
        .def("__str__", &Object::toString)
        ;

    // Box.h
    pybind11::class_<BoxCoordinate> (m, "BoxCoordinate")
        .def(pybind11::init<>())
        .def(pybind11::init<BoxCoordinate::coordinate_type, BoxCoordinate::coordinate_type, BoxCoordinate::coordinate_type, BoxCoordinate::coordinate_type>())
        //.def("unset", &BoxCoordinate::unset, "set to uninitialized status")
        //.def("set", (BoxCoordinate& (BoxCoordinate::*)(BoxCoordinate::coordinate_type, BoxCoordinate::coordinate_type, BoxCoordinate::coordinate_type, BoxCoordinate::coordinate_type)) &BoxCoordinate::set, "set xl, yl, xh, yh of the box")
        .def("xl", &BoxCoordinate::xl)
        .def("yl", &BoxCoordinate::yl)
        .def("xh", &BoxCoordinate::xh)
        .def("yh", &BoxCoordinate::yh)
        .def("width", &BoxCoordinate::width)
        .def("height", &BoxCoordinate::height)
        .def("area", &BoxCoordinate::area)
        .def("__str__", &BoxCoordinate::toString)
        ;
    pybind11::class_<BoxIndex> (m, "BoxIndex")
        .def(pybind11::init<>())
        .def(pybind11::init<BoxIndex::coordinate_type, BoxIndex::coordinate_type, BoxIndex::coordinate_type, BoxIndex::coordinate_type>())
        //.def("unset", &BoxIndex::unset, "set to uninitialized status")
        //.def("set", (BoxIndex& (BoxIndex::*)(BoxIndex::coordinate_type, BoxIndex::coordinate_type, BoxIndex::coordinate_type, BoxIndex::coordinate_type)) &BoxIndex::set, "set xl, yl, xh, yh of the box")
        .def("xl", &BoxIndex::xl)
        .def("yl", &BoxIndex::yl)
        .def("xh", &BoxIndex::xh)
        .def("yh", &BoxIndex::yh)
        .def("width", &BoxIndex::width)
        .def("height", &BoxIndex::height)
        .def("area", &BoxIndex::area)
        .def("__str__", &BoxIndex::toString)
        ;
    pybind11::bind_vector<std::vector<BoxCoordinate> >(m, "VectorBoxCoordinate");
    pybind11::bind_vector<std::vector<BoxIndex> >(m, "VectorBoxIndex");

    // Site.h
    pybind11::class_<Site> (m, "Site")
        .def(pybind11::init<>())
        .def("name", &Site::name)
        .def("className", &Site::className)
        .def("symmetry", &Site::symmetry)
        .def("size", &Site::size)
        .def("width", &Site::width)
        .def("height", &Site::height)
        ;

    // Pin.h
    pybind11::class_<Pin, Object> (m, "Pin")
        .def(pybind11::init<>())
        .def("macroPinId", &Pin::macroPinId)
        .def("nodeId", &Pin::nodeId)
        .def("netId", &Pin::netId)
        .def("offset", &Pin::offset)
        .def("direct", &Pin::direct)
        ;
    pybind11::bind_vector<std::vector<Pin> >(m, "VectorPin");

    // Node.h
    pybind11::class_<Node, BoxCoordinate, Object> (m, "Node")
        .def(pybind11::init<>())
        .def("status", &Node::status)
        .def("multiRowAttr", &Node::multiRowAttr)
        .def("orient", &Node::orient)
        .def("initPos", &Node::initPos)
        .def("pins", (std::vector<Node::index_type> const& (Node::*)() const) &Node::pins)
        .def("pinPos", (Node::point_type (Node::*)(Pin const&, Node::point_type const&) const) &Node::pinPos)
        .def("pinPos", (Node::point_type (Node::*)(Pin const&, Node::coordinate_type, Node::coordinate_type) const) &Node::pinPos)
        .def("pinPos", (Node::point_type (Node::*)(Pin const&) const) &Node::pinPos)
        .def("pinPos", (Node::coordinate_type (Node::*)(Pin const&, Direction1DType) const) &Node::pinPos)
        .def("pinX", &Node::pinX)
        .def("pinY", &Node::pinY)
        .def("siteArea", &Node::siteArea)
        ;
    pybind11::bind_vector<std::vector<Node> >(m, "VectorNode");

    pybind11::class_<NodeProperty> (m, "NodeProperty")
        .def(pybind11::init<>())
        .def("name", &NodeProperty::name)
        .def("macroId", &NodeProperty::macroId)
        ;
    pybind11::bind_vector<std::vector<NodeProperty> >(m, "VectorNodeProperty");

    // Net.h
    pybind11::class_<Net, Object> (m, "Net")
        .def(pybind11::init<>())
        .def("bbox", (Net::box_type const& (Net::*)() const) &Net::bbox)
        .def("weight", &Net::weight)
        .def("pins", (std::vector<Net::index_type> const& (Net::*)() const) &Net::pins)
        ;
    pybind11::bind_vector<std::vector<Net> >(m, "VectorNet");

    pybind11::class_<NetProperty> (m, "NetProperty")
        .def(pybind11::init<>())
        .def("name", &NetProperty::name)
        ;
    pybind11::bind_vector<std::vector<NetProperty> >(m, "VectorNetProperty");

    // MacroPin.h
    pybind11::class_<MacroPort, Object> (m, "MacroPort")
        .def(pybind11::init<>())
        .def("bbox", &MacroPort::bbox)
        .def("boxes", (std::vector<MacroPort::box_type> const& (MacroPort::*)() const) &MacroPort::boxes)
        .def("layers", (std::vector<std::string> const& (MacroPort::*)() const) &MacroPort::layers)
        ;
    pybind11::bind_vector<std::vector<MacroPort> >(m, "VectorMacroPort");

    pybind11::class_<MacroPin, Object> (m, "MacroPin")
        .def(pybind11::init<>())
        .def("name", &MacroPin::name)
        .def("direct", &MacroPin::direct)
        .def("bbox", &MacroPin::bbox)
        .def("macroPorts", (std::vector<MacroPort> const& (MacroPin::*)() const) &MacroPin::macroPorts)
        .def("macroPort", (MacroPort const& (MacroPin::*)(MacroPin::index_type) const) &MacroPin::macroPort)
        ;
    pybind11::bind_vector<std::vector<MacroPin> >(m, "VectorMacroPin");

    // MacroObs.h
    pybind11::class_<MacroObs, Object> (m, "MacroObs")
        .def(pybind11::init<>())
        .def("obsMap", (MacroObs::obs_map_type const& (MacroObs::*)() const) &MacroObs::obsMap)
        ;
    pybind11::bind_vector<std::vector<MacroObs> >(m, "VectorMacroObs");

    // Macro.h 
    pybind11::class_<Macro, BoxCoordinate, Object> (m, "Macro")
        .def(pybind11::init<>())
        .def("name", &Macro::name)
        .def("className", &Macro::className)
        .def("siteName", &Macro::siteName)
        .def("edgeName", &Macro::edgeName)
        .def("symmetry", &Macro::symmetry)
        .def("initOrigin", &Macro::initOrigin)
        .def("obs", (MacroObs const& (Macro::*)() const) &Macro::obs)
        .def("macroPins", (std::vector<MacroPin> const& (Macro::*)() const) &Macro::macroPins)
        .def("macroPinName2Index", (Macro::string2index_map_type const& (Macro::*)() const) &Macro::macroPinName2Index)
        .def("macroPin", (MacroPin const& (Macro::*)(Macro::index_type) const) &Macro::macroPin)
        ;
    pybind11::bind_vector<std::vector<Macro> >(m, "VectorMacro");

    // Row.h
    pybind11::class_<Row, BoxCoordinate, Object> (m, "Row")
        .def(pybind11::init<>())
        .def("name", &Row::name)
        .def("macroName", &Row::macroName)
        .def("orient", &Row::orient)
        .def("step", &Row::step)
        .def("numSites", &Row::numSites)
        ;
    pybind11::bind_vector<std::vector<Row> >(m, "VectorRow");

    pybind11::class_<SubRow, BoxCoordinate > (m, "SubRow")
        .def(pybind11::init<>())
        .def("index1D", &SubRow::index1D)
        .def("rowId", &SubRow::rowId)
        .def("subRowId", &SubRow::subRowId)
        .def("binRows", (std::vector<SubRow::index_type> const& (SubRow::*)() const) &SubRow::binRows)
        ;
    pybind11::bind_vector<std::vector<SubRow> >(m, "VectorSubRow");

    pybind11::class_<BinRow, BoxCoordinate> (m, "BinRow")
        .def(pybind11::init<>())
        .def("index1D", &BinRow::index1D)
        .def("binId", &BinRow::binId)
        .def("subRowId", &BinRow::subRowId)
        ;
    pybind11::bind_vector<std::vector<BinRow> >(m, "VectorBinRow");

    // PlaceDB.h
    pybind11::class_<PlaceDB> (m, "PlaceDB")
        .def(pybind11::init<>())
        .def("nodes", (std::vector<Node> const& (PlaceDB::*)() const) &PlaceDB::nodes)
        .def("node", (Node const& (PlaceDB::*)(PlaceDB::index_type) const) &PlaceDB::node)
        .def("nodeProperty", (NodeProperty const& (PlaceDB::*)(PlaceDB::index_type) const) &PlaceDB::nodeProperty)
        .def("nodeProperty", (NodeProperty const& (PlaceDB::*)(Node const&) const) &PlaceDB::nodeProperty)
        .def("nets", (std::vector<Net> const& (PlaceDB::*)() const) &PlaceDB::nets)
        .def("net", (Net const& (PlaceDB::*)(PlaceDB::index_type) const) &PlaceDB::net)
        .def("netProperty", (NetProperty const& (PlaceDB::*)(PlaceDB::index_type) const) &PlaceDB::netProperty)
        .def("netProperty", (NetProperty const& (PlaceDB::*)(Net const&) const) &PlaceDB::netProperty)
        .def("pins", (std::vector<Pin> const& (PlaceDB::*)() const) &PlaceDB::pins)
        .def("pin", (Pin const& (PlaceDB::*)(PlaceDB::index_type) const) &PlaceDB::pin)
        .def("macros", (std::vector<Macro> const& (PlaceDB::*)() const) &PlaceDB::macros)
        .def("macro", (Macro const& (PlaceDB::*)(PlaceDB::index_type) const) &PlaceDB::macro)
        .def("rows", (std::vector<Row> const& (PlaceDB::*)() const) &PlaceDB::rows)
        .def("row", (Row const& (PlaceDB::*)(PlaceDB::index_type) const) &PlaceDB::row)
        .def("placeBlockages", (std::vector<DREAMPLACE_NAMESPACE::Box<PlaceDB::coordinate_type> > const& (PlaceDB::*)() const) &PlaceDB::placeBlockages)
        .def("site", &PlaceDB::site)
        .def("siteArea", &PlaceDB::siteArea)
        .def("dieArea", &PlaceDB::dieArea)
        .def("macroName2Index", (PlaceDB::string2index_map_type const& (PlaceDB::*)() const) &PlaceDB::macroName2Index)
        .def("nodeName2Index", (PlaceDB::string2index_map_type const& (PlaceDB::*)() const) &PlaceDB::nodeName2Index)
        .def("numMovable", &PlaceDB::numMovable)
        .def("numFixed", &PlaceDB::numFixed)
        .def("numMacro", &PlaceDB::numMacro)
        .def("numIOPin", &PlaceDB::numIOPin)
        .def("numIgnoredNet", &PlaceDB::numIgnoredNet)
        .def("movableNodeIndices", (std::vector<PlaceDB::index_type> const& (PlaceDB::*)() const) &PlaceDB::movableNodeIndices)
        .def("fixedNodeIndices", (std::vector<PlaceDB::index_type> const& (PlaceDB::*)() const) &PlaceDB::fixedNodeIndices)
        .def("lefUnit", &PlaceDB::lefUnit)
        .def("lefVersion", &PlaceDB::lefVersion)
        .def("defUnit", &PlaceDB::defUnit)
        .def("defVersion", &PlaceDB::defVersion)
        .def("designName", &PlaceDB::designName)
        .def("userParam", (DREAMPLACE_NAMESPACE::UserParam const& (PlaceDB::*)() const) &PlaceDB::userParam)
        .def("benchMetrics", (DREAMPLACE_NAMESPACE::BenchMetrics const& (PlaceDB::*)() const) &PlaceDB::benchMetrics)
        .def("getNode", (Node const& (PlaceDB::*)(PlaceDB::index_type) const) &PlaceDB::getNode)
        .def("getNode", (Node const& (PlaceDB::*)(Pin const&) const) &PlaceDB::getNode)
        .def("getNet", (Net const& (PlaceDB::*)(PlaceDB::index_type) const) &PlaceDB::getNet)
        .def("getNet", (Net const& (PlaceDB::*)(Pin const&) const) &PlaceDB::getNet)
        .def("pinPos", (PlaceDB::coordinate_type (PlaceDB::*)(PlaceDB::index_type, Direction1DType) const) &PlaceDB::pinPos)
        .def("pinPos", (PlaceDB::coordinate_type (PlaceDB::*)(Pin const&, Direction1DType) const) &PlaceDB::pinPos)
        .def("pinBbox", (BoxCoordinate (PlaceDB::*)(PlaceDB::index_type) const) &PlaceDB::pinBbox)
        .def("pinBbox", (BoxCoordinate (PlaceDB::*)(Pin const&) const) &PlaceDB::pinBbox)
        .def("macroPin", (MacroPin const& (PlaceDB::*)(PlaceDB::index_type) const) &PlaceDB::macroPin)
        .def("macroPin", (MacroPin const& (PlaceDB::*)(Pin const&) const) &PlaceDB::macroPin)
        .def("nodeName", (std::string const& (PlaceDB::*)(PlaceDB::index_type) const) &PlaceDB::nodeName)
        .def("nodeName", (std::string const& (PlaceDB::*)(Node const&) const) &PlaceDB::nodeName)
        .def("macroId", (PlaceDB::index_type (PlaceDB::*)(PlaceDB::index_type) const) &PlaceDB::macroId)
        .def("macroId", (PlaceDB::index_type (PlaceDB::*)(Node const&) const) &PlaceDB::macroId)
        .def("netName", (std::string const& (PlaceDB::*)(Net const&) const) &PlaceDB::netName)
        .def("macroName", (std::string const& (PlaceDB::*)(Node const&) const) &PlaceDB::macroName)
        .def("macroObs", (MacroObs const& (PlaceDB::*)(Node const&) const) &PlaceDB::macroObs)
        .def("xl", &PlaceDB::xl)
        .def("yl", &PlaceDB::yl)
        .def("xh", &PlaceDB::xh)
        .def("yh", &PlaceDB::yh)
        .def("width", &PlaceDB::width)
        .def("height", &PlaceDB::height)
        .def("rowHeight", &PlaceDB::rowHeight)
        .def("rowBbox", &PlaceDB::rowBbox)
        .def("rowXL", &PlaceDB::rowXL)
        .def("rowYL", &PlaceDB::rowYL)
        .def("rowXH", &PlaceDB::rowXH)
        .def("rowYH", &PlaceDB::rowYH)
        .def("siteWidth", &PlaceDB::siteWidth)
        .def("siteHeight", &PlaceDB::siteHeight)
        .def("maxDisplace", &PlaceDB::maxDisplace)
        .def("minMovableNodeWidth", &PlaceDB::minMovableNodeWidth)
        .def("maxMovableNodeWidth", &PlaceDB::maxMovableNodeWidth)
        .def("avgMovableNodeWidth", &PlaceDB::avgMovableNodeWidth)
        .def("totalMovableNodeArea", &PlaceDB::totalMovableNodeArea)
        .def("totalFixedNodeArea", &PlaceDB::totalFixedNodeArea)
        .def("totalRowArea", &PlaceDB::totalRowArea)
        .def("computeMovableUtil", &PlaceDB::computeMovableUtil)
        .def("computePinUtil", &PlaceDB::computePinUtil)
        .def("numMultiRowMovable", &PlaceDB::numMultiRowMovable)
        .def("numKRowMovable", &PlaceDB::numKRowMovable)
        .def("isMultiRowMovable", (bool (PlaceDB::*)(PlaceDB::index_type) const) &PlaceDB::isMultiRowMovable)
        .def("isMultiRowMovable", (bool (PlaceDB::*)(Node const&) const) &PlaceDB::isMultiRowMovable)
        .def("isIgnoredNet", (bool (PlaceDB::*)(PlaceDB::index_type) const) &PlaceDB::isIgnoredNet)
        .def("isIgnoredNet", (bool (PlaceDB::*)(Net const&) const) &PlaceDB::isIgnoredNet)
        .def("netIgnoreFlag", &PlaceDB::netIgnoreFlag)
        ;

    pybind11::class_<DREAMPLACE_NAMESPACE::PyPlaceDB>(m, "PyPlaceDB")
        .def(pybind11::init<>())
        .def_readwrite("num_nodes", &DREAMPLACE_NAMESPACE::PyPlaceDB::num_nodes)
        .def_readwrite("num_terminals", &DREAMPLACE_NAMESPACE::PyPlaceDB::num_terminals)
        .def_readwrite("node_name2id_map", &DREAMPLACE_NAMESPACE::PyPlaceDB::node_name2id_map)
        .def_readwrite("node_names", &DREAMPLACE_NAMESPACE::PyPlaceDB::node_names)
        .def_readwrite("node_x", &DREAMPLACE_NAMESPACE::PyPlaceDB::node_x)
        .def_readwrite("node_y", &DREAMPLACE_NAMESPACE::PyPlaceDB::node_y)
        .def_readwrite("node_orient", &DREAMPLACE_NAMESPACE::PyPlaceDB::node_orient)
        .def_readwrite("node_size_x", &DREAMPLACE_NAMESPACE::PyPlaceDB::node_size_x)
        .def_readwrite("node_size_y", &DREAMPLACE_NAMESPACE::PyPlaceDB::node_size_y)
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
        ;

    m.def("forward", &place_io_forward, "PlaceDB IO Read");
    m.def("pydb", [](DREAMPLACE_NAMESPACE::PlaceDB const& db){return DREAMPLACE_NAMESPACE::PyPlaceDB(db);}, "Convert PlaceDB to PyPlaceDB");
    m.def("write", [](DREAMPLACE_NAMESPACE::PlaceDB const& db, 
                std::string const& filename, SolutionFileFormat ff, 
                pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast> const& x, 
                pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast> const& y) {return write(db, filename, ff, x, y);}, 
            "Write Placement Solution (float)");
    m.def("write", [](DREAMPLACE_NAMESPACE::PlaceDB const& db, 
                std::string const& filename, SolutionFileFormat ff, 
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

