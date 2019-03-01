/*************************************************************************
    > File Name: place_io.cpp
    > Author: Yibo Lin
    > Mail: yibolin@utexas.edu
    > Created Time: Thu Jun 18 23:08:28 2015
 ************************************************************************/

#include "PlaceDB.h"
#include <sstream>
//#include <boost/timer/timer.hpp>
#include <torch/torch.h>
#include <pybind11/stl.h>

GPF_BEGIN_NAMESPACE

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
        gpfPrint(kINFO, "reading %s\n", filename.c_str());
        bool flag = LefParser::read(db, filename);
        if (!flag) 
        {
            gpfPrint(kERROR, "LEF file parsing failed: %s\n", filename.c_str());
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
        gpfPrint(kINFO, "reading %s\n", filename.c_str());
        // a pre-reading phase to grep number of components, nets, and pins 
        prereadDef(db, filename);
        bool flag = DefParser::read(db, filename);
        if (!flag) 
        {
            gpfPrint(kERROR, "DEF file parsing failed: %s\n", filename.c_str());
            return false;
        }
    }
	else gpfPrint(kWARN, "no DEF file specified\n");

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

    gpfPrint(kINFO, "detect %u rows, %u components, %u IO pins, %u nets, %u blockages\n", numRows, numNodes, numIOPin, numNets, numBlockages);
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
        gpfPrint(kINFO, "reading %s\n", filename.c_str());
        bool flag = VerilogParser::read(db, filename);
        if (!flag)
        {
            gpfPrint(kERROR, "Verilog file parsing failed: %s\n", filename.c_str());
            return false;
        }
    }
    else gpfPrint(kWARN, "no Verilog file specified\n");

    return true;
}

bool readBookshelf(PlaceDB& db)
{
    // read bookshelf 
    std::string const& bookshelfAuxInput = db.userParam().bookshelfAuxInput;
    if (!bookshelfAuxInput.empty())
    {
        std::string const& filename = bookshelfAuxInput;
        gpfPrint(kINFO, "reading %s\n", filename.c_str());
        bool flag = BookshelfParser::read(db, filename);
        if (!flag)
        {
            gpfPrint(kERROR, "Bookshelf file parsing failed: %s\n", filename.c_str());
            return false;
        }
    }
    else gpfPrint(kWARN, "no Bookshelf file specified\n");

    // read additional .pl file 
    std::string const& bookshelfPlInput = db.userParam().bookshelfPlInput;
    if (!bookshelfPlInput.empty())
    {
        std::string const& filename = bookshelfPlInput;
        gpfPrint(kINFO, "reading %s\n", filename.c_str());
        bool flag = BookshelfParser::readPl(db, filename);
        if (!flag)
        {
            gpfPrint(kERROR, "Bookshelf additional .pl file parsing failed: %s\n", filename.c_str());
            return false;
        }
    }
    else gpfPrint(kWARN, "no additional Bookshelf .pl file specified\n");

    return true;
}

bool write(PlaceDB& db, std::string const& filename) 
{
    return db.write(filename);
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

GPF_END_NAMESPACE

gpf::PyPlaceDB place_io_forward(std::vector<std::string> const& args)
{
    //char buf[256];
    //gpf::gpfSPrint(gpf::kINFO, buf, "reading input files takes %%t seconds CPU, %%w seconds real\n");
	//boost::timer::auto_cpu_timer timer (buf);

    gpf::PlaceDB db; 

    int argc = args.size(); 
    char** argv = new char* [argc]; 
    for (int i = 0; i < argc; ++i)
    {
        argv[i] = new char [args[i].size()+1];
        std::copy(args[i].begin(), args[i].end(), argv[i]); 
        argv[i][args[i].size()] = '\0';
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
	flag = gpf::readLef(db);
    gpfAssertMsg(flag, "failed to read input LEF files");

	// read def 
	flag = gpf::readDef(db);
    gpfAssertMsg(flag, "failed to read input DEF files");

    // if netlist is not set by DEF, read verilog 
    if (db.nets().empty()) 
    {
        // read verilog 
        flag = gpf::readVerilog(db);
        gpfAssertMsg(flag, "failed to read input Verilog files");
    }

    // read bookshelf 
    flag = gpf::readBookshelf(db);
    gpfAssertMsg(flag, "failed to read input Bookshelf files");

    // adjust input parameters 
    db.adjustParams();

    return gpf::PyPlaceDB(db); 
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &place_io_forward, "PlaceDB IO Read");
    pybind11::class_<gpf::PyPlaceDB>(m, "PyPlaceDB")
        .def(pybind11::init<>())
        .def_readwrite("num_nodes", &gpf::PyPlaceDB::num_nodes)
        .def_readwrite("num_terminals", &gpf::PyPlaceDB::num_terminals)
        .def_readwrite("node_name2id_map", &gpf::PyPlaceDB::node_name2id_map)
        .def_readwrite("node_names", &gpf::PyPlaceDB::node_names)
        .def_readwrite("node_x", &gpf::PyPlaceDB::node_x)
        .def_readwrite("node_y", &gpf::PyPlaceDB::node_y)
        .def_readwrite("node_orient", &gpf::PyPlaceDB::node_orient)
        .def_readwrite("node_size_x", &gpf::PyPlaceDB::node_size_x)
        .def_readwrite("node_size_y", &gpf::PyPlaceDB::node_size_y)
        .def_readwrite("pin_direct", &gpf::PyPlaceDB::pin_direct)
        .def_readwrite("pin_offset_x", &gpf::PyPlaceDB::pin_offset_x)
        .def_readwrite("pin_offset_y", &gpf::PyPlaceDB::pin_offset_y)
        .def_readwrite("net_name2id_map", &gpf::PyPlaceDB::net_name2id_map)
        .def_readwrite("net_names", &gpf::PyPlaceDB::net_names)
        .def_readwrite("net2pin_map", &gpf::PyPlaceDB::net2pin_map)
        .def_readwrite("flat_net2pin_map", &gpf::PyPlaceDB::flat_net2pin_map)
        .def_readwrite("flat_net2pin_start_map", &gpf::PyPlaceDB::flat_net2pin_start_map)
        .def_readwrite("node2pin_map", &gpf::PyPlaceDB::node2pin_map)
        .def_readwrite("flat_node2pin_map", &gpf::PyPlaceDB::flat_node2pin_map)
        .def_readwrite("flat_node2pin_start_map", &gpf::PyPlaceDB::flat_node2pin_start_map)
        .def_readwrite("pin2node_map", &gpf::PyPlaceDB::pin2node_map)
        .def_readwrite("pin2net_map", &gpf::PyPlaceDB::pin2net_map)
        .def_readwrite("rows", &gpf::PyPlaceDB::rows)
        .def_readwrite("xl", &gpf::PyPlaceDB::xl)
        .def_readwrite("yl", &gpf::PyPlaceDB::yl)
        .def_readwrite("xh", &gpf::PyPlaceDB::xh)
        .def_readwrite("yh", &gpf::PyPlaceDB::yh)
        .def_readwrite("row_height", &gpf::PyPlaceDB::row_height)
        .def_readwrite("site_width", &gpf::PyPlaceDB::site_width)
        .def_readwrite("num_movable_pins", &gpf::PyPlaceDB::num_movable_pins)
        ;
}

