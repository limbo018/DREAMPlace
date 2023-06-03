/**
 * @file   place_io.cpp
 * @author Yibo Lin
 * @date   Apr 2020
 * @brief  Python binding 
 */

#include "PyPlaceDB.h"

DREAMPLACE_BEGIN_NAMESPACE

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
            vx[i] = std::round(x.at(i)); 
        }
    }
    PlaceDB::index_type leny = y.size(); 
    if (leny >= db.numMovable())
    {
        vy = new PlaceDB::coordinate_type [leny];
        for (PlaceDB::index_type i = 0; i < leny; ++i)
        {
            vy[i] = std::round(y.at(i)); 
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
            PlaceDB::coordinate_type xx = std::round(x.at(node.id())); 
            PlaceDB::coordinate_type yy = std::round(y.at(node.id())); 
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

PlaceDB place_io_forward(pybind11::list const& args)
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

DREAMPLACE_END_NAMESPACE

// create Python binding 

void bind_PlaceDB(pybind11::module&);
void bind_PyPlaceDB(pybind11::module&);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    bind_PlaceDB(m); 
    bind_PyPlaceDB(m);

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
    m.def("pydb", [](DREAMPLACE_NAMESPACE::PlaceDB const& db){return DREAMPLACE_NAMESPACE::PyPlaceDB(db);}, "Convert PlaceDB to PyPlaceDB");
    m.def("forward", &DREAMPLACE_NAMESPACE::place_io_forward, "PlaceDB IO Read");
}

