/**
 * @file   PybindPlaceDB.cpp
 * @author Yibo Lin
 * @date   Apr 2020
 * @brief  Python binding for PlaceDB 
 */

#include "PyPlaceDB.h"

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

void bind_PlaceDB(pybind11::module& m) 
{
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
}
