/*************************************************************************
    > File Name: PlaceDB.cpp
    > Author: Yibo Lin
    > Mail: yibolin@utexas.edu
    > Created Time: Wed Jun 17 22:29:55 2015
 ************************************************************************/

#include "PlaceDB.h"
#include <numeric>
#include "BookshelfWriter.h"
#include "DefWriter.h"
#include "Iterators.h"
#include "LefCbkHelper.h"
#include "RowMap.h"
#include "utility/src/defs.h"
//#include <boost/timer/timer.hpp>

#include <boost/version.hpp>

#if (BOOST_VERSION/100)%1000 > 55
// this is to fix the problem in boost 1.57.0 (1.55.0 works fine)
// it reports problem to find abs 
namespace boost { namespace polygon {
  using std::abs;
}} // namespace boost // namespace polygon
#endif

#include <boost/geometry.hpp>
// use adapted boost.polygon in boost.geometry, which is compatible to rtree
#include <boost/geometry/geometries/adapted/boost_polygon.hpp>
namespace gtl = boost::polygon;
using namespace boost::polygon::operators;

DREAMPLACE_BEGIN_NAMESPACE

/// default constructor
PlaceDB::PlaceDB() {
  m_coreSiteId = 0;
  m_numMovable = 0;
  m_numFixed = 0;
  m_numMacro = 0;
  m_numIOPin = 0;
  m_numIgnoredNet = 0;
  m_numPlaceBlockages = 0;

  m_lefUnit = 0;
  m_defUnit = 0;

  m_numNetsWithDuplicatePins = 0;
  m_numPinsDuplicatedInNets = 0;

  m_numRoutingGrids[0] = m_numRoutingGrids[1] = m_numRoutingGrids[2] = 0;
}

///==== LEF Callbacks ====
void PlaceDB::lef_version_cbk(std::string const& v) { m_lefVersion = v; }
void PlaceDB::lef_version_cbk(double v) {
  if (v < 5.6)
    dreamplacePrint(kWARN, "current LEF version %g may be not well supported\n",
                    v);
}
void PlaceDB::lef_casesensitive_cbk(int v) { lefNamesCaseSensitive = v; }
void PlaceDB::lef_dividerchar_cbk(std::string const&) {}
void PlaceDB::lef_units_cbk(LefParser::lefiUnits const& v) {
  if (v.hasDatabase()) m_lefUnit = v.databaseNumber();
}
void PlaceDB::lef_manufacturing_cbk(double) {}
void PlaceDB::lef_useminspacing_cbk(LefParser::lefiUseMinSpacing const&) {}
void PlaceDB::lef_clearancemeasure_cbk(std::string const&) {}
void PlaceDB::lef_busbitchars_cbk(std::string const&) {}
void PlaceDB::lef_layer_cbk(LefParser::lefiLayer const&) {}
void PlaceDB::lef_via_cbk(LefParser::lefiVia const&) {}
void PlaceDB::lef_viarule_cbk(LefParser::lefiViaRule const&) {}
void PlaceDB::lef_spacing_cbk(LefParser::lefiSpacing const&) {}
void PlaceDB::lef_site_cbk(LefParser::lefiSite const& s) {
  if (m_mSiteName2Index.find(s.name()) != m_mSiteName2Index.end()) {
    dreamplacePrint(kERROR, "Site %s has already been defined, ignored", s.name()); 
  } else {
    m_vSite.push_back(Site());
    Site& site = m_vSite.back();
    site.setId(m_vSite.size() - 1);
    site.setName(s.name());
    if (s.hasClass()) site.setClassName(s.siteClass());
    if (s.hasSize()) {
      site.setSize(kX, round(s.sizeX() * m_lefUnit));
      site.setSize(kY, round(s.sizeY() * m_lefUnit));
    }
    m_mSiteName2Index[site.name()] = site.id();
    m_vSiteUsedCount.push_back(0); 

    if (limbo::iequals(site.className(), "CORE")) {
      // a heuristic to guess core site, may change later after reading LEF 
      // we only update if the recorded core site is not named "core" 
      // choose the site with the lowest height and smallest area  
      Site const& coreSite = m_vSite[m_coreSiteId];
      if (m_coreSiteId != site.id()) {
        if (coreSite.height() > site.height() 
            || (coreSite.height() == site.height() && coreSite.width() > site.width())) {
          m_coreSiteId = site.id();
          dreamplacePrint(kINFO, "set smallest CORE site to %s, %d x %d, id = %u\n",
              site.name().c_str(), site.width(), site.height(), m_coreSiteId);
        }
      }
    }
  }
}
void PlaceDB::lef_macrobegin_cbk(std::string const& n) {
  // create and add macro
  std::pair<index_type, bool> insertMacroRet = addMacro(n);
  // check duplicate
  if (!insertMacroRet.second) {
    dreamplacePrint(kWARN, "duplicate macro found in LEF file: %s\n",
                    n.c_str());
  }
}
void PlaceDB::lef_macro_cbk(LefParser::lefiMacro const& m) {
  // assume current macro is the last macro added
  Macro& macro = m_vMacro.back();

  if (m.hasClass()) macro.setClassName(m.macroClass());
  if (m.hasSiteName()) {
    if (m_mSiteName2Index.find(m.siteName()) != m_mSiteName2Index.end()) {
      index_type siteId = m_mSiteName2Index.at(m.siteName());
      m_vSiteUsedCount[siteId] += 1; 
    } else {
      dreamplaceAssertMsg(m_coreSiteId < m_vSite.size(), "core SITE not found (%u < %lu). Probably MISSING technology LEF (tlef) or SITE definition in LEF", m_coreSiteId, m_vSite.size());
      dreamplacePrint(kWARN, "Macro site name %s is NOT DEFINED in site names, add to default site %s\n", 
          m.siteName(), m_vSite[m_coreSiteId].name().c_str());
      m_vSiteUsedCount[m_coreSiteId] += 1; 
    }
  }
  // all other coordinates corresponding to origins are mapped to that
  if (m.hasOrigin())
    macro.setInitOrigin(round(m.originX() * m_lefUnit),
                        round(m.originY() * m_lefUnit));
  else
    macro.setInitOrigin(0, 0);
  macro.set(kXLOW, round(0)).set(kYLOW, round(0));
  if (m.hasSize()) {
    // remember we have set origin to (0, 0)
    macro.set(kXHIGH, round(m.sizeX() * m_lefUnit))
        .set(kYHIGH, round(m.sizeY() * m_lefUnit));
  }
  if (m.hasSiteName()) macro.setSiteName(m.siteName());
}
void PlaceDB::lef_pin_cbk(LefParser::lefiPin const& p) {
  // assume current macro is the last macro added
  Macro& macro = m_vMacro.back();

  // skip possible vdd and gnd
  if (p.hasUse() &&
      (limbo::iequals(p.use(), "POWER") || limbo::iequals(p.use(), "GROUND")))
    return;

  // create and add pin
  std::pair<index_type, bool> insertMacroPinRet = macro.addMacroPin(p.name());
  if (!insertMacroPinRet.second) {
    dreamplacePrint(kWARN,
                    "duplicate macro pin found in LEF file: %s.(%s, %d)\n",
                    macro.name().c_str(), p.name(), insertMacroPinRet.first);
    return;
  }
  MacroPin& mPin = macro.macroPin(insertMacroPinRet.first);
  if (p.hasDirection()) {
    std::string direct = p.direction(); 
    for (auto& c : direct) {
      if (c == ' ') {
        c = '_';
      }
    }
    mPin.setDirect(direct);
  }

  for (int j = 0; j < p.numPorts(); ++j) {
    // create and add port
    index_type macroPortId = mPin.addMacroPort();
    MacroPort& macroPort = mPin.macroPort(macroPortId);
    const LefParser::lefiGeometries* port = p.port(j);
    LefCbkGeometryHelper<MacroPort>()(macroPort, *port, macro.initOrigin(),
                                      lefUnit());
    // compute bounding box of port
    deriveMacroPortBbox(macroPort);
  }
  // compute bounding box of pin
  deriveMacroPinBbox(mPin);
}
void PlaceDB::lef_obstruction_cbk(LefParser::lefiObstruction const& o) {
  // assume current macro is the last macro added
  Macro& macro = m_vMacro.back();

  // obstruction
  LefCbkGeometryHelper<MacroObs>()(macro.obs(), *(o.geometries()),
                                   macro.initOrigin(), lefUnit());
}
void PlaceDB::lef_prop_cbk(LefParser::lefiProp const&) {}
void PlaceDB::lef_maxstackvia_cbk(LefParser::lefiMaxStackVia const&) {}

///==== DEF Callbacks ====
void PlaceDB::set_def_busbitchars(std::string const&) {}
void PlaceDB::set_def_dividerchar(std::string const&) {}
void PlaceDB::set_def_version(std::string const& v) { m_defVersion = v; }
void PlaceDB::set_def_unit(int u) { m_defUnit = u; }
void PlaceDB::set_def_design(std::string const& d) { 
  m_designName = d; 

  // A heuristic to set core site according to number of occurrence in macro definitions 
  std::vector<std::size_t>::const_iterator itMax = std::max_element(m_vSiteUsedCount.begin(), m_vSiteUsedCount.end()); 
  if (itMax != m_vSiteUsedCount.end()) {
    m_coreSiteId = itMax - m_vSiteUsedCount.begin(); 
  }
  Site const& site = m_vSite[m_coreSiteId]; 
  dreamplacePrint(kINFO, "set CORE site to %s, %d x %d, id = %u\n",
      site.name().c_str(), site.width(), site.height(), m_coreSiteId);
}
void PlaceDB::set_def_diearea(int xl, int yl, int xh, int yh) {
  m_dieArea.set(
      xl * lefDefUnitRatio(), 
      yl * lefDefUnitRatio(), 
      xh * lefDefUnitRatio(), 
      yh * lefDefUnitRatio()
      );
}
void PlaceDB::set_def_diearea(int n, const int* x, const int* y) {
  // construct bounding box of diearea and polygon shape 
  int dieareaXL = std::numeric_limits<int>::max(); 
  int dieareaYL = std::numeric_limits<int>::max(); 
  int dieareaXH = std::numeric_limits<int>::min(); 
  int dieareaYH = std::numeric_limits<int>::min(); 
  std::vector<gtl::point_data<int>> vPoint;
  vPoint.reserve(n); 
  for (int i = 0; i < n; ++i) {
    vPoint.emplace_back(x[i], y[i]);
    dieareaXL = std::min(dieareaXL, x[i]); 
    dieareaYL = std::min(dieareaYL, y[i]); 
    dieareaXH = std::max(dieareaXH, x[i]); 
    dieareaYH = std::max(dieareaYH, y[i]); 
  }
  gtl::polygon_90_data<int> polygon; 
  polygon.set(vPoint.begin(), vPoint.end());
  gtl::rectangle_data<int> dieareaBbox (dieareaXL, dieareaYL, dieareaXH, dieareaYH);
  gtl::polygon_90_set_data<int> dieareaBboxSet;
  dieareaBboxSet.insert(dieareaBbox);

  // add to polygon set 
  gtl::polygon_90_set_data<int> dieareaSet; 
  dieareaSet.insert(polygon); 

  // the complimentary polygon is the blockage set
  // blockageSet = dieareaBboxSet - dieareaSet
  gtl::polygon_90_set_data<int> blockageSet; 
  blockageSet = dieareaBboxSet; 
  blockageSet -= dieareaSet; 

  // get rectangles from blockage set 
  std::vector<gtl::rectangle_data<int>> vBlockageBox; 
  gtl::get_rectangles(vBlockageBox, blockageSet);
  std::vector<std::vector<int>> vBbox (vBlockageBox.size(), std::vector<int>(4));
  for (std::size_t i = 0; i < vBlockageBox.size(); ++i) {
    vBbox[i][0] = gtl::xl(vBlockageBox[i]);
    vBbox[i][1] = gtl::yl(vBlockageBox[i]);
    vBbox[i][2] = gtl::xh(vBlockageBox[i]);
    vBbox[i][3] = gtl::yh(vBlockageBox[i]);

    dreamplacePrint(kDEBUG, "Extra blockage for non-rectangular diearea (%d, %d, %d, %d)\n", vBbox[i][0], vBbox[i][1], vBbox[i][2], vBbox[i][3]);
  }

  // set diearea as the bounding box and add placement blockages 
  if (!vBbox.empty()) {
    add_def_placement_blockage(vBbox);
  }
  dreamplacePrint(kDEBUG, "Bounding box of diearea (%d, %d, %d, %d)\n", dieareaXL, dieareaYL, dieareaXH, dieareaYH);
  set_def_diearea(dieareaXL, dieareaYL, dieareaXH, dieareaYH);
}
void PlaceDB::add_def_row(DefParser::Row const& r) {
  // create and add row
  m_vRow.push_back(Row());
  Row& row = m_vRow.back();
  row.setId(m_vRow.size() - 1);

  row.setName(r.row_name);
  row.setMacroName(r.macro_name);
  row.setOrient(r.orient);
  auto siteIter = m_mSiteName2Index.find(row.macroName());
  dreamplaceAssertMsg(m_mSiteName2Index.find(row.macroName()) != m_mSiteName2Index.end(), 
      "Site name %s in Row %s is not defined in LEF", row.macroName().c_str(), row.name().c_str());
  index_type siteId = m_mSiteName2Index.at(row.macroName());
  Site const& site = m_vSite.at(siteId);
  // only support N and FS, because I'm not sure what the format should be for
  // other orient
  if (r.orient == "N" || r.orient == "FS") {
    row.set(
        r.origin[0] * lefDefUnitRatio(), 
        r.origin[1] * lefDefUnitRatio(), 
        (r.origin[0] + r.repeat[0] * r.step[0]) * lefDefUnitRatio(),
        r.origin[1] * lefDefUnitRatio() + site.size(kY)
        );
  } else {
    dreamplacePrint(kWARN, "unsupported row orientation %s\n",
                    r.orient.c_str());
    row.set(
        r.origin[0] * lefDefUnitRatio(), 
        r.origin[1] * lefDefUnitRatio(), 
        (r.origin[0] + r.repeat[0] * r.step[0]) * lefDefUnitRatio(),
        r.origin[1] * lefDefUnitRatio() + site.size(kY)
        );
  }

  row.setStep(
      r.step[0] * lefDefUnitRatio(), 
      r.step[1] * lefDefUnitRatio()
      );

  m_rowBbox.encompass(row);
}
void PlaceDB::resize_def_component(int s) {
  // save space for io pins
  // usually there are not so many io pins
  if ((long)m_vNode.capacity() <
      s)  // estimate it if PlaceDB::prepare() is not called is not called
  {
    m_vNode.reserve(s * 1.006);
    m_vNodeProperty.reserve(m_vNode.capacity());
  }
  m_numMovable = 0;
  m_numFixed = 0;
  m_vMovableNodeIndex.clear();
  m_vFixedNodeIndex.clear();
}
void PlaceDB::add_def_component(DefParser::Component const& c) {
  if (m_userParam.sDefIgnoreCellType.count(c.macro_name)) return;

  // create and add node
  std::pair<index_type, bool> insertRet = addNode(c.comp_name);
  // check duplicate
  if (!insertRet.second) {
    dreamplacePrint(kWARN, "duplicate component found in DEF file: %s\n",
                    c.comp_name.c_str());
    return;
  }

  Node& node = m_vNode.at(insertRet.first);
  NodeProperty& property = m_vNodeProperty.at(node.id());
  property.setMacroId(m_mMacroName2Index[c.macro_name]);
  Macro const& macro = m_vMacro.at(property.macroId());
  node.set(
      c.origin[0] * lefDefUnitRatio(), 
      c.origin[1] * lefDefUnitRatio(), 
      c.origin[0] * lefDefUnitRatio() + macro.width(),
      c.origin[1] * lefDefUnitRatio() + macro.height()
      );  // must update width and height
  // update status
  node.setStatus(PlaceStatusEnum::UNPLACED); 
  if (!c.status.empty()) {
    node.setStatus(c.status); 
  }
  if (!limbo::iequals(macro.className(), "CORE") && !limbo::iequals(macro.className(), "BLOCK")) {
    // always fix cells whose macro class is not CORE or BLOCK
    dreamplaceAssertMsg(node.status() == PlaceStatusEnum::FIXED ||
                            node.status() == PlaceStatusEnum::PLACED,
                        "non-CORE or non-BLOCK class cells must be FIXED or PLACED: %s %s",
                        c.comp_name.c_str(), macro.className().c_str());
    node.setStatus(PlaceStatusEnum::FIXED);
  }
  if (node.status() != PlaceStatusEnum::UNPLACED) {
    node.setOrient(c.orient);  // update orient
  }
  deriveMultiRowAttr(node);  // update MultiRowAttr
  if (node.status() == PlaceStatusEnum::FIXED ||
      node.status() == PlaceStatusEnum::DUMMY_FIXED ||
      node.status() == PlaceStatusEnum::PLACED)
    node.setInitPos(ll(node));

  // update statistics
  // may need to change the criteria of fixed cells according to benchmarks
  if (node.status() == PlaceStatusEnum::FIXED) {
    m_numFixed += 1;
    m_vFixedNodeIndex.push_back(node.id());
  } else {
    m_numMovable += 1;
    m_vMovableNodeIndex.push_back(node.id());
  }

  // reserve space for pins in add_def_net
  node.pins().reserve(m_vMacro.at(property.macroId()).macroPins().size());
}
void PlaceDB::resize_def_pin(int s) {
  m_vMacro.reserve(m_vMacro.size() + s);
  m_numIOPin = s;
}
void PlaceDB::add_def_pin(DefParser::Pin const& p) {
  // containing dirty processing to consider various situations
  bool hasLayer = false; 
  // check pin layer
  if (!p.vLayer.empty()) {
    hasLayer = true; 
  } else if (!p.vPinPort.empty()) {
    // check pin port 
    for (std::size_t i = 0; i < p.vPinPort.size(); ++i) {
      // check pin port has box 
      if (!p.vPinPort[i].vLayer.empty()) {
        hasLayer = true; 
        break; 
      }
    }
  }
  if (!hasLayer) {
    dreamplacePrint(kWARN, "IO pin %s: no layer specified\n",
                    p.pin_name.c_str());
  }
  // create virtual macro
  std::pair<index_type, bool> insertMacroRet = addMacro(p.pin_name);
  dreamplaceAssertMsg(insertMacroRet.second,
                      "IO pin %s: failed to create virtual macro",
                      p.pin_name.c_str());
  Macro& macro = m_vMacro.at(insertMacroRet.first);
  // indicate this is an IO pin
  macro.setClassName("DREAMPlace.IOPin");

  // create and add virtual node
  std::pair<index_type, bool> insertNodeRet = addNode(p.pin_name);
  dreamplaceAssertMsg(insertNodeRet.second,
                      "failed to create virtual node for io pin %s",
                      p.pin_name.c_str());
  Node& node = m_vNode.at(insertNodeRet.first);
  NodeProperty& property = m_vNodeProperty.at(node.id());

  property.setMacroId(macro.id());
  node.setStatus(PlaceStatusEnum::FIXED);  // io pin should always be fixed
  if (!p.orient.empty()) {
    node.setOrient(p.orient);
  } else if (!p.vPinPort.empty()) {
    node.setOrient(p.vPinPort.front().orient); 
  } else {
    node.setOrient(OrientEnum::N);
  }
  deriveMultiRowAttr(node);
  if (node.status() == PlaceStatusEnum::FIXED ||
      node.status() == PlaceStatusEnum::DUMMY_FIXED ||
      node.status() == PlaceStatusEnum::PLACED) {
    // check pin origin initialized 
    bool hasOrigin = false; 
    // construct an invalid box 
    node.set(
        std::numeric_limits<coordinate_type>::max(), 
        std::numeric_limits<coordinate_type>::max(), 
        std::numeric_limits<coordinate_type>::min(), 
        std::numeric_limits<coordinate_type>::min()
        );
    if (!p.vBbox.empty()) {
      for (std::size_t i = 0; i < p.vLayer.size(); ++i) {
        if (!(p.origin[0] == -1 && p.origin[1] == -1)) {
          node.encompass(MacroPort::box_type(
                (p.origin[0] + p.vBbox[i][0]) * lefDefUnitRatio(), 
                (p.origin[1] + p.vBbox[i][1]) * lefDefUnitRatio(), 
                (p.origin[0] + p.vBbox[i][2]) * lefDefUnitRatio(),
                (p.origin[1] + p.vBbox[i][3]) * lefDefUnitRatio()
                )); 
          hasOrigin = true; 
        }
      }
    } else if (!p.vPinPort.empty()) {
      for (std::size_t i = 0; i < p.vPinPort.size(); ++i) {
        DefParser::PinPort const& pport = p.vPinPort[i]; 
        for (std::size_t j = 0; j < pport.vLayer.size(); ++j) {
          if (!(pport.origin[0] == -1 && pport.origin[1] == -1)) {
            node.encompass(MacroPort::box_type(
                  (pport.origin[0] + pport.vBbox[j][0]) * lefDefUnitRatio(), 
                  (pport.origin[1] + pport.vBbox[j][1]) * lefDefUnitRatio(), 
                  (pport.origin[0] + pport.vBbox[j][2]) * lefDefUnitRatio(),
                  (pport.origin[1] + pport.vBbox[j][3]) * lefDefUnitRatio()
                  ));
            hasOrigin = true; 
          }
        }
      }
    }
    /*
    if (!(p.origin[0] == -1 && p.origin[1] == -1)) {
      node.set(
          p.origin[0] * lefDefUnitRatio(), 
          p.origin[1] * lefDefUnitRatio(),
          p.origin[0] * lefDefUnitRatio(), 
          p.origin[1] * lefDefUnitRatio()
          );
      hasOrigin = true; 
    } else {
      // check pin port 
      for (std::size_t i = 0; i < p.vPinPort.size(); ++i) {
        // check pin port origin initialized 
        DefParser::PinPort const& pport = p.vPinPort[i]; 
        if (!(pport.origin[0] == -1 && pport.origin[1] == -1)) {
          node.set(
              pport.origin[0] * lefDefUnitRatio(), 
              pport.origin[1] * lefDefUnitRatio(),
              pport.origin[0] * lefDefUnitRatio(), 
              pport.origin[1] * lefDefUnitRatio()
              ); 
          hasOrigin = true; 
          // only consider first port 
          break; 
        }
      }
    }
    */
    if (!hasOrigin) {
      node.set(0, 0, 0, 0);
      dreamplacePrint(
          kWARN, "IO pin: no position specified, set to (%d, %d, %d, %d)\n",
          p.pin_name.c_str(), node.xl(), node.yl(), node.xh(), node.yh());
    } /* else { // IO pins are considered as terminal NI. Their sizes should not affect the functionality of placement.
      node.set(
          center(node).x(), 
          center(node).y(), 
          center(node).x(), 
          center(node).y()
          ); 
    } */
    node.setInitPos(ll(node));
  }

  // initialize macro bounding box 
  macro.setInitOrigin(0, 0);
  macro.set(0, 0, node.width(), node.height());  

  // create and add io pin
  std::pair<index_type, bool> insertMacroPinRet = macro.addMacroPin(p.pin_name);
  dreamplaceAssertMsg(insertMacroPinRet.second,
                      "failed to create virtual io pin: %s.%s",
                      macro.name().c_str(), p.pin_name.c_str());
  MacroPin& iopin = macro.macroPin(insertMacroPinRet.first);
  // reverse direction for primary input and output
  // I hope this will not cause incompatible issues with writer of other formats
  if (limbo::iequals(p.direct, "INPUT"))
    iopin.setDirect(std::string("OUTPUT"));
  else if (limbo::iequals(p.direct, "OUTPUT"))
    iopin.setDirect(std::string("INPUT"));
  else
    iopin.setDirect(p.direct);

  // create and add port for io pin
  if (!p.vLayer.empty()) {
    iopin.macroPorts().push_back(MacroPort());
    MacroPort& macroPort = iopin.macroPorts().back();
    macroPort.setId(iopin.macroPorts().size() - 1);
    for (std::size_t i = 0; i < p.vLayer.size(); ++i) {
      if (!(p.origin[0] == -1 && p.origin[1] == -1)) {
        macroPort.layers().push_back(p.vLayer[i]);
        macroPort.boxes().push_back(MacroPort::box_type(
              (p.origin[0] + p.vBbox[i][0] - node.xl()) * lefDefUnitRatio(), 
              (p.origin[1] + p.vBbox[i][1] - node.yl()) * lefDefUnitRatio(), 
              (p.origin[0] + p.vBbox[i][2] - node.xl()) * lefDefUnitRatio(),
              (p.origin[1] + p.vBbox[i][3] - node.yl()) * lefDefUnitRatio()
              )); 
      }
    }
    deriveMacroPortBbox(macroPort);
  } else if (!p.vPinPort.empty()) {
    for (std::size_t i = 0; i < p.vPinPort.size(); ++i) {
      iopin.macroPorts().push_back(MacroPort());
      MacroPort& macroPort = iopin.macroPorts().back();
      macroPort.setId(iopin.macroPorts().size() - 1);
      DefParser::PinPort const& pport = p.vPinPort[i]; 
      for (std::size_t j = 0; j < pport.vLayer.size(); ++j) {
        if (!(pport.origin[0] == -1 && pport.origin[1] == -1)) {
          macroPort.layers() .push_back(pport.vLayer[j]); 
          macroPort.boxes().push_back(MacroPort::box_type(
                (pport.origin[0] + pport.vBbox[j][0] - node.xl()) * lefDefUnitRatio(), 
                (pport.origin[1] + pport.vBbox[j][1] - node.yl()) * lefDefUnitRatio(), 
                (pport.origin[0] + pport.vBbox[j][2] - node.xl()) * lefDefUnitRatio(),
                (pport.origin[1] + pport.vBbox[j][3] - node.yl()) * lefDefUnitRatio()
                ));
        }
      }
      deriveMacroPortBbox(macroPort);
    }
  }
  deriveMacroPinBbox(iopin);

}
void PlaceDB::resize_def_net(int s) {
  if ((long)m_vNet.capacity() < s)  // only if PlaceDB::prepare() is not called
  {
    m_vNet.reserve(s);
    m_vNetProperty.reserve(s);
  }
}
void PlaceDB::add_def_net(DefParser::Net const& n) {
  // check the validity of nets

  bool ignoreFlag = false;
  // ignore nets with pins less than 2
  if (n.vNetPin.size() < 2)
    ignoreFlag = true;
  else {
    // ignore nets that may cause problems
    bool all_pin_in_one_node = true;
    for (unsigned i = 1, ie = n.vNetPin.size(); i < ie; ++i) {
      if (n.vNetPin[i - 1].first != n.vNetPin[i].first) {
        all_pin_in_one_node = false;
        break;
      }
    }
    if (all_pin_in_one_node) {
      // dreamplacePrint(kWARN, "net %s has all pins belong to the same node or
      // io pins: ignored\n", n.net_name.c_str());  return;
      ignoreFlag = true;
    }
  }

  // create and add net
  std::pair<index_type, bool> insertNetRet = addNet(n.net_name);
  // check duplicate
  if (!insertNetRet.second) {
    m_vDuplicateNet.push_back(n.net_name);
    // dreamplacePrint(kWARN, "duplicate net found in Verilog file: %s\n",
    // n.net_name.c_str());
    return;
  }
  Net& net = m_vNet.at(insertNetRet.first);

  // update ignore flag
  if (ignoreFlag) {
    m_vNetIgnoreFlag[net.id()] = true;
    m_numIgnoredNet += 1;
  }
  // nodes in a net may be IOPin
  net.pins().reserve(n.vNetPin.size());  // reserve enough space
  for (unsigned i = 0, ie = n.vNetPin.size(); i < ie; ++i) {
    index_type nodeId;
    // io pin or node
    string2index_map_type::const_iterator foundNode = m_mNodeName2Index.find(
        (n.vNetPin[i].first == "PIN") ? n.vNetPin[i].second
                                      : n.vNetPin[i].first);
    if (foundNode != m_mNodeName2Index.end())
      nodeId = foundNode->second;
    else {
      dreamplacePrint(kWARN, "Pin not found: %s.%s\n",
                      n.vNetPin[i].first.c_str(), n.vNetPin[i].second.c_str());
      continue;
    }
    Node& node = m_vNode[nodeId];

    // create and add pin
    addPin(n.vNetPin[i].second, net, node);
  }
  net.setWeight(n.net_weight); 
}
void PlaceDB::resize_def_blockage(int n) {
  m_numPlaceBlockages = 0;
  m_vPlaceBlockageIndex.reserve(n);
}
void PlaceDB::add_def_placement_blockage(
    std::vector<std::vector<int> > const& vBbox) {
  char buf[128];
  std::string name;
  for (std::vector<std::vector<int> >::const_iterator it = vBbox.begin(),
                                                      ite = vBbox.end();
       it != ite; ++it) {
    std::vector<int> const& bbox = *it;

    // create virtual macro
    dreamplaceSPrint(kNONE, buf, "DREAMPlacePlaceBlockage%lu",
                     m_vPlaceBlockageIndex.size());
    name = buf;
    std::pair<index_type, bool> insertMacroRet = addMacro(name);
    dreamplaceAssertMsg(
        insertMacroRet.second,
        "failed to create virtual macro for placement blockage: %s",
        name.c_str());
    Macro& macro = m_vMacro.at(insertMacroRet.first);
    // indicate this is placement blockage
    macro.setClassName("DREAMPlace.PlaceBlockage");

    macro.setInitOrigin(
        bbox[0] * lefDefUnitRatio(), 
        bbox[1] * lefDefUnitRatio()
        );
    macro.set(
        0, 
        0, 
        (bbox[2] - bbox[0]) * lefDefUnitRatio(),
        (bbox[3] - bbox[1]) * lefDefUnitRatio()
        );  // adjust to origin (0, 0)

    // create and add virtual node
    std::pair<index_type, bool> insertNodeRet = addNode(name);
    dreamplaceAssertMsg(
        insertNodeRet.second,
        "failed to create virtual node for placement blockage %s",
        name.c_str());
    Node& node = m_vNode.at(insertNodeRet.first);
    NodeProperty& property = m_vNodeProperty.at(node.id());

    property.setMacroId(macro.id());
    node.setStatus(
        PlaceStatusEnum::FIXED);  // placement blockages should always be fixed
    node.setOrient(OrientEnum::UNKNOWN);
    deriveMultiRowAttr(node);
    node.set(
        bbox[0] * lefDefUnitRatio(), 
        bbox[1] * lefDefUnitRatio(), 
        bbox[2] * lefDefUnitRatio(), 
        bbox[3] * lefDefUnitRatio()
        );
    node.setInitPos(ll(node));

    m_vPlaceBlockageIndex.push_back(node.id());
    ++m_numPlaceBlockages;
  }
}
void PlaceDB::resize_def_region(int n) { m_vRegion.reserve(n); }
void PlaceDB::add_def_region(DefParser::Region const& r) {
  std::pair<index_type, bool> insertRet = addRegion(r.region_name);
  index_type regionId = insertRet.first;
  Region& region = m_vRegion.at(regionId);
  region.setType(r.region_type);
  std::vector<Region::box_type>& boxes = region.boxes();
  boxes.reserve(r.vRectangle.size());
  for (index_type i = 0, ie = r.vRectangle.size(); i < ie; ++i) {
    boxes.push_back(
        Region::box_type(
          r.vRectangle[i].at(0) * lefDefUnitRatio(), 
          r.vRectangle[i].at(1) * lefDefUnitRatio(),
          r.vRectangle[i].at(2) * lefDefUnitRatio(), 
          r.vRectangle[i].at(3) * lefDefUnitRatio()
          ));
  }
  // check whether boxes in region overlap with each other
  // as I have the assumption that the boxes should not overlap
  for (index_type i = 0, ie = boxes.size(); i < ie; ++i) {
    Region::box_type box1 = boxes[i];
    for (index_type j = i + 1; j < ie; ++j) {
      Region::box_type box2 = boxes[j];
      Region::box_type::coordinate_type dx =
          std::min(box1.xh(), box2.xh()) - std::max(box1.xl(), box2.xl());
      Region::box_type::coordinate_type dy =
          std::min(box1.yh(), box2.yh()) - std::max(box1.yl(), box2.yl());
      if (dx > 0 && dy > 0) {
        dreamplacePrint(kWARN,
                        "region %s (%u) has overlapped boxes (%d, %d, %d, %d) "
                        "(%d, %d, %d, %d)\n",
                        region.name().c_str(), regionId, box1.xl(), box1.yl(),
                        box1.xh(), box1.yh(), box2.xl(), box2.yl(), box2.xh(),
                        box2.yh());
      }
    }
  }
}
std::pair<PlaceDB::index_type, bool> PlaceDB::addRegion(std::string const& r) {
  index_type regionId;
  string2index_map_type::iterator foundRegion = m_mRegionName2Index.find(r);
  if (foundRegion != m_mRegionName2Index.end())  // already exists
  {
    dreamplacePrint(kWARN, "duplicate region %s found\n", r.c_str());
    regionId = foundRegion->second;
    return std::make_pair(regionId, false);
  } else  // create
  {
    regionId = m_vRegion.size();
    m_vRegion.push_back(Region());
    Region& region = m_vRegion.back();
    region.setName(r);
    region.setId(regionId);
    std::pair<string2index_map_type::iterator, bool> insertRet =
        m_mRegionName2Index.insert(std::make_pair(r, regionId));
    dreamplaceAssertMsg(insertRet.second,
                        "failed to insert region %s to m_mRegionName2Index",
                        r.c_str());
    return std::make_pair(regionId, true);
  }
}
void PlaceDB::resize_def_group(int n) { m_vGroup.reserve(n); }
void PlaceDB::add_def_group(DefParser::Group const& g) {
  index_type groupId;
  string2index_map_type::iterator foundGroup =
      m_mGroupName2Index.find(g.group_name);
  if (foundGroup != m_mGroupName2Index.end()) {
    dreamplacePrint(kWARN, "duplicate group %s found\n", g.group_name.c_str());
    groupId = foundGroup->second;
  } else {
    groupId = m_vGroup.size();
    std::pair<string2index_map_type::iterator, bool> insertRet =
        m_mGroupName2Index.insert(std::make_pair(g.group_name, groupId));
    dreamplaceAssertMsg(insertRet.second,
                        "failed to insert group %s to m_mGroupName2Index",
                        g.group_name.c_str());
    m_vGroup.push_back(Group());
  }
  Group& group = m_vGroup.at(groupId);
  group.setId(groupId);
  group.setName(g.group_name);
  group.nodeNames() = g.vGroupMember;

  // a region may not exist or already exist
  // retrieve the index if already exist
  // create if not exist
  std::pair<index_type, bool> insertRegionRet = addRegion(g.region_name);
  group.setRegion(insertRegionRet.first);

  // node indices in group are not set yet
  // they need to be set in the adjustParams() function
}
void PlaceDB::add_def_track(defiTrack const& t) {
  dreamplacePrint(kWARN, "Track definition in DEF ignored\n");
}
void PlaceDB::add_def_via(defiVia const& v) {
  dreamplacePrint(kWARN, "Via definition in DEF ignored\n");
}
void PlaceDB::add_def_snet(defiNet const& n) {
  dreamplacePrint(kWARN, "SPECIALNET definition in DEF ignored\n");
}
void PlaceDB::add_def_gcellgrid(DefParser::GCellGrid const& g) {
  dreamplacePrint(kWARN, "GCELLGRID definition in DEF ignored\n");
}
void PlaceDB::add_def_route_blockage(std::vector<std::vector<int>> const&, std::string const&) {
  dreamplacePrint(kWARN, "ROUTE BLOCKAGE definition in DEF ignored\n");
}
void PlaceDB::end_def_design() {
  // make sure rows are sorted from bottom to up
  std::sort(m_vRow.begin(), m_vRow.end(), CompareByRowBottomCoord());
  // reset id
  for (unsigned int i = 0, ie = m_vRow.size(); i < ie; ++i) m_vRow[i].setId(i);
#ifdef DEBUG
  for (unsigned int i = 1, ie = m_vRow.size(); i < ie; ++i) {
    dreamplaceAssert(m_vRow[i - 1].yl() < m_vRow[i].yl());
    dreamplaceAssert(m_vRow[i - 1].yh() == m_vRow[i].yl());
  }
#endif
}

///==== Verilog Callbacks ====
void PlaceDB::verilog_module_declaration_cbk(std::string const& module_name, 
        std::vector<VerilogParser::GeneralName> const& vPinName) {
    dreamplaceAssertMsg(module_name == m_designName, 
            "verilog module name %s must match design name %s", 
            module_name.c_str(), 
            m_designName.c_str()
            );
} 
void PlaceDB::verilog_net_declare_cbk(std::string const& netName,
                                      VerilogParser::Range const& range) {
  dreamplaceAssertMsg(range.low == range.high, "do not support bus yet");

  std::pair<index_type, bool> insertNetRet = addNet(netName);
  // check duplicate
  if (!insertNetRet.second)
    dreamplacePrint(kWARN, "duplicate net found in Verilog file: %s\n",
                    netName.c_str());
}
void PlaceDB::verilog_pin_declare_cbk(std::string const& pinName,
                                      unsigned /*type*/,
                                      VerilogParser::Range const& range) {
  // find virtual node for io pin
  index_type nodeId;
  string2index_map_type::const_iterator foundNode =
      m_mNodeName2Index.find(pinName);
  if (foundNode != m_mNodeName2Index.end())
    nodeId = foundNode->second;
  else {
    dreamplacePrint(kWARN, "IO pin not found: %s\n", pinName.c_str());
    return;
  }
  Node& node = m_vNode[nodeId];

  // for io pin, it has the net with the same name as pin name
  std::string const& netName = pinName;
  dreamplaceAssertMsg(range.low == range.high, "do not support bus yet");

  // for io pin, the net name is the same as pin name
  std::pair<index_type, bool> insertNetRet = addNet(netName);
  // check duplicate
  if (!insertNetRet.second) {
    dreamplacePrint(kWARN, "duplicate net found in Verilog file: %s\n",
                    netName.c_str());
    return;
  }
  Net& net = m_vNet.at(insertNetRet.first);
  // add pin
  // macro pin name for virtual node is the same as pin name
  addPin(pinName, net, node);
}
void PlaceDB::verilog_instance_cbk(
    std::string const& macroName, std::string const& instName,
    std::vector<VerilogParser::NetPin> const& vNetPin) {
  string2index_map_type::iterator foundNode = m_mNodeName2Index.find(instName);
  dreamplaceAssertMsg(foundNode != m_mNodeName2Index.end(),
                      "failed to find instance name %s", instName.c_str());
  Node& node = m_vNode.at(foundNode->second);
  Macro const& macro = m_vMacro.at(macroId(node));
  dreamplaceAssertMsg(macro.name() == macroName, "macro name mismatch %s != %s",
                      macroName.c_str(), macro.name().c_str());
  for (std::vector<VerilogParser::NetPin>::const_iterator it = vNetPin.begin(),
                                                          ite = vNetPin.end();
       it != ite; ++it) {
    VerilogParser::NetPin const& np = *it;
    string2index_map_type::iterator foundNet = m_mNetName2Index.find(np.net);
    dreamplaceAssertMsg(foundNet != m_mNetName2Index.end(),
                        "failed to find net %s", np.net.c_str());
    Net& net = m_vNet.at(foundNet->second);

    // add pin
    addPin(np.pin, net, node, instName);
  }
}
///==== Bookshelf Callbacks ====
void PlaceDB::resize_bookshelf_node_terminals(int nn, int nt) {
  m_vNode.reserve(nn + nt);
  m_vNodeProperty.reserve(m_vNode.capacity());
}
void PlaceDB::resize_bookshelf_net(int n) {
  m_vNet.reserve(n);
  m_vNetProperty.reserve(n);
  m_vNetIgnoreFlag.reserve(n);
}
void PlaceDB::resize_bookshelf_pin(int n) { m_vPin.reserve(n); }
void PlaceDB::resize_bookshelf_row(int n) {
  m_vRow.reserve(n);

  // create site
  dreamplaceAssert(m_vSite.empty());
  m_vSite.push_back(Site());
  Site& site = m_vSite.back();
  site.setId(m_vSite.size() - 1);
  site.setName("CoreSite");
  site.setClassName("CORE");
  m_mSiteName2Index[site.name()] = site.id();
  m_coreSiteId = site.id();
}
void PlaceDB::resize_bookshelf_shapes(int /*n*/) {}
void PlaceDB::resize_bookshelf_niterminal_layers(int) {}
void PlaceDB::resize_bookshelf_blockage_layers(int) {}
void PlaceDB::add_bookshelf_terminal(std::string& name, int w, int h) {
  // it seems no difference
  add_bookshelf_node(name, w, h, false);
}
void PlaceDB::add_bookshelf_terminal_NI(std::string& name, int w, int h) {
  // it seems no difference
  add_bookshelf_node(name, w, h, false);
  // regard terminal_NI as IO pins
  Macro& macro = m_vMacro.back();
  macro.setClassName("DREAMPlace.IOPin");

  // bookshelf use terminal_NI and FIXED_NI to denotes IO pins
  // I assume IO pins are appended to the list
  m_numIOPin += 1;
}
void PlaceDB::add_bookshelf_node(std::string& name, int w, int h, bool /*is_cell*/) {
  // TODO: is_cell not used yet, wait for Zizheng's fix in another branch 

  // create and add node
  std::pair<index_type, bool> insertRet = addNode(name);
  // check duplicate
  if (!insertRet.second) {
    dreamplacePrint(kWARN, "duplicate component found in .nodes file: %s\n",
                    name.c_str());
    return;
  }

  Node& node = m_vNode.at(insertRet.first);
  node.set(0, 0, w, h);  // must update width and height

  // create dummy macro
  std::pair<index_type, bool> insertMacroRet = addMacro("DREAMPlace." + name);
  // check duplicate
  dreamplaceAssert(insertMacroRet.second);
  Macro& macro = m_vMacro.at(insertMacroRet.first);
  // only CORE class is considered as the target diearea to optimize
  macro.setClassName("CORE");

  NodeProperty& property = m_vNodeProperty.at(node.id());
  property.setMacroId(insertMacroRet.first);
}
// sort NetPin by node name to avoid a net containing multiple pins from the
// same node
struct SortNetPinByNode {
  bool operator()(BookshelfParser::NetPin const& np1,
                  BookshelfParser::NetPin const& np2) const {
    return np1.node_name < np2.node_name ||
           (np1.node_name == np2.node_name && np1.pin_name < np2.pin_name);
  }
};
struct CompareNetPinByNode {
  bool operator()(BookshelfParser::NetPin const& np1,
                  BookshelfParser::NetPin const& np2) const {
    return np1.node_name == np2.node_name;
  }
};
void PlaceDB::add_bookshelf_net(BookshelfParser::Net const& n) {
  // check the validity of nets

  // if a node has multiple pins in the net, only one is kepted
  std::vector<BookshelfParser::NetPin> vNetPin = n.vNetPin;
#if 1
  std::sort(vNetPin.begin(), vNetPin.end(), SortNetPinByNode());
  std::vector<BookshelfParser::NetPin>::iterator itnp =
      std::unique(vNetPin.begin(), vNetPin.end(), CompareNetPinByNode());
  vNetPin.resize(std::distance(vNetPin.begin(), itnp));
  if (vNetPin.size() < n.vNetPin.size()) {
    // dreamplacePrint(kWARN, "net %s ignore %d pins from same nodes\n",
    // n.net_name.c_str(), n.vNetPin.size()-vNetPin.size());
    m_numNetsWithDuplicatePins += 1;
    m_numPinsDuplicatedInNets += n.vNetPin.size() - vNetPin.size();
  }
#endif

  bool ignoreFlag = false;
  // ignore nets with pins less than 2
  if (vNetPin.size() < 2) ignoreFlag = true;
  // ignore nets that may cause problems
  bool all_pin_in_one_node = true;
  for (unsigned i = 1, ie = vNetPin.size(); i < ie; ++i) {
    if (vNetPin[i - 1].node_name != vNetPin[i].node_name) {
      all_pin_in_one_node = false;
      break;
    }
  }
  if (all_pin_in_one_node) {
    // dreamplacePrint(kWARN, "net %s has all pins belong to the same node or io
    // pins: ignored\n", n.net_name.c_str());  return;
    ignoreFlag = true;
  }

  // create and add net
  std::pair<index_type, bool> insertNetRet = addNet(n.net_name);
  // check duplicate
  if (!insertNetRet.second) {
    dreamplacePrint(kWARN, "duplicate net found in Verilog file: %s\n",
                    n.net_name.c_str());
    return;
  }
  Net& net = m_vNet.at(insertNetRet.first);

  // update ignore flag
  if (ignoreFlag) {
    m_vNetIgnoreFlag[net.id()] = true;
    m_numIgnoredNet += 1;
  }
  // nodes in a net may be IOPin
  net.pins().reserve(vNetPin.size());  // reserve enough space
  for (unsigned i = 0, ie = vNetPin.size(); i < ie; ++i) {
    BookshelfParser::NetPin const& netPin = vNetPin[i];
    index_type nodeId;
    // io pin or node
    string2index_map_type::const_iterator foundNode =
        m_mNodeName2Index.find(netPin.node_name);
    if (foundNode != m_mNodeName2Index.end())
      nodeId = foundNode->second;
    else {
      dreamplacePrint(kWARN, "Pin not found: %s.%s\n", netPin.node_name.c_str(),
                      netPin.pin_name.c_str());
      continue;
    }
    Node& node = m_vNode.at(nodeId);

    // create and add pin
    // assume pin offset starts from center
    createPin(
        net, node,
        SignalDirect((netPin.direct == 'I')
                         ? SignalDirectEnum::INPUT
                         : (netPin.direct == 'O') ? SignalDirectEnum::OUTPUT
                                                  : SignalDirectEnum::INOUT),
        // Point<coordinate_type>(round(netPin.offset[0]+netPin.size[0]/2),
        // round(netPin.offset[1]+netPin.size[1]/2)),
        Point<coordinate_type>(round(netPin.offset[0] + node.width() * 0.5),
                               round(netPin.offset[1] + node.height() * 0.5)),
        std::numeric_limits<index_type>::max());
  }
}
void PlaceDB::add_bookshelf_row(BookshelfParser::Row const& r) {
  // create and add row
  m_vRow.push_back(Row());
  Row& row = m_vRow.back();
  row.setId(m_vRow.size() - 1);

  // only support HORIZONTAL row, because I don't know how to deal with vertical
  // rows
  if (r.orient == "HORIZONTAL") {
    if (r.site_orient_str.empty()) {
      // currently only support 0 and 1
      switch (r.site_orient) {
        case 0:
          row.setOrient(OrientEnum::FS);
          break;
        case 1:
          row.setOrient(OrientEnum::N);
          break;
        default:
          dreamplaceAssertMsg(0, "unknown row orientation %d", r.site_orient);
      }
    } else {
      row.setOrient(r.site_orient_str);
    }
  } else
    dreamplacePrint(kWARN, "unsupported row orientation %s\n",
                    r.orient.c_str());
  row.set(r.origin[0], r.origin[1], r.origin[0] + r.site_width * r.site_num,
          r.origin[1] + r.height);

  row.setStep(r.site_width, 0);

  m_rowBbox.encompass(row);

  // set site
  Site& site = m_vSite.at(m_coreSiteId);
  site.setSize(kX, r.site_width);
  site.setSize(kY, r.height);
}
void PlaceDB::set_bookshelf_node_position(std::string const& name, double x,
                                          double y, std::string const& orient,
                                          std::string const& status,
                                          bool plFlag) {
  string2index_map_type::iterator found = m_mNodeName2Index.find(name);
  if (found == m_mNodeName2Index.end()) {
    dreamplacePrint(kWARN, "component not found from .pl file: %s\n",
                    name.c_str());
    return;
  }
  Node& node = m_vNode.at(found->second);
  moveTo(node, round(x), round(y));  // update position
  node.setOrient(orient);            // update orient
  bool iopinFlag = false;
  if (!plFlag)  // only update when plFlag is false
  {
    if (status.empty())
      node.setStatus(PlaceStatusEnum::PLACED);    // update status
    else if (limbo::iequals(status, "FIXED_NI"))  // IO pin
    {
      iopinFlag = true;
      node.setStatus(PlaceStatusEnum::FIXED);
    } else
      node.setStatus(status);  // update status
    // a heuristic fix for some special cases, first move it to a legal position
    // and then fix it I found in the benchmark from Dr. Chris Chu, there are
    // very big movable macros simply fix them
    if (node.status() != PlaceStatusEnum::FIXED &&
        node.height() > (rowHeight() * DUMMY_FIXED_NUM_ROWS)) {
      dreamplacePrint(kWARN,
                      "detect large movable macros that will be handled "
                      "differently from standard cells: %s %ldx%ld @(%d,%d) "
                      "with %lu pins\n",
                      nodeName(node).c_str(), node.width(), node.height(),
                      node.xl(), node.yl(), node.pins().size());
      node.setStatus(PlaceStatusEnum::DUMMY_FIXED);
    }
    deriveMultiRowAttr(node);  // update MultiRowAttr

    // update statistics
    // may need to change the criteria of fixed cells according to benchmarks
    if (!iopinFlag)  // exclude io pins
    {
      if (node.status() == PlaceStatusEnum::FIXED) {
        m_numFixed += 1;
        m_vFixedNodeIndex.push_back(node.id());
      } else {
        m_numMovable += 1;
        m_vMovableNodeIndex.push_back(node.id());
      }
    }
  }
  if (node.status() == PlaceStatusEnum::FIXED ||
      node.status() == PlaceStatusEnum::DUMMY_FIXED ||
      node.status() == PlaceStatusEnum::PLACED)
    node.setInitPos(ll(node));
}
void PlaceDB::set_bookshelf_net_weight(std::string const& name, double w) {
  string2index_map_type::iterator found = m_mNetName2Index.find(name);
  dreamplaceAssertMsg(found != m_mNetName2Index.end(), "failed to find net %s",
                      name.c_str());
  Net& net = this->net(found->second);
  net.setWeight(w);
}
void PlaceDB::set_bookshelf_shape(BookshelfParser::NodeShape const& shape) {
  string2index_map_type::iterator found =
      m_mNodeName2Index.find(shape.node_name);
  if (found == m_mNodeName2Index.end()) {
    dreamplacePrint(kWARN, "component not found from .shapes file: %s\n",
                    shape.node_name.c_str());
    return;
  }
  Node const& node = m_vNode.at(found->second);
  Macro& macro = m_vMacro.at(this->macroId(node));
  // regard shape boxes as obstruction as a dummy layer called Bookshelf.Shape
  for (index_type i = 0, ie = shape.vShapeBox.size(); i != ie; ++i) {
    coordinate_type xl =
        shape.vShapeBox[i].origin[0] - node.xl() - macro.initOrigin().x();
    coordinate_type yl =
        shape.vShapeBox[i].origin[1] - node.yl() - macro.initOrigin().y();
    macro.obs().add(
        "Bookshelf.Shape",
        Box<coordinate_type>(xl, yl, xl + shape.vShapeBox[i].size[0],
                             yl + shape.vShapeBox[i].size[1]));
  }
}
void PlaceDB::set_bookshelf_route_info(BookshelfParser::RouteInfo const& info) {
  m_numRoutingGrids[kX] = info.numGrids[0];
  m_numRoutingGrids[kY] = info.numGrids[1];
  m_numRoutingGrids[2] = info.numLayers;
  m_vRoutingCapacity[PlanarDirectEnum::VERTICAL].assign(
      info.vVerticalCapacity.begin(), info.vVerticalCapacity.end());
  m_vRoutingCapacity[PlanarDirectEnum::HORIZONTAL].assign(
      info.vHorizontalCapacity.begin(), info.vHorizontalCapacity.end());
  m_vMinWireWidth.assign(info.vMinWireWidth.begin(), info.vMinWireWidth.end());
  m_vMinWireSpacing.assign(info.vMinWireSpacing.begin(),
                           info.vMinWireSpacing.end());
  m_vViaSpacing.assign(info.vViaSpacing.begin(), info.vViaSpacing.end());
  m_routingGridOrigin[kX] = info.gridOrigin[0];
  m_routingGridOrigin[kY] = info.gridOrigin[1];
  m_routingTileSize[kX] = info.tileSize[0];
  m_routingTileSize[kY] = info.tileSize[1];
  m_routingBlockagePorosity = info.blockagePorosity;

  char buf[64];
  for (index_type layer = 0; layer < (index_type)info.numLayers; ++layer) {
    dreamplaceSPrint(kNONE, buf, "%u", layer + 1);
    std::string layerName = buf;
    m_vLayerName.push_back(layerName);
    dreamplaceAssertMsg(
        m_mLayerName2Index.insert(std::make_pair(std::string(layerName), layer))
            .second,
        "failed to insert layer (%s, %u)", layerName.c_str(), layer);
  }
}
void PlaceDB::add_bookshelf_niterminal_layer(std::string const&,
                                             std::string const&) {}
void PlaceDB::add_bookshelf_blockage_layers(
    std::string const& name, std::vector<std::string> const& vLayer) {
  string2index_map_type::iterator found = m_mNodeName2Index.find(name);
  if (found == m_mNodeName2Index.end()) {
    dreamplacePrint(kWARN, "component not found from .shapes file: %s\n",
                    name.c_str());
    return;
  }
  Node const& node = m_vNode.at(found->second);
  Macro& macro = m_vMacro.at(this->macroId(node));

  if (macro.obs().empty())  // no shape
  {
    // necessary to add a shape indicator
    macro.obs().add("Bookshelf.Shape",
                    Box<coordinate_type>(0, 0, node.width(), node.height()));
    for (std::vector<std::string>::const_iterator it = vLayer.begin();
         it != vLayer.end(); ++it) {
      std::string const& layerName = *it;
      macro.obs().add(layerName,
                      Box<coordinate_type>(0, 0, node.width(), node.height()));
    }
  } else  // has shapes
  {
    MacroObs::ObsConstIterator foundObs =
        macro.obs().obsMap().find("Bookshelf.Shape");
    dreamplaceAssertMsg(foundObs != macro.obs().obsMap().end(),
                        "Node %s must have Bookshelf.Shape layer defined in "
                        "obstruction if obstruction exists",
                        name.c_str());
    std::vector<MacroObs::box_type> const& vBox = foundObs->second;
    for (std::vector<MacroObs::box_type>::const_iterator itb = vBox.begin();
         itb != vBox.end(); ++itb) {
      for (std::vector<std::string>::const_iterator it = vLayer.begin();
           it != vLayer.end(); ++it) {
        std::string const& layerName = *it;
        macro.obs().add(layerName, Box<coordinate_type>(itb->xl(), itb->yl(),
                                                        itb->xh(), itb->yh()));
      }
    }
  }
}
void PlaceDB::set_bookshelf_design(std::string& name) {
  m_designName.swap(name);
}
void PlaceDB::bookshelf_end() {
  // parsing bookshelf format finishes
  // now it is necessary to init data that is not set in bookshelf
  m_dieArea = m_rowBbox;  // set die area
  // iterate through all fixed cells to encompass them in the die area
  // we use the bounding box of die area to check whether all pins of a net is
  // ignored so the die area should be large enough to be differentiated from
  // all other nets
  for (FixedNodeConstIterator it = fixedNodeBegin(); it.inRange(); ++it)
    m_dieArea.encompass(*it);

  // update lef/def unit
  m_lefUnit = 1000;
  m_defUnit = 1000;
}

void PlaceDB::reportStats() {
  dreamplacePrint(kNONE,
                  "========================= benchmark statistics "
                  "=========================\n");
  reportStatsKernel();
  m_benchMetrics.print();
  dreamplacePrint(kNONE,
                  "============================================================"
                  "============\n");
}

void PlaceDB::reportStatsKernel() {
  if (!m_benchMetrics.initPlaceDBFlag) {
    dreamplacePrint(kINFO,
                    "size of Box object %u bytes, Object object %u bytes\n",
                    sizeof(Box<coordinate_type>), sizeof(Object));
    dreamplacePrint(
        kINFO, "size of Node object %u bytes, NodeProperty object %u bytes\n",
        sizeof(Node), sizeof(NodeProperty));
    dreamplacePrint(
        kINFO, "size of Net object %u bytes, NetProperty object %u bytes\n",
        sizeof(Net), sizeof(NetProperty));
    m_benchMetrics.designName = designName();
    m_benchMetrics.lefUnit = lefUnit();
    m_benchMetrics.defUnit = defUnit();
    m_benchMetrics.numMacro = numMacro();
    m_benchMetrics.numNodes = nodes().size();
    m_benchMetrics.numMovable = numMovable();
    m_benchMetrics.numFixed = numFixed();
    m_benchMetrics.numIOPin = numIOPin();
    m_benchMetrics.numPlaceBlockage = numPlaceBlockages();
    m_benchMetrics.numMultiRowMovable = numMultiRowMovable();
    if (m_benchMetrics.numMultiRowMovable)  // only evaluate when there are
                                            // multi-row cells
    {
      m_benchMetrics.num2RowMovable = numKRowMovable(2);
      m_benchMetrics.num3RowMovable = numKRowMovable(3);
      m_benchMetrics.num4RowMovable = numKRowMovable(4);
    } else {
      m_benchMetrics.num2RowMovable = 0;
      m_benchMetrics.num3RowMovable = 0;
      m_benchMetrics.num4RowMovable = 0;
    }
    m_benchMetrics.numNets = nets().size();
    m_benchMetrics.numRows = rows().size();
    m_benchMetrics.numPins = pins().size();
    m_benchMetrics.siteWidth = siteWidth();
    m_benchMetrics.rowHeight = rowHeight();
    m_benchMetrics.dieArea.set(dieArea().xl(), dieArea().yl(), dieArea().xh(),
                               dieArea().yh());
    m_benchMetrics.rowBbox.set(rowBbox().xl(), rowBbox().yl(), rowBbox().xh(),
                               rowBbox().yh());
    m_benchMetrics.movableUtil = computeMovableUtil();
    m_benchMetrics.numIgnoredNet = numIgnoredNet();
    ;
    m_benchMetrics.numDuplicateNet = m_vDuplicateNet.size();

    m_benchMetrics.initPlaceDBFlag = true;

    dreamplaceAssertMsg(numMovable() == m_vMovableNodeIndex.size(),
                        "inconsistent number of movable cells: %lu != %lu\n",
                        numMovable(), m_vMovableNodeIndex.size());
    dreamplaceAssertMsg(numFixed() == m_vFixedNodeIndex.size(),
                        "inconsistent number of fixed cells: %lu != %lu\n",
                        numFixed(), m_vFixedNodeIndex.size());
    std::size_t countNum =
        std::count(m_vNetIgnoreFlag.begin(), m_vNetIgnoreFlag.end(), true);
    dreamplaceAssertMsg(numIgnoredNet() == countNum,
                        "inconsistent number of ignored nets: %lu != %lu\n",
                        numIgnoredNet(), countNum);
  }
}

bool PlaceDB::write(std::string const& filename) const {
  // char buf[256];
  // dreamplaceSPrint(kINFO, buf, "writing placement solution takes %%t seconds
  // CPU, %%w seconds real\n");  boost::timer::auto_cpu_timer timer (buf);

  return write(filename, userParam().fileFormat, NULL, NULL);
}

bool PlaceDB::write(std::string const& filename, SolutionFileFormat ff,
                    PlaceDB::coordinate_type const* x,
                    PlaceDB::coordinate_type const* y) const {
  bool flag = false;
  switch (ff) {
    case DEF: {
      std::vector<index_type> vNodeIndex = m_vMovableNodeIndex;
      vNodeIndex.insert(vNodeIndex.end(), m_vFixedNodeIndex.begin(),
                        m_vFixedNodeIndex.end());
      flag = DefWriter(*this).write(filename, userParam().defInput, m_vNode,
                                    vNodeIndex, x, y);
    } break;
    case DEFSIMPLE: {
      std::vector<index_type> vNodeIndex = m_vMovableNodeIndex;
      vNodeIndex.insert(vNodeIndex.end(), m_vFixedNodeIndex.begin(),
                        m_vFixedNodeIndex.end());
      flag = DefWriter(*this).writeSimple(filename, defVersion(), designName(),
                                          m_vNode, vNodeIndex, x, y);
    } break;
    case BOOKSHELF:
      flag = BookShelfWriter(*this).write(filename, x, y);
      break;
    case BOOKSHELFALL:
      flag = BookShelfWriter(*this).writeAll(filename, designName(), x, y);
      break;
    default:
      dreamplacePrint(kERROR, "unknown solution format at line %u\n", __LINE__);
      break;
  }
  return flag;
}

std::pair<PlaceDB::index_type, bool> PlaceDB::addNode(std::string const& n) {
  string2index_map_type::iterator found = m_mNodeName2Index.find(n);
  if (found != m_mNodeName2Index.end())  // already exists
    return std::make_pair(found->second, false);
  else  // create
  {
    m_vNode.push_back(Node());
    m_vNodeProperty.push_back(NodeProperty());
    Node& node = m_vNode.back();
    NodeProperty& property = m_vNodeProperty.back();
    property.setName(n);
    node.setId(m_vNode.size() - 1);
    std::pair<string2index_map_type::iterator, bool> insertRet =
        m_mNodeName2Index.insert(std::make_pair(property.name(), node.id()));
    dreamplaceAssertMsg(insertRet.second, "failed to insert node (%s, %d)",
                        property.name().c_str(), node.id());

    return std::make_pair(node.id(), true);
  }
}

std::pair<PlaceDB::index_type, bool> PlaceDB::addMacro(std::string const& n) {
  string2index_map_type::iterator found = m_mMacroName2Index.find(n);
  if (found != m_mMacroName2Index.end())  // already exists
    return std::make_pair(found->second, false);
  else  // create
  {
    m_vMacro.push_back(Macro());
    Macro& macro = m_vMacro.back();
    macro.setName(n);
    macro.setId(m_vMacro.size() - 1);
    std::pair<string2index_map_type::iterator, bool> insertRet =
        m_mMacroName2Index.insert(std::make_pair(macro.name(), macro.id()));
    dreamplaceAssertMsg(insertRet.second, "failed to insert macro (%s, %d)",
                        macro.name().c_str(), macro.id());

    // TODO: this may not be correct
    // wait for Zizheng's fix in another branch
    m_numMacro = m_vMacro.size();  // update number of macros

    return std::make_pair(macro.id(), true);
  }
}

std::pair<PlaceDB::index_type, bool> PlaceDB::addNet(std::string const& n) {
  string2index_map_type::iterator found = m_mNetName2Index.find(n);
  if (found != m_mNetName2Index.end())  // already exists
    return std::make_pair(found->second, false);
  else  // create
  {
    m_vNet.push_back(Net());
    m_vNetProperty.push_back(NetProperty());
    m_vNetIgnoreFlag.push_back(false);
    Net& net = m_vNet.back();
    NetProperty& property = m_vNetProperty.back();
    property.setName(n);
    net.setId(m_vNet.size() - 1);
    std::pair<string2index_map_type::iterator, bool> insertRet =
        m_mNetName2Index.insert(std::make_pair(property.name(), net.id()));
    dreamplaceAssertMsg(insertRet.second, "failed to insert net (%s, %d)",
                        property.name().c_str(), net.id());

    return std::make_pair(net.id(), true);
  }
}

void PlaceDB::addPin(std::string const& macroPinName, Net& net, Node& node, std::string instName) {
  Macro const& macro = m_vMacro.at(macroId(node));
  index_type macroPinId = macro.macroPinIndex(macroPinName);
  dreamplaceAssertMsg(macroPinId < std::numeric_limits<index_type>::max(),
                      "failed to find pin %s in macro %s", macroPinName.c_str(),
                      macro.name().c_str());
  if (instName.empty())
    addPin(macroPinId, net, node, macroPinName);
  else
    addPin(macroPinId, net, node, instName + ":" + macroPinName);
}

void PlaceDB::addPin(index_type macroPinId, Net& net, Node& node, std::string pinName) {
  Macro const& macro = m_vMacro.at(macroId(node));
  MacroPin const& mpin = macro.macroPin(macroPinId);

  // create and add pin
  createPin(net, node, mpin.direct(), center(mpin.bbox()), macroPinId, pinName);
}
Pin& PlaceDB::createPin(Net& net, Node& node, SignalDirect const& direct,
                        Point<PlaceDB::coordinate_type> const& offset,
                        PlaceDB::index_type macroPinId,
                        std::string pinName) {
  // create and add pin
  m_vPin.push_back(Pin());
  Pin& pin = m_vPin.back();
  pin.setId(m_vPin.size() - 1);

  // Assign attributes to the current pin.
  pin.setNodeId(node.id())
     .setNetId(net.id())
     .setMacroPinId(macroPinId)
     .setOffset(offset)
     .setDirect(direct)
     .setName(pinName);

  // add pin index to net and node
  node.pins().push_back(pin.id());
  net.pins().push_back(pin.id());
  if (pin.direct() ==
      SignalDirectEnum::OUTPUT)  // set the first pin in the net to be source
    std::swap(net.pins().front(), net.pins().back());

  return pin;
}
void PlaceDB::deriveMultiRowAttr(Node& node) {
  // assume node status and sizes have already been set
  // the node here may be fixed instances, or io pins, or cells

  // currently there is no sign to tell whether a multi-row cell should align to
  // power line assume all even-row cells align to N/FN rows
  if (node.height() ==
      rowHeight())  // single-row cell, may be some fixed instances
    node.setMultiRowAttr(MultiRowAttrEnum::SINGLE_ROW);
  else if (node.status() == PlaceStatusEnum::FIXED)  // large fixed instances
                                                     // can be aligned to any
                                                     // row
    node.setMultiRowAttr(MultiRowAttrEnum::MULTI_ROW_ANY);
  else if (node.height() % rowHeight())  // odd-row cells
    node.setMultiRowAttr(MultiRowAttrEnum::MULTI_ROW_ANY);
  else  // even-row cells
  {
    Row const& row = this->row(0);  // check first row
    if (row.orient() == OrientEnum::N ||
        row.orient() == OrientEnum::FN)  // assume align to N/FN rows
      node.setMultiRowAttr(MultiRowAttrEnum::MULTI_ROW_N);
    else  // assume align to S/FS rows
      node.setMultiRowAttr(MultiRowAttrEnum::MULTI_ROW_S);
  }
}
PlaceDB::index_type PlaceDB::computeFlipFlag(Orient const& origOrient,
                                             Orient const& newOrient) const {
  bool vflip = false;
  bool hflip = false;

  if (newOrient == Orient::vflip(origOrient))  // only vertically flipped
    vflip = true;
  else if (newOrient == Orient::hflip(origOrient))  // only horizontally flipped
    hflip = true;
  else if (newOrient ==
           Orient::hflip(Orient::vflip(
               origOrient)))  // both vertically and horizontally flipped
    vflip = hflip = true;

  // return encoded (hflip, vflip) pair
  return (hflip << 1) + vflip;
}
Point<PlaceDB::coordinate_type> PlaceDB::getNodePinOffset(
    Pin const& pin, Orient const& origOrient, Orient const& newOrient) const {
  index_type hvflip = computeFlipFlag(origOrient, newOrient);
  Node const& node = this->node(pin.nodeId());
  return Point<coordinate_type>(
      (hvflip & 2) ? node.width() - pin.offset().x() : pin.offset().x(),
      (hvflip & 1) ? node.height() - pin.offset().y() : pin.offset().y());
}
void PlaceDB::updateNodePinOffset(Node const& node, Orient const& origOrient,
                                  Orient const& newOrient) {
  index_type hvflip = computeFlipFlag(origOrient, newOrient);

  for (std::vector<index_type>::const_iterator it = node.pins().begin(),
                                               ite = node.pins().end();
       it != ite; ++it) {
    Pin& pin = this->pin(*it);
    pin.setOffset(Point<coordinate_type>(
        (hvflip & 2) ? node.width() - pin.offset().x() : pin.offset().x(),
        (hvflip & 1) ? node.height() - pin.offset().y() : pin.offset().y()));
  }
}

void PlaceDB::prepare(unsigned numRows, unsigned numNodes, unsigned numIOPin,
                      unsigned numNets, unsigned numBlockages) {
  m_vRow.reserve(numRows);
  m_vNode.reserve(numNodes + numIOPin);
  m_vNodeProperty.reserve(m_vNode.capacity());
  m_vNet.reserve(numNets);
  m_vNetProperty.reserve(numNets);
  m_vNetIgnoreFlag.reserve(numNets);
  m_vPlaceBlockageIndex.reserve(numBlockages);
}

PlaceDB::coordinate_type PlaceDB::pinPos(PlaceDB::index_type pinId,
                                         Direction1DType d) const {
  return pinPos(m_vPin.at(pinId), d);
}

PlaceDB::coordinate_type PlaceDB::pinPos(Pin const& pin,
                                         Direction1DType d) const {
  Node const& node = m_vNode.at(pin.nodeId());
  return node.pinPos(pin, d);
}

Point<PlaceDB::coordinate_type> PlaceDB::pinPos(
    PlaceDB::index_type pinId) const {
  return pinPos(m_vPin.at(pinId));
}
Point<PlaceDB::coordinate_type> PlaceDB::pinPos(Pin const& pin) const {
  Node const& node = m_vNode.at(pin.nodeId());
  return node.pinPos(pin);
}
Box<PlaceDB::coordinate_type> PlaceDB::pinBbox(
    PlaceDB::index_type pinId) const {
  return pinBbox(m_vPin.at(pinId));
}
Box<PlaceDB::coordinate_type> PlaceDB::pinBbox(Pin const& pin) const {
  MacroPin const& mPin = macroPin(pin);
  Node const& node = m_vNode.at(pin.nodeId());
  Box<coordinate_type> box = mPin.bbox();
  return move(box, node.xl(), node.yl());
}
MacroPin const& PlaceDB::macroPin(PlaceDB::index_type pinId) const {
  return macroPin(m_vPin.at(pinId));
}
MacroPin const& PlaceDB::macroPin(Pin const& pin) const {
  Node const& node = m_vNode.at(pin.nodeId());
  Macro const& macro = m_vMacro.at(macroId(node));
  return macro.macroPin(pin.macroPinId());
}
MovableNodeIterator PlaceDB::movableNodeBegin() {
  return MovableNodeIterator(0, 0, m_numMovable, this);
}
MovableNodeIterator PlaceDB::movableNodeEnd() {
  return MovableNodeIterator(m_numMovable, 0, m_numMovable, this);
}
MovableNodeConstIterator PlaceDB::movableNodeBegin() const {
  return MovableNodeConstIterator(0, 0, m_numMovable, this);
}
MovableNodeConstIterator PlaceDB::movableNodeEnd() const {
  return MovableNodeConstIterator(m_numMovable, 0, m_numMovable, this);
}
FixedNodeIterator PlaceDB::fixedNodeBegin() {
  return FixedNodeIterator(0, 0, m_numFixed, this);
}
FixedNodeIterator PlaceDB::fixedNodeEnd() {
  return FixedNodeIterator(m_numFixed, 0, m_numFixed, this);
}
FixedNodeConstIterator PlaceDB::fixedNodeBegin() const {
  return FixedNodeConstIterator(0, 0, m_numFixed, this);
}
FixedNodeConstIterator PlaceDB::fixedNodeEnd() const {
  return FixedNodeConstIterator(m_numFixed, 0, m_numFixed, this);
}
PlaceBlockageIterator PlaceDB::placeBlockageBegin() {
  return PlaceBlockageIterator(0, 0, m_numPlaceBlockages, this);
}
PlaceBlockageIterator PlaceDB::placeBlockageEnd() {
  return PlaceBlockageIterator(m_numPlaceBlockages, 0, m_numPlaceBlockages,
                               this);
}
PlaceBlockageConstIterator PlaceDB::placeBlockageBegin() const {
  return PlaceBlockageConstIterator(0, 0, m_numPlaceBlockages, this);
}
PlaceBlockageConstIterator PlaceDB::placeBlockageEnd() const {
  return PlaceBlockageConstIterator(m_numPlaceBlockages, 0, m_numPlaceBlockages,
                                    this);
}
IOPinNodeIterator PlaceDB::iopinNodeBegin() {
  index_type first = m_numMovable + m_numFixed + m_numPlaceBlockages;
  return IOPinNodeIterator(first, first, m_vNode.size(), this);
}
IOPinNodeIterator PlaceDB::iopinNodeEnd() {
  index_type first = m_numMovable + m_numFixed + m_numPlaceBlockages;
  index_type last = m_vNode.size();
  return IOPinNodeIterator(last, first, last, this);
}
IOPinNodeConstIterator PlaceDB::iopinNodeBegin() const {
  index_type first = m_numMovable + m_numFixed + m_numPlaceBlockages;
  index_type last = m_vNode.size();
  return IOPinNodeConstIterator(first, first, last, this);
}
IOPinNodeConstIterator PlaceDB::iopinNodeEnd() const {
  index_type first = m_numMovable + m_numFixed + m_numPlaceBlockages;
  index_type last = m_vNode.size();
  return IOPinNodeConstIterator(last, first, last, this);
}
CellMacroIterator PlaceDB::cellMacroBegin() {
  return CellMacroIterator(0, 0, m_numMacro, this);
}
CellMacroIterator PlaceDB::cellMacroEnd() {
  return CellMacroIterator(m_numMacro, 0, m_numMacro, this);
}
CellMacroConstIterator PlaceDB::cellMacroBegin() const {
  return CellMacroConstIterator(0, 0, m_numMacro, this);
}
CellMacroConstIterator PlaceDB::cellMacroEnd() const {
  return CellMacroConstIterator(m_numMacro, 0, m_numMacro, this);
}
IOPinMacroIterator PlaceDB::iopinMacroBegin() {
  index_type last = m_vMacro.size();
  return IOPinMacroIterator(m_numMacro, m_numMacro, last, this);
}
IOPinMacroIterator PlaceDB::iopinMacroEnd() {
  index_type last = m_vMacro.size();
  return IOPinMacroIterator(last, m_numMacro, last, this);
}
IOPinMacroConstIterator PlaceDB::iopinMacroBegin() const {
  index_type last = m_vMacro.size();
  return IOPinMacroConstIterator(m_numMacro, m_numMacro, last, this);
}
IOPinMacroConstIterator PlaceDB::iopinMacroEnd() const {
  index_type last = m_vMacro.size();
  return IOPinMacroConstIterator(last, m_numMacro, last, this);
}
PlaceDB::index_type PlaceDB::getLayer(std::string const& layerName) const {
  string2index_map_type::const_iterator found =
      m_mLayerName2Index.find(layerName);
  dreamplaceAssertMsg(found != m_mLayerName2Index.end(),
                      "Layer not found: %s\n", layerName.c_str());
  return found->second;
}
std::string PlaceDB::getLayerName(PlaceDB::index_type layer) const {
  return m_vLayerName.at(layer);
}

void PlaceDB::adjustParams() {
  dreamplacePrint(kWARN, "%lu nets with %lu pins from same nodes\n",
                  m_numNetsWithDuplicatePins, m_numPinsDuplicatedInNets);
  dreamplacePrint(
      kWARN, "%lu nets should be ignored due to not enough pins\n",
      std::count(m_vNetIgnoreFlag.begin(), m_vNetIgnoreFlag.end(), true));

  // sort nodes such that
  // movable cells are followed by fixed cells
  sortNodeByPlaceStatus();
  // sort nets and pins such that
  // nets are ordered from small to large degrees
  // pins are ordered to have bulk locations for each net
  if (userParam().sortNetsByDegree) {
    sortNetByDegree();
  }

  // some input parameters are not compatible
  // set max displacement to database unit
  m_maxDisplace = (coordinate_type)floor(userParam().maxDisplace * defUnit());

#if 0  // moved to AlgoDB for more accurate estimation
    // compute target utilizations if not set
    if (userParam().targetUtil < std::numeric_limits<double>::epsilon())
        userParam().targetUtil = computeMovableUtil(); // set to average utilization
    if (userParam().targetPinUtil < std::numeric_limits<double>::epsilon())
        userParam().targetPinUtil = std::max(computePinUtil(), 0.3); // set to average utilization
    if (userParam().targetPPR < std::numeric_limits<double>::epsilon())
        userParam().targetPPR = 0.5; // set to empirical utilization
#endif

  // will be processed later in PyPlaceDB::convertOrient()
  //// must adjust the pin offset to orientation for movable and fixed nodes
  //// since the offset is w.r.t orientation N
  //for (index_type i = 0, ie = numMovable() + numFixed(); i < ie; ++i) {
  //  Node const& node = this->node(i);
  //  updateNodePinOffset(node, OrientEnum::N, node.orient());
  //}

  // process region groups, like fence region 
  processGroups(); 
}

PlaceDB::manhattan_distance_type PlaceDB::minMovableNodeWidth() const {
  // something tricky here
  // it will be faster to go through all cell types instead of all nodes
  // but sometimes, in a layout, not all cell types are adopted
  // so it is more accurate to iterate through all nodes
  manhattan_distance_type width =
      std::numeric_limits<manhattan_distance_type>::max();
  for (MovableNodeConstIterator it = movableNodeBegin(); it.inRange(); ++it)
    width = std::min(width, it->width());
  return width;
}
PlaceDB::manhattan_distance_type PlaceDB::maxMovableNodeWidth() const {
  manhattan_distance_type width =
      std::numeric_limits<manhattan_distance_type>::min();
  for (MovableNodeConstIterator it = movableNodeBegin(); it.inRange(); ++it)
    width = std::max(width, it->width());
  return width;
}
PlaceDB::manhattan_distance_type PlaceDB::avgMovableNodeWidth() const {
  manhattan_distance_type width = 0;
  for (MovableNodeConstIterator it = movableNodeBegin(); it.inRange(); ++it)
    width += it->width();
  return width / numMovable();
}
PlaceDB::index_type PlaceDB::totalMovableNodeArea() const {
  index_type area = 0;
  for (MovableNodeConstIterator it = movableNodeBegin(); it.inRange(); ++it)
    area += (it->width() / siteWidth()) *
            (it->height() / rowHeight());  // avoid overflow
  return area;
}
PlaceDB::index_type PlaceDB::totalFixedNodeArea() const {
  index_type area = 0;
  for (FixedNodeConstIterator it = fixedNodeBegin(); it.inRange(); ++it)
    area += (it->width() / siteWidth()) *
            (it->height() / rowHeight());  // avoid overflow
  return area;
}
PlaceDB::index_type PlaceDB::totalRowArea() const {
  index_type area = 0;
  for (std::vector<Row>::const_iterator it = m_vRow.begin(), ite = m_vRow.end();
       it != ite; ++it)
    area += it->width() / siteWidth();  // avoid overflow
  return area;
}
double PlaceDB::computeMovableUtil() const {
  return (totalMovableNodeArea() + std::numeric_limits<double>::epsilon()) /
         (totalRowArea() - totalFixedNodeArea() +
          std::numeric_limits<double>::epsilon());
}
double PlaceDB::computePinUtil() const {
  std::size_t numSites = 0;
  for (std::vector<Row>::const_iterator it = rows().begin(); it != rows().end();
       ++it) {
    Row const& row = *it;
    numSites += row.width() / siteWidth();
  }
  // it should be noted that we already know the total number of pins in the
  // layout
  return (double)pins().size() / numSites;
}
std::size_t PlaceDB::numMultiRowMovable() const {
  index_type num = 0;
  for (MovableNodeConstIterator it = movableNodeBegin(); it.inRange(); ++it)
    if (isMultiRowMovable(*it)) num += 1;
  return num;
}
std::size_t PlaceDB::numKRowMovable(PlaceDB::index_type k) const {
  index_type num = 0;
  for (MovableNodeConstIterator it = movableNodeBegin(); it.inRange(); ++it)
    if (it->height() == rowHeight() * k) num += 1;
  return num;
}
void PlaceDB::printNode(PlaceDB::index_type id) const {
  Node const& node = nodes().at(id);
  dreamplacePrint(kNONE, "node %u: \n", node.id());
  for (index_type i = 0; i < node.pins().size(); ++i) {
    Pin const& pin = pins().at(node.pins().at(i));
    dreamplacePrint(kNONE, "[%u] pin %u, net %u, offset (%d,%d)\n", i, pin.id(),
                    pin.netId(), pin.offset().x(), pin.offset().y());
  }
}
void PlaceDB::printNet(PlaceDB::index_type id) const {
  Net const& net = nets().at(id);
  dreamplacePrint(kNONE, "net %u: \n", net.id());
  for (index_type i = 0; i < net.pins().size(); ++i) {
    Pin const& pin = pins().at(net.pins().at(i));
    dreamplacePrint(kNONE, "[%u] pin %u, node %u, offset (%d,%d)\n", i,
                    pin.id(), pin.nodeId(), pin.offset().x(), pin.offset().y());
  }
}

struct ArgSortNetByDegree {
  std::vector<Net> const& vNet;

  ArgSortNetByDegree(std::vector<Net> const& v) : vNet(v) {}
  bool operator()(PlaceDB::index_type i, PlaceDB::index_type j) const {
    PlaceDB::index_type degree1 = vNet[i].pins().size();
    PlaceDB::index_type degree2 = vNet[j].pins().size();
    return degree1 < degree2 || (degree1 == degree2 && i < j);
  }
};

struct ArgSortPinByNet {
  std::vector<Pin> const& vPin;

  ArgSortPinByNet(std::vector<Pin> const& v) : vPin(v) {}
  bool operator()(PlaceDB::index_type i, PlaceDB::index_type j) const {
    PlaceDB::index_type net_id1 = vPin[i].netId();
    PlaceDB::index_type net_id2 = vPin[j].netId();
    return net_id1 < net_id2 || (net_id1 == net_id2 && i < j);
  }
};

void PlaceDB::sortNetByDegree() {
  dreamplacePrint(kINFO,
                  "sort nets from small degree to large degree and pins with "
                  "neighboring pins belonging to the same net\n");
  // sort m_vNet, m_vNetProperty, m_mNetName2Index
  // map order to net id
  std::vector<index_type> vNetOrder(m_vNet.size());
  for (index_type i = 0, ie = vNetOrder.size(); i != ie; ++i) vNetOrder[i] = i;

  std::sort(vNetOrder.begin(), vNetOrder.end(), ArgSortNetByDegree(m_vNet));

  // map net id to order
  std::vector<index_type> vNetId2Order(m_vNet.size());
  for (index_type i = 0, ie = vNetOrder.size(); i != ie; ++i)
    vNetId2Order[vNetOrder[i]] = i;
  // update m_mNetName2Index
  for (string2index_map_type::iterator it = m_mNetName2Index.begin(),
                                       ite = m_mNetName2Index.end();
       it != ite; ++it) {
    it->second = vNetId2Order[it->second];
  }

  // for all elements to put in place
  for (index_type i = 0; i < m_vNet.size() - 1; ++i) {
    // while the element i is not yet in place
    while (i != vNetId2Order[i]) {
      // swap it with the element at its final place
      index_type alt = vNetId2Order[i];

      std::swap(m_vNet[i], m_vNet[alt]);
      std::swap(m_vNetProperty[i], m_vNetProperty[alt]);

      std::swap(vNetId2Order[i], vNetId2Order[alt]);
    }
  }
  for (index_type i = 1, ie = vNetOrder.size(); i != ie; ++i) {
    dreamplaceAssert(m_vNet[i].id() == vNetOrder[i]);
    dreamplaceAssertMsg(m_vNet[i - 1].pins().size() <= m_vNet[i].pins().size(),
                        "permuting nets error");
  }
  // update net id and pin to net id
  for (index_type i = 0, ie = m_vNet.size(); i != ie; ++i) {
    Net& net = m_vNet[i];
    for (std::vector<index_type>::const_iterator it = net.pins().begin(),
                                                 ite = net.pins().end();
         it != ite; ++it) {
      // we have not update the net id yet
      // so it should be consistent
      dreamplaceAssert(m_vPin[*it].netId() == net.id());
      m_vPin[*it].setNetId(i);
    }
    net.setId(i);
  }

  // sort m_vPin, m_vNode
  std::vector<index_type> vPinOrder(m_vPin.size());
  for (index_type i = 0, ie = vPinOrder.size(); i != ie; ++i) vPinOrder[i] = i;

  std::sort(vPinOrder.begin(), vPinOrder.end(), ArgSortPinByNet(m_vPin));

  // map net id to order
  std::vector<index_type> vPinId2Order(m_vPin.size());
  for (index_type i = 0, ie = vPinOrder.size(); i != ie; ++i)
    vPinId2Order[vPinOrder[i]] = i;

  // for all elements to put in place
  for (index_type i = 0; i < m_vPin.size() - 1; ++i) {
    // while the element i is not yet in place
    while (i != vPinId2Order[i]) {
      // swap it with the element at its final place
      index_type alt = vPinId2Order[i];

      std::swap(m_vPin[i], m_vPin[alt]);

      std::swap(vPinId2Order[i], vPinId2Order[alt]);
    }
  }
  for (index_type i = 1, ie = vPinOrder.size(); i != ie; ++i) {
    dreamplaceAssert(m_vPin[i].id() == vPinOrder[i]);
    dreamplaceAssertMsg(m_vPin[i - 1].netId() <= m_vPin[i].netId(),
                        "permuting pins error");
  }
  // update pins in node
  for (index_type i = 0, ie = vPinOrder.size(); i != ie; ++i)
    vPinId2Order[vPinOrder[i]] = i;
  for (std::vector<Node>::iterator it = m_vNode.begin(), ite = m_vNode.end();
       it != ite; ++it) {
    Node& node = *it;
    for (std::vector<index_type>::iterator itp = node.pins().begin(),
                                           itpe = node.pins().end();
         itp != itpe; ++itp) {
      // since the pin id has not been updated yet
      // we can check the correctness
      dreamplaceAssert(m_vPin[vPinId2Order[*itp]].id() == *itp);
      *itp = vPinId2Order[*itp];
    }
  }
  // update pins in net
  for (std::vector<Net>::iterator it = m_vNet.begin(), ite = m_vNet.end();
       it != ite; ++it) {
    Net& net = *it;
    for (std::vector<index_type>::iterator itp = net.pins().begin(),
                                           itpe = net.pins().end();
         itp != itpe; ++itp) {
      // since the pin id has not been updated yet
      // we can check the correctness
      dreamplaceAssert(m_vPin[vPinId2Order[*itp]].id() == *itp);
      *itp = vPinId2Order[*itp];
    }
  }
  // update pin id
  for (index_type i = 0, ie = m_vPin.size(); i != ie; ++i) {
    m_vPin[i].setId(i);
  }

#ifdef DEBUG
  // check pins, nodes, and nets
  for (std::vector<Node>::const_iterator it = m_vNode.begin(),
                                         ite = m_vNode.end();
       it != ite; ++it) {
    Node const& node = *it;
    for (std::vector<index_type>::const_iterator itp = node.pins().begin(),
                                                 itpe = node.pins().end();
         itp != itpe; ++itp) {
      Pin const& pin = m_vPin[*itp];
      dreamplaceAssert(pin.nodeId() == node.id());
    }
  }
  for (std::vector<Net>::const_iterator it = m_vNet.begin(), ite = m_vNet.end();
       it != ite; ++it) {
    Net const& net = *it;
    for (std::vector<index_type>::const_iterator itp = net.pins().begin(),
                                                 itpe = net.pins().end();
         itp != itpe; ++itp) {
      Pin const& pin = m_vPin[*itp];
      dreamplaceAssert(pin.netId() == net.id());
    }
  }
#endif
}

void PlaceDB::sortNodeByPlaceStatus() {
  dreamplacePrint(kINFO, "sort nodes in the order of movable and fixed\n");

  // I assume the total number does not change,
  // but the number of movable and fixed cells may change
  // m_vMovableNodeIndex.clear();
  // m_vFixedNodeIndex.clear();
  dreamplaceAssert(m_vNode.size() == numMovable() + numFixed() + numIOPin() + numPlaceBlockages());
  // for (std::vector<Node>::const_iterator
  //          it = m_vNode.begin(),
  //          ite = m_vNode.begin() + numMovable() + numFixed() + numIOPin();
  //      it != ite; ++it) {
  //   Node const& node = *it;
  //   Macro const& macro = m_vMacro.at(macroId(node));
  //   // exclude io pins
  //   if (not limbo::iequals(macro.className(), "DREAMPlace.IOPin") ) {
  //     if (node.status() == PlaceStatusEnum::FIXED) {
  //       m_vFixedNodeIndex.push_back(node.id());
  //     } else {
  //       m_vMovableNodeIndex.push_back(node.id());
  //     }
  //   }
  // }
  dreamplaceAssert(m_numMovable == m_vMovableNodeIndex.size());
  dreamplaceAssert(m_numFixed == m_vFixedNodeIndex.size());
  // m_numMovable = m_vMovableNodeIndex.size();
  // m_numFixed = m_vFixedNodeIndex.size();
  // sort m_vNode, m_vNodeProperty, m_mNodeName2Index
  // map order to node id
  // only work on movable and fixed cells, excluding IO pins
  std::vector<index_type> vNodeOrder(m_vNode.size());
  for (index_type i = 0, ie = vNodeOrder.size(); i != ie; ++i)
    vNodeOrder[i] = i;

  // so far the data layout in m_vNode
  // mixed UNPLACED/PLACED, DUMMY_FIXED, and FIXED, IO pins, placement blockages
  // Target order is UNPLACED, PLACED, DUMMY_FIXED, FIXED,
  // placement blockages, IO pins

  // 1. we first work on movable and fixed cells
  dreamplaceAssert(vNodeOrder.size() <= m_vNode.size());
  auto statusOrder = [&](index_type i) {
    auto const& node = m_vNode.at(i);
    auto const& macro = m_vMacro.at(nodeProperty(i).macroId());
    PlaceStatusEnum::PlaceStatusType status = node.status();
    index_type order = (status == PlaceStatusEnum::FIXED) * 512 +
                       (status == PlaceStatusEnum::DUMMY_FIXED) * 64 +
                       (status == PlaceStatusEnum::PLACED) * 8 +
                       (status == PlaceStatusEnum::UNPLACED);
    if (macro.className() == "DREAMPlace.IOPin") {
      order += 2048;
    } else if (macro.className() == "DREAMPlace.PlaceBlockage") {
      order += 1024;
    }
    return order;
  };
  std::sort(vNodeOrder.begin(), vNodeOrder.end(),
            [&](index_type i, index_type j) {
              index_type order1 = statusOrder(i);
              index_type order2 = statusOrder(j);
              return (order1 < order2 || (order1 == order2 && i < j));
            });

  // map node id to order
  std::vector<index_type> vNodeId2Order(vNodeOrder.size());
  for (index_type i = 0, ie = vNodeOrder.size(); i != ie; ++i)
    vNodeId2Order[vNodeOrder[i]] = i;
  // update m_mNodeName2Index
  for (string2index_map_type::iterator it = m_mNodeName2Index.begin(),
                                       ite = m_mNodeName2Index.end();
       it != ite; ++it) {
    if (it->second < vNodeOrder.size()) {
      it->second = vNodeId2Order[it->second];
    }
  }

  // for all elements to put in place
  for (index_type i = 0; i < vNodeOrder.size() - 1; ++i) {
    // while the element i is not yet in place
    while (i != vNodeId2Order[i]) {
      // swap it with the element at its final place
      index_type alt = vNodeId2Order[i];

      std::swap(m_vNode[i], m_vNode[alt]);
      std::swap(m_vNodeProperty[i], m_vNodeProperty[alt]);

      std::swap(vNodeId2Order[i], vNodeId2Order[alt]);
    }
  }
  for (index_type i = 1, ie = vNodeOrder.size(); i != ie; ++i) {
    dreamplaceAssert(m_vNode[i].id() == vNodeOrder[i]);
    // careful, node.id() has not been changed yet; use i as node id
    // all the functions called here need to consider this
    dreamplaceAssertMsg(
        statusOrder(i - 1) <= statusOrder(i),
        "permuting nodes error: i = %u, order %u (%s, %d) vs %u (%s, %d)", i,
        statusOrder(i - 1), nodeName(i - 1).c_str(),
        int(m_vNode[i - 1].status()), statusOrder(i), nodeName(i).c_str(),
        int(m_vNode[i].status()));
  }
  // update node id and pin to node id
  for (index_type i = 0, ie = vNodeOrder.size(); i != ie; ++i) {
    Node& node = m_vNode[i];
    for (std::vector<index_type>::const_iterator it = node.pins().begin(),
                                                 ite = node.pins().end();
         it != ite; ++it) {
      // we have not update the node id yet
      // so it should be consistent
      dreamplaceAssert(m_vPin[*it].nodeId() == node.id());
      m_vPin[*it].setNodeId(i);
    }
    node.setId(i);
  }

  m_vMovableNodeIndex.clear();
  m_vFixedNodeIndex.clear();
  for (std::vector<Node>::const_iterator it = m_vNode.begin(),
                                         ite = m_vNode.begin() + numMovable() +
                                               numFixed() + numPlaceBlockages();
       it != ite; ++it) {
    Node const& node = *it;
    Macro const& macro = m_vMacro[macroId(node)];
    if (macro.className() == "DREAMPlace.PlaceBlockage") {
      m_vPlaceBlockageIndex.push_back(node.id());
    } else {
      if (node.status() == PlaceStatusEnum::FIXED) {
        m_vFixedNodeIndex.push_back(node.id());
      } else {
        m_vMovableNodeIndex.push_back(node.id());
      }
    }
  }

  dreamplaceAssert(m_numMovable == m_vMovableNodeIndex.size());
  dreamplaceAssert(m_numFixed == m_vFixedNodeIndex.size());

#ifdef DEBUG
  // check pins, nodes, and nets
  for (std::vector<Node>::const_iterator it = m_vNode.begin(),
                                         ite = m_vNode.end();
       it != ite; ++it) {
    Node const& node = *it;
    for (std::vector<index_type>::const_iterator itp = node.pins().begin(),
                                                 itpe = node.pins().end();
         itp != itpe; ++itp) {
      Pin const& pin = m_vPin.at(*itp);
      dreamplaceAssert(pin.nodeId() == node.id());
    }
  }
  for (std::vector<Net>::const_iterator it = m_vNet.begin(), ite = m_vNet.end();
       it != ite; ++it) {
    Net const& net = *it;
    for (std::vector<index_type>::const_iterator itp = net.pins().begin(),
                                                 itpe = net.pins().end();
         itp != itpe; ++itp) {
      Pin const& pin = m_vPin.at(*itp);
      dreamplaceAssert(pin.netId() == net.id());
    }
  }
  for (unsigned int i = 1; i < m_vMovableNodeIndex.size(); ++i) {
    dreamplaceAssert(m_vMovableNodeIndex[i - 1] + 1 == m_vMovableNodeIndex[i]);
  }
  for (unsigned int i = 1; i < m_vFixedNodeIndex.size(); ++i) {
    dreamplaceAssert(m_vFixedNodeIndex[i - 1] + 1 == m_vFixedNodeIndex[i]);
  }
  for (unsigned int i = 1; i < m_vPlaceBlockageIndex.size(); ++i) {
    dreamplaceAssert(m_vPlaceBlockageIndex[i - 1] + 1 ==
                     m_vPlaceBlockageIndex[i]);
  }
#endif
}


void PlaceDB::processGroups() {
  dreamplacePrint(kINFO, "Group cells for fence regions\n");
  std::vector<unsigned char> markers(numMovable() + numFixed(), 0);
  // add a node to a group 
  auto addNode2Group = [&](Group& group, index_type node_id) {
    Node const& node = this->node(node_id); 
    if (node.id() >= numMovable() + numFixed()) {
      dreamplacePrint(kWARN, "node %s in group %s (%u) not movable, ignored\n", 
          nodeName(node).c_str(), group.name().c_str(), group.id()); 
    } else if (markers.at(node.id())) {
      dreamplacePrint(
          kWARN, "node %u in multiple groups, currently add to group %u\n",
          node.id(), group.id());
    } else {
      group.nodes().push_back(node.id());
      markers.at(node.id()) = 1;
    }
  };
  // set node indices in groups
  // according to node names

  // process non-wildcard patterns with exact names shown 
  // filter out the wildcard patterns 
  std::vector<std::pair<index_type, std::string>> vGroupWildcardPatternPair; 
  // find node from group 
  for (index_type i = 0, je = m_vGroup.size(); i < je; ++i) {
    Group& group = m_vGroup[i];
    for (std::vector<std::string>::const_iterator
        it = group.nodeNames().begin(),
        ite = group.nodeNames().end();
        it != ite; ++it) {
      std::string const& pattern = *it;
      // find wildcard characters *, ?
      auto found_star = pattern.find('*');
      auto found_question = pattern.find('?'); 
      // not wildcard pattern
      if (found_star == std::string::npos && found_question == std::string::npos) {
        auto found = m_mNodeName2Index.find(pattern); 
        if (found == m_mNodeName2Index.end()) {
          dreamplacePrint(kWARN, "Group %s (%u), pattern %s not found, ignored\n", 
              group.name().c_str(), i, pattern.c_str());
        } else {
          addNode2Group(group, found->second); 
        }
      } else {
        // wildcard pattern 
        vGroupWildcardPatternPair.emplace_back(group.id(), pattern); 
      }
    }
  }
  // process rest wildcard patterns 
  // find group from node 
  WildcardMatch matcher;
  for (index_type i = 0, ie = numMovable() + numFixed(); i < ie; ++i) {
    Node const& node = this->node(i);
    std::string const& name = this->nodeName(node);

    for (auto const& kvp : vGroupWildcardPatternPair) {
      auto& group = m_vGroup[kvp.first]; 
      auto const& pattern = kvp.second; 
      if (matcher(name.c_str(), pattern.c_str(), name.size(),
            pattern.size())) {
        addNode2Group(group, node.id()); 
      }
    }
  }
  dreamplacePrint(kINFO, "Construct %lu groups\n", m_vGroup.size()); 
  for (auto const& group : m_vGroup) {
    auto const& region = m_vRegion.at(group.region()); 
    dreamplacePrint(kINFO, "Group %s (%u), region %s (%u), %lu boxes, contains %lu nodes\n", 
        group.name().c_str(), group.id(), region.name().c_str(), group.region(), 
        region.boxes().size(), group.nodes().size()); 
  }
  dreamplacePrint(kINFO, "Fence region groups done\n");
#ifdef DEBUG
  for (index_type i = 0; i < m_vRegion.size(); ++i) {
    Region const& region = m_vRegion[i];
    dreamplacePrint(kDEBUG, "region[%u] %s: ", region.id(),
                    region.name().c_str());
    for (index_type j = 0; j < region.boxes().size(); ++j) {
      Region::box_type const& box = region.boxes().at(j);
      dreamplacePrint(kNONE, "(%d, %d, %d, %d) ", box.xl(), box.yl(), box.xh(),
                      box.yh());
    }
    dreamplacePrint(kNONE, "\n");
  }
#endif
}

DREAMPLACE_END_NAMESPACE
