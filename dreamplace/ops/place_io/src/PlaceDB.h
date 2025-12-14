/*************************************************************************
    > File Name: PlaceDB.h
    > Author: Yibo Lin
    > Mail: yibolin@utexas.edu
    > Created Time: Wed Jun 17 21:09:24 2015
 ************************************************************************/

#ifndef DREAMPLACE_PLACEDB_H
#define DREAMPLACE_PLACEDB_H

#include <limbo/parsers/lef/adapt/LefDriver.h> // LEF parser 
#include <limbo/parsers/def/adapt/DefDriver.h> // DEF parser 
#include <limbo/parsers/verilog/bison/VerilogDriver.h> // verilog parser
#include <limbo/parsers/bookshelf/bison/BookshelfDriver.h> // bookshelf parser 
#include <limbo/parsers/gdsii/stream/GdsWriter.h> // GDSII writer 
#include <limbo/string/String.h>

#include "Node.h"
#include "Net.h"
#include "Pin.h"
#include "Macro.h"
#include "Row.h"
#include "Region.h"
#include "Group.h"
#include "Site.h"
#include "Params.h"
#include "BenchMetrics.h"

DREAMPLACE_BEGIN_NAMESPACE

/// different tags for data traversal 
struct MovableNodeIteratorTag;
struct FixedNodeIteratorTag;
struct PlaceBlockageIteratorTag;
struct IOPinNodeIteratorTag;
struct CellMacroIteratorTag;
struct IOPinMacroIteratorTag;
struct SubRowMap2DIteratorTag;

/// forward declaration of data iterator 
template <typename PlaceDBType, typename IteratorTagType>
class DBIterator;

class PlaceDB;

/// iterator 
typedef DBIterator<PlaceDB, MovableNodeIteratorTag> MovableNodeIterator;
typedef DBIterator<PlaceDB, FixedNodeIteratorTag> FixedNodeIterator;
typedef DBIterator<PlaceDB, PlaceBlockageIteratorTag> PlaceBlockageIterator;
typedef DBIterator<PlaceDB, IOPinNodeIteratorTag> IOPinNodeIterator;
typedef DBIterator<PlaceDB, CellMacroIteratorTag> CellMacroIterator;
typedef DBIterator<PlaceDB, IOPinMacroIteratorTag> IOPinMacroIterator;
/// const iterator 
typedef DBIterator<const PlaceDB, MovableNodeIteratorTag> MovableNodeConstIterator;
typedef DBIterator<const PlaceDB, FixedNodeIteratorTag> FixedNodeConstIterator;
typedef DBIterator<const PlaceDB, PlaceBlockageIteratorTag> PlaceBlockageConstIterator;
typedef DBIterator<const PlaceDB, IOPinNodeIteratorTag> IOPinNodeConstIterator;
typedef DBIterator<const PlaceDB, CellMacroIteratorTag> CellMacroConstIterator;
typedef DBIterator<const PlaceDB, IOPinMacroIteratorTag> IOPinMacroConstIterator;

class PlaceDB : public DefParser::DefDataBase 
                , public LefParser::LefDataBase
                , public VerilogParser::VerilogDataBase
                , public BookshelfParser::BookshelfDataBase
{
    public:
        typedef Object::coordinate_type coordinate_type;
        typedef coordinate_traits<coordinate_type>::manhattan_distance_type manhattan_distance_type;
        typedef coordinate_traits<coordinate_type>::index_type index_type;
        typedef coordinate_traits<coordinate_type>::area_type area_type;
        typedef hashspace::unordered_map<std::string, index_type> string2index_map_type;
        typedef Box<coordinate_type> diearea_type;

        /// default constructor
        PlaceDB(); 
        /// copy constructor, forbidden
        //PlaceDB(PlaceDB const& rhs); 
        /// assignment, forbidden
        //PlaceDB& operator=(PlaceDB const& rhs);

        /// destructor
        virtual ~PlaceDB() {}

        /// member functions 
        /// data access
        std::vector<Node> const& nodes() const {return m_vNode;}
        std::vector<Node>& nodes() {return m_vNode;}
        Node const& node(index_type id) const {return m_vNode.at(id);}
        Node& node(index_type id) {return m_vNode.at(id);}
        NodeProperty const& nodeProperty(index_type id) const {return m_vNodeProperty.at(id);}
        NodeProperty const& nodeProperty(Node const& n) const {return nodeProperty(n.id());}
        /// some shortcut functions to set nodes for python binding 
        Node const& setNodeStatus(index_type id, PlaceStatusEnum::PlaceStatusType s) {return m_vNode.at(id).setStatus(s);}
        Node const& setNodeMultiRowAttr(index_type id, MultiRowAttrEnum::MultiRowAttrType a) {return m_vNode.at(id).setMultiRowAttr(a);}
        Node const& setNodeOrient(index_type id, OrientEnum::OrientType o) {return m_vNode.at(id).setOrient(o);}

        std::vector<Net> const& nets() const {return m_vNet;}
        std::vector<Net>& nets() {return m_vNet;}
        Net const& net(index_type id) const {return m_vNet.at(id);}
        Net& net(index_type id) {return m_vNet.at(id);}
        NetProperty const& netProperty(index_type id) const {return m_vNetProperty.at(id);}
        NetProperty const& netProperty(Net const& n) const {return netProperty(n.id());}
        /// some shortcut functions to set nets for python binding 
        Net const& setNetWeight(index_type id, Net::weight_type w) {return m_vNet.at(id).setWeight(w);}

        std::vector<Pin> const& pins() const {return m_vPin;}
        std::vector<Pin>& pins() {return m_vPin;}
        Pin const& pin(index_type id) const {return m_vPin.at(id);}
        Pin& pin(index_type id) {return m_vPin.at(id);}

        std::vector<Macro> const& macros() const {return m_vMacro;}
        std::vector<Macro>& macros() {return m_vMacro;}
        Macro const& macro(index_type id) const {return m_vMacro.at(id);}
        Macro& macro(index_type id) {return m_vMacro.at(id);}

        std::vector<Row> const& rows() const {return m_vRow;}
        std::vector<Row>& rows() {return m_vRow;}
        Row const& row(index_type id) const {return m_vRow.at(id);}
        Row& row(index_type id) {return m_vRow.at(id);}

        Site const& site() const {return m_vSite[m_coreSiteId];}
        area_type siteArea() const {return siteWidth()*rowHeight();}
        
        /// be careful to use die area because it is larger than the actual rowBbox() which is the placement area 
        /// it is safer to use rowBbox()
        diearea_type const& dieArea() const {return m_dieArea;}

        string2index_map_type const& macroName2Index() const {return m_mMacroName2Index;}
        string2index_map_type& macroName2Index() {return m_mMacroName2Index;}

        string2index_map_type const& nodeName2Index() const {return m_mNodeName2Index;}
        string2index_map_type& nodeName2Index() {return m_mNodeName2Index;}

        std::size_t numMovable() const {return m_numMovable;}
        std::size_t numFixed() const {return m_numFixed;}
        std::size_t numMacro() const {return m_numMacro;}
        std::size_t numIOPin() const {return m_numIOPin;}
        std::size_t numIgnoredNet() const {return m_numIgnoredNet;}
        std::size_t numPlaceBlockages() const {return m_numPlaceBlockages;}

        std::vector<index_type> const& movableNodeIndices() const {return m_vMovableNodeIndex;}
        std::vector<index_type>& movableNodeIndices() {return m_vMovableNodeIndex;}

        std::vector<index_type> const& fixedNodeIndices() const {return m_vFixedNodeIndex;}
        std::vector<index_type>& fixedNodeIndices() {return m_vFixedNodeIndex;}

        std::vector<index_type> const& placeBlockageIndices() const {return m_vPlaceBlockageIndex;}
        std::vector<index_type>& placeBlockageIndices() {return m_vPlaceBlockageIndex;}

        std::vector<Region> const& regions() const {return m_vRegion;}
        std::vector<Region>& regions() {return m_vRegion;}
        Region const& region(index_type i) const {return m_vRegion.at(i);}
        Region& region(index_type i) {return m_vRegion.at(i);}

        std::vector<Group> const& groups() const {return m_vGroup;}
        std::vector<Group>& groups() {return m_vGroup;}
        Group const& group(index_type i) const {return m_vGroup.at(i);}
        Group& group(index_type i) {return m_vGroup.at(i);}

        int lefUnit() const {return m_lefUnit;}
        std::string lefVersion() const {return m_lefVersion;}

        int defUnit() const {return m_defUnit;}
        std::string defVersion() const {return m_defVersion;}
        std::string designName() const {return m_designName;}

        /// \brief sometimes the units may be different 
        /// Need to scale to LEF unit 
        double lefDefUnitRatio() const {return lefUnit() / defUnit();}

        UserParam const& userParam() const {return m_userParam;}
        UserParam& userParam() {return m_userParam;}

        BenchMetrics const& benchMetrics() const {return m_benchMetrics;}
        BenchMetrics& benchMetrics() {return m_benchMetrics;}

        /// helper functions 
        /// \return node from a pin 
        Node const& getNode(index_type pinId) const {return m_vNode.at(m_vPin.at(pinId).nodeId());}
        Node& getNode(index_type pinId) {return m_vNode.at(m_vPin.at(pinId).nodeId());}
        Node const& getNode(Pin const& pin) const {return m_vNode.at(pin.nodeId());}
        Node& getNode(Pin const& pin) {return m_vNode.at(pin.nodeId());}
        /// \return net from a pin 
        Net const& getNet(index_type pinId) const {return m_vNet.at(m_vPin.at(pinId).netId());}
        Net& getNet(index_type pinId) {return m_vNet.at(m_vPin.at(pinId).netId());}
        Net const& getNet(Pin const& pin) const {return m_vNet.at(pin.netId());}
        Net& getNet(Pin const& pin) {return m_vNet.at(pin.netId());}
        /// absolute position of a pin 
        coordinate_type pinPos(index_type pinId, Direction1DType d) const; 
        coordinate_type pinPos(Pin const& pin, Direction1DType d) const; 
        Point<coordinate_type> pinPos(index_type pinId) const;
        Point<coordinate_type> pinPos(Pin const& pin) const;
        /// absolute bounding box of a pin 
        Box<coordinate_type> pinBbox(index_type pinId) const;
        Box<coordinate_type> pinBbox(Pin const& pin) const;
        /// find macro pin from pin  
        MacroPin const& macroPin(index_type pinId) const;
        MacroPin const& macroPin(Pin const& pin) const;

        /// functions for routing information 
        index_type numRoutingGrids(Direction1DType d) const {return m_numRoutingGrids[d];}
        index_type numRoutingLayers() const {return m_numRoutingGrids[2];}
        std::vector<index_type> const& routingCapacity(PlanarDirectEnum::PlanarDirectType d) const {return m_vRoutingCapacity[d];}
        std::vector<index_type> const& minWireWidth() const {return m_vMinWireWidth;}
        std::vector<index_type> const& minWireSpacing() const {return m_vMinWireSpacing;}
        std::vector<index_type> const& viaSpacing() const {return m_vViaSpacing;}
        coordinate_type routingGridOrigin(Direction1DType d) const {return m_routingGridOrigin[d];}
        coordinate_type routingTileSize(Direction1DType d) const {return m_routingTileSize[d];}
        index_type routingBlockagePorosity() const {return m_routingBlockagePorosity;}
        /// @brief compute number of routing tracks per tile 
        index_type numRoutingTracks(PlanarDirectEnum::PlanarDirectType d, index_type layer) const {return routingCapacity(d).at(layer) / (minWireWidth().at(layer) + minWireSpacing().at(layer));}
        index_type getLayer(std::string const& layerName) const; 
        std::string getLayerName(index_type layer) const; 

        /// traverse movable node 
        MovableNodeIterator movableNodeBegin();
        MovableNodeIterator movableNodeEnd();
        MovableNodeConstIterator movableNodeBegin() const;
        MovableNodeConstIterator movableNodeEnd() const;
        /// traverse fixed node 
        FixedNodeIterator fixedNodeBegin();
        FixedNodeIterator fixedNodeEnd();
        FixedNodeConstIterator fixedNodeBegin() const;
        FixedNodeConstIterator fixedNodeEnd() const;
        /// traverse placement blockage 
        PlaceBlockageIterator placeBlockageBegin();
        PlaceBlockageIterator placeBlockageEnd();
        PlaceBlockageConstIterator placeBlockageBegin() const;
        PlaceBlockageConstIterator placeBlockageEnd() const;
        /// traverse io pin virtual node 
        IOPinNodeIterator iopinNodeBegin();
        IOPinNodeIterator iopinNodeEnd();
        IOPinNodeConstIterator iopinNodeBegin() const;
        IOPinNodeConstIterator iopinNodeEnd() const;
        /// traverse cell macro 
        CellMacroIterator cellMacroBegin();
        CellMacroIterator cellMacroEnd();
        CellMacroConstIterator cellMacroBegin() const;
        CellMacroConstIterator cellMacroEnd() const;
        /// traverse io pin virtual macro 
        IOPinMacroIterator iopinMacroBegin();
        IOPinMacroIterator iopinMacroEnd();
        IOPinMacroConstIterator iopinMacroBegin() const;
        IOPinMacroConstIterator iopinMacroEnd() const;

        /// \return name of a node 
        std::string const& nodeName(index_type id) const {return nodeProperty(node(id)).name();}
        std::string const& nodeName(Node const& n) const {return nodeProperty(n).name();}
        /// \return macro id of a node 
        index_type macroId(index_type id) const {return nodeProperty(node(id)).macroId();}
        index_type macroId(Node const& n) const {return nodeProperty(n).macroId();}
        /// \return name of a net 
        std::string const& netName(Net const& n) const {return netProperty(n).name();}
        /// \return macro name with a node 
        std::string const& macroName(Node const& n) const {return m_vMacro.at(macroId(n)).name();}
        /// \return obstruction in macro with a node 
        MacroObs const& macroObs(Node const& n) const {return m_vMacro.at(macroId(n)).obs();}
        /// \return die area information of layout 
        coordinate_type xl() const {return m_dieArea.xl();}
        coordinate_type yl() const {return m_dieArea.yl();}
        coordinate_type xh() const {return m_dieArea.xh();}
        coordinate_type yh() const {return m_dieArea.yh();}
        manhattan_distance_type width() const {return m_dieArea.width();}
        manhattan_distance_type height() const {return m_dieArea.height();}

        /// \return index of row from position in y direction 
        index_type getRowIndex(coordinate_type y) const;
        /// \return a range of row indices 
        /// yl+1 and yh-1 avoid redundant bottom or top rows 
        Interval<index_type> getRowIndexRange(coordinate_type yl, coordinate_type yh) const {return Interval<index_type>(getRowIndex(yl+1), getRowIndex(yh-1));}

        /// \return height of a row, assume to be the same as site height 
        coordinate_type rowHeight() const {return m_vSite[m_coreSiteId].height();}
        /// \return the region of rows, it may be different from die area 
        Box<coordinate_type> const& rowBbox() const {return m_rowBbox;}
        coordinate_type rowXL() const {return (m_vRow.empty())? xl() : m_rowBbox.xl();}
        coordinate_type rowYL() const {return (m_vRow.empty())? yl() : m_rowBbox.yl();}
        coordinate_type rowYH() const {return (m_vRow.empty())? yh() : m_rowBbox.yh();}
        coordinate_type rowXH() const {return (m_vRow.empty())? xh() : m_rowBbox.xh();}
        /// \return true if macros are defined 
        bool hasMacros() const {return !m_vMacro.empty();} 

        /// adjust user input parameters 
        /// must be called after parsing input files 
        void adjustParams();
        /// sort net from small degrees to large degrees 
        /// sort pins such that all pins belonging to the same net is adjacent 
        void sortNetByDegree();
        /// sort nodes such that 
        /// movable cells are followed by fixed cells 
        void sortNodeByPlaceStatus();

        /// \return site width 
        coordinate_type siteWidth() const {return m_vSite[m_coreSiteId].width();}
        /// \return site height 
        coordinate_type siteHeight() const {return m_vSite[m_coreSiteId].height();}
        /// \return max displacement in database unit 
        coordinate_type maxDisplace() const {return m_maxDisplace;}
        /// \return minimum width of movable nodes 
        manhattan_distance_type minMovableNodeWidth() const;
        /// \return maximum width of movable nodes 
        manhattan_distance_type maxMovableNodeWidth() const;
        /// \return average width of movable nodes 
        manhattan_distance_type avgMovableNodeWidth() const;
        /// \return area of all movable nodes, normalized to site count 
        index_type totalMovableNodeArea() const;
        /// \return area of all fixed nodes, normalized to site count 
        index_type totalFixedNodeArea() const;
        /// \return total row area, normalized to site count 
        index_type totalRowArea() const;
        /// \return the utilization ratio for movable cells 
        /// (movable node area) / (row area - fixed node area)
        double computeMovableUtil() const;
        /// \return average pin utilization per site 
        double computePinUtil() const;
        /// \return total number of movable multi-row cells 
        std::size_t numMultiRowMovable() const;
        /// \return total number of k-row height cells 
        std::size_t numKRowMovable(index_type k) const;
        /// \return true if it is a multi-row cell 
        bool isMultiRowMovable(index_type nodeId) const {return isMultiRowMovable(nodes().at(nodeId));}
        bool isMultiRowMovable(Node const& node) const {return node.status() != PlaceStatusEnum::FIXED && node.multiRowAttr() != MultiRowAttrEnum::SINGLE_ROW;}
        /// \return true if the net is ignored 
        bool isIgnoredNet(index_type netId) const {return m_vNetIgnoreFlag.at(netId);}
        bool isIgnoredNet(Net const& net) const {return isIgnoredNet(net.id());}
        std::vector<bool> const& netIgnoreFlag() const {return m_vNetIgnoreFlag;}

        /// parser callback functions 
        ///==== LEF Callbacks ====
        virtual void lef_version_cbk(std::string const& v);
        virtual void lef_version_cbk(double v); 
        virtual void lef_casesensitive_cbk(int v); 
        virtual void lef_dividerchar_cbk(std::string const& ); 
        virtual void lef_units_cbk(LefParser::lefiUnits const& v);
        virtual void lef_manufacturing_cbk(double );
        virtual void lef_useminspacing_cbk(LefParser::lefiUseMinSpacing const&);
        virtual void lef_clearancemeasure_cbk(std::string const&);
        virtual void lef_busbitchars_cbk(std::string const& );
        virtual void lef_layer_cbk(LefParser::lefiLayer const& );
        virtual void lef_via_cbk(LefParser::lefiVia const& );
        virtual void lef_viarule_cbk(LefParser::lefiViaRule const& );
        virtual void lef_spacing_cbk(LefParser::lefiSpacing const& );
        virtual void lef_site_cbk(LefParser::lefiSite const& s);
        virtual void lef_macrobegin_cbk(std::string const& n); 
        virtual void lef_macro_cbk(LefParser::lefiMacro const& m);
        virtual void lef_pin_cbk(LefParser::lefiPin const& p);
        virtual void lef_obstruction_cbk(LefParser::lefiObstruction const& o);
        virtual void lef_prop_cbk(LefParser::lefiProp const&);
        virtual void lef_maxstackvia_cbk(LefParser::lefiMaxStackVia const&);
        ///==== DEF Callbacks ====
        virtual void set_def_busbitchars(std::string const&);
        virtual void set_def_dividerchar(std::string const&);
        virtual void set_def_version(std::string const& v);
        virtual void set_def_unit(int u);
        virtual void set_def_design(std::string const& d);
        virtual void set_def_diearea(int xl, int yl, int xh, int yh);
        virtual void set_def_diearea(int n, const int* x, const int* y);
        virtual void add_def_row(DefParser::Row const& r);
        virtual void resize_def_component(int s);
        virtual void add_def_component(DefParser::Component const& c);
        virtual void resize_def_pin(int s);
        virtual void add_def_pin(DefParser::Pin const& p);
        virtual void resize_def_net(int s);
        virtual void add_def_net(DefParser::Net const& n);
        virtual void resize_def_blockage(int);
        virtual void add_def_placement_blockage(std::vector<std::vector<int> > const&);
        virtual void resize_def_region(int);
        virtual void add_def_region(DefParser::Region const& r);
        virtual void resize_def_group(int);
        virtual void add_def_group(DefParser::Group const& g);
        virtual void add_def_track(defiTrack const& t);
        virtual void add_def_via(defiVia const& v);
        virtual void add_def_snet(defiNet const& n);
        virtual void add_def_gcellgrid(DefParser::GCellGrid const& g);
        virtual void add_def_route_blockage(std::vector<std::vector<int>> const&, std::string const&); 
        virtual void end_def_design(); 
        ///==== Verilog Callbacks ==== 
        virtual void verilog_module_declaration_cbk(std::string const& module_name, std::vector<VerilogParser::GeneralName> const& vPinName); 
        virtual void verilog_net_declare_cbk(std::string const&, VerilogParser::Range const&);
        virtual void verilog_pin_declare_cbk(std::string const&, unsigned, VerilogParser::Range const&);
        virtual void verilog_instance_cbk(std::string const&, std::string const&, std::vector<VerilogParser::NetPin> const& vNetPin);
        ///==== Bookshelf Callbacks ====
        virtual void resize_bookshelf_node_terminals(int nn, int nt);
        virtual void resize_bookshelf_net(int n);
        virtual void resize_bookshelf_pin(int n);
        virtual void resize_bookshelf_row(int n);
        virtual void resize_bookshelf_shapes(int n);
        virtual void resize_bookshelf_niterminal_layers(int);
        virtual void resize_bookshelf_blockage_layers(int);
        virtual void add_bookshelf_terminal(std::string& name, int w, int h);
        virtual void add_bookshelf_terminal_NI(std::string& name, int w, int h);
        virtual void add_bookshelf_node(std::string& name, int w, int h, bool is_cell);
        virtual void add_bookshelf_net(BookshelfParser::Net const& n);
        virtual void add_bookshelf_row(BookshelfParser::Row const& r);
        virtual void set_bookshelf_node_position(std::string const& name, double x, double y, std::string const& orient, std::string const& status, bool plFlag);
        virtual void set_bookshelf_net_weight(std::string const& name, double w); 
        virtual void set_bookshelf_shape(BookshelfParser::NodeShape const& shape); 
        virtual void set_bookshelf_route_info(BookshelfParser::RouteInfo const&);
        virtual void add_bookshelf_niterminal_layer(std::string const&, std::string const&);
        virtual void add_bookshelf_blockage_layers(std::string const&, std::vector<std::string> const&);
        virtual void set_bookshelf_design(std::string& name);
        virtual void bookshelf_end(); 

        /// derive MultiRowAttr of a cell 
        void deriveMultiRowAttr(Node& node);
        /// when a node is moved to a new position, its orient is changed, so we need to update the pin offsets to its origin  
        /// \param origOrient is the original orientation before movement 
        /// \param newOrient is the new orientation after movement
        /// \return encoded (hflip, vflip) flags 
        index_type computeFlipFlag(Orient const& origOrient, Orient const& newOrient) const; 
        Point<coordinate_type> getNodePinOffset(Pin const& pin, Orient const& origOrient, Orient const& newOrient) const; 
        void updateNodePinOffset(Node const& node, Orient const& origOrient, Orient const& newOrient);

        ///==== prepare data ==== 
        /// mainly used to reserve spaces 
        virtual void prepare(unsigned numRows, unsigned numNodes, unsigned numIOPin, unsigned numNets, unsigned numBlockages);

        /// report statistics 
        virtual void reportStats();
        virtual void reportStatsKernel();
        /// write placement solutions 
        virtual bool write(std::string const& filename) const;
        virtual bool write(std::string const& filename, SolutionFileFormat ff, coordinate_type const* x = NULL, coordinate_type const* y = NULL) const;

        /// for debug 
        virtual void printNode(index_type id) const;
        virtual void printNet(index_type id) const;
    protected:
        /// add node to m_vNode and m_mNodeName2Index
        /// \param n denotes name 
        /// \return index in m_vNode and successful flag 
        std::pair<index_type, bool> addNode(std::string const& n);
        /// add node to m_vMacro and m_mMacroName2Index
        /// \param n denotes name 
        /// \return index in m_vMacro and successful flag 
        std::pair<index_type, bool> addMacro(std::string const& n);
        /// add net to m_vNet and m_mNetName2Index
        /// \param n denotes name 
        /// \return index in m_vNet and successful flag 
        std::pair<index_type, bool> addNet(std::string const& n);
        /// add pin to m_vPin, node and net 
        /// \param pinName denotes name of corresponding macro pin 
        /// \param net and \param node are corresponding net and node 
        void addPin(std::string const& macroPinName, Net& net, Node& node,
                    std::string instName = "");
        void addPin(index_type macroPinId, Net& net, Node& node, std::string pinName);
        /// lower level helper to addPin()
        Pin& createPin(Net& net, Node& node, SignalDirect const& direct,
                       Point<coordinate_type> const& offset, index_type macroPinId,
                       std::string pinName = "");
        /// add region to m_vRegion 
        /// \param r region name 
        /// \return index in m_vRegion and successful flag 
        std::pair<index_type, bool> addRegion(std::string const& r);
        /// collect nodes for groups and summarize the statistics for fence region 
        void processGroups(); 

        /// kernel data for placement 
        std::vector<Node> m_vNode; ///< instances, including movable and fixed instances, virtual placement blockages, and virtual io pins (appended) 
        std::vector<NodeProperty> m_vNodeProperty; ///< some unimportant properties for instances, together with m_vNode
        std::vector<Net> m_vNet; ///< nets 
        std::vector<NetProperty> m_vNetProperty; ///< some unimportant properties for nets, together with m_vNet
        std::vector<Pin> m_vPin; ///< pins for instances and nets, the offset of a pin must be adjusted when a node is moved 
        std::vector<Macro> m_vMacro; ///< macros for standard cells, for io pins, virtual macros are appended  
        std::vector<Row> m_vRow; ///< placement rows 
        std::vector<Site> m_vSite; ///< all sites defined 
        std::vector<std::size_t> m_vSiteUsedCount; ///< count how many macros in LEF refer each site 
        index_type m_coreSiteId; ///< id of core placement site, determine by m_vSiteUsedCount
        diearea_type m_dieArea; ///< die area, it can be larger than actual placement area 
        std::vector<bool> m_vNetIgnoreFlag; ///< whether the net should be ignored due to pins belonging to the same cell 
        std::vector<std::string> m_vDuplicateNet; ///< name of duplicate nets found in verilog file 

        string2index_map_type m_mMacroName2Index; ///< map name of macro to index of m_vMacro
        string2index_map_type m_mNodeName2Index; ///< map instance name to index of m_vNode
        string2index_map_type m_mNetName2Index; ///< map net name to index of m_vNet 
        string2index_map_type m_mLayerName2Index; ///< map layer name to layer 
        std::vector<std::string> m_vLayerName; ///< layer to layer name 
        string2index_map_type m_mSiteName2Index; ///< map site name to index of m_vSite 

        Box<coordinate_type> m_rowBbox; ///< bounding box of row regions, it may be different from die area  
                                        ///< different rows may have different width, this is the largest box 

        std::size_t m_numMovable; ///< number of movable cells 
        std::size_t m_numFixed; ///< number of fixed cells 
        std::size_t m_numMacro; ///< number of standard cells in the library (0~m_numMacro-1 in m_vMacro) 
        std::size_t m_numIOPin; ///< number of io pins (m_numMacro~m_numMacro+m_numIOPin-1 in m_vMacro)
        std::size_t m_numIgnoredNet; ///< number of nets ignored 
        std::size_t m_numPlaceBlockages; ///< number of placement blockages 
    
        std::vector<index_type> m_vMovableNodeIndex; ///< movable node index 
        std::vector<index_type> m_vFixedNodeIndex; ///< fixed node index 
        std::vector<index_type> m_vPlaceBlockageIndex; ///< placement blockages are stored in m_vNode, we record the index 

        std::vector<Region> m_vRegion; ///< placement regions like FENCE or GUIDE 
        std::vector<Group> m_vGroup; ///< cell groups for placement regions 

        string2index_map_type m_mRegionName2Index; ///< map region name to index 
        string2index_map_type m_mGroupName2Index; ///< map group name to index 

        /// data for routing configuration 
        index_type m_numRoutingGrids[3]; ///< global routing grids in X, Y and number of layers 
        std::vector<index_type> m_vRoutingCapacity[2]; ///< horizontal and vertical capacity at each layer 
        std::vector<index_type> m_vMinWireWidth; ///< min wire width for each layer 
        std::vector<index_type> m_vMinWireSpacing; ///< min wire spacing for each layer 
        std::vector<index_type> m_vViaSpacing; ///< via spacing per layer 
        coordinate_type m_routingGridOrigin[2]; ///< Absolute coordinates of the origin of the grid (grid_lowerleft_X grid_lowerleft_Y)
        coordinate_type m_routingTileSize[2]; ///< tile_width, tile_height
        index_type m_routingBlockagePorosity; ///< Porosity for routing blockages (Zero implies the blockage completely blocks overlapping routing tracks. Default = 0)

        /// data only used in parsers
        int m_lefUnit;
        std::string m_lefVersion; 
        int m_defUnit;
        std::string m_defVersion;
        std::string m_designName; ///< for writing def file

        /// parameters 
        UserParam m_userParam; ///< user defined parameters 

        /// scaled parameters from UserParam 
        coordinate_type m_maxDisplace; ///< max displacement constraint in coordinate_type unit 

        BenchMetrics m_benchMetrics; ///< benchmark metrics 

        /// used to print warnings 
        std::size_t m_numNetsWithDuplicatePins; ///< nets with pins from the same nodes, count nets 
        std::size_t m_numPinsDuplicatedInNets; ///< nets with pins from the same nodes, count pins  
};

inline PlaceDB::index_type PlaceDB::getRowIndex(PlaceDB::coordinate_type y) const
{
    // use row region instead of die area 
    // because the starting point of rows may be different from that of die area 
    if (y <= rowYL()) return 0;
    else if (y >= rowYH()) return m_vRow.size()-1;
    else return (y-rowYL())/rowHeight(); // bottom or top row may be redundant 
}

DREAMPLACE_END_NAMESPACE

#endif
