/*************************************************************************
    > File Name: DrawPlace.cpp
    > Author: Yibo Lin
    > Mail: yibolin@utexas.edu
    > Created Time: Sat 20 Jun 2015 12:25:20 AM CDT
 ************************************************************************/

#include "DrawPlace.h"
#include "AlgoDB.h"
#include "DrawPlaceExt.h"
#include <fstream>

#if DRAWPLACE == 1
#include <cairo/cairo.h>
#include <cairo/cairo-pdf.h>
#include <cairo/cairo-ps.h>
#include <cairo/cairo-svg.h>
#endif

#include<cstdio>
#include<cstdlib>
#include <limbo/parsers/gdsii/stream/GdsWriter.h>


GPF_BEGIN_NAMESPACE

/// helper object of ObsWraper for drawing fixed cells 
struct DrawFixedNodeHelper
{
    typedef AlgoDB::index_type index_type;
    typedef AlgoDB::box_type box_type;

    index_type const fixedCellBboxLayer; 
    GdsParser::GdsWriter& gw; 

    DrawFixedNodeHelper(index_type const layer, GdsParser::GdsWriter& g)
        : fixedCellBboxLayer(layer)
        , gw(g)
    {
    }
    DrawFixedNodeHelper(DrawFixedNodeHelper const& rhs)
        : fixedCellBboxLayer(rhs.fixedCellBboxLayer)
        , gw(rhs.gw)
    {
    }
    inline void operator()(Node const& /*node*/, box_type const& box)
    {
        gw.write_box(fixedCellBboxLayer, 0, box.xl(), box.yl(), box.xh(), box.yh());
    }
};

PlaceDrawer::PlaceDrawer(AlgoDB const& db, PlaceDrawerExt* ext, int content)
    : m_db(db)
    , m_ext(ext)
    , m_content(content)
{
}

bool PlaceDrawer::run(std::string const& filename, PlaceDrawer::FileFormat ff) const
{
    gpfPrint(kINFO, "writing placement to %s\n", filename.c_str());
    bool flag = false;

    //PlaceDB const& placeDB = m_db.placeDB();

    switch (ff)
    {
        case EPS:
        case PDF:
        case SVG:
        case PNG:
            flag = writeFig(filename.c_str(), 800, 800, ff);
            break;
        case GDSII:
            flag = writeGdsii(filename);
            break;
        default:
            gpfPrint(kERROR, "unknown writing format at line %u\n", __LINE__);
            break;
    }

    return flag;
}

void PlaceDrawer::paintCairo(cairo_surface_t* cs, double width, double height) const
{
#if DRAWPLACE == 1
    PlaceDB const& placeDB = m_db.placeDB();

    std::vector<Node> const& vNode = placeDB.nodes();
    std::vector<Net> const& vNet = placeDB.nets();
    std::vector<Pin> const& vPin = placeDB.pins();

    double ratio[2] = {
        width/placeDB.dieArea().width(),
        height/placeDB.dieArea().height()
    };
    char buf[16];
	cairo_t *c;
    cairo_text_extents_t extents;

	c=cairo_create(cs);
    cairo_save(c); // save status 
    cairo_translate(c, 0-placeDB.xl()*ratio[kX], height+placeDB.yl()*ratio[kY]); // translate is additive
    cairo_scale(c, ratio[kX], -ratio[kY]); // scale is additive 

    // background 
	cairo_rectangle(c, placeDB.xl(), placeDB.yl(), placeDB.dieArea().width(), placeDB.dieArea().height());
	cairo_set_source_rgb(c, 1.0, 1.0, 1.0);
	cairo_fill(c);

    // nodes 
    cairo_set_line_width(c, 2);
    cairo_set_source_rgb(c, 0.0, 0.0, 0.0);
    cairo_select_font_face (c, "Sans",
            CAIRO_FONT_SLANT_NORMAL,
            CAIRO_FONT_WEIGHT_NORMAL);
    if (m_content&NODE)
    {
        for (std::vector<Node>::const_iterator it = vNode.begin(), ite = vNode.end(); it != ite; ++it)
        {
            Node const& node = *it;
            cairo_rectangle(c, node.xl(), node.yl(), node.width(), node.height());
            //printf("node %d, %d, %ld, %ld\n", node.xl(), node.yl(), node.width(), node.height());
            cairo_stroke(c);
            if (m_content&NODETEXT)
            {
                sprintf(buf, "%u", node.id());
                cairo_set_font_size (c, node.height()/20);
                cairo_text_extents (c, buf, &extents);
                cairo_move_to(c, center(node, kX)-(extents.width/2+extents.x_bearing), center(node, kY)-(extents.height/2+extents.y_bearing));
                cairo_show_text(c, buf);
            }
        }
    }
    // pins 
    if (m_content&PIN)
    {
        cairo_set_line_width(c, 1);
        for (std::vector<Pin>::const_iterator it = vPin.begin(), ite = vPin.end(); it != ite; ++it)
        {
            Pin const& pin = *it;
            Box<AlgoDB::coordinate_type> box = placeDB.pinBbox(pin);
            cairo_rectangle(c, box.xl(), box.yl(), box.width(), box.height());
            //printf("pin %d, %d, %ld, %ld\n", box.xl(), box.yl(), box.width(), box.height());
            cairo_stroke(c);
        }
    }
    // nets 
    if (m_content&NET)
    {
        cairo_set_line_width(c, 1);
        for (std::vector<Net>::const_iterator it = vNet.begin(), ite = vNet.end(); it != ite; ++it)
        {
            Net const& net = *it;
            // only draw nets for marked nodes 
            bool drawFlag = false;
            for (std::vector<AlgoDB::index_type>::const_iterator itp = net.pins().begin(), itpe = net.pins().end(); itp != itpe; ++itp)
            {
                Pin const& pin = vPin.at(*itp);
                if (m_sMarkNode.count(pin.nodeId()))
                {
                    drawFlag = true;
                    break;
                }
            }
            if (!drawFlag) continue;

            Point<AlgoDB::coordinate_type> srcPoint;
            for (std::vector<AlgoDB::index_type>::const_iterator itp = net.pins().begin(), itpe = net.pins().end(); itp != itpe; ++itp)
            {
                Pin const& pin = vPin.at(*itp);
                if (itp == net.pins().begin())
                    srcPoint = placeDB.pinPos(pin);
                else 
                {
                    Point<AlgoDB::coordinate_type> sinkPoint = placeDB.pinPos(pin);
                    Box<AlgoDB::coordinate_type> box = placeDB.pinBbox(pin);
                    cairo_move_to(c, srcPoint.x(), srcPoint.y());
                    cairo_line_to(c, sinkPoint.x(), sinkPoint.y());
                    cairo_stroke(c);
                    cairo_rectangle(c, sinkPoint.x()-box.width()/10, sinkPoint.y()-box.width()/10, box.width()/5, box.width()/5); // small rect denotes sink 
                    cairo_stroke(c);
                    cairo_fill(c);
                }
            }
        }
    }
    cairo_restore(c);
    // call extension 
    if (m_ext)
        m_ext->paintCairo(m_db, c, width, height);

	cairo_show_page(c);

	cairo_destroy(c);
#else 
    gpfPrint(kWARN, "cs = %p, width = %g, height = %g are not used, as DRAWPLACE not enabled\n", cs, width, height);
#endif
}

double PlaceDrawer::scaleToScreen(double coord, double srcOffset, double srcSize, double tgtOffset, double tgtSize) const 
{
    double ratio = tgtSize/srcSize;
    return tgtOffset + (coord-srcOffset)*ratio;
}

bool PlaceDrawer::writeFig(const char *fname, double width, double height, PlaceDrawer::FileFormat ff) const
{
#if DRAWPLACE == 1
	cairo_surface_t *cs;

    switch (ff)
    {
        case PNG:
            cs=cairo_image_surface_create(CAIRO_FORMAT_ARGB32, width, height);
            break;
        case PDF:
            cs=cairo_pdf_surface_create(fname, width, height);
            break;
        case EPS:
            cs=cairo_ps_surface_create(fname, width, height);
            break;
        case SVG:
            cs=cairo_svg_surface_create(fname, width, height);
            break;
        default:
            gpfPrint(kERROR, "unknown file format in %s\n", __func__);
            return false;
    }

	paintCairo(cs, width, height);

	cairo_surface_flush(cs);
    // need additional writing call for PNG 
    if (ff == PNG)
        cairo_surface_write_to_png(cs, fname);
	cairo_surface_destroy(cs);
    return true;
#else 
    gpfPrint(kWARN, "filename = %s, width = %g, height = %g, file format = %d not used, as DRAWPLACE not enabled\n", fname, width, height, (int)ff);
    return false;
#endif 
}

bool PlaceDrawer::writeGdsii(std::string const& filename) const 
{
    GdsParser::GdsWriter gw (filename.c_str());
    gw.create_lib("TOP", 0.001, 1e-6/m_db.placeDB().lefUnit());
    gw.gds_write_bgnstr();
    gw.gds_write_strname(m_db.placeDB().designName().c_str());

    // kernel function to fill in contents 
    writeGdsiiContent(gw); 

    gw.gds_write_endstr();
    gw.gds_write_endlib();

    return true;
}

void PlaceDrawer::writeGdsiiContent(GdsParser::GdsWriter& gw) const
{
    // layer specification 
    // it is better to use even layers, because text appears on odd layers
    const unsigned dieAreaLayer = getLayer(true);
    const unsigned rowLayer = getLayer(false);
    const unsigned subRowLayer = getLayer(false);
    const unsigned binRowLayer = getLayer(false);
    const unsigned binLayer = getLayer(false);
    const unsigned sbinLayer = getLayer(false);
    const unsigned movableCellBboxLayer = getLayer(false);
    const unsigned fixedCellBboxLayer = getLayer(false);
    const unsigned blockageBboxLayer = getLayer(false);
    const unsigned pinLayer = getLayer(false);
    const unsigned multiRowCellBboxLayer = getLayer(false);
    const unsigned movePathLayer = getLayer(false);
    const unsigned markedNodeLayer = getLayer(false); // together with netLayer 
    const unsigned netLayer = getLayer(false);

    gpfPrint(kINFO, "Layer: dieArea:%u, row:%u, subRow:%u, binRow:%u, bin:%u, sbin:%u, movableCellBbox:%u, fixedCellBbox:%u, blockageBbox:%u, pin:%u, multiRowCellBbox:%u, movePathLayer:%u, markedNodeLayer:%u, net:from %u\n", 
            dieAreaLayer, rowLayer, subRowLayer, binRowLayer, binLayer, sbinLayer, movableCellBboxLayer, fixedCellBboxLayer, blockageBboxLayer, pinLayer, multiRowCellBboxLayer, movePathLayer, markedNodeLayer, netLayer);

    PlaceDB const& placeDB = m_db.placeDB();
    char buf[1024];

    std::vector<Node> const& vNode = placeDB.nodes();
    std::vector<Net> const& vNet = placeDB.nets();
    std::vector<Pin> const& vPin = placeDB.pins();
    std::vector<Row> const& vRow = placeDB.rows();

    // write dieArea
    gw.write_box(dieAreaLayer, 0, placeDB.xl(), placeDB.yl(), placeDB.xh(), placeDB.yh());
    // write rows 
    for (std::vector<Row>::const_iterator it = vRow.begin(), ite = vRow.end(); it != ite; ++it)
    {
        Row const& row = *it;
        gw.write_box(rowLayer, 0, row.xl(), row.yl(), row.xh(), row.yh());
        gpfSPrint(kNONE, buf, "%u", row.id()); // write row index at left site 
        gw.gds_create_text(buf, row.xl()-100, center(row, kY), rowLayer+1, 5);
    }
    // write subrows 
    for (SubRowMap1DConstIterator it = m_db.subRowMap().begin1D(), ite = m_db.subRowMap().end1D(); it != ite; ++it)
    {
        SubRow const& srow = *it;
        gw.write_box(subRowLayer, 0, srow.xl(), srow.yl(), srow.xh(), srow.yh());
        gpfSPrint(kNONE, buf, "%u", srow.index1D());
        gw.gds_create_text(buf, center(srow, kX), center(srow, kY), subRowLayer+1, 5);
    }
    // write binrows 
    for (BinRowMap1DConstIterator it = m_db.binRowMap().begin1D(), ite = m_db.binRowMap().end1D(); it != ite; ++it)
    {
        BinRow const& brow = *it;
        gw.write_box(binRowLayer, 0, brow.xl(), brow.yl(), brow.xh(), brow.yh());
        gpfSPrint(kNONE, buf, "%u", brow.index1D());
        gw.gds_create_text(buf, center(brow, kX), center(brow, kY), binRowLayer+1, 5);
    }
    // write bins 
    for (BinMap1DConstIterator it = m_db.binMap(kBin).begin1D(), ite = m_db.binMap(kBin).end1D(); it != ite; ++it)
    {
        Bin const& bin = *it;
        gw.write_box(binLayer, 0, bin.xl(), bin.yl(), bin.xh(), bin.yh());
        gpfSPrint(kNONE, buf, "%u", bin.index1D());
        gw.gds_create_text(buf, center(bin, kX), center(bin, kY), binLayer+1, 5);
    }
    // write bins 
    for (BinMap1DConstIterator it = m_db.binMap(kSBin).begin1D(), ite = m_db.binMap(kSBin).end1D(); it != ite; ++it)
    {
        Bin const& bin = *it;
        gw.write_box(sbinLayer, 0, bin.xl(), bin.yl(), bin.xh(), bin.yh());
        gpfSPrint(kNONE, buf, "%u", bin.index1D());
        gw.gds_create_text(buf, center(bin, kX), center(bin, kY), sbinLayer+1, 5);
    }
    // write cells 
    ObsWraper<DrawFixedNodeHelper> dfnHelper(placeDB, DrawFixedNodeHelper(fixedCellBboxLayer, gw));
    for (std::vector<Node>::const_iterator it = vNode.begin(), ite = vNode.end(); it != ite; ++it)
    {
        Node const& node = *it;

        // bounding box of cells and its name 
        if (node.status() == PlaceStatusEnum::FIXED)
        {
            // draw fixed cell 
            dfnHelper(node);

            gpfSPrint(kNONE, buf, "(%u)%s", node.id(), getTextOnNode(node).c_str());
            gw.gds_create_text(buf, center(node, kX), center(node, kY), fixedCellBboxLayer+1, 5);
        }
        else if (m_sMarkNode.empty()) // do not write cells if there are marked cells 
        {
            gw.write_box(movableCellBboxLayer, 0, node.xl(), node.yl(), node.xh(), node.yh());
            gpfSPrint(kNONE, buf, "(%u)%s", node.id(), getTextOnNode(node).c_str());
            gw.gds_create_text(buf, center(node, kX), center(node, kY), movableCellBboxLayer+1, 5);
            if (placeDB.isMultiRowMovable(node)) // multi-row cell 
            {
                gw.write_box(multiRowCellBboxLayer, 0, node.xl(), node.yl(), node.xh(), node.yh());
                gw.gds_create_text(buf, center(node, kX), center(node, kY), multiRowCellBboxLayer+1, 5);
            }
            // write movement path 
            if (manhattanDistance(node.initPos(), ll(node)) > 0) // only draw if a cell is moved 
            {
                gw.gds_write_path();
                gw.gds_write_layer(movePathLayer);        // layer 
                gw.gds_write_datatype(0);            // datatype 
                gw.gds_write_pathtype(2);            // extended square ends
                gw.gds_write_width(5);               // 5 nm wide
                AlgoDB::coordinate_type x[2] = {node.initPos().x(), node.xl()};
                AlgoDB::coordinate_type y[2] = {node.initPos().y(), node.yl()};
                gw.gds_write_xy(x, y, 2);
                gw.gds_write_endel();
            }
        }
        if (m_sMarkNode.count(node.id())) // highlight marked nodes 
        {
            gw.write_box(markedNodeLayer, 0, node.xl(), node.yl(), node.xh(), node.yh());
            gpfSPrint(kNONE, buf, "(%u)%s", node.id(), getTextOnNode(node).c_str());
            gw.gds_create_text(buf, center(node, kX), center(node, kY), markedNodeLayer+1, 5);
        }
    }
    // write placement blockages 
    for (std::vector<Box<coordinate_type> >::const_iterator it = placeDB.placeBlockages().begin(); it != placeDB.placeBlockages().end(); ++it)
    {
        gw.write_box(blockageBboxLayer, 0, it->xl(), it->yl(), it->xh(), it->yh());
    }
    // write pins 
    for (std::vector<Pin>::const_iterator it = vPin.begin(), ite = vPin.end(); it != ite; ++it)
    {
        Pin const& pin = *it;
        Box<AlgoDB::coordinate_type> box = getPinBbox(pin);
        // bounding box of pins and its macropin name 
        gw.write_box(pinLayer, 0, box.xl(), box.yl(), box.xh(), box.yh());
        gw.gds_create_text(getTextOnPin(pin).c_str(), center(box, kX), center(box, kY), pinLayer+1, 5);
    }
    // write nets 
    unsigned count = 0;
    for (std::vector<Net>::const_iterator it = vNet.begin(), ite = vNet.end(); it != ite; ++it)
    {
        Net const& net = *it;
        // ignore large nets for net analysis 
        if (net.pins().size() > 100)
            continue; 
        // only draw nets for marked nodes 
        bool drawFlag = false;
        for (std::vector<AlgoDB::index_type>::const_iterator itp = net.pins().begin(), itpe = net.pins().end(); itp != itpe; ++itp)
        {
            Pin const& pin = vPin.at(*itp);
            if (m_sMarkNode.count(pin.nodeId()))
            {
                drawFlag = true;
                break;
            }
        }
        if (!drawFlag) continue;

        Point<AlgoDB::coordinate_type> srcPoint;
        for (std::vector<AlgoDB::index_type>::const_iterator itp = net.pins().begin(), itpe = net.pins().end(); itp != itpe; ++itp)
        {
            Pin const& pin = vPin.at(*itp);
            if (itp == net.pins().begin())
                srcPoint = placeDB.pinPos(pin);
            else 
            {
                Point<AlgoDB::coordinate_type> sinkPoint = placeDB.pinPos(pin);
                // write path 
                gw.gds_write_path();
                gw.gds_write_layer(netLayer+count);        // layer 
                gw.gds_write_datatype(0);            // datatype 
                gw.gds_write_pathtype(2);            // extended square ends
                gw.gds_write_width(1);               // 1 nm wide
                AlgoDB::coordinate_type x[2] = {srcPoint.x(), sinkPoint.x()};
                AlgoDB::coordinate_type y[2] = {srcPoint.y(), sinkPoint.y()};
                gw.gds_write_xy(x, y, 2);
                gw.gds_write_endel();
            }
        }
        // write net bbox 
        AlgoDB::box_type box;
        m_db.computeHPWLBox(box, net);
        gw.write_box(netLayer+count+1, 0, box.xl(), box.yl(), box.xh(), box.yh());
        count += 2;
    }

    // call extension 
    if (m_ext)
        m_ext->writeGdsii(m_db, gw);
}

unsigned PlaceDrawer::getLayer(bool reset) const 
{
    static unsigned count = 0;
    if (reset) 
        count = 0;
    return (++count)<<1;
}

std::string PlaceDrawer::getTextOnNode(Node const& node) const 
{
    if (m_db.placeDB().hasMacros())
        return m_db.placeDB().macroName(node);
    else 
        return m_db.placeDB().nodeName(node);
}

std::string PlaceDrawer::getTextOnPin(Pin const& pin) const 
{
    if (m_db.placeDB().hasMacros())
    {
        MacroPin const& mpin = m_db.placeDB().macroPin(pin);
        return mpin.name();
    }
    else 
        return "NA";
}

Box<PlaceDrawer::coordinate_type> PlaceDrawer::getPinBbox(Pin const& pin) const
{
    if (m_db.placeDB().hasMacros())
        return m_db.placeDB().pinBbox(pin);
    else 
    {
        Point<coordinate_type> pinPos = m_db.placeDB().pinPos(pin);
        return Box<coordinate_type>(pinPos.x()-5, pinPos.y()-5, pinPos.x()+5, pinPos.y()+5);
    }
}

GPF_END_NAMESPACE
