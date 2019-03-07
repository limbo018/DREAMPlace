/*************************************************************************
    > File Name: DrawPlaceExt.cpp
    > Author: Yibo Lin
    > Mail: yibolin@utexas.edu
    > Created Time: Sat Nov  5 16:51:59 2016
 ************************************************************************/

#include "DrawPlaceExt.h"
#include "AlgoDB.h"
#include "GlobalMoveCellChain.h"

#if DRAWPLACE == 1
#include <cairo/cairo.h>
#include <cairo/cairo-pdf.h>
#include <cairo/cairo-ps.h>
#include <cairo/cairo-svg.h>
#endif


DREAMPLACE_BEGIN_NAMESPACE

void ChainMoveDrawerExt::paintCairo(AlgoDB const& db, cairo_t* c, double width, double height) const 
{
#if DRAWPLACE == 1
    PlaceDB const& placeDB = db.placeDB();
    std::vector<Node> const& vNode = placeDB.nodes();

    AlgoDB::box_type region = intersection(m_region, placeDB.dieArea()).first;
    double regionRatio = region.area()/(double)placeDB.dieArea().area();
    dreamplacePrint(kINFO, "draw region (%d, %d, %d, %d) ratio = %g\n", region.xl(), region.yl(), region.xh(), region.yh(), regionRatio);
    cairo_save(c);

    // the order of transform matters 
    cairo_translate(c, 0-region.xl()*width/region.width(), height+region.yl()*height/region.height()); // translate is additive
    cairo_scale(c, width/region.width(), -height/region.height()); // scale is additive 
    // nodes 
    cairo_set_line_width(c, 1);
    cairo_set_source_rgb(c, 0.1, 0.1, 0.1);
    for (std::vector<Node>::const_iterator it = vNode.begin(), ite = vNode.end(); it != ite; ++it)
    {
        Node const& node = *it;
        // only draw nodes in the region 
        if (!intersects(region, node))
            continue; 
        cairo_rectangle(c, node.xl(), node.yl(), node.width(), node.height());
        cairo_stroke(c);
    }
    // chain moves 
    cairo_set_line_width(c, 3);
    cairo_set_source_rgb(c, 0.0, 0.0, 0.0);
    for (std::vector<NodeMove>::const_iterator it = m_entry.vNodeMove.begin(), ite = m_entry.vNodeMove.end(); it != ite; ++it)
    {
        NodeMove const& nm = *it;
        Node const& node = db.node(nm.nodeId);
        // only draw nodes in the region 
        if (!intersects(region, node))
            continue; 
        // draw node at original position 
        cairo_set_source_rgb(c, 0.0, 0.0, 0.0);
        cairo_rectangle(c, nm.origPos.x(), nm.origPos.y(), node.width(), node.height());
        cairo_stroke(c);
        cairo_set_source_rgb(c, 0/255.0, 191/255.0, 255/255.0); // fill color 
        cairo_rectangle(c, nm.origPos.x(), nm.origPos.y(), node.width(), node.height());
        cairo_fill(c);

        // draw arrow from center of original to current 
        // draw line 
        Point<AlgoDB::coordinate_type> srcPoint (nm.origPos.x()+node.width()/2, nm.origPos.y()+node.height()/2);
        Point<AlgoDB::coordinate_type> sinkPoint (center(node));
        cairo_set_source_rgb(c, 0.0, 0.0, 0.0);
        cairo_move_to(c, srcPoint.x(), srcPoint.y());
        cairo_line_to(c, sinkPoint.x(), sinkPoint.y());
        cairo_stroke(c);
        // draw triangle 
        // (x1, y1)
        //        \ arrowLength
        // --------- 2*arrowDegree
        //        /
        // (x2, y2)
        // we need to compute (x1, y1), (x2, y2) for the ending point of arrow triangle 
        double arrowLength = db.placeDB().siteWidth()/3; 
        double arrowDegree = (45/180)*M_PI; 
        double angle = atan2 (sinkPoint.y() - srcPoint.y(), sinkPoint.x() - srcPoint.x()) + M_PI;
        double x1 = sinkPoint.x() + arrowLength * cos(angle - arrowDegree);
        double y1 = sinkPoint.y() + arrowLength * sin(angle - arrowDegree);
        double x2 = sinkPoint.x() + arrowLength * cos(angle + arrowDegree);
        double y2 = sinkPoint.y() + arrowLength * sin(angle + arrowDegree);
        cairo_move_to(c, x1, y1);
        cairo_line_to(c, sinkPoint.x(), sinkPoint.y());
        cairo_stroke(c);
        cairo_move_to(c, x2, y2);
        cairo_line_to(c, sinkPoint.x(), sinkPoint.y());
        cairo_set_source_rgb(c, 255/255.0, 0/255.0, 0/255.0); // fill color 
        cairo_stroke(c);
    }
#else 
    dreamplacePrint(kWARN, "&db = %p, c = %p, width = %g, height = %g are not used, as DRAWPLACE not enabled\n", &db, c, width, height);
#endif
}

void ChainMoveDrawerExt::writeGdsii(AlgoDB const& /*db*/, GdsParser::GdsWriter& /*gw*/) const
{
}

void RowDPDrawerExt::paintCairo(AlgoDB const& db, cairo_t* c, double width, double height) const 
{
#if DRAWPLACE == 1
    PlaceDB const& placeDB = db.placeDB();
    std::vector<Node> const& vNode = placeDB.nodes();

    AlgoDB::box_type region = intersection(m_region, placeDB.dieArea()).first;
    double regionRatio = region.area()/(double)placeDB.dieArea().area();
    dreamplacePrint(kINFO, "draw region (%d, %d, %d, %d) ratio = %g\n", region.xl(), region.yl(), region.xh(), region.yh(), regionRatio);
    cairo_save(c);

    // the order of transform matters 
    cairo_translate(c, 0-region.xl()*width/region.width(), height+region.yl()*height/region.height()); // translate is additive
    cairo_scale(c, width/region.width(), -height/region.height()); // scale is additive 
    // nodes 
    cairo_set_line_width(c, 1);
    cairo_set_source_rgb(c, 0.1, 0.1, 0.1);
    for (std::vector<Node>::const_iterator it = vNode.begin(), ite = vNode.end(); it != ite; ++it)
    {
        Node const& node = *it;
        // only draw nodes in the region 
        if (!intersects(region, node))
            continue; 
        cairo_rectangle(c, node.xl(), node.yl(), node.width(), node.height());
        cairo_stroke(c);
    }
    // row placement  
    cairo_set_line_width(c, 3);
    cairo_set_source_rgb(c, 0.0, 0.0, 0.0);
    AlgoDB::index_type const numTotalNodes = vNode.size();
    for (AlgoDB::index_type rowIdx = m_rowBegin; rowIdx <= m_rowEnd; ++rowIdx)
    {
        std::vector<map_element_type> const& vRowNode = m_mRowNode.at(rowIdx);
        for (std::vector<map_element_type>::const_iterator it = vRowNode.begin(); it != vRowNode.end(); ++it)
        {
            map_element_type const& element = *it;
            if (element.nodeId >= numTotalNodes) // skip invalid entries 
                continue;
            Node const& node = vNode.at(element.nodeId);
            if (!intersects(region, node))
                continue; 
            // draw node at original position 
            cairo_set_source_rgb(c, 0.0, 0.0, 0.0);
            cairo_rectangle(c, element.inv.low(), node.yl(), node.width(), node.height());
            cairo_stroke(c);
            cairo_set_source_rgb(c, 0/255.0, 191/255.0, 255/255.0); // fill color 
            cairo_rectangle(c, element.inv.low(), node.yl(), node.width(), node.height());
            cairo_fill(c);

            // draw arrow from center of original to current 
            // draw line 
            Point<AlgoDB::coordinate_type> srcPoint (element.inv.low()+node.width()/2, node.yl()+node.height()/2);
            Point<AlgoDB::coordinate_type> sinkPoint (center(node));
            cairo_set_source_rgb(c, 0.0, 0.0, 0.0);
            cairo_move_to(c, srcPoint.x(), srcPoint.y());
            cairo_line_to(c, sinkPoint.x(), sinkPoint.y());
            cairo_stroke(c);
            // draw triangle 
            // (x1, y1)
            //        \ arrowLength
            // --------- 2*arrowDegree
            //        /
            // (x2, y2)
            // we need to compute (x1, y1), (x2, y2) for the ending point of arrow triangle 
            double arrowLength = db.placeDB().siteWidth()/3; 
            double arrowDegree = (45/180)*M_PI; 
            double angle = atan2 (sinkPoint.y() - srcPoint.y(), sinkPoint.x() - srcPoint.x()) + M_PI;
            double x1 = sinkPoint.x() + arrowLength * cos(angle - arrowDegree);
            double y1 = sinkPoint.y() + arrowLength * sin(angle - arrowDegree);
            double x2 = sinkPoint.x() + arrowLength * cos(angle + arrowDegree);
            double y2 = sinkPoint.y() + arrowLength * sin(angle + arrowDegree);
            cairo_move_to(c, x1, y1);
            cairo_line_to(c, sinkPoint.x(), sinkPoint.y());
            cairo_stroke(c);
            cairo_move_to(c, x2, y2);
            cairo_line_to(c, sinkPoint.x(), sinkPoint.y());
            cairo_set_source_rgb(c, 255/255.0, 0/255.0, 0/255.0); // fill color 
            cairo_stroke(c);
        }
    }
#else 
    dreamplacePrint(kWARN, "&db = %p, c = %p, width = %g, height = %g are not used, as DRAWPLACE not enabled\n", &db, c, width, height);
#endif
}

void RowDPDrawerExt::writeGdsii(AlgoDB const& /*db*/, GdsParser::GdsWriter& /*gw*/) const
{
}

DREAMPLACE_END_NAMESPACE
