/*************************************************************************
    > File Name: LefCbkHelper.h
    > Author: Yibo Lin
    > Mail: yibolin@utexas.edu
    > Created Time: Wed 22 Jul 2015 11:13:39 PM CDT
 ************************************************************************/

#ifndef DREAMPLACE_LEFCBKHELPER_H
#define DREAMPLACE_LEFCBKHELPER_H

#include "Macro.h"
#include "GeometryApi.h"
#include <limbo/geometry/Polygon2Rectangle.h>

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
struct LefCbkGeometryHelperTraits;

/// specialization for MacroPort
template <>
struct LefCbkGeometryHelperTraits<MacroPort>
{
    typedef MacroPort::coordinate_type coordinate_type;
    typedef Box<coordinate_type> box_type;

    static void add(MacroPort& macroPort, std::string const& layerName, box_type const& box)
    {
        macroPort.boxes().push_back(box);
        macroPort.layers().push_back(layerName);
    }
    /// random access iterator is required 
    template <typename Iterator>
    static void add(MacroPort& macroPort, std::string const& layerName, Iterator first, Iterator last)
    {
        macroPort.boxes().insert(macroPort.boxes().end(), first, last);
        macroPort.layers().insert(macroPort.layers().end(), (last-first), layerName);
    }
};

/// specialization for MacroObs
template <>
struct LefCbkGeometryHelperTraits<MacroObs>
{
    typedef MacroObs::coordinate_type coordinate_type;
    typedef Box<coordinate_type> box_type;

    static void add(MacroObs& macroObs, std::string const& layerName, box_type const& box)
    {
        macroObs.add(layerName, box);
    }
    /// random access iterator is required 
    template <typename Iterator>
    static void add(MacroObs& macroObs, std::string const& layerName, Iterator first, Iterator last)
    {
        macroObs.add(layerName, first, last);
    }
};

/// a helper struct to add LEF geometry
template <typename T>
struct LefCbkGeometryHelper
{
    typedef typename T::coordinate_type coordinate_type;
    typedef Box<coordinate_type> box_type;
    typedef Point<coordinate_type> point_type;
    typedef LefCbkGeometryHelperTraits<T> traits_type;

    void operator()(T& target, LefParser::lefiGeometries const& geos, point_type const& initOffset, coordinate_type lefUnit) const
    {
        std::string layerName;
        for (int k = 0; k < geos.numItems(); ++k)
        {
            switch (geos.itemType(k))
            {
                case LefParser::lefiGeomLayerE:
                    {
                        layerName = geos.getLayer(k);
                        break;
                    }
                case LefParser::lefiGeomPathE:
                    {
                        dreamplacePrint(kWARN, "unsupported shape lefiGeomPathE in LEF file\n");
                        break;
                    }
                case LefParser::lefiGeomPathIterE:
                    {
                        dreamplacePrint(kWARN, "unsupported shape lefiGeomPathIterE in LEF file\n");
                        break;
                    }
                case LefParser::lefiGeomRectE:
                    {
                        LefParser::lefiGeomRect* rect = geos.getRect(k);
                        traits_type::add(target, layerName, box_type(
                                    (coordinate_type)round(rect->xl*lefUnit)-initOffset.x(),
                                    (coordinate_type)round(rect->yl*lefUnit)-initOffset.y(),
                                    (coordinate_type)round(rect->xh*lefUnit)-initOffset.x(),
                                    (coordinate_type)round(rect->yh*lefUnit)-initOffset.y()
                                    ));
                        break;
                    }
                case LefParser::lefiGeomRectIterE:
                    {
                        dreamplacePrint(kWARN, "unsupported shape lefiGeomRectIterE in LEF file\n");
                        break;
                    }
                case LefParser::lefiGeomPolygonE:
                    {
                        LefParser::lefiGeomPolygon* poly = geos.getPolygon(k);
                        std::vector<Point<coordinate_type> > points (poly->numPoints);
                        for (int l = 0; l < poly->numPoints; l++)
                        {
                            points[l].set(
                                    round(poly->x[l]*lefUnit)-initOffset.x(), 
                                    round(poly->y[l]*lefUnit)-initOffset.y()
                                    );
                        }
                        std::vector<box_type> vRect;
                        bool success = limbo::geometry::polygon2rectangle(
                                points.begin(), points.end(), 
                                std::vector<Point<coordinate_type> >(), 
                                vRect,
                                limbo::geometry::HORIZONTAL_SLICING);
                        dreamplaceAssertMsg(success, "failed to convert polygon to rectangles");
                        traits_type::add(target, layerName, vRect.begin(), vRect.end());
                        break;
                    }
                case LefParser::lefiGeomPolygonIterE:
                    {
                        dreamplacePrint(kWARN, "unsupported shape lefiGeomPolygonIterE in LEF file\n");
                        break;
                    }
                default: 
                    {
                        dreamplacePrint(kWARN, "unsupported shape %d in LEF file\n", geos.itemType(k));
                        break;
                    }
            }
        }
    }
};

DREAMPLACE_END_NAMESPACE

#endif
