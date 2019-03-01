/*************************************************************************
    > File Name: PlaceDrawerExt.h
    > Author: Yibo Lin
    > Mail: yibolin@utexas.edu
    > Created Time: Sat Nov  5 16:35:08 2016
 ************************************************************************/

#ifndef GPF_DRAWPLACEEXT_H
#define GPF_DRAWPLACEEXT_H

#include <vector>
#include "Object.h"
#include "Box.h"
#include "NodeMapElement.h"

typedef struct _cairo cairo_t;
namespace GdsParser {
struct GdsWriter; 
} // namespace GdsParser 

GPF_BEGIN_NAMESPACE

class AlgoDB;
struct ScoreBoardEntry; 

/// base class for DrawPlace extensions 
/// this is also the default one 
class PlaceDrawerExt
{
    public:
        /// constructor 
        PlaceDrawerExt() {}
        /// destructor 
        virtual ~PlaceDrawerExt() {}

        /// writing fig 
        virtual void paintCairo(AlgoDB const& /*db*/, cairo_t* /*c*/, double /*width*/, double /*height*/) const {} 
        /// writing gdsii 
        virtual void writeGdsii(AlgoDB const& /*db*/, GdsParser::GdsWriter& /*gw*/) const {}
};

/// an extension for drawing chain move algorithm  
class ChainMoveDrawerExt : public PlaceDrawerExt
{
    public:
        typedef PlaceDrawerExt base_type; 

        ChainMoveDrawerExt(ScoreBoardEntry const& entry, Box<Object::coordinate_type> const& region) : base_type(), m_entry(entry), m_region(region) {}

        /// writing fig 
        virtual void paintCairo(AlgoDB const& db, cairo_t* c, double width, double height) const;  
        /// writing gdsii 
        virtual void writeGdsii(AlgoDB const& db, GdsParser::GdsWriter& gw) const; 
    protected:
        ScoreBoardEntry const& m_entry; 
        Box<Object::coordinate_type> m_region; ///< drawing region  
};

/// an extension for drawing row placement algorithm  
class RowDPDrawerExt : public PlaceDrawerExt
{
    public:
        typedef PlaceDrawerExt base_type; 
        typedef NodeMapElement map_element_type; 

        RowDPDrawerExt(std::vector<std::vector<map_element_type> > const& mRowNode, unsigned int rb, unsigned int re, Box<Object::coordinate_type> const& region) 
            : base_type(), m_mRowNode(mRowNode), m_rowBegin(rb), m_rowEnd(re), m_region(region) {}

        /// writing fig 
        virtual void paintCairo(AlgoDB const& db, cairo_t* c, double width, double height) const;  
        /// writing gdsii 
        virtual void writeGdsii(AlgoDB const& db, GdsParser::GdsWriter& gw) const; 
    protected:
        std::vector<std::vector<map_element_type> > const& m_mRowNode; ///< cells on rows including fixed cells , indexed by row id, use map_element_type because of obstruction 
                                                            ///< multi-row cells will be distributed to multiple rows 
        unsigned int m_rowBegin; 
        unsigned int m_rowEnd; 
        Box<Object::coordinate_type> m_region; ///< drawing region 
};

GPF_END_NAMESPACE

#endif
