/*************************************************************************
    > File Name: DrawPlace.h
    > Author: Yibo Lin
    > Mail: yibolin@utexas.edu
    > Created Time: Sat 20 Jun 2015 12:18:01 AM CDT
 ************************************************************************/

#ifndef DREAMPLACE_DRAWPLACE_H
#define DREAMPLACE_DRAWPLACE_H

#include <string>
#include <ostream>
#include <set>
#include "Object.h"
#include "Box.h"

typedef struct _cairo_surface cairo_surface_t;
namespace GdsParser {
struct GdsWriter; 
} // namespace GdsParser 

DREAMPLACE_BEGIN_NAMESPACE

class Node;
class Pin;
class AlgoDB;
class PlaceDrawerExt; 

/// PlaceDrawer write files in various formats from DREAMPLACE_NAMESPACE::PlaceDB
class PlaceDrawer
{
    public:
        typedef Object::coordinate_type coordinate_type;

        enum FileFormat {
            EPS = 0, // handle by cairo
            PDF = 1, // handle by cairo
            SVG = 2, // handle by cairo 
            PNG = 3,  // handle by cairo
            GDSII = 4
        };
        enum DrawContent {
            NONE = 0, 
            NODE = 1, 
            NODETEXT = 2, 
            PIN = 4, 
            NET = 8, 
            ALL = NODE|NODETEXT|PIN|NET
        };
        /// constructor 
        PlaceDrawer(AlgoDB const& db, PlaceDrawerExt* ext = NULL, int content = ALL);

        bool run(std::string const& filename, FileFormat ff) const;
        /// \param first and last mark nodes whose nets will be drawn 
        template <typename Iterator>
        bool run(std::string const& filename, FileFormat ff, Iterator first, Iterator last);
    protected:
        /// write formats supported by cairo 
        /// \param width of screen 
        /// \param height of screen 
        void paintCairo(cairo_surface_t* cs, double width, double height) const;
        bool writeFig(const char* fname, double width, double height, FileFormat ff) const;
        /// scale source coordinate to target screen 
        double scaleToScreen(double coord, double srcOffset, double srcSize, double tgtOffset, double tgtSize) const;

        /// write gdsii format 
        virtual bool writeGdsii(std::string const& filename) const;
        /// write contents to GDSII 
        virtual void writeGdsiiContent(GdsParser::GdsWriter& gw) const; 
        /// automatically increment by 2
        /// \param reset controls whehter restart from 1 
        unsigned getLayer(bool reset = false) const;
        /// \return text to be shown on cell 
        std::string getTextOnNode(Node const& node) const;
        /// \return text to be shown on pin 
        std::string getTextOnPin(Pin const& pin) const;
        Box<coordinate_type> getPinBbox(Pin const& pin) const;

        AlgoDB const& m_db; 
        std::set<unsigned> m_sMarkNode; ///< marked nodes whose net will be drawn
        PlaceDrawerExt* m_ext; ///< an extension for flexibility 
        int m_content; ///< content for DrawContent

        friend struct DrawFixedNodeHelper;
};

template <typename Iterator>
bool PlaceDrawer::run(std::string const& filename, PlaceDrawer::FileFormat ff, Iterator first, Iterator last)
{
    m_sMarkNode.insert(first, last);
    bool flag = run(filename, ff);
    m_sMarkNode.clear();
    return flag;
}

DREAMPLACE_END_NAMESPACE

#endif
