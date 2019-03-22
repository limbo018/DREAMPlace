/*************************************************************************
    > File Name: MacroPin.h
    > Author: Yibo Lin
    > Mail: yibolin@utexas.edu
    > Created Time: Tue Jun 16 21:08:59 2015
 ************************************************************************/

#ifndef DREAMPLACE_MACROPIN_H
#define DREAMPLACE_MACROPIN_H

#include <string>
#include <vector>
#include "Object.h"
#include "Box.h"

DREAMPLACE_BEGIN_NAMESPACE

/// class MacroPort describes the ports in a macro pin 
/// it consists of rectangles and polygons 
class MacroPort : public Object
{
    public:
        typedef Object base_type;
        typedef base_type::coordinate_type coordinate_type;
        typedef Box<coordinate_type> box_type;

        /// default constructor 
        MacroPort();
        /// copy constructor
        MacroPort(MacroPort const& rhs);
        /// assignment
        MacroPort& operator=(MacroPort const& rhs);

        box_type const& bbox() const {return m_bbox;}
        MacroPort& setBbox(box_type const& b) {m_bbox = b; return *this;}

        std::vector<box_type> const& boxes() const {return m_vBox;}
        std::vector<box_type>& boxes() {return m_vBox;}

        std::vector<std::string> const& layers() const {return m_vLayer;}
        std::vector<std::string>& layers() {return m_vLayer;}
    protected:
        void copy(MacroPort const& rhs);

        box_type m_bbox; ///< bounding box of port 
        std::vector<box_type> m_vBox; ///< decomposed rectangles from the polygon of port 
        std::vector<std::string> m_vLayer; ///< layers, the same number as boxes 
};

inline MacroPort::MacroPort() 
    : MacroPort::base_type()
    , m_bbox()
    , m_vBox()
    , m_vLayer()
{
}
inline MacroPort::MacroPort(MacroPort const& rhs)
    : MacroPort::base_type(rhs)
{
    copy(rhs);
}
inline MacroPort& MacroPort::operator=(MacroPort const& rhs)
{
    if (this != &rhs)
    {
        this->base_type::operator=(rhs);
        copy(rhs);
    }
    return *this;
}
inline void MacroPort::copy(MacroPort const& rhs)
{
    m_bbox = rhs.m_bbox;
    m_vBox = rhs.m_vBox;
    m_vLayer = rhs.m_vLayer;
}

/// class MacroPin describes the pins of a standard cell 
/// it contains detailed physical information such as name, shape, and direction
/// since a pin is usually a rectilinear polygon, the decomposed rectangles are introduced to store its shape
class MacroPin : public Object
{
    public:
        typedef Object base_type;
        typedef base_type::coordinate_type coordinate_type;
        typedef coordinate_traits<coordinate_type>::index_type index_type;
        typedef Box<coordinate_type> box_type;

        /// default constructor 
        MacroPin();
        /// copy constructor
        MacroPin(MacroPin const& rhs);
        /// assignment
        MacroPin& operator=(MacroPin const& rhs);

        /// member functions
        std::string const& name() const {return m_name;}
        MacroPin& setName(std::string const& s) {m_name = s; return *this;}

        SignalDirect const& direct() const {return m_direct;}
        MacroPin& setDirect(SignalDirect const& d) {m_direct = d; return *this;}

        box_type const& bbox() const {return m_bbox;}
        MacroPin& setBbox(box_type const& b) {m_bbox = b; return *this;}

        std::vector<MacroPort> const& macroPorts() const {return m_vMacroPort;}
        std::vector<MacroPort>& macroPorts() {return m_vMacroPort;}

        MacroPort const& macroPort(index_type id) const {return m_vMacroPort.at(id);}
        MacroPort& macroPort(index_type id) {return m_vMacroPort.at(id);}

        /// add macro port and set index 
        index_type addMacroPort();
    protected:
        void copy(MacroPin const& rhs);

        std::string m_name; ///< pin name 
        SignalDirect m_direct; ///< signal direction of pin 
        box_type m_bbox; ///< bounding box of pin 
        std::vector<MacroPort> m_vMacroPort; ///< ports in a pin 
};

inline MacroPin::MacroPin() 
    : MacroPin::base_type()
    , m_name("")
    , m_direct()
    , m_bbox()
    , m_vMacroPort()
{
}
inline MacroPin::MacroPin(MacroPin const& rhs)
    : MacroPin::base_type(rhs)
{
    copy(rhs);
}
inline MacroPin& MacroPin::operator=(MacroPin const& rhs)
{
    if (this != &rhs)
    {
        this->base_type::operator=(rhs);
        copy(rhs);
    }
    return *this;
}
inline void MacroPin::copy(MacroPin const& rhs)
{
    m_name = rhs.m_name;
    m_direct = rhs.m_direct;
    m_bbox = rhs.m_bbox;
    m_vMacroPort = rhs.m_vMacroPort;
}

inline MacroPin::index_type MacroPin::addMacroPort()
{
    m_vMacroPort.push_back(MacroPort());
    MacroPort& mp = m_vMacroPort.back();
    mp.setId(m_vMacroPort.size()-1);

    return mp.id();
}

///====  helper functions ====
/// compute bounding box from a port shape 
inline void deriveMacroPortBbox(MacroPort& mp)
{
    typedef MacroPort::coordinate_type coordinate_type;
    typedef MacroPort::box_type box_type;
    // construct an invalid box 
    box_type box (
            std::numeric_limits<coordinate_type>::max(), 
            std::numeric_limits<coordinate_type>::max(), 
            std::numeric_limits<coordinate_type>::min(), 
            std::numeric_limits<coordinate_type>::min()
            );
    // compute bounding box 
    std::vector<box_type> const& vBox = mp.boxes();
    for (std::vector<box_type>::const_iterator it = vBox.begin(), ite = vBox.end(); it != ite; ++it)
        box.encompass(*it);
    // update bounding box 
    mp.setBbox(box);
}
/// compute bounding box from pin shape
inline void deriveMacroPinBbox(MacroPin& mp)
{
    // assume bounding box in macro port is update-to-date 
    typedef MacroPin::coordinate_type coordinate_type;
    typedef MacroPin::box_type box_type;
    // construct an invalid box 
    box_type box (
            std::numeric_limits<coordinate_type>::max(), 
            std::numeric_limits<coordinate_type>::max(), 
            std::numeric_limits<coordinate_type>::min(), 
            std::numeric_limits<coordinate_type>::min()
            );
    // compute bounding box 
    std::vector<MacroPort> const& vMacroPort = mp.macroPorts();
    for (std::vector<MacroPort>::const_iterator it = vMacroPort.begin(), ite = vMacroPort.end(); it != ite; ++it)
        box.encompass(it->bbox());
    // update bounding box 
    mp.setBbox(box);
}

DREAMPLACE_END_NAMESPACE

#endif
