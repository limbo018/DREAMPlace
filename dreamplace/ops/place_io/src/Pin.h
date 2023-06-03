/*************************************************************************
    > File Name: Pin.h
    > Author: Yibo Lin
    > Mail: yibolin@utexas.edu
    > Created Time: Mon Jun 15 21:48:45 2015
 ************************************************************************/

#ifndef DREAMPLACE_PIN_H
#define DREAMPLACE_PIN_H

#include <string>
#include "Object.h"
#include "Point.h"
#include "Enums.h"

DREAMPLACE_BEGIN_NAMESPACE

class Pin : public Object
{
    public:
        typedef Object base_type;
        typedef base_type::coordinate_type coordinate_type;
        typedef coordinate_traits<coordinate_type>::index_type index_type;
        typedef Point<coordinate_type> point_type;

        /// default constructor 
        Pin();
        /// copy constructor
        Pin(Pin const& rhs);
        /// assignment
        Pin& operator=(Pin const& rhs);

        /// member functions 
        index_type const& macroPinId() const {return m_macroPinId;}
        Pin& setMacroPinId(index_type id) {m_macroPinId = id; return *this;}

        index_type nodeId() const {return m_nodeId;}
        Pin& setNodeId(index_type id) {m_nodeId = id; return *this;}

        index_type netId() const {return m_netId;}
        Pin& setNetId(index_type id) {m_netId = id; return *this;}

        point_type const& offset() const {return m_offset;}
        Pin& setOffset(point_type const& p) {m_offset = p; return *this;}

        SignalDirect const& direct() const {return m_direct;}
        Pin& setDirect(SignalDirect const& d) {m_direct = d; return *this;} 

        std::string name() const { return m_name; }
        Pin& setName(std::string const& n) { m_name = n; return *this; }
    protected:
        void copy(Pin const& rhs);

        index_type m_macroPinId; ///< index to the macro pin list of corresponding macro 
        index_type m_nodeId; ///< corresponding node  
        index_type m_netId; ///< corresponding net 
        point_type m_offset; ///< offset based on the origin of node 
        SignalDirect m_direct; ///< direction of signal 
        std::string m_name; ///< name of this pin
};

inline Pin::Pin() 
    : Pin::base_type()
    , m_macroPinId(std::numeric_limits<Pin::index_type>::max())
    , m_nodeId(std::numeric_limits<Pin::index_type>::max())
    , m_netId(std::numeric_limits<Pin::index_type>::max())
    , m_offset()
    , m_direct()
    , m_name()
{
}
inline Pin::Pin(Pin const& rhs)
    : Pin::base_type(rhs)
{
    copy(rhs);
}
inline Pin& Pin::operator=(Pin const& rhs)
{
    if (this != &rhs)
    {
        this->base_type::operator=(rhs);
        copy(rhs);
    }
    return *this;
}
inline void Pin::copy(Pin const& rhs)
{
    m_macroPinId = rhs.m_macroPinId;
    m_nodeId = rhs.m_nodeId; 
    m_netId = rhs.m_netId; 
    m_offset = rhs.m_offset; 
    m_direct = rhs.m_direct;
    m_name = rhs.m_name;
}


DREAMPLACE_END_NAMESPACE

#endif
