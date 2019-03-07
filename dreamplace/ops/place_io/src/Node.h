/*************************************************************************
    > File Name: Node.h
    > Author: Yibo Lin
    > Mail: yibolin@utexas.edu
    > Created Time: Mon Jun 15 21:23:38 2015
 ************************************************************************/

#ifndef DREAMPLACE_NODE_H
#define DREAMPLACE_NODE_H

#include <vector>
#include "Pin.h"
#include "Box.h"

DREAMPLACE_BEGIN_NAMESPACE

/// class Node denotes an instantiation of a standard cell 
class Node : public Box<Object::coordinate_type>, public Object
{
	public:
        typedef Object base_type2;
        typedef base_type2::coordinate_type coordinate_type;
        typedef Box<coordinate_type> base_type1;
        typedef coordinate_traits<coordinate_type>::index_type index_type;
        typedef Point<coordinate_type> point_type;

        /// default constructor 
        Node();
        /// copy constructor
        Node(Node const& rhs);
        /// assignment
        Node& operator=(Node const& rhs);

        /// member functions 
        PlaceStatusEnum::PlaceStatusType status() const {return (PlaceStatusEnum::PlaceStatusType)m_status;}
        Node& setStatus(PlaceStatusEnum::PlaceStatusType s) {m_status = s; return *this;}
        Node& setStatus(PlaceStatus const& s) {return setStatus(s.value());}

        MultiRowAttrEnum::MultiRowAttrType multiRowAttr() const {return (MultiRowAttrEnum::MultiRowAttrType)m_multiRowAttr;}
        Node& setMultiRowAttr(MultiRowAttrEnum::MultiRowAttrType a) {m_multiRowAttr = a; return *this;}
        Node& setMultiRowAttr(MultiRowAttr const& a) {return setMultiRowAttr(a.value());}

        OrientEnum::OrientType orient() const {return (OrientEnum::OrientType)m_orient;}
        Node& setOrient(OrientEnum::OrientType o) {m_orient = o; return *this;}
        Node& setOrient(Orient const& o) {return setOrient(o.value());}

        point_type const& initPos() const {return m_initPos;}
        Node& setInitPos(point_type const& p) {m_initPos = p; return *this;}

        std::vector<index_type> const& pins() const {return m_vPinId;}
        std::vector<index_type>& pins() {return m_vPinId;}

        ///====  helper functions ====
        /// \return absolute position of a pin with given position of the node 
        point_type pinPos(Pin const& p, point_type const& pos) const {return pos+p.offset();}
        /// \return absolute position of a pin with given position (xl, yl) of the node 
        point_type pinPos(Pin const& p, coordinate_type xl, coordinate_type yl) const {return pinPos(p, point_type(xl, yl));}
        /// \return absolute position of a pin 
        point_type pinPos(Pin const& p) const {return pinPos(p, ll(*this));}
        /// \return absolute position of a pin in one direction (xy)
        coordinate_type pinPos(Pin const& p, Direction1DType d) const {return get(d, kLOW)+p.offset().get(d);}
        /// \return absolute position of a pin in x direction
        coordinate_type pinX(Pin const& p) const {return pinPos(p, kX);}
        /// \return absolute position of a pin in y direction
        coordinate_type pinY(Pin const& p) const {return pinPos(p, kY);}
        /// \return number of sites with given site width and site height 
        /// may be smaller than actual sizes if either width or height is fractional 
        std::size_t siteArea(coordinate_type siteWidth, coordinate_type rowHeight) const {return (width()/siteWidth)*(height()/rowHeight);}

    protected:
        void copy(Node const& rhs);

        /// I found this order gives the minimum object size 
        point_type m_initPos; ///< initial position 
        /// although single char is enough for these three enums, it seems openmp in gcc may crash for that 
        /// probably it is due to the problem of bit-fields 
        /// see https://lwn.net/Articles/478657/
        char m_status; ///< placement status, 2 bits are enough 
        char m_multiRowAttr; ///< multi-row attributes, 2 bits are enough 
        char m_orient; ///< orientation, 4 bits are enough
        std::vector<index_type> m_vPinId; ///< index of pins 
};

/// cell property class 
/// attributes of cells, but may not be often used 
class NodeProperty
{
    public:
        typedef Object::coordinate_type coordinate_type;
        typedef coordinate_traits<coordinate_type>::index_type index_type;

        /// default constructor 
        NodeProperty();
        /// copy constructor
        NodeProperty(NodeProperty const& rhs);
        /// assignment
        NodeProperty& operator=(NodeProperty const& rhs);

        /// member functions 
        std::string const& name() const {return m_name;}
        NodeProperty& setName(std::string const& s) {m_name = s; return *this;}

        index_type macroId() const {return m_macroId;}
        NodeProperty& setMacroId(index_type id) {m_macroId = id; return *this;}

    protected:
        void copy(NodeProperty const& rhs);

        std::string m_name; ///< instance name 
        index_type m_macroId; ///< standard cell id 
};

DREAMPLACE_END_NAMESPACE

#endif
