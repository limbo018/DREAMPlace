/*************************************************************************
    > File Name: Net.h
    > Author: Yibo Lin
    > Mail: yibolin@utexas.edu
    > Created Time: Mon Jun 15 22:37:37 2015
 ************************************************************************/

#ifndef DREAMPLACE_NET_H
#define DREAMPLACE_NET_H

#include <vector>
#include "Pin.h"
#include "Box.h"

DREAMPLACE_BEGIN_NAMESPACE

class Net : public Object
{
    public:
        typedef Object base_type;
        typedef base_type::coordinate_type coordinate_type;
        typedef coordinate_traits<coordinate_type>::index_type index_type;
        typedef coordinate_traits<coordinate_type>::weight_type weight_type;
        typedef Point<coordinate_type> point_type;
        typedef Box<coordinate_type> box_type;

        /// default constructor 
        Net();
        /// copy constructor
        Net(Net const& rhs);
        /// assignment
        Net& operator=(Net const& rhs);

        /// member functions
        box_type const& bbox() const {return m_bbox;}
        box_type& bbox() {return m_bbox;}
        Net& setBbox(box_type const& b) {m_bbox = b; return *this;}

        weight_type weight() const {return m_weight;}
        Net& setWeight(weight_type w) {m_weight = w; return *this;}

        std::vector<index_type> const& pins() const {return m_vPinId;}
        std::vector<index_type>& pins() {return m_vPinId;}

        /// \return the source pin index of the net 
        index_type source() const {return (!m_vPinId.empty())? m_vPinId.front() : std::numeric_limits<index_type>::max();}
    protected:
        void copy(Net const& rhs);

        box_type m_bbox; ///< bounding box of net 
        weight_type m_weight; ///< weight of net 
        std::vector<index_type> m_vPinId; ///< index of pins, the first one is source  
};

class NetProperty
{
    public:
        /// default constructor 
        NetProperty();
        /// copy constructor
        NetProperty(NetProperty const& rhs);
        /// assignment
        NetProperty& operator=(NetProperty const& rhs);

        /// member functions 
        std::string const& name() const {return m_name;}
        NetProperty& setName(std::string const& s) {m_name = s; return *this;}

    protected:
        void copy(NetProperty const& rhs);

        std::string m_name; ///< net name 
};

DREAMPLACE_END_NAMESPACE

#endif
