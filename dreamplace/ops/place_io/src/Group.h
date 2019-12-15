/**
 * @file   Group.h
 * @author Yibo Lin
 * @date   Dec 2019
 */

#ifndef DREAMPLACE_GROUP_H
#define DREAMPLACE_GROUP_H

#include <vector>
#include "Object.h"

DREAMPLACE_BEGIN_NAMESPACE

class Group : public Object
{
	public:
        typedef Object base_type;
        typedef base_type::coordinate_type coordinate_type;
        typedef coordinate_traits<coordinate_type>::index_type index_type;

        /// default constructor 
        Group();
        /// copy constructor
        Group(Group const& rhs);
        /// assignment
        Group& operator=(Group const& rhs);

        /// node functions 
        std::string const& name() const {return m_name;}
        Group& setName(std::string const& name) {m_name = name; return *this;}

        std::vector<std::string> const& nodeNames() const {return m_vNodeName;}
        std::vector<std::string>& nodeNames() {return m_vNodeName;}

        std::vector<index_type> const& nodes() const {return m_vNodeId;}
        std::vector<index_type>& nodes() {return m_vNodeId;}

        index_type region() const {return m_region_id;}
        Group& setRegion(index_type region_id) {m_region_id = region_id; return *this;}

    protected:
        void copy(Group const& rhs);

        std::string m_name; ///< group name 
        std::vector<std::string> m_vNodeName; ///< group node names, they names may be regex 
        std::vector<index_type> m_vNodeId; ///< group node indices 
        index_type m_region_id; ///< region index 
};

DREAMPLACE_END_NAMESPACE

#endif
