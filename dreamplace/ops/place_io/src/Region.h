/**
 * @file   Region.h
 * @author Yibo Lin
 * @mail   yibolin@pku.edu.cn
 * @date   Dec 2019
 */

#ifndef DREAMPLACE_REGION_H
#define DREAMPLACE_REGION_H

#include <vector>
#include "Object.h"
#include "Box.h"
#include "Enums.h"

DREAMPLACE_BEGIN_NAMESPACE

/// class Region denotes a region like fence or guide 
class Region : public Object
{
	public:
        typedef Object base_type;
        typedef base_type::coordinate_type coordinate_type;
        typedef Box<coordinate_type> box_type;
        typedef coordinate_traits<coordinate_type>::index_type index_type;
        typedef Point<coordinate_type> point_type;

        /// default constructor 
        Region();
        /// copy constructor
        Region(Region const& rhs);
        /// assignment
        Region& operator=(Region const& rhs);

        /// member functions 
        std::vector<box_type> const& boxes() const {return m_vBox;}
        std::vector<box_type>& boxes() {return m_vBox;}
        Region& addBox(box_type const& box) {m_vBox.push_back(box); return *this;}
        Region& setBox(index_type i, box_type const& box) {m_vBox.at(i) = box; return *this;}

        std::string const& name() const {return m_name;}
        Region& setName(std::string const& name) {m_name = name; return *this;}

        RegionTypeEnum::RegionEnumType type() const {return (RegionTypeEnum::RegionEnumType)m_type;}
        Region& setType(RegionTypeEnum::RegionEnumType t) {m_type = t; return *this;}
        Region& setType(RegionType const& t) {return setType(t.value());}

    protected:
        void copy(Region const& rhs);

        std::vector<box_type> m_vBox; ///< rectangles for the region 
        std::string m_name; ///< region name 
        char m_type; ///< region type
};

DREAMPLACE_END_NAMESPACE

#endif
