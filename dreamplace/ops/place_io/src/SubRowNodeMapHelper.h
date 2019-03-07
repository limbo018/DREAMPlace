/*************************************************************************
    > File Name: SubRowNodeMapHelper.h
    > Author: Yibo Lin
    > Mail: yibolin@utexas.edu
    > Created Time: Sun 14 Feb 2016 09:24:33 PM CST
 ************************************************************************/

#ifndef DREAMPLACE_SUBROWNODEMAPHELPER_H
#define DREAMPLACE_SUBROWNODEMAPHELPER_H

#include <boost/geometry.hpp>
#include <boost/geometry/index/rtree.hpp>
#include "Interval.h"
#include "IntervalHashMap.h"

/// API for Boost.Geometry 
namespace boost { namespace geometry { namespace index {

template <typename CoordinateType, typename T>
struct indexable< std::pair<boost::geometry::model::box<boost::geometry::model::point<CoordinateType, 1, boost::geometry::cs::cartesian> >, T> >
{
    typedef std::pair<boost::geometry::model::box<boost::geometry::model::point<CoordinateType, 1, boost::geometry::cs::cartesian> >, T> V;

    typedef boost::geometry::model::point<CoordinateType, 1, boost::geometry::cs::cartesian> point_type;
    typedef boost::geometry::model::box<point_type> const& result_type;

    result_type operator()(V const& v) const { return v.first; }
};

template <>
struct indexable<DREAMPLACE_NAMESPACE::Node const*>
{
    typedef DREAMPLACE_NAMESPACE::Node const* value_type;
    typedef DREAMPLACE_NAMESPACE::Interval<GPF_NAMESPACE::Node::coordinate_type> const& result_type;

    result_type operator()(value_type const& v) const {return v->get(DREAMPLACE_NAMESPACE::kX);}
};

template <>
struct indexable<DREAMPLACE_NAMESPACE::NodeMapElement>
{
    typedef DREAMPLACE_NAMESPACE::NodeMapElement value_type;
    typedef DREAMPLACE_NAMESPACE::NodeMapElement::interval_type const& result_type;

    result_type operator()(value_type const& v) const {return v.inv;}
};

}}} // namespace boost // namespace geometry // namespace index

namespace boost { namespace geometry { namespace traits {

//////// for intervals ////////
template <typename CoordinateType>
struct tag<DREAMPLACE_NAMESPACE::Interval<CoordinateType> > 
{
    typedef box_tag type;
};

template <typename CoordinateType>
struct point_type<DREAMPLACE_NAMESPACE::Interval<CoordinateType> >
{
    // a 1D point 
    typedef boost::geometry::model::point<CoordinateType, 1, boost::geometry::cs::cartesian> type;
};

template <typename CoordinateType, std::size_t Dimension>
struct indexed_access
<
    DREAMPLACE_NAMESPACE::Interval<CoordinateType>,
    min_corner, Dimension
> 
{
    typedef CoordinateType coordinate_type;

    static inline coordinate_type get(DREAMPLACE_NAMESPACE::Interval<coordinate_type> const& inv)
    {
        return inv.low();
    }
    static inline void set(DREAMPLACE_NAMESPACE::Interval<coordinate_type>& inv, coordinate_type const& value)
    {
        inv.set(DREAMPLACE_NAMESPACE::kLOW, value);
    }
};


template <typename CoordinateType, std::size_t Dimension>
struct indexed_access
<
    DREAMPLACE_NAMESPACE::Interval<CoordinateType>,
    max_corner, Dimension
> 
{
    typedef CoordinateType coordinate_type;

    static inline coordinate_type get(DREAMPLACE_NAMESPACE::Interval<coordinate_type> const& inv)
    {
        return inv.high();
    }
    static inline void set(DREAMPLACE_NAMESPACE::Interval<coordinate_type>& inv, coordinate_type const& value)
    {
        inv.set(DREAMPLACE_NAMESPACE::kHIGH, value);
    }
};

}}} // namespace boost // namespace geometry // namespace traits

/// API for IntervalHashMap
DREAMPLACE_BEGIN_NAMESPACE

template <>
struct IntervalHashMapTraits<NodeMapElement> 
{
    typedef NodeMapElement value_type; 
    typedef value_type::coordinate_type coordinate_type; 
    typedef unsigned index_type; 

    static coordinate_type low(value_type const& v) {return v.inv.low(); } 
    static coordinate_type high(value_type const& v) {return v.inv.high();}
    static bool equal(value_type const& v1, value_type const& v2) {return v1.nodeId == v2.nodeId;}

    /// compare object for sorting 
    struct CompareByLow
    {
        bool operator()(value_type const& v1, value_type const& v2) const {return low(v1) < low(v2) || (low(v1) == low(v2) && v1.nodeId < v2.nodeId);}
        bool operator()(coordinate_type v1, value_type const& v2) const {return v1 < low(v2);}
        bool operator()(value_type const& v1, coordinate_type v2) const {return low(v1) < v2;}
    }; 
};

DREAMPLACE_END_NAMESPACE


#endif
