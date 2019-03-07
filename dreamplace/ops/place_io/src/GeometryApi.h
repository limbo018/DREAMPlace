/*************************************************************************
    > File Name: GeometryApi.h
    > Author: Yibo Lin
    > Mail: yibolin@utexas.edu
    > Created Time: Tue Jun 30 16:57:11 2015
 ************************************************************************/

#ifndef DREAMPLACE_GEOMETRYAPI_H
#define DREAMPLACE_GEOMETRYAPI_H

#include <limits>
#include <limbo/geometry/Geometry.h>
#include <boost/geometry.hpp>
#include "Util.h"
#include "Point.h"
#include "Box.h"

/// limbo.geometry api
/// with this api, algorithms in limbo.geometry are available 
namespace limbo { namespace geometry {

/// \brief specialization for boost::polygon::point_data
template <typename T>
struct point_traits<DREAMPLACE_NAMESPACE::Point<T> >
{
	typedef T coordinate_type;
	typedef DREAMPLACE_NAMESPACE::Point<T> point_type;

	static coordinate_type get(const point_type& point, orientation_2d const& orient) 
	{
		if (orient == HORIZONTAL) return point.x();
		else if (orient == VERTICAL) return point.y();
		else {assert(0); return 0;}
	}
	static void set(point_type& point, orientation_2d const& orient, coordinate_type const& value) 
	{
		if (orient == HORIZONTAL) point.set(DREAMPLACE_NAMESPACE::kX, value);
		else if (orient == VERTICAL) point.set(DREAMPLACE_NAMESPACE::kY, value);
		else dreamplaceAssertMsg(0, "unknown orient");
	}
	static point_type construct(coordinate_type const& x, coordinate_type const& y) 
	{
		return point_type(x, y);
	}
};

/// \brief specialization for boost::polygon::rectangle_data
template <typename T>
struct rectangle_traits<DREAMPLACE_NAMESPACE::Box<T> >
{
	typedef T coordinate_type;
	typedef DREAMPLACE_NAMESPACE::Box<T> rectangle_type;

	static coordinate_type get(const rectangle_type& rect, direction_2d const& dir) 
	{
		switch (dir)
		{
			case LEFT: return rect.xl();
			case BOTTOM: return rect.yl();
			case RIGHT: return rect.xh();
			case TOP: return rect.yh();
			default: dreamplaceAssertMsg(0, "unknown orient");
		}
        return std::numeric_limits<coordinate_type>::max();
	}
	static void set(rectangle_type& rect, direction_2d const& dir, coordinate_type const& value) 
	{
		switch (dir)
		{
            case LEFT: rect.set(DREAMPLACE_NAMESPACE::kXLOW, value); break;
			case BOTTOM: rect.set(DREAMPLACE_NAMESPACE::kYLOW, value); break;
			case RIGHT: rect.set(DREAMPLACE_NAMESPACE::kXHIGH, value); break;
			case TOP: rect.set(DREAMPLACE_NAMESPACE::kYHIGH, value); break;
			default: dreamplaceAssertMsg(0, "unknown orient");
		}
	}
	static rectangle_type construct(coordinate_type const& xl, coordinate_type const& yl, 
			coordinate_type const& xh, coordinate_type const& yh) 
	{
		return rectangle_type(xl, yl, xh, yh); 
	}
};

}}// namespace limbo // namespace geometry


#endif
