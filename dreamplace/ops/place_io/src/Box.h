/*************************************************************************
    > File Name: Box.h
    > Author: Yibo Lin
    > Mail: yibolin@utexas.edu
    > Created Time: Sun 14 Jun 2015 04:05:16 PM CDT
 ************************************************************************/

#ifndef DREAMPLACE_BOX_H
#define DREAMPLACE_BOX_H

#include "Point.h"
#include "Interval.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
class Box 
{
	public:
		typedef T coordinate_type;
		typedef typename coordinate_traits<coordinate_type>::manhattan_distance_type manhattan_distance_type;
		typedef typename coordinate_traits<coordinate_type>::area_type area_type;
        typedef Interval<coordinate_type> interval_type;
		typedef Point<coordinate_type> point_type;

        ///==== constructors ====
		/// default constructor will result in an invalid box 
		/// but it is more compatible with functions such as encompass()
        Box()
        {
            set(
                    std::numeric_limits<coordinate_type>::max(), 
                    std::numeric_limits<coordinate_type>::max(), 
                    std::numeric_limits<coordinate_type>::min(), 
                    std::numeric_limits<coordinate_type>::min() 
                    );
        }
        Box(coordinate_type xl, 
                coordinate_type yl, 
				coordinate_type xh, 
                coordinate_type yh)
        {
            set(xl, yl, xh, yh);
        }
		Box(interval_type const& ix, interval_type const& iy)
		{
			set(kX, ix);
			set(kY, iy);
		}
        Box(Box const& rhs)
        {
            copy(rhs);
        }
        Box& operator=(Box const& rhs)
        {
            copy(rhs);
            return *this;
        }
        template <typename BoxType>
        explicit Box(BoxType const& rhs)
        {
            copy(rhs);
        }
        template <typename BoxType>
        Box& operator=(BoxType const& rhs)
        {
            copy(rhs);
            return *this;
        }

        ///==== member functions ====
        /// set to uninitialized status 
        Box& unset()
        {
            set(std::numeric_limits<coordinate_type>::max(), 
                    std::numeric_limits<coordinate_type>::max(), 
                    std::numeric_limits<coordinate_type>::min(), 
                    std::numeric_limits<coordinate_type>::min());
            return *this;
        }
        Box& set(coordinate_type xl, coordinate_type yl, coordinate_type xh, coordinate_type yh)
        {
            m_range[kX].set(xl, xh);
            m_range[kY].set(yl, yh);
            return *this;
        }
        Box& set(Direction1DType d, interval_type const& range)
        {
            m_range[d] = range;
            return *this;
        }
        Box& set(Direction1DType xy, Direction1DType lh, coordinate_type v)
        {
            m_range[xy].set(lh, v);
            return *this;
        }
        Box& set(Direction2DType d, coordinate_type v)
        {
            return set(getXY(d), getLH(d), v);
        }
		/// encompass box with an input point 
		Box& encompass(point_type const& p)
		{
			m_range[kX].encompass(p.get(kX));
			m_range[kY].encompass(p.get(kY));
			return *this;
		}
        /// encompass box with an input box 
        Box& encompass(Box const& b)
        {
            m_range[kX].encompass(b.get(kX));
            m_range[kY].encompass(b.get(kY));
            return *this;
        }
		/// make sure a valid box 
		Box& adjust()
		{
			m_range[kX].adjust();
			m_range[kY].adjust();
			return *this;
		}
        interval_type const& get(Direction1DType xy) const
        {
            return m_range[xy];
        }
        coordinate_type get(Direction1DType xy, Direction1DType lh) const 
        {
            return m_range[xy].get(lh);
        }
        coordinate_type get(Direction2DType d) const
        {
            return get(getXY(d), getLH(d));
        }
        coordinate_type xl() const 
        {
            return get(kX, kLOW);
        }
        coordinate_type yl() const 
        {
            return get(kY, kLOW);
        }
        coordinate_type xh() const 
        {
            return get(kX, kHIGH);
        }
        coordinate_type yh() const 
        {
            return get(kY, kHIGH);
        }
        manhattan_distance_type delta(Direction1DType d) const 
        {
            return m_range[d].delta();
        }
        manhattan_distance_type width() const 
        {
            return m_range[kX].delta();
        }
        manhattan_distance_type height() const 
        {
            return m_range[kY].delta();
        }
        /// \return area
        area_type area() const 
        {
            // be careful about overflow 
            return (area_type)width()*(area_type)height();
        }

        ///==== overload operators ====
        template <typename BoxType>
        bool operator==(const BoxType& rhs) const
        {
            return m_range[kX] == rhs.m_range[kX] && m_range[kY] == rhs.m_range[kY];
        }
        template <typename BoxType>
        bool operator!=(const BoxType& rhs) const { return !((*this) == rhs); }

        std::string toString() const {return "(" + limbo::to_string(xl()) + ", " + limbo::to_string(yl()) + ", " + limbo::to_string(xh()) + ", " + limbo::to_string(yh()) + ")";}
	protected:
        template <typename BoxType>
        void copy(BoxType const& rhs)
        {
            m_range[kX] = rhs.m_range[kX];
            m_range[kY] = rhs.m_range[kY];
        }

		interval_type m_range[2]; ///< two ranges define a box  
};

/// \return lower left point 
template <typename T>
inline typename Box<T>::point_type 
ll(Box<T> const& b)
{
	return typename Box<T>::point_type(b.xl(), b.yl());
}
/// \return lower right point 
template <typename T>
inline typename Box<T>::point_type 
lr(Box<T> const& b)
{
	return typename Box<T>::point_type(b.xh(), b.yl());
}
/// \return upper left point 
template <typename T>
inline typename Box<T>::point_type 
ul(Box<T> const& b)
{
	return typename Box<T>::point_type(b.xl(), b.yh());
}
/// \return upper right point 
template <typename T>
inline typename Box<T>::point_type 
ur(Box<T> const& b)
{
	return typename Box<T>::point_type(b.xh(), b.yh());
}
/// \return center coordinate 
template <typename T>
inline typename coordinate_traits<T>::coordinate_type
center(Box<T> const& b, Direction1DType d)
{
	return center(b.get(d));
}
/// \return center point 
template <typename T>
inline typename Box<T>::point_type 
center(Box<T> const& b)
{
	return typename Box<T>::point_type(center(b, kX), center(b, kY));
}
/// \return half perimeter
template <typename T>
inline typename coordinate_traits<T>::manhattan_distance_type
halfPerimeter(Box<T> const& b)
{
	return b.delta(kX)+b.delta(kY);
}
/// \return perimeter
template <typename T>
inline typename coordinate_traits<T>::manhattan_distance_type
perimeter(Box<T> const& b)
{
	return halfPerimeter(b)*2;
}
/// specialization for integers 
template <>
inline coordinate_traits<int>::manhattan_distance_type
perimeter(Box<int> const& b)
{
	return halfPerimeter(b)<<1;
}
/// \return true if a point is inside a box 
template <typename T>
inline bool contain(Box<T> const& b, Point<T> const& p)
{
    return contain(b.get(kX), p.x()) && contain(b.get(kY), p.y());
}
/// \return true if a box contains the other 
template <typename T>
inline bool contain(Box<T> const& b1, Box<T> const& b2)
{
    return contain(b1.get(kX), b2.get(kX)) && contain(b1.get(kY), b2.get(kY));
}
/// move a box in x or y direction 
/// \param v denotes displacement 
template <typename T>
inline Box<T>& move(Box<T>& b, Direction1DType d, typename coordinate_traits<T>::coordinate_type v)
{
	typename Box<T>::interval_type ivl = b.get(d);
	move(ivl, v);
	return b.set(d, ivl);
}
/// move a box in two directions 
/// \param vx denotes displacement in x 
/// \param vy denotes displacement in y 
template <typename T>
inline Box<T>& move(Box<T>& b, typename coordinate_traits<T>::coordinate_type vx, typename coordinate_traits<T>::coordinate_type vy)
{
	return move(move(b, kX, vx), kY, vy);
}
/// move a box in two directions 
/// \param p.x() denotes destination in x 
/// \param p.y() denotes destination in y 
template <typename T>
inline Box<T>& move(Box<T>& b, Point<T> const& p)
{
    return move(b, p.x(), p.y());
}
/// move a box to a position in x or y direction
/// \param v denotes destination of left lower corner 
template <typename T>
inline Box<T>& moveTo(Box<T>& b, Direction1DType d, typename coordinate_traits<T>::coordinate_type v)
{
	typename Box<T>::interval_type ivl = b.get(d);
    moveTo(ivl, v);
    return b.set(d, ivl);
}
/// move a box in two directions 
/// \param vx denotes destination in x 
/// \param vy denotes destination in y 
template <typename T>
inline Box<T>& moveTo(Box<T>& b, typename coordinate_traits<T>::coordinate_type vx, typename coordinate_traits<T>::coordinate_type vy)
{
	return moveTo(moveTo(b, kX, vx), kY, vy);
}
/// move a box in two directions 
/// \param p.x() denotes destination in x 
/// \param p.y() denotes destination in y 
template <typename T>
inline Box<T>& moveTo(Box<T>& b, Point<T> const& p)
{
    return moveTo(b, p.x(), p.y());
}
/// \return manhattan distance in a direction between a box and a point 
template <typename T>
inline typename coordinate_traits<T>::manhattan_distance_type 
manhattanDistance(Box<T> const& b, typename Box<T>::point_type const& p, Direction1DType d)
{
	return distance(b.get(d), p.get(d));
}
/// \return manhattan distance between a box and a point 
template <typename T>
inline typename coordinate_traits<T>::manhattan_distance_type 
manhattanDistance(Box<T> const& b, typename Box<T>::point_type const& p)
{
	return manhattanDistance(b, p, kX)+manhattanDistance(b, p, kY);
}
/// \return square distance between a box and a point 
template <typename T>
inline typename coordinate_traits<T>::euclidean_distance_type
squareDistance(Box<T> const& b, typename Box<T>::point_type const& p)
{
    typedef typename coordinate_traits<T>::euclidean_distance_type euclidean_distance_type;
	return (euclidean_distance_type)(
			pow(manhattanDistance(b, p, kX), 2)
			+ pow(manhattanDistance(b, p, kY), 2)
		);
}
/// \return euclidean distance between a box and a point 
template <typename T>
inline typename coordinate_traits<T>::euclidean_distance_type
euclideanDistance(Box<T> const& b, typename Box<T>::point_type const& p)
{
	return sqrt(squareDistance(b, p));
}
/// \return manhattan distance in a direction between two boxes  
template <typename T>
inline typename coordinate_traits<T>::manhattan_distance_type 
manhattanDistance(Box<T> const& b1, Box<T> const& b2, Direction1DType d)
{
	return distance(b1.get(d), b2.get(d));
}
/// \return manhattan distance between two boxes  
template <typename T>
inline typename coordinate_traits<T>::manhattan_distance_type 
manhattanDistance(Box<T> const& b1, Box<T> const& b2)
{
	return manhattanDistance(b1, b2, kX) + manhattanDistance(b1, b2, kY);
}
/// \return square distance between two boxes  
template <typename T>
inline typename coordinate_traits<T>::euclidean_distance_type
squareDistance(Box<T> const& b1, Box<T> const& b2)
{
    typedef typename coordinate_traits<T>::euclidean_distance_type euclidean_distance_type;
	return (euclidean_distance_type)(
			pow(manhattanDistance(b1, b2, kX), 2)
			+ pow(manhattanDistance(b1, b2, kY), 2)
		);
}
/// \return euclidean distance between two boxes  
template <typename T>
inline typename coordinate_traits<T>::euclidean_distance_type
euclideanDistance(Box<T> const& b1, Box<T> const& b2)
{
	return sqrt(squareDistance(b1, b2));
}
/// \return the intersection of two boxes 
template <typename T>
inline std::pair<Box<T>, bool>
intersection(Box<T> const& b1, Box<T> const& b2, bool consider_touch = true)
{
	std::pair<Interval<T>, bool> ivl[2] = {
		intersection(b1.get(kX), b2.get(kX), consider_touch), 
		intersection(b1.get(kY), b2.get(kY), consider_touch) 
	};
	return std::make_pair(Box<T>(ivl[kX].first, ivl[kY].first), ivl[kX].second && ivl[kY].second);
}
/// \return true if two boxes have intersection 
template <typename T>
inline bool intersects(Box<T> const& b1, Box<T> const& b2, bool consider_touch = true)
{
	return intersects(b1.get(kX), b2.get(kX), consider_touch) 
        && intersects(b1.get(kY), b2.get(kY), consider_touch);
}
/// \return the intersection area of two boxes 
template <typename T>
inline typename coordinate_traits<T>::area_type
intersectArea(Box<T> const& b1, Box<T> const& b2)
{
    typedef typename coordinate_traits<T>::manhattan_distance_type manhattan_distance_type;
    typedef typename coordinate_traits<T>::area_type area_type;
    manhattan_distance_type dist[2] = {
		intersectDistance(b1.get(kX), b2.get(kX)), 
		intersectDistance(b1.get(kY), b2.get(kY)) 
    };
    return (area_type)dist[kX]*dist[kY];
}

/// \return true if a point is on boundary of the box 
template <typename T>
inline bool onBoundary(Box<T> const& b, Point<T> const& p)
{
    return (onBoundary(b.get(kX), p.x()) && contain(b.get(kY), p.y()))
        || (onBoundary(b.get(kY), p.y()) && contain(b.get(kX), p.x()));
}

DREAMPLACE_END_NAMESPACE

#endif
