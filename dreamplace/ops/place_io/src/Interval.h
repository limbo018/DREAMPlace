/*************************************************************************
    > File Name: Interval.h
    > Author: Yibo Lin
    > Mail: yibolin@utexas.edu
    > Created Time: Sun 14 Jun 2015 08:52:22 PM CDT
 ************************************************************************/

#ifndef DREAMPLACE_INTERVAL_H
#define DREAMPLACE_INTERVAL_H

#include <limits>
#include <limbo/math/Math.h>
#include "Util.h"

DREAMPLACE_BEGIN_NAMESPACE

/// an interval class 
/// it consists of a low value and a high value 
template <typename T>
class Interval 
{
    public:
        typedef T coordinate_type;
		typedef typename coordinate_traits<coordinate_type>::manhattan_distance_type manhattan_distance_type;

		///==== constructors ====
		/// default constructor will result in an invalid interval 
		/// but it is more compatible to functions such as encompass()
        Interval(coordinate_type l = std::numeric_limits<coordinate_type>::max(), 
				coordinate_type h = std::numeric_limits<coordinate_type>::min()) 
        {
            set(l, h);
        }

        Interval(Interval const& rhs) 
        {
            copy(rhs);
        }

        Interval& operator=(Interval const& rhs) 
        {
            copy(rhs);
            return *this;
        }

        template <typename IntervalType>
        Interval& operator=(IntervalType const& rhs) 
        {
            copy(rhs);
            return *this;
        }

        ///==== public member functions ==== 
        Interval& set(coordinate_type l, coordinate_type h) 
        {
            m_coords[kLOW] = l;
            m_coords[kHIGH] = h;
            return *this;
        }
        Interval& set(Direction1DType d, coordinate_type v) 
        {
            m_coords[d] = v;
            return *this;
        }
        /// encompass interval with an input value 
        Interval& encompass(coordinate_type v)
        {
            if (m_coords[kLOW] > v) m_coords[kLOW] = v;
            if (m_coords[kHIGH] < v) m_coords[kHIGH] = v;
            return *this;
        }
        /// encompass interval with an input interval 
        Interval& encompass(Interval const& i)
        {
            encompass(i.get(kLOW));
            encompass(i.get(kHIGH));
            return *this;
        }
		/// make sure low is no larger than high 
		Interval& adjust()
		{
			if (m_coords[kLOW] > m_coords[kHIGH])
				swap(m_coords[kLOW], m_coords[kHIGH]);
			return *this;
		}
        coordinate_type get(Direction1DType d) const 
        {
            return m_coords[d];
        }
        coordinate_type low() const 
        {
            return m_coords[kLOW];
        }
        coordinate_type high() const 
        {
            return m_coords[kHIGH];
        }
        manhattan_distance_type delta() const 
        {
            return high() - low();
        }

        ///==== overload operators ====
        bool operator==(Interval const& rhs) const 
        {
            return low() == rhs.low() && high() == rhs.high();
        }

        bool operator!=(Interval const& rhs) const 
        {
            return low() != rhs.low() || high() != rhs.high();
        }

        bool operator<(Interval const& rhs) const 
        {
            if (m_coords[0] != rhs.m_coords[0]) {
                return m_coords[0] < rhs.m_coords[0];
            }
            return m_coords[1] < rhs.m_coords[1];
        }

        bool operator<=(Interval const& rhs) const 
        {
            return !(rhs < *this);
        }

        bool operator>(Interval const& rhs) const 
        {
            return rhs < *this;
        }

        bool operator>=(Interval const& rhs) const 
        {
            return !((*this) < rhs);
        }

        /// comparison function objects 
        struct CompareByLow 
        {
            bool operator()(Interval const& i1, Interval const& i2) const {return i1.low() < i2.low();}
        };
        struct CompareByHigh
        {
            bool operator()(Interval const& i1, Interval const& i2) const {return i1.high() < i2.high();}
        };
        struct CompareByCenter
        {
            bool operator()(Interval const& i1, Interval const& i2) const {return i1.center() < i2.center();}
        };

    protected:
        template <typename IntervalType>
        void copy(IntervalType const& rhs)
        {
            m_coords[0] = rhs.m_coords[0];
            m_coords[1] = rhs.m_coords[1];
        }

        coordinate_type m_coords[2]; ///< low and high values 
};

/// \return center of an interval 
template <typename T>
inline typename coordinate_traits<T>::coordinate_type
center(Interval<T> const& i1)
{
    return (i1.low()+i1.high())/2;
}
/// specialization for integers
template <>
inline coordinate_traits<int>::coordinate_type
center(Interval<int> const& i1)
{
    return (i1.low()+i1.high())>>1;
}
/// \return the intersection of two intervals 
/// false denotes no intersection 
template <typename T>
inline std::pair<Interval<T>, bool>
intersection(Interval<T> const& i1, Interval<T> const& i2, bool consider_touch = true)
{
    typedef typename coordinate_traits<T>::coordinate_type coordinate_type;
    coordinate_type l = std::max(i1.low(), i2.low());
    coordinate_type h = std::min(i1.high(), i2.high());
    bool valid  = (consider_touch)? l <= h : l < h;

    return std::make_pair(Interval<T>(l, h), valid);
}
/// \return true if two intervals have intersection 
template <typename T>
inline bool intersects(Interval<T> const& i1, Interval<T> const& i2, bool consider_touch = true)
{
    typedef typename coordinate_traits<T>::coordinate_type coordinate_type;
    coordinate_type l = std::max(i1.low(), i2.low());
    coordinate_type h = std::min(i1.high(), i2.high());
    return (consider_touch)? l <= h : l < h;
}
/// \return the intersection distance of two intervals 
template <typename T>
inline typename coordinate_traits<T>::manhattan_distance_type 
intersectDistance(Interval<T> const& i1, Interval<T> const& i2)
{
    typedef typename coordinate_traits<T>::coordinate_type coordinate_type;
    typedef typename coordinate_traits<T>::manhattan_distance_type manhattan_distance_type;
    coordinate_type l = std::max(i1.low(), i2.low());
    coordinate_type h = std::min(i1.high(), i2.high());
    return (l < h)? (manhattan_distance_type)h-l : 0;
}

/// \return true if an interval contains a value 
template <typename T>
inline bool contain(Interval<T> const& i1, typename coordinate_traits<T>::coordinate_type v)
{
    return i1.low() <= v && v <= i1.high();
}
/// \return true if an interval contains the other  
template <typename T>
inline bool contain(Interval<T> const& i1, Interval<T> const& i2)
{
    return i1.low() <= i2.low() && i2.high() <= i1.high();
}
/// move an interval 
template <typename T>
inline Interval<T>& move(Interval<T>& i1, typename coordinate_traits<T>::coordinate_type v)
{
	return i1.set(i1.low()+v, i1.high()+v);
}
/// move an interval to a destination 
/// \param v is the final lower value, the delta of interval does not change 
template <typename T>
inline Interval<T>& moveTo(Interval<T>& i1, typename coordinate_traits<T>::coordinate_type v)
{
    typename coordinate_traits<T>::manhattan_distance_type delta = i1.delta();
	return i1.set(v, v+delta);
}
/// \return distance 
template <typename T>
inline typename coordinate_traits<T>::manhattan_distance_type
distance(Interval<T> const& i1, typename coordinate_traits<T>::coordinate_type v)
{
	if (v < i1.low()) return i1.low()-v;
	else if (v > i1.high()) return v-i1.high();
	else return 0;
}
/// \return the distance between two intervals 
/// \return 0 if two intervals have intersection
template <typename T>
inline typename coordinate_traits<T>::manhattan_distance_type
distance(Interval<T> const& i1, Interval<T> const& i2)
{
    typedef typename coordinate_traits<T>::coordinate_type coordinate_type;
    coordinate_type l = std::max(i1.low(), i2.low());
    coordinate_type h = std::min(i1.high(), i2.high());
    return (l > h)? l-h : 0; // different from computing intersection
}

/// \return true if the value is on the boundary of the interval 
template <typename T>
inline bool onBoundary(Interval<T> const& t, typename coordinate_traits<T>::coordinate_type v)
{
    return t.low() == v || t.high() == v;
}

DREAMPLACE_END_NAMESPACE

#endif
