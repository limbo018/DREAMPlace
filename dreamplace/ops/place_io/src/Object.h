/*************************************************************************
    > File Name: Object.h
    > Author: Yibo Lin
    > Mail: yibolin@utexas.edu
    > Created Time: Sun 14 Jun 2015 04:07:46 PM CDT
 ************************************************************************/

#ifndef DREAMPLACE_OBJECT_H
#define DREAMPLACE_OBJECT_H

#include <limits>
#include "Util.h"

DREAMPLACE_BEGIN_NAMESPACE

/// base class for all objects 
class Object 
{
	public:
		typedef int coordinate_type;
        typedef coordinate_traits<coordinate_type>::index_type index_type;

        /// default constructor 
		Object() : m_id(std::numeric_limits<index_type>::max()) {}
        /// copy constructor
        Object(Object const& rhs) : m_id(rhs.m_id) {}
        /// assignment 
        Object& operator=(Object const& rhs)
        {
            m_id = rhs.m_id;
            return *this;
        }
        /// destructor 
        ~Object() {}

		index_type id() const {return m_id;}
		void setId(index_type i) {m_id = i;}

        std::string toString() const {return limbo::to_string(m_id);}

	protected:
		index_type m_id; ///< index of object 
};

DREAMPLACE_END_NAMESPACE

#endif
