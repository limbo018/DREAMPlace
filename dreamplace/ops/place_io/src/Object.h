/*************************************************************************
    > File Name: Object.h
    > Author: Yibo Lin
    > Mail: yibolin@utexas.edu
    > Created Time: Sun 14 Jun 2015 04:07:46 PM CDT
 ************************************************************************/

#ifndef GPF_OBJECT_H
#define GPF_OBJECT_H

#include <limits>
#include "Util.h"

GPF_BEGIN_NAMESPACE

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

	protected:
		index_type m_id; ///< index of object 
};

GPF_END_NAMESPACE

#endif
