/*************************************************************************
    > File Name: Site.h
    > Author: Yibo Lin
    > Mail: yibolin@utexas.edu
    > Created Time: Wed Jun 17 22:09:40 2015
 ************************************************************************/

#ifndef DREAMPLACE_SITE_H
#define DREAMPLACE_SITE_H

#include "Object.h" 

DREAMPLACE_BEGIN_NAMESPACE

class Site : public Object
{
    public:
        typedef Object base_type;
        typedef base_type::coordinate_type coordinate_type;

        /// default constructor 
        Site();
        /// copy constructor
        Site(Site const& rhs);
        /// assignment
        Site& operator=(Site const& rhs);

        /// member functions 
        std::string const& name() const {return m_name;}
        Site& setName(std::string const& s) {m_name = s; return *this;}

        std::string const& className() const {return m_className;}
        Site& setClassName(std::string const& s) {m_className = s; return *this;}

        unsigned char symmetry() const {return m_symmetry;}
        Site& setSymmetry(unsigned char s) {m_symmetry = s; return *this;}

        coordinate_type size(Direction1DType d) const {return m_size[d];}
        Site& setSize(Direction1DType d, coordinate_type v) {m_size[d] = v; return *this;}
        coordinate_type width() const {return m_size[kX];}
        coordinate_type height() const {return m_size[kY];}

    protected:
        void copy(Site const& rhs);

        std::string m_name; ///< site name 
        std::string m_className; ///< class name 
        unsigned char m_symmetry; ///< 3-bit: x, y, R90
        coordinate_type m_size[2]; ///< width and height 
};

inline Site::Site() 
    : Site::base_type()
    , m_name("")
    , m_className("")
    , m_symmetry(std::numeric_limits<unsigned char>::max())
{
    m_size[kX] = std::numeric_limits<Row::coordinate_type>::max();
    m_size[kY] = std::numeric_limits<Row::coordinate_type>::max();
}
inline Site::Site(Site const& rhs)
    : Site::base_type(rhs)
{
    copy(rhs);
}
inline Site& Site::operator=(Site const& rhs)
{
    if (this != &rhs)
    {
        this->base_type::operator=(rhs);
        copy(rhs);
    }
    return *this;
}
inline void Site::copy(Site const& rhs)
{
    m_name = rhs.m_name;
    m_className = rhs.m_className;
    m_symmetry = rhs.m_symmetry;
    m_size[kX] = rhs.m_size[kX];
    m_size[kY] = rhs.m_size[kY];
}


DREAMPLACE_END_NAMESPACE

#endif
