/**
 * @file   Region.cpp
 * @author Yibo Lin
 * @date   Dec 2019
 */
#include "Region.h"

DREAMPLACE_BEGIN_NAMESPACE

Region::Region() 
    : Region::base_type()
    , m_vBox()
    , m_name("")
    , m_type(RegionTypeEnum::UNKNOWN)
{
}
Region::Region(Region const& rhs)
    : Region::base_type(rhs)
{
    copy(rhs);
}
Region& Region::operator=(Region const& rhs)
{
    if (this != &rhs)
    {
        this->base_type::operator=(rhs);
        copy(rhs);
    }
    return *this;
}
void Region::copy(Region const& rhs)
{
    m_vBox = rhs.m_vBox;
    m_name = rhs.m_name;
    m_type = rhs.m_type;
}

DREAMPLACE_END_NAMESPACE
