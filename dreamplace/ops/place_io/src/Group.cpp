/**
 * @file   Group.cpp
 * @author Yibo Lin
 * @date   Dec 2019
 */
#include "Group.h"

DREAMPLACE_BEGIN_NAMESPACE

Group::Group() 
    : Group::base_type()
    , m_name("")
    , m_vNodeName()
    , m_vNodeId()
    , m_region_id(std::numeric_limits<Group::index_type>::max())
{
}
Group::Group(Group const& rhs)
    : Group::base_type(rhs)
{
    copy(rhs);
}
Group& Group::operator=(Group const& rhs)
{
    if (this != &rhs)
    {
        this->base_type::operator=(rhs);
        copy(rhs);
    }
    return *this;
}
void Group::copy(Group const& rhs)
{
    m_name = rhs.m_name;
    m_vNodeName = rhs.m_vNodeName; 
    m_vNodeId = rhs.m_vNodeId;
    m_region_id = rhs.m_region_id;
}


DREAMPLACE_END_NAMESPACE
