/*************************************************************************
    > File Name: Net.cpp
    > Author: Yibo Lin
    > Mail: yibolin@utexas.edu
    > Created Time: Wed Mar 23 23:02:28 2016
 ************************************************************************/

#include "Net.h"

DREAMPLACE_BEGIN_NAMESPACE

Net::Net() 
    : Net::base_type()
    , m_bbox()
    , m_weight(1)
    , m_vPinId()
{
}
Net::Net(Net const& rhs)
    : Net::base_type(rhs)
{
    copy(rhs);
}
Net& Net::operator=(Net const& rhs)
{
    if (this != &rhs)
    {
        this->base_type::operator=(rhs);
        copy(rhs);
    }
    return *this;
}
void Net::copy(Net const& rhs)
{
    m_bbox = rhs.m_bbox;
    m_weight = rhs.m_weight; 
    m_vPinId = rhs.m_vPinId;
}

NetProperty::NetProperty() 
    : m_name("")
{
}
NetProperty::NetProperty(NetProperty const& rhs)
{
    copy(rhs);
}
NetProperty& NetProperty::operator=(NetProperty const& rhs)
{
    if (this != &rhs)
        copy(rhs);
    return *this;
}
void NetProperty::copy(NetProperty const& rhs)
{
    m_name = rhs.m_name;
}

DREAMPLACE_END_NAMESPACE
