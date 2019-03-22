/*************************************************************************
    > File Name: Node.cpp
    > Author: Yibo Lin
    > Mail: yibolin@utexas.edu
    > Created Time: Tue 22 Mar 2016 07:27:48 PM CDT
 ************************************************************************/

#include "Node.h"

DREAMPLACE_BEGIN_NAMESPACE

Node::Node() 
    : Node::base_type1()
    , Node::base_type2()
    , m_initPos()
    , m_status(PlaceStatusEnum::UNKNOWN)
    , m_multiRowAttr(MultiRowAttrEnum::SINGLE_ROW)
    , m_orient(OrientEnum::UNKNOWN)
    , m_vPinId()
{
}
Node::Node(Node const& rhs)
    : Node::base_type1(rhs)
    , Node::base_type2(rhs)
{
    copy(rhs);
}
Node& Node::operator=(Node const& rhs)
{
    if (this != &rhs)
    {
        this->base_type1::operator=(rhs);
        this->base_type2::operator=(rhs);
        copy(rhs);
    }
    return *this;
}
void Node::copy(Node const& rhs)
{
    m_initPos = rhs.m_initPos;
    m_status = rhs.m_status;
    m_multiRowAttr = rhs.m_multiRowAttr;
    m_orient = rhs.m_orient;
    m_vPinId = rhs.m_vPinId;
}

NodeProperty::NodeProperty() 
    : m_name("")
    , m_macroId(std::numeric_limits<NodeProperty::index_type>::max())
{
}
NodeProperty::NodeProperty(NodeProperty const& rhs)
{
    copy(rhs);
}
NodeProperty& NodeProperty::operator=(NodeProperty const& rhs)
{
    if (this != &rhs)
        copy(rhs);
    return *this;
}
void NodeProperty::copy(NodeProperty const& rhs)
{
    m_name = rhs.m_name;
    m_macroId = rhs.m_macroId;
}

DREAMPLACE_END_NAMESPACE
