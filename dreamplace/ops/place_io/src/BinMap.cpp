/*************************************************************************
    > File Name: BinMap.cpp
    > Author: Yibo Lin
    > Mail: yibolin@utexas.edu
    > Created Time: Tue 23 Jun 2015 08:57:40 PM CDT
 ************************************************************************/

#include "BinMap.h"

DREAMPLACE_BEGIN_NAMESPACE

BinMap& BinMap::set(BinType type, BinMap::index_type xNum, BinMap::index_type yNum)
{
    m_dimension[kX] = xNum;
    m_dimension[kY] = yNum;
    m_vBin.resize(xNum*yNum);

    // only set bin indices 
    // bin coordinates are set outside since layout information is not known 
    index_type id2DX = 0;
    index_type id2DY = 0;
    for (index_type id1D = 0, id1De = m_vBin.size(); id1D < id1De; ++id1D)
    {
        Bin& bin = m_vBin.at(id1D);
        bin.setType(type);
        bin.setIndex1D(id1D);
        bin.setIndex2D(kX, id2DX)
            .setIndex2D(kY, id2DY);

        id2DX += 1;
        if (id2DX == m_dimension[kX])
        {
            id2DX = 0;
            id2DY += 1;
        }
    }

    return *this;
}

BinMap& BinMap::resetBinDemand()
{
    for (BinMap1DIterator it = begin1D(), ite = end1D(); it != ite; ++it)
    {
        it->setDemand(0);
        it->setPinDemand(0);
    }
    return *this;
}

DREAMPLACE_END_NAMESPACE
