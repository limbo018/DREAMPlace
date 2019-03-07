/*************************************************************************
    > File Name: SubRowNodeMap.cpp
    > Author: Yibo Lin
    > Mail: yibolin@utexas.edu
    > Created Time: Sun 14 Feb 2016 11:16:56 AM CST
 ************************************************************************/

#include "SubRowNodeMap.h"
#include "AlgoDB.h"
#include "Iterators.h"
#include <algorithm>

DREAMPLACE_BEGIN_NAMESPACE

SubRowNodeMap::SubRowNodeMap(AlgoDB const* db)
{
    if (db) // only initialize if it is set 
        set(db);
}

SubRowNodeMap::SubRowNodeMap(SubRowNodeMap const& rhs)
{
    copy(rhs);
}

SubRowNodeMap& SubRowNodeMap::operator=(SubRowNodeMap const& rhs)
{
    if (this != &rhs)
        copy(rhs);
    return *this;
}

SubRowNodeMap::~SubRowNodeMap()
{
}

SubRowNodeMap& SubRowNodeMap::set(AlgoDB const* db)
{
    m_algoDB = db;
    m_vMap.clear();
    m_vMap.resize(m_algoDB->subRowMap().numSubRows());
    m_avgMovableWidth = m_algoDB->placeDB().avgMovableNodeWidth(); 

    return *this;
}

void SubRowNodeMap::copy(SubRowNodeMap const& rhs)
{
    m_algoDB = rhs.m_algoDB;
    m_vMap = rhs.m_vMap;
    m_avgMovableWidth = rhs.m_avgMovableWidth; 
}

void SubRowNodeMap::setSubRowNodes(bool moveDummyFixedCellOnly) 
{
    // reset maps 
    for (std::vector<map_type>::iterator it = m_vMap.begin(), ite = m_vMap.end(); it != ite; ++it)
        it->clear();
    // distribute nodes 
    coordinate_type rowHeight = m_algoDB->placeDB().rowHeight();
    std::vector<std::vector<map_element_type> > mNode (m_algoDB->subRowMap().numSubRows());
    for (MovableNodeConstIterator it = m_algoDB->placeDB().movableNodeBegin(); it.inRange(); ++it)
    {
        Node const& node = *it;
        if (moveDummyFixedCellOnly && node.status() != PlaceStatusEnum::DUMMY_FIXED)
            continue; 
        // this for loop only works for movable cells, because fixed instances may occupy partial row  
        coordinate_type y = node.yl();
        while (y < node.yh())
        {
            index_type idx = m_algoDB->getSubRowIndexSafe(node.xl(), y+1); // compute sub row index 
            mNode[idx].push_back(map_element_type(node.id(), node.initPos().x(), node.get(kX)));

            y += rowHeight; // next row 
        }
    }
    for (index_type i = 0, ie = m_vMap.size(); i < ie; ++i)
    {
#ifdef USE_RTREE
        map_type(mNode[i].begin(), mNode[i].end()).swap(m_vMap.at(i));
#elif defined(USE_INTERVALHASHMAP)
        map_type(m_algoDB->getSubRow(i).xl(), m_algoDB->getSubRow(i).xh(), m_avgMovableWidth, mNode[i].begin(), mNode[i].end()).swap(m_vMap.at(i)); 
#endif
    }
}

std::vector<SubRowNodeMap::index_type> SubRowNodeMap::queryRange(Box<SubRowNodeMap::coordinate_type> const& box) const
{
    // go through bins and extract sub rows 
    Box<index_type> idxBinBox = m_algoDB->getBinIndexRange(kSBin, box.xl(), box.yl(), box.xh(), box.yh());
    std::vector<index_type> vSubRow;
    for (index_type idxX = idxBinBox.xl(); idxX <= idxBinBox.xh(); ++idxX)
        for (index_type idxY = idxBinBox.yl(); idxY <= idxBinBox.yh(); ++idxY)
        {
            Bin const& bin = m_algoDB->getBinByIndex(kSBin, idxX, idxY);
            for (HrchyList<index_type>::const_iterator_type itBR = bin.binRows().begin(), itBRe = bin.binRows().end(); itBR != itBRe; ++itBR)
            {
                BinRow const& brow = m_algoDB->getBinRow(*itBR);
                vSubRow.push_back(brow.subRowId());
            }
        }
    // remove duplicates of sub rows 
    removeDuplicates(vSubRow);

    // go through sub rows and extract nodes 
    std::vector<index_type> vNodeInBox; // result 
    for (std::vector<index_type>::const_iterator itSR = vSubRow.begin(), itSRe = vSubRow.end(); itSR != itSRe; ++itSR)
    {
        SubRow const& srow = m_algoDB->getSubRow(*itSR);
        if (intersects(srow, box, false)) // consider sub row that has overlap with the box 
        {
            std::pair<map_const_iterator_type, map_const_iterator_type> found = queryRange(srow.index1D(), box.xl(), box.xh(), true);
            for (map_const_iterator_type itn = found.first; itn != found.second; ++itn)
            {
#ifdef DEBUG
#ifdef USE_RTREE
                dreamplaceAssert(getMapElementHigh(itn) > box.xl()); // skip cells that do not have overlap with the range 
#endif
#endif
#ifdef USE_RTREE
                vNodeInBox.push_back(getMapElementId(itn));
#elif defined(USE_INTERVALHASHMAP)
                if (getMapElementHigh(itn) > box.xl()) // skip cells that do not have overlap with the range 
                    vNodeInBox.push_back(getMapElementId(itn));
#endif
            }
        }
    }
    return vNodeInBox;
}

std::vector<SubRowNodeMap::index_type> SubRowNodeMap::queryRange(
        SubRowNodeMap::coordinate_type xl, SubRowNodeMap::coordinate_type yl, 
        SubRowNodeMap::coordinate_type xh, SubRowNodeMap::coordinate_type yh) const 
{
    return queryRange(Box<SubRowNodeMap::coordinate_type>(xl, yl, xh, yh));
}

std::pair<SubRowNodeMap::map_const_iterator_type, SubRowNodeMap::map_const_iterator_type> 
SubRowNodeMap::queryRange(SubRowNodeMap::index_type idx, 
        SubRowNodeMap::coordinate_type xl, SubRowNodeMap::coordinate_type xh, bool noBoundary) const 
{
    map_type const& map = subRowMap(idx);
#ifdef USE_RTREE
    if (noBoundary)
        return std::make_pair(
                map.qbegin(bgi::intersects(interval_type(xl, xh)) 
                    && bgi::satisfies(NoBoundaryPredicate(xl, xh))), 
                map.qend()
                );
    else 
        return std::make_pair(
                map.qbegin(bgi::intersects(interval_type(xl, xh))), 
                map.qend()
                );
#elif defined(USE_INTERVALHASHMAP)
    return query(map, xl, xh, !noBoundary); 
#endif
}

void SubRowNodeMap::queryRange(SubRowNodeMap::index_type idx, SubRowNodeMap::coordinate_type xl, SubRowNodeMap::coordinate_type xh, bool noBoundary, std::vector<map_element_type>& vNode) const
{
    map_type const& map = subRowMap(idx);
#ifdef USE_RTREE
    if (noBoundary)
        map.query(bgi::intersects(interval_type(xl, xh)), 
                std::back_inserter(vNode));
    else 
        map.query(bgi::intersects(interval_type(xl, xh)) && bgi::satisfies(NoBoundaryPredicate(xl, xh)), 
                std::back_inserter(vNode));
#elif defined(USE_INTERVALHASHMAP)
    std::pair<map_const_iterator_type, map_const_iterator_type> found = query(map, xl, xh, !noBoundary); 
    vNode.assign(found.first, found.second); 
#endif
}

void SubRowNodeMap::queryRange(SubRowNodeMap::index_type idx, SubRowNodeMap::coordinate_type xl, SubRowNodeMap::coordinate_type xh, bool noBoundary, SubRowNodeMap::map_type& targetMap) const
{
    map_type const& map = subRowMap(idx);
#ifdef USE_RTREE
    if (noBoundary)
        map.query(bgi::intersects(interval_type(xl, xh)), 
                bgi::inserter(targetMap));
    else 
        map.query(bgi::intersects(interval_type(xl, xh)) && bgi::satisfies(NoBoundaryPredicate(xl, xh)), 
                bgi::inserter(targetMap));
#elif defined(USE_INTERVALHASHMAP)
    std::pair<map_const_iterator_type, map_const_iterator_type> found = query(map, xl, xh, !noBoundary); 
    for (map_const_iterator_type it = found.first; it != found.second; ++it)
        targetMap.insert(*it); 
#endif
}

bool SubRowNodeMap::count(SubRowNodeMap::index_type idx, Node const& node) const 
{
    return subRowMap(idx).count(
            map_element_type(node.id(), node.initPos().x(), node.get(kX))
            );
}

bool SubRowNodeMap::erase(SubRowNodeMap::index_type idx, Node const& node)
{
    return subRowMap(idx).remove(
            map_element_type(node.id(), node.initPos().x(), node.get(kX))
            );
}

void SubRowNodeMap::insert(SubRowNodeMap::index_type idx, Node const& node)
{
    subRowMap(idx).insert(
            map_element_type(node.id(), node.initPos().x(), node.get(kX))
            );
}

void SubRowNodeMap::print(SubRowNodeMap::index_type idx) const
{
    printRange(idx, m_algoDB->placeDB().rowXL(), m_algoDB->placeDB().rowXH()); 
}

void SubRowNodeMap::printRange(SubRowNodeMap::index_type idx, 
        SubRowNodeMap::coordinate_type xl, SubRowNodeMap::coordinate_type xh) const 
{
    char prefix[16];
    dreamplaceSPrint(kNONE, prefix, "r%u: ", idx);
#ifdef USE_RTREE
    for (map_const_iterator_type it = subRowMap(idx).qbegin(
                bgi::intersects(
                    interval_type(xl, xh)
                    ) 
                && bgi::satisfies(
                    NoBoundaryPredicate(xl, xh)
                    )); it != subRowMap(idx).qend(); ++it)
    {
        dreamplacePrint(kNONE, "%s%u@%d", prefix, getMapElementId(it), getMapElementLow(it));
        dreamplaceSPrint(kNONE, prefix, ", ");
    }
#elif defined(USE_INTERVALHASHMAP)
    std::pair<map_const_iterator_type, map_const_iterator_type> found = query(subRowMap(idx), xl, xh, true); 
    for (map_const_iterator_type it = found.first; it != found.second; ++it)
    {
        dreamplacePrint(kNONE, "%s%u@%d", prefix, getMapElementId(it), getMapElementLow(it));
        dreamplaceSPrint(kNONE, prefix, ", ");
    }
#endif 
    dreamplacePrint(kNONE, "\n");
}

#if 0
Node const& SubRowNodeMap::getMapElement(SubRowNodeMap::map_const_iterator_type const& it) const
{
    return getMapElement(*it);
}

Node const& SubRowNodeMap::getMapElement(SubRowNodeMap::map_element_type const& v) const 
{
    return m_algoDB->node(getMapElementId(v));
}
#endif

DREAMPLACE_END_NAMESPACE
