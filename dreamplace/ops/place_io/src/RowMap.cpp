/*************************************************************************
    > File Name: RowMap.cpp
    > Author: Yibo Lin
    > Mail: yibolin@utexas.edu
    > Created Time: Wed Jun 24 21:19:32 2015
 ************************************************************************/

#include "RowMap.h"
#include "AlgoDB.h"
#include "AlgoHelpers.h"
#include "Iterators.h"
#include <algorithm>
#include <boost/algorithm/cxx11/is_sorted.hpp>

DREAMPLACE_BEGIN_NAMESPACE

/// helper object of ObsWraper for collectBlockageIntervals()
struct CollectFixedCellIntervalHelper
{
    typedef AlgoDB::index_type index_type;
    typedef AlgoDB::interval_type interval_type;
    typedef AlgoDB::box_type box_type;

    PlaceDB const& placeDB; 
    SubRowMap const& subRowMap;
    std::vector<std::vector<interval_type> >& mBlkInterval;

    CollectFixedCellIntervalHelper(PlaceDB const& db, SubRowMap const& map, std::vector<std::vector<interval_type> >& mInterval)
        : placeDB(db)
        , subRowMap(map)
        , mBlkInterval(mInterval)
    {
    }
    CollectFixedCellIntervalHelper(CollectFixedCellIntervalHelper const& rhs)
        : placeDB(rhs.placeDB)
        , subRowMap(rhs.subRowMap)
        , mBlkInterval(rhs.mBlkInterval)
    {
    }
    inline void operator()(Node const& /*node*/, box_type const& box)
    {
        subRowMap.collectBlockageIntervals(placeDB, box, mBlkInterval);
    }
};

SubRowMap& SubRowMap::set(PlaceDB const& db)
{
    std::vector<Row> const& vRow = db.rows();
    index_type numRow = vRow.size();
    m_vSubRow.clear();
    m_mSubRowId.resize(numRow);
    // create sub rows 
    // construct a 2D blockage map 
    std::vector<std::vector<interval_type> > mBlkInterval (numRow); 
    ObsWraper<CollectFixedCellIntervalHelper> cfciHelper (db, CollectFixedCellIntervalHelper(db, *this, mBlkInterval));
    for (FixedNodeConstIterator it = db.fixedNodeBegin(); it.inRange(); ++it)
    {
        Node const& node = *it;
        cfciHelper(node);
    }
    for (std::vector<box_type>::const_iterator it = db.placeBlockages().begin(); it != db.placeBlockages().end(); ++it)
        collectBlockageIntervals(db, *it, mBlkInterval);
    // clean blockage map so that there is no overlap between row slots 
    for (std::vector<std::vector<Interval<coordinate_type> > >::iterator it1 = mBlkInterval.begin(), it1e = mBlkInterval.end(); it1 != it1e; ++it1)
    {
        std::vector<Interval<coordinate_type> > vMergedBlkInterval;
        // sort by low coordinates 
        std::sort(it1->begin(), it1->end(), Interval<coordinate_type>::CompareByLow());
        // merge intervals 
        for (index_type i = 0, ie = it1->size(); i < ie; ++i)
        {
            Interval<coordinate_type> const& icur = it1->at(i);
            if (!vMergedBlkInterval.empty() && intersects(vMergedBlkInterval.back(), icur, true))
                vMergedBlkInterval.back().encompass(icur);
            else 
                vMergedBlkInterval.push_back(icur);
        }
        // apply vMergedBlkInterval to mBlkInterval
        it1->swap(vMergedBlkInterval);
    }
    // create sub rows in each row 
    for (index_type i = 0; i < numRow; ++i)
    {
        Row const& row = db.rows().at(i);
        std::vector<Interval<coordinate_type> > const& vBlkInterval = mBlkInterval.at(i);
        std::vector<index_type>& vSubRowId = m_mSubRowId.at(i);
        vSubRowId.reserve(vBlkInterval.size()+1);
        for (index_type j = 0, je = vBlkInterval.size(); j <= je; ++j)
        {
            coordinate_type xl = (j == 0)? row.xl() : vBlkInterval[j-1].high();
            coordinate_type xh = (j < vBlkInterval.size())? vBlkInterval[j].low() : row.xh();

            if (xl+db.siteWidth() <= xh) // at least 1 site 
            {
                // create and add sub row 
                // set data that is not going to change 
                m_vSubRow.push_back(SubRow());
                SubRow& srow = m_vSubRow.back();
                srow.setIndex1D(m_vSubRow.size()-1); // object id 
                srow.set(xl, row.yl(), xh, row.yh());
                srow.setRowId(row.id()); // id for level-1 indexing 
                srow.setSubRowId(vSubRowId.size()); // id for level-2 indexing 
                vSubRowId.push_back(srow.index1D());
            }
        }
    }

    return *this;
}

void SubRowMap::collectBlockageIntervals(PlaceDB const& db, SubRowMap::box_type const& box, std::vector<std::vector<SubRowMap::interval_type> >& mBlkInterval) const
{
    std::vector<Row> const& vRow = db.rows();
    // collect intervals to blockage map 
    // it is possible that the blockage may not align to sites 
    // scale it up so that all the sub rows start and end to sites 
    box_type adjustBox (
            db.rowXL()+floor((box.xl()-db.rowXL())/db.siteWidth())*db.siteWidth(), 
            box.yl(),
            db.rowXL()+ceil((double)(box.xh()-db.rowXL())/db.siteWidth())*db.siteWidth(), 
            box.yh()
            );
    Interval<index_type> idxRange (db.getRowIndexRange(adjustBox.yl(), adjustBox.yh()));
    for (index_type i = idxRange.low(); i <= idxRange.high(); ++i)
    {
        if (i < mBlkInterval.size() && intersects(vRow[i], adjustBox, false)) // only collect valid intervals 
            mBlkInterval[i].push_back(adjustBox.get(kX));
    }
}

BinRowMap& BinRowMap::set(AlgoDB& algo)
{
    BinType bt = kSBin; // use sbin to initialize bin row 
    SubRowMap const& subRowMap = algo.subRowMap();
    m_vBinRow.clear();
    m_mBinRowId.resize(subRowMap.numRows());
    // create bin rows according to sub row map and bin map 
    // traverse through all sub rows 
    for (SubRowMap1DConstIterator it = subRowMap.begin1D(), ite = subRowMap.end1D(); it != ite; ++it)
    {
        SubRow const& srow = *it;
        std::vector<index_type>& vBinRowId = m_mBinRowId.at(srow.rowId());
        Box<index_type> idxBox (algo.getBinIndexRange(bt, srow.xl(), srow.yl(), srow.xh(), srow.yh()));
        for (index_type iy = idxBox.yl(); iy <= idxBox.yh(); ++iy)
        {
            vBinRowId.reserve(idxBox.width()+1);
            for (index_type ix = idxBox.xl(); ix <= idxBox.xh(); ++ix)
            {
                Bin& bin = algo.getBinByIndex(bt, ix, iy);
                std::pair<Box<coordinate_type>, bool> intersectBox = intersection(srow, bin, false);
                if (intersectBox.second) // at least have intersection 
                {
                    // create bin row 
                    m_vBinRow.push_back(BinRow());
                    BinRow& brow = m_vBinRow.back();
                    brow.set(kX, intersectBox.first.get(kX)).set(kY, srow.get(kY));
                    brow.setIndex1D(m_vBinRow.size()-1); // object id 
                    brow.setBinId(bin.index1D());
                    brow.setSubRowId(srow.index1D()); // id to find parent sub row 
                    vBinRowId.push_back(brow.index1D());
                    // add to bin
                    // first step 
                    bin.binRows().setPartial1(brow.index1D());
                }
            }
        }
    }
    // iterate through all bins to construct m_vBinRowId in bins  
    for (BinMap1DIterator it = algo.binMap(bt).begin1D(), ite = algo.binMap(bt).end1D(); it != ite; ++it)
    {
        Bin& bin = *it;
        std::vector<index_type> vRowIdOfBinRows (bin.binRows().size()); // row id of bin rows, because row id is continuous vertically 
        index_type count = 0;
        for (HrchyList<index_type>::const_iterator_type itBR = bin.binRows().begin(), itBRe = bin.binRows().end(); itBR != itBRe; ++itBR)
            vRowIdOfBinRows[count++] = algo.getSubRow(getBinRow(*itBR).subRowId()).rowId(); // assume sub row is initialized 

        Interval<index_type> rowIdxInv = algo.placeDB().getRowIndexRange(bin.yl(), bin.yh()-1); // min, max row id 
        for (std::vector<index_type>::iterator itSRID = vRowIdOfBinRows.begin(), itSRIDe = vRowIdOfBinRows.end(); itSRID != itSRIDe; ++itSRID)
        {
           *itSRID -= rowIdxInv.low(); // normalize to 0 
        }
        std::vector<std::vector<index_type> > mData;
        // second step 
        bin.binRows().setPartial2(vRowIdOfBinRows, mData, rowIdxInv.delta()+1);
        // the target is to sort bin rows from left to right within each row 
        // considering that bin row id is ordered from bottom to top, left to right, by construction 
        // it should be enough to sort by the indices 
        for (std::vector<std::vector<index_type> >::iterator itd = mData.begin(), itde = mData.end(); itd != itde; ++itd)
            std::sort(itd->begin(), itd->end());
        // third step 
        bin.binRows().setPartial3(mData);
    }
#ifdef DEBUG
    // maybe no longer reasonable check 
    for (BinMap1DConstIterator it = algo.binMap(bt).begin1D(), ite = algo.binMap(bt).end1D(); it != ite; ++it)
    {
        Bin const& bin = *it;
        // check whether bin rows in a sbin are sorted 
        dreamplaceAssert(boost::algorithm::is_sorted(bin.binRows().begin(), bin.binRows().end()));
    }
#endif

    return *this;
}

DREAMPLACE_END_NAMESPACE
