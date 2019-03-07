/*************************************************************************
    > File Name: RowMap.h
    > Author: Yibo Lin
    > Mail: yibolin@utexas.edu
    > Created Time: Wed Jun 24 21:03:54 2015
 ************************************************************************/

#ifndef DREAMPLACE_ROWMAP_H
#define DREAMPLACE_ROWMAP_H

#include "Row.h"

DREAMPLACE_BEGIN_NAMESPACE

/// forward declaration of data 
class BinMap;
class PlaceDB;
class AlgoDB;

/// sub row map 
/// sub rows should not contain any region of fixed cells 
class SubRowMap 
{
    public:
        typedef Object::index_type index_type;
        typedef Object::coordinate_type coordinate_type;
        typedef Interval<coordinate_type> interval_type;
        typedef Box<coordinate_type> box_type;
        typedef std::vector<SubRow>::iterator SubRowMap1DIterator; ///< 1D iterator 
        typedef std::vector<SubRow>::const_iterator SubRowMap1DConstIterator; ///< 1D const iterator 

        /// constructor 
        SubRowMap();
        /// copy constructor 
        SubRowMap(SubRowMap const& rhs);
        /// assignment 
        SubRowMap& operator=(SubRowMap const& rhs);

        /// member functions 
        /// initialize 
        SubRowMap& set(PlaceDB const& db);

        /// \return level-1 indexing number 
        index_type numRows() const {return m_mSubRowId.size();}

        /// \return total number of sub rows 
        index_type numSubRows() const {return m_vSubRow.size();}
        /// \return sub rows 
        std::vector<SubRow> const& getSubRows() const {return m_vSubRow;}
        std::vector<SubRow>& getSubRows() {return m_vSubRow;}

        /// \return sub row index array in a row 
        std::vector<index_type> const& getSubRowsByRowIndex(index_type i) const {return m_mSubRowId.at(i);}
        /// \return sub row array iterators in a row 
        /// this is more convenient than getSubRowsByRowIndex()
        std::pair<std::vector<SubRow>::const_iterator, std::vector<SubRow>::const_iterator> getSubRowItersByRowIndex(index_type i) const 
        {
            if (i+1 < numRows()) 
                return std::make_pair(m_vSubRow.begin()+m_mSubRowId[i].front(), m_vSubRow.begin()+m_mSubRowId[i+1].front());
            else // last row 
                return std::make_pair(m_vSubRow.begin()+m_mSubRowId[i].front(), m_vSubRow.end());
        }
        /// index version of getSubRowItersByRowIndex()
        std::pair<index_type, index_type> getSubRowIndicesByRowIndex(index_type i) const 
        {
            if (i+1 < numRows()) 
                return std::make_pair(m_mSubRowId[i].front(), m_mSubRowId[i+1].front());
            else // last row 
                return std::make_pair(m_mSubRowId[i].front(), numSubRows());
        }

        /// \return sub row from object id 
        SubRow const& getSubRow(index_type id) const {return m_vSubRow.at(id);}
        SubRow& getSubRow(index_type id) {return m_vSubRow.at(id);}

        /// \return sub row from row id and sub row id 
        SubRow const& getSubRow(index_type rowId, index_type subRowId) const {return m_vSubRow.at(m_mSubRowId[rowId][subRowId]);}
        SubRow& getSubRow(index_type rowId, index_type subRowId) {return m_vSubRow.at(m_mSubRowId[rowId][subRowId]);}

        SubRowMap1DIterator begin1D() {return m_vSubRow.begin();}
        SubRowMap1DIterator end1D() {return m_vSubRow.end();}

        SubRowMap1DConstIterator begin1D() const {return m_vSubRow.begin();}
        SubRowMap1DConstIterator end1D() const {return m_vSubRow.end();}
#if 0
        /// TO DO: create iterator class for such kind of data structure 
        /// traversal through level-1
        SubRowMap1DIterator1 begin1() {return m_mSubRow.begin();}
        SubRowMap1DIterator1 end1() {return m_mSubRow.end();}
        SubRowMap1DConstIterator1 begin1() const {return m_mSubRow.begin();}
        SubRowMap1DConstIterator1 end1() const {return m_mSubRow.end();}
        /// traversal through level-2
        SubRowMap1DIterator2 begin2(SubRowMap1DIterator1 it) {return it->begin();}
        SubRowMap1DIterator2 end2(SubRowMap1DIterator1 it) {return it->end();}
        SubRowMap1DConstIterator2 begin2(SubRowMap1DConstIterator1 it) const {return it->begin();}
        SubRowMap1DConstIterator2 end2(SubRowMap1DConstIterator1 it) const {return it->end();}
#endif
    protected:
        void copy(SubRowMap const& rhs);
        /// a helper function to set()
        void collectBlockageIntervals(PlaceDB const& db, box_type const& box, std::vector<std::vector<interval_type> >& mBlkInterval) const;

        std::vector<SubRow> m_vSubRow; ///< actual sub rows are saved in a 1D array, must keep low to high, left to right order  
        std::vector<std::vector<index_type> > m_mSubRowId; ///< 2D bin row map, level-1 indexing is number of rows 
                                                    ///< number of sub rows may be different between rows 

        friend struct CollectFixedCellIntervalHelper;
};

inline SubRowMap::SubRowMap() 
    : m_vSubRow ()
    , m_mSubRowId()
{
}
inline SubRowMap::SubRowMap(SubRowMap const& rhs)
{
    copy(rhs);
}
inline SubRowMap& SubRowMap::operator=(SubRowMap const& rhs)
{
    if (this != &rhs)
        copy(rhs);
    return *this;
}
inline void SubRowMap::copy(SubRowMap const& rhs)
{
    m_vSubRow = rhs.m_vSubRow;
    m_mSubRowId = rhs.m_mSubRowId;
}

typedef SubRowMap::SubRowMap1DIterator SubRowMap1DIterator;
typedef SubRowMap::SubRowMap1DConstIterator SubRowMap1DConstIterator;

/// bin row map 
/// bin rows should not contain any region of fixed cells 
class BinRowMap 
{
    public:
        typedef Object::index_type index_type;
        typedef BinRow::coordinate_type coordinate_type;
        typedef std::vector<BinRow>::iterator BinRowMap1DIterator; ///< 1D iterator 
        typedef std::vector<BinRow>::const_iterator BinRowMap1DConstIterator; ///< 1D const iterator 

        /// constructor 
        BinRowMap();
        /// copy constructor 
        BinRowMap(BinRowMap const& rhs);
        /// assignment 
        BinRowMap& operator=(BinRowMap const& rhs);

        /// member functions 
        /// initialize  
        BinRowMap& set(AlgoDB& algo);

        index_type numRows() const {return m_mBinRowId.size();}
        index_type numBinRows() const {return m_vBinRow.size();}

        /// \return bin rows 
        std::vector<BinRow> const& getBinRows() const {return m_vBinRow;}
        std::vector<BinRow>& getBinRows() {return m_vBinRow;}

        /// \return bin row array in a row 
        std::vector<index_type> const& getBinRowsByRowIndex(index_type i) const {return m_mBinRowId.at(i);}

        /// \return bin row from object id 
        BinRow const& getBinRow(index_type id) const {return m_vBinRow.at(id);}
        BinRow& getBinRow(index_type id) {return m_vBinRow.at(id);}

        BinRowMap1DIterator begin1D() {return m_vBinRow.begin();}
        BinRowMap1DIterator end1D() {return m_vBinRow.end();}

        BinRowMap1DConstIterator begin1D() const {return m_vBinRow.begin();}
        BinRowMap1DConstIterator end1D() const {return m_vBinRow.end();}
    protected:
        void copy(BinRowMap const& rhs);

        std::vector<BinRow> m_vBinRow; ///< actual bin rows are saved in a 1D array, must be in low to high, left to right order  
        std::vector<std::vector<index_type> > m_mBinRowId; ///< 2D bin row map for indexing, level-1 indexing is number of rows 
                                                    ///< number of bin rows may be different between rows 
};

inline BinRowMap::BinRowMap() 
    : m_vBinRow ()
    , m_mBinRowId()
{
}
inline BinRowMap::BinRowMap(BinRowMap const& rhs)
{
    copy(rhs);
}
inline BinRowMap& BinRowMap::operator=(BinRowMap const& rhs)
{
    if (this != &rhs)
        copy(rhs);
    return *this;
}
inline void BinRowMap::copy(BinRowMap const& rhs)
{
    m_vBinRow = rhs.m_vBinRow;
    m_mBinRowId = rhs.m_mBinRowId;
}

typedef BinRowMap::BinRowMap1DIterator BinRowMap1DIterator;
typedef BinRowMap::BinRowMap1DConstIterator BinRowMap1DConstIterator;

DREAMPLACE_END_NAMESPACE

#endif
