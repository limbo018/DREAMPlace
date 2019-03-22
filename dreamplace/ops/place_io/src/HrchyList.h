/*************************************************************************
  > File Name: HrchyList.h
  > Author: Yibo Lin
  > Mail: yibolin@utexas.edu
  > Created Time: Mon 01 Feb 2016 02:17:29 PM CST
 ************************************************************************/

#ifndef DREAMPLACE_HRCHYLIST_H
#define DREAMPLACE_HRCHYLIST_H

#include <iostream>
#include <vector>
#include <limits>
#include "Util.h"

DREAMPLACE_BEGIN_NAMESPACE

/// HrchyList defines a hierarchical list for data storage. 
/// Data is stored like a matrix with 2-dimension (row/column). 
/// The second dimension may have different size with different first dimension. 
/// So it is not actually a matrix. 
/// The api allows to iterate the HrchyList like a 1D array. 

template <typename T>
class HrchyList
{
    public:
        typedef T value_type;
        typedef unsigned int index_type;
        typedef std::vector<value_type> container_type;
        typedef typename container_type::iterator iterator_type;
        typedef typename container_type::const_iterator const_iterator_type;

        /// constructor 
        HrchyList()
        {
        }
        /// copy constructor 
        HrchyList(HrchyList const& rhs)
        {
            copy(rhs);
        }
        /// assignment 
        HrchyList& operator=(HrchyList const& rhs)
        {
            if (this != &rhs)
                copy(rhs);
            return *this;
        }
        /// destructor
        ~HrchyList() {}

        /// \return size of all data 
        index_type size() const {return m_vData.size();}
        /// \return number of rows 
        index_type numRows() const {return m_vRowBeginIdx.size();}
        /// \return next row begin index  
        /// overflow to m_vData.size()
        index_type getNextRowBeginIndex(index_type rowIdx) const 
        {
            index_type nr = numRows();
            index_type nextRowIdx = rowIdx+1;
            index_type nextRowBeginIdx;
            if (nextRowIdx >= nr) // rowIdx is last row 
                nextRowBeginIdx = m_vData.size();
            else 
            {
                nextRowBeginIdx = m_vRowBeginIdx[nextRowIdx];
                if (nextRowBeginIdx >= s_largeIdx)// nextRowIdx is empty 
                    nextRowBeginIdx -= s_largeIdx;
            }
            return nextRowBeginIdx;
        }
        /// \return number of columns in a specific row 
        index_type numCols(index_type rowIdx) const 
        {
            index_type rowBeginIdx = m_vRowBeginIdx[rowIdx];
            index_type nextRowBeginIdx;
            if (rowBeginIdx < s_largeIdx) // non-empty row 
                nextRowBeginIdx = getNextRowBeginIndex(rowIdx);
            else // empty row 
                nextRowBeginIdx = rowBeginIdx;
            return nextRowBeginIdx-rowBeginIdx;
        }

        /// \return a row of data 
        std::pair<iterator_type, iterator_type> row(index_type rowIdx) 
        {
            index_type rowBeginIdx = m_vRowBeginIdx[rowIdx];
            index_type nextRowBeginIdx;
            if (rowBeginIdx < s_largeIdx) // non-empty row 
                nextRowBeginIdx = getNextRowBeginIndex(rowIdx);
            else // empty row 
                nextRowBeginIdx = rowBeginIdx = rowBeginIdx-s_largeIdx;
            return std::make_pair(m_vData.begin()+rowBeginIdx, m_vData.begin()+nextRowBeginIdx);
        }
        std::pair<const_iterator_type, const_iterator_type> row(index_type rowIdx) const
        {
            index_type rowBeginIdx = m_vRowBeginIdx[rowIdx];
            index_type nextRowBeginIdx;
            if (rowBeginIdx < s_largeIdx) // non-empty row 
                nextRowBeginIdx = getNextRowBeginIndex(rowIdx);
            else // empty row 
                nextRowBeginIdx = rowBeginIdx = rowBeginIdx-s_largeIdx;
            return std::make_pair(m_vData.begin()+rowBeginIdx, m_vData.begin()+nextRowBeginIdx);
        }
        /// \return range of rows 
        /// assume \param rowIdxL < \param rowIdxH and they are all within the range 
        std::pair<iterator_type, iterator_type> rowRange(index_type rowIdxL, index_type rowIdxH) 
        {
            index_type rowBeginIdxL = m_vRowBeginIdx[rowIdxL];
            if (rowBeginIdxL >= s_largeIdx) // empty row of rowIdxL
                rowBeginIdxL -= s_largeIdx;
            index_type nextRowBeginIdxH = getNextRowBeginIndex(rowIdxH);

            return std::make_pair(m_vData.begin()+rowBeginIdxL, m_vData.begin()+nextRowBeginIdxH);
        }
        std::pair<const_iterator_type, const_iterator_type> rowRange(index_type rowIdxL, index_type rowIdxH) const
        {
            index_type rowBeginIdxL = m_vRowBeginIdx[rowIdxL];
            if (rowBeginIdxL >= s_largeIdx) // empty row of rowIdxL
                rowBeginIdxL -= s_largeIdx;
            index_type nextRowBeginIdxH = getNextRowBeginIndex(rowIdxH);

            return std::make_pair(m_vData.begin()+rowBeginIdxL, m_vData.begin()+nextRowBeginIdxH);
        }

        /// \return the index of first non-empty row 
        index_type frontNonemptyRowIndex() const 
        {
            index_type rowIdx = 0;
            index_type nr = numRows();
            while (rowIdx < nr && m_vRowBeginIdx[rowIdx] >= s_largeIdx)
                ++rowIdx;
            return rowIdx;
        }
        /// \return the index of last non-empty row 
        index_type backNonemptyRowIndex() const 
        {
            index_type rowIdx = numRows();
            while (rowIdx > 0 && m_vRowBeginIdx[rowIdx-1] >= s_largeIdx)
                --rowIdx;
            return rowIdx-1; // overflow if no non-empty row 
        }

        /// \return a single value 
        /// assume row is not empty 
        T& at(index_type rowIdx, index_type colIdx) {return m_vData[m_vRowBeginIdx[rowIdx]+colIdx];}
        T const& at(index_type rowIdx, index_type colIdx) const {return m_vData[m_vRowBeginIdx[rowIdx]+colIdx];}

        /// \return begin iterator of data 
        iterator_type begin() {return m_vData.begin();}
        const_iterator_type begin() const {return m_vData.begin();}
        /// \return end iterator of data 
        iterator_type end() {return m_vData.end();}
        const_iterator_type end() const {return m_vData.end();}
        /// \return data 
        std::vector<value_type>& data() {return m_vData;}
        std::vector<value_type> const& data() const {return m_vData;}

        /// \return the begin iterator of m_vRowBeginIdx
        std::vector<index_type>::iterator beginRowIndex() {return m_vRowBeginIdx.begin();}
        std::vector<index_type>::const_iterator beginRowIndex() const {return m_vRowBeginIdx.begin();}
        /// \return the end iterator of m_vRowBeginIdx
        std::vector<index_type>::iterator endRowIndex() {return m_vRowBeginIdx.end();}
        std::vector<index_type>::const_iterator endRowIndex() const {return m_vRowBeginIdx.end();}

        /// construct HrchyList with a 2D array 
        void set(std::vector<std::vector<value_type> > const& mData)
        {
            index_type numData = 0;
            for (typename std::vector<std::vector<value_type> >::const_iterator it1 = mData.begin(), it1e = mData.end(); it1 != it1e; ++it1)
                numData += it1->size();
            m_vData.resize(numData);
            m_vRowBeginIdx.resize(mData.size(), 0);

            index_type count = 0;
            for (index_type rowIdx = 0; rowIdx < mData.size(); ++rowIdx)
            {
                m_vRowBeginIdx[rowIdx] = count;
                if (mData[rowIdx].empty())
                    m_vRowBeginIdx[rowIdx] += s_largeIdx;

                for (index_type colIdx = 0; colIdx < mData[rowIdx].size(); ++colIdx)
                    m_vData[count++] = mData[rowIdx][colIdx];
            }
        }
        /// construct HrchyList by three steps 
        /// this is the first step that append all the data to m_vData
        void setPartial1(value_type v) 
        {
            m_vData.push_back(v);
        }
        /// this is the second step that given all the row index of existing data
        /// construct the rest, \param vRowId must have the same order with m_vData
        /// \param rn denotes number of rows in the list 
        /// \param vRowId assumes the row id starts from 0 
        void setPartial2(std::vector<index_type> const& vRowId, std::vector<std::vector<value_type> >& mData, index_type rn)
        {
            mData.resize(rn);
            for (index_type i = 0, ie = m_vData.size(); i < ie; ++i)
                mData[vRowId[i]].push_back(m_vData[i]);
            // users are allowed to manipulate mData before calling the third step 
        }
        /// this is the third step 
        /// actually an alias of set()
        void setPartial3(std::vector<std::vector<value_type> > const& mData)
        {
            set(mData);
        }
        /// clear all data 
        void clear()
        {
            m_vData.clear();
            m_vRowBeginIdx.clear();
        }

        /// for debug 
        void printHrchy(std::ostream& os = std::cout) const 
        {
            for (index_type rowIdx = 0; rowIdx < numRows(); ++rowIdx)
            {
                std::pair<const_iterator_type, const_iterator_type> found = row(rowIdx);
                os << "#" << rowIdx << ": (" << numCols(rowIdx) << ")";
                for (const_iterator_type it = found.first; it != found.second; ++it)
                    os << *it << " ";
                os << "\n";
            }
         }
        void print(std::ostream& os = std::cout) const 
        {
            os << "data: ";
            for (const_iterator_type it = begin(), ite = end(); it != ite; ++it)
                os << *it << " ";
            os << "\n";
            os << "row begin idx: ";
            for (std::vector<index_type>::const_iterator it = beginRowIndex(), ite = endRowIndex(); it != ite; ++it)
                os << *it << " ";
            os << "\n";
            os << "first non-empty row index: " << frontNonemptyRowIndex() << "\n";
            os << "last non-empty row index: " << backNonemptyRowIndex() << "\n";
        }
    protected:
        void copy(HrchyList const& rhs)
        {
            m_vData = rhs.m_vData;
            m_vRowBeginIdx = rhs.m_vRowBeginIdx;
        }

        std::vector<value_type> m_vData;
        std::vector<index_type> m_vRowBeginIdx; ///< the index of row beginners 
                                                ///< if a row is empty, m_vRowBeginIdx[]  is set to s_largeIdx + m_vRowBeginIdx[non-empty index] 
                                                ///< e.g. row 1 is empty, row 2 is not empty, then m_vRowBeginIdx[1] = s_largeIdx + m_vRowBeginIdx[2]
        static const index_type s_largeIdx; ///< a very large number for index 
};

template <typename T>
const typename HrchyList<T>::index_type HrchyList<T>::s_largeIdx = std::numeric_limits<typename HrchyList<T>::index_type>::max()>>1;

DREAMPLACE_END_NAMESPACE

#endif
