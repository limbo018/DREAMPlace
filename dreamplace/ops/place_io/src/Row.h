/*************************************************************************
    > File Name: Row.h
    > Author: Yibo Lin
    > Mail: yibolin@utexas.edu
    > Created Time: Wed Jun 17 21:54:17 2015
 ************************************************************************/

#ifndef DREAMPLACE_ROW_H
#define DREAMPLACE_ROW_H

#include "Object.h"
#include "Box.h"
#include "Enums.h"
#include <vector>

DREAMPLACE_BEGIN_NAMESPACE

class Row : public Box<Object::coordinate_type>, public Object
{
	public:
        typedef Object base_type2;
        typedef base_type2::coordinate_type coordinate_type;
        typedef Box<coordinate_type> base_type1;
        typedef coordinate_traits<coordinate_type>::index_type index_type;

        /// default constructor 
        Row();
        /// copy constructor
        Row(Row const& rhs);
        /// assignment
        Row& operator=(Row const& rhs);

        /// member functions
        std::string const& name() const {return m_name;}
        Row& setName(std::string const& s) {m_name = s; return *this;}

        std::string const& macroName() const {return m_macroName;}
        Row& setMacroName(std::string const& s) {m_macroName = s; return *this;}

        Orient const& orient() const {return m_orient;}
        Row& setOrient(Orient const& o) {m_orient = o; return *this;}

        coordinate_type const& step(Direction1DType d) const {return m_step[d];}
        Row& setStep(Direction1DType d, coordinate_type v) {m_step[d] = v; return *this;}
        Row& setStep(coordinate_type vx, coordinate_type vy) {m_step[kX] = vx; m_step[kY] = vy; return *this;}

        index_type numSites(Direction1DType d) const {return delta(d)/step(d);}
    protected:
        void copy(Row const& rhs);

        std::string m_name; ///< name of row 
        std::string m_macroName; ///< macro name of row, usually not used  
		Orient m_orient;
        coordinate_type m_step[2]; ///< step of rows in x and y direction, usually same as site width and height 
};

inline Row::Row() 
    : Row::base_type1()
    , Row::base_type2()
    , m_name("")
    , m_macroName("")
    , m_orient()
{
    m_step[kX] = std::numeric_limits<Row::coordinate_type>::max();
    m_step[kY] = std::numeric_limits<Row::coordinate_type>::max();
}
inline Row::Row(Row const& rhs)
    : Row::base_type1(rhs)
    , Row::base_type2(rhs)
{
    copy(rhs);
}
inline Row& Row::operator=(Row const& rhs)
{
    if (this != &rhs)
    {
        this->base_type1::operator=(rhs);
        this->base_type2::operator=(rhs);
        copy(rhs);
    }
    return *this;
}
inline void Row::copy(Row const& rhs)
{
    m_name = rhs.m_name;
    m_macroName = rhs.m_macroName;
    m_orient = rhs.m_orient;
    m_step[kX] = rhs.m_step[kX];
    m_step[kY] = rhs.m_step[kY];
}

/// compare by row bottom edge 
/// tie break by row id 
struct CompareByRowBottomCoord
{
    bool operator()(Row const& row1, Row const& row2) const
    {
        return row1.yl() < row2.yl() || (row1.yl() == row2.yl() && row1.id() < row2.id()); 
    }
};

/// a row is divided into sub row due to fixed cells or blockages 
class SubRow : public Box<Object::coordinate_type>
{
	public:
        typedef Object::coordinate_type coordinate_type;
        typedef Box<coordinate_type> base_type;
        typedef coordinate_traits<coordinate_type>::index_type index_type;
        typedef coordinate_traits<coordinate_type>::area_type area_type;

        /// default constructor 
        SubRow();
        /// copy constructor
        SubRow(SubRow const& rhs);
        /// assignment
        SubRow& operator=(SubRow const& rhs);

        /// member functions 
        index_type index1D() const {return m_index1D;}
        SubRow& setIndex1D(index_type id) {m_index1D = id; return *this;}

        index_type rowId() const {return m_rowId;}
        SubRow& setRowId(index_type id) {m_rowId = id; return *this;}

        /// \return sub row id in a row for sub row map indexing 
        index_type subRowId() const {return m_subRowId;}
        SubRow& setSubRowId(index_type id) {m_subRowId = id; return *this;}

        std::vector<index_type> const& binRows() const {return m_vBinRowId;}
        std::vector<index_type>& binRows() {return m_vBinRowId;}

    protected:
        void copy(SubRow const& rhs);

        index_type m_index1D; ///< index in sub row array 
        index_type m_rowId; ///< parent row index 
        index_type m_subRowId; ///< sub row id in a row 
        std::vector<index_type> m_vBinRowId; ///< indices of bin sub rows 
};
inline SubRow::SubRow() 
    : SubRow::base_type()
    , m_index1D (std::numeric_limits<SubRow::index_type>::max())
    , m_rowId(std::numeric_limits<SubRow::index_type>::max())
    , m_subRowId(std::numeric_limits<SubRow::index_type>::max())
{
}
inline SubRow::SubRow(SubRow const& rhs)
    : SubRow::base_type(rhs)
{
    copy(rhs);
}
inline SubRow& SubRow::operator=(SubRow const& rhs)
{
    if (this != &rhs)
    {
        this->base_type::operator=(rhs);
        copy(rhs);
    }
    return *this;
}
inline void SubRow::copy(SubRow const& rhs)
{
    m_index1D = rhs.m_index1D;
    m_rowId = rhs.m_rowId;
    m_subRowId = rhs.m_subRowId;
    m_vBinRowId = rhs.m_vBinRowId;
}

/// row in a bin 
class BinRow : public Box<Object::coordinate_type>
{
	public:
        typedef Object::coordinate_type coordinate_type;
        typedef Box<coordinate_type> base_type;
        typedef coordinate_traits<coordinate_type>::index_type index_type;
        typedef coordinate_traits<coordinate_type>::area_type area_type;

        /// default constructor 
        BinRow();
        /// copy constructor
        BinRow(BinRow const& rhs);
        /// assignment
        BinRow& operator=(BinRow const& rhs);

        /// member functions 
        index_type index1D() const {return m_index1D;}
        BinRow& setIndex1D(index_type id) {m_index1D = id; return *this;}

        index_type binId() const {return m_binId;}
        BinRow& setBinId(index_type id) {m_binId = id; return *this;}

        index_type subRowId() const {return m_subRowId;}
        BinRow& setSubRowId(index_type id) {m_subRowId = id; return *this;}

        //std::vector<index_type> const& binSubRows() const {return m_vBSRowId;}
        //std::vector<index_type>& binSubRows() {return m_vBSRowId;}

    protected:
        void copy(BinRow const& rhs);

        index_type m_index1D; ///< bin row index in 1D array 
        index_type m_binId; ///< parent bin 1D index 
        index_type m_subRowId; ///< parent sub row 1D index
        //std::vector<index_type> m_vBSRowId; ///< indices of bin sub rows 
};

inline BinRow::BinRow() 
    : BinRow::base_type()
    , m_index1D(std::numeric_limits<BinRow::index_type>::max())
    , m_binId(std::numeric_limits<BinRow::index_type>::max())
    , m_subRowId(std::numeric_limits<BinRow::index_type>::max())
{
}
inline BinRow::BinRow(BinRow const& rhs)
    : BinRow::base_type(rhs)
{
    copy(rhs);
}
inline BinRow& BinRow::operator=(BinRow const& rhs)
{
    if (this != &rhs)
    {
        this->base_type::operator=(rhs);
        copy(rhs);
    }
    return *this;
}
inline void BinRow::copy(BinRow const& rhs)
{
    m_index1D = rhs.m_index1D;
    m_binId = rhs.m_binId;
    m_subRowId = rhs.m_subRowId;
    //m_vBSRowId = rhs.m_vBSRowId;
}


DREAMPLACE_END_NAMESPACE

#endif
