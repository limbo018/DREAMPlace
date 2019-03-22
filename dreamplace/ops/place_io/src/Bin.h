/*************************************************************************
    > File Name: Bin.h
    > Author: Yibo Lin
    > Mail: yibolin@utexas.edu
    > Created Time: Sun Jun 21 20:06:50 2015
 ************************************************************************/

#ifndef DREAMPLACE_BIN_H
#define DREAMPLACE_BIN_H

#include "Box.h"
#include "Object.h"
#include "HrchyList.h"

DREAMPLACE_BEGIN_NAMESPACE

enum BinType
{
    kBin = 0,
    kSBin = 1, 
    kBinTypeNum = 2
};

inline BinType& operator++(BinType& t)
{
    if (t == kBin) return (t = kSBin);
    else if (t == kSBin) return (t = kBinTypeNum);
    dreamplaceAssertMsg(0, "overflow for increment BinType");
    return t;
}

class Bin : public Box<Object::coordinate_type>
{
    public:
        typedef Object::coordinate_type coordinate_type;
        typedef Box<coordinate_type> base_type;
        typedef coordinate_traits<coordinate_type>::index_type index_type;
        typedef coordinate_traits<coordinate_type>::area_type area_type;
        typedef coordinate_traits<coordinate_type>::site_area_type site_area_type;

        /// constructor
        Bin (coordinate_type xl = std::numeric_limits<coordinate_type>::max(), 
                coordinate_type yl = std::numeric_limits<coordinate_type>::max(), 
                coordinate_type xh = std::numeric_limits<coordinate_type>::min(), 
                coordinate_type yh = std::numeric_limits<coordinate_type>::min(), 
                BinType t = kBin,
                index_type id1D = std::numeric_limits<index_type>::max(), 
                index_type id2DX = std::numeric_limits<index_type>::max(), 
                index_type id2DY = std::numeric_limits<index_type>::max()); 
        /// copy constructor 
        Bin (Bin const& rhs);
        /// assignment
        Bin& operator=(Bin const& rhs);

        /// member functions
        BinType type() const {return m_type;}
        Bin& setType(BinType t) {m_type = t; return *this;}

        index_type index1D() const {return m_index1D;}
        Bin& setIndex1D(index_type id) {m_index1D = id; return *this;}

        index_type index2D(Direction1DType d) const {return m_index2D[d];}
        Bin& setIndex2D(Direction1DType d, index_type v) {m_index2D[d] = v; return *this;}

        index_type indexX() const {return index2D(kX);}
        index_type indexY() const {return index2D(kY);}

        /// \return demand of cell area 
        area_type demand() const {return m_demand;}
        Bin& setDemand(area_type v) {m_demand = v; return *this;}
        Bin& incrDemand(area_type v) {m_demand += v; return *this;}
        Bin& decrDemand(area_type v) {m_demand -= v; return *this;}

        /// \return capacity of area for cells in the bin, excluding fixed cells and forbidden regions  
        area_type capacity() const {return m_capacity;}
        Bin& setCapacity(area_type v) {m_capacity = v; return *this;}
        Bin& incrCapacity(area_type v) {m_capacity += v; return *this;}
        Bin& decrCapacity(area_type v) {m_capacity -= v; return *this;}

        /// \return pin demand 
        index_type pinDemand() const {return m_pinDemand;}
        Bin& setPinDemand(index_type v) {m_pinDemand = v; return *this;}
        Bin& incrPinDemand(index_type v) {m_pinDemand += v; return *this;}
        Bin& decrPinDemand(index_type v) {m_pinDemand -= v; return *this;}

        /// \return site capacity 
        site_area_type siteCapacity() const {return m_siteCapacity;}
        Bin& setSiteCapacity(site_area_type v) {m_siteCapacity = v; return *this;}
        Bin& incrSiteCapacity(site_area_type v) {m_siteCapacity += v; return *this;}
        Bin& decrSiteCapacity(site_area_type v) {m_siteCapacity -= v; return *this;}

        /// \return density, area density 
        double density() const; 
        /// given a demand, \return density 
        double density(area_type d) const;
        /// \return pin density, pin number per site  
        double pinDensity() const;
        /// given a pin demand, \return pin density 
        double pinDensity(index_type d) const;

        HrchyList<index_type> const& binRows() const {return m_vBinRowId;}
        HrchyList<index_type>& binRows() {return m_vBinRowId;}

        std::vector<index_type> const& nodes() const {return m_vNodeId;}
        std::vector<index_type>& nodes() {return m_vNodeId;}

    protected:
        void copy(Bin const& rhs);

        BinType m_type; ///< type of bin 
        index_type m_index1D; ///< index in bin array 
        index_type m_index2D[2]; ///< index in x and y direction of bin map 
        area_type m_capacity; ///< area available for cells 
        site_area_type m_siteCapacity; ///< number of sites available for cells 
        area_type m_demand; ///< area taken by cells 
        index_type m_pinDemand; ///< number of pins 

        /// may not be used 
        std::vector<index_type> m_vNodeId; ///< nodes in the bin, initialize iff it is used 
        /// for sub bins 
        HrchyList<index_type> m_vBinRowId; ///< bin rows in the bin, from low to high, left to right 
};

inline Bin::Bin(Bin::coordinate_type xl, Bin::coordinate_type yl, 
        Bin::coordinate_type xh, Bin::coordinate_type yh, 
        BinType t,
        index_type id1D, index_type id2DX, index_type id2DY) 
    : Bin::base_type(xl, yl, xh, yh)
    , m_type (t)
    , m_index1D(id1D)
{
    m_index2D[kX] = id2DX;
    m_index2D[kY] = id2DY;
    m_capacity = 0;
    m_siteCapacity = 0;
    m_demand = 0;
    m_pinDemand = 0;
}
inline Bin::Bin(Bin const& rhs)
    : Bin::base_type(rhs)
{
    copy(rhs);
}
inline Bin& Bin::operator=(Bin const& rhs)
{
    if (this != &rhs)
    {
        this->base_type::operator=(rhs);
        copy(rhs);
    }
    return *this;
}
inline void Bin::copy(Bin const& rhs)
{
    m_type = rhs.m_type;
    m_index1D = rhs.m_index1D;
    m_index2D[kX] = rhs.m_index2D[kX];
    m_index2D[kY] = rhs.m_index2D[kY];
    m_capacity = rhs.m_capacity;
    m_siteCapacity = rhs.m_siteCapacity;
    m_demand = rhs.m_demand;
    m_pinDemand = rhs.m_pinDemand;
    m_vBinRowId = rhs.m_vBinRowId;
    m_vNodeId = rhs.m_vNodeId;
}
inline double Bin::density() const 
{
    if (m_capacity == 0)
        return (m_demand == 0)? 0.0 : std::numeric_limits<double>::max();
    else 
        return (double)m_demand/m_capacity;
}
inline double Bin::density(Bin::area_type d) const 
{
    if (m_capacity == 0)
        return (d == 0)? 0.0 : std::numeric_limits<double>::max();
    else 
        return (double)d/m_capacity;
}
inline double Bin::pinDensity() const 
{
    if (m_siteCapacity == 0)
        return (m_pinDemand == 0)? 0.0 : std::numeric_limits<double>::max();
    else 
        return (double)m_pinDemand/m_siteCapacity;
}
inline double Bin::pinDensity(Bin::index_type d) const 
{
    if (m_siteCapacity == 0)
        return (d == 0)? 0.0 : std::numeric_limits<double>::max();
    else 
        return (double)d/m_siteCapacity;
}

DREAMPLACE_END_NAMESPACE

#endif
