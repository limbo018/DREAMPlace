/*************************************************************************
    > File Name: BinMap.h
    > Author: Yibo Lin
    > Mail: yibolin@utexas.edu
    > Created Time: Sun Jun 21 21:23:08 2015
 ************************************************************************/

#ifndef DREAMPLACE_BINMAP_H
#define DREAMPLACE_BINMAP_H

#include <vector>
#include "Bin.h"

DREAMPLACE_BEGIN_NAMESPACE

/// class BinMap holds all bins 
class BinMap 
{
    public:
        typedef Object::index_type index_type;
        typedef std::vector<Bin>::iterator BinMap1DIterator;
        typedef std::vector<Bin>::const_iterator BinMap1DConstIterator;

        /// constructor 
        BinMap(BinType t = kBin, index_type xNum = 0, index_type yNum = 0);
        /// copy constructor 
        BinMap(BinMap const& rhs);
        /// assignment 
        BinMap& operator=(BinMap const& rhs);

        /// member functions
        /// initialize bin map 
        BinMap& set(BinType type, index_type xNum, index_type yNum);

        /// \return dimensions 
        index_type dimension(Direction1DType d) const {return m_dimension[d];}
        index_type dimensionX() const {return dimension(kX);}
        index_type dimensionY() const {return dimension(kY);}

        /// \return total number of bins 
        index_type size() const {return m_vBin.size();}

        /// \return bin with index 1D
        Bin const& getBin(index_type id1D) const {return m_vBin.at(id1D);}
        Bin& getBin(index_type id1D) {return m_vBin.at(id1D);}

        /// \return bin with index 2D (x and y)
        Bin const& getBin(index_type id2DX, index_type id2DY) const {return m_vBin.at(m_dimension[kX]*id2DY + id2DX);}
        Bin& getBin(index_type id2DX, index_type id2DY) {return m_vBin.at(m_dimension[kX]*id2DY + id2DX);}

        /// \return left bin of current bin 
        /// if current bin is leftmost, return itself  
        Bin const& getLeftBin(Bin const& bin) const {return (bin.index1D()%dimensionX() == 0)? bin : getBin(bin.index1D()-1);}
        Bin& getLeftBin(Bin& bin) {return (bin.index1D()%dimensionX() == 0)? bin : getBin(bin.index1D()-1);}

        /// \return right bin of current bin 
        /// if current bin is right most, return itself 
        Bin const& getRightBin(Bin const& bin) const {return ((bin.index1D()+1)%dimensionX() == 0)? bin : getBin(bin.index1D()+1);}
        Bin& getRightBin(Bin& bin) {return ((bin.index1D()+1)%dimensionX() == 0)? bin : getBin(bin.index1D()+1);}
        
        /// \return lower bin of current bin 
        /// if current bin is lowest, return itself 
        Bin const& getLowerBin(Bin const& bin) const {return (bin.index1D() < dimensionX())? bin : getBin(bin.index1D()-dimensionX());}
        Bin& getLowerBin(Bin& bin) {return (bin.index1D() < dimensionX())? bin : getBin(bin.index1D()-dimensionX());}

        /// \return upper bin of current bin 
        /// if current bin is top, return itself 
        Bin const& getUpperBin(Bin const& bin) const {return (bin.index1D()+dimensionX() >= size())? bin : getBin(bin.index1D()+dimensionX());}
        Bin& getUpperBin(Bin& bin) {return (bin.index1D()+dimensionX() >= size())? bin : getBin(bin.index1D()+dimensionX());}

        /// \return 1D iterators of bins 
        BinMap1DIterator begin1D() {return m_vBin.begin();}
        BinMap1DIterator end1D() {return m_vBin.end();}
        BinMap1DConstIterator begin1D() const {return m_vBin.begin();}
        BinMap1DConstIterator end1D() const {return m_vBin.end();}

        /// reset bin demand to zero 
        BinMap& resetBinDemand();

    protected:
        void copy(BinMap const& rhs);

        std::vector<Bin> m_vBin; ///< bin array, corresponds to Bin::m_index1D
                                ///< the indexing order is locally horizontal and globally vertical 
        index_type m_dimension[2]; ///< 2D dimension in x and y direction of bin map, number of bins in x and y directions  
};

inline BinMap::BinMap(BinType t, BinMap::index_type xNum, BinMap::index_type yNum)
{
    set(t, xNum, yNum);
}
inline BinMap::BinMap(BinMap const& rhs)
{
    copy(rhs);
}
inline BinMap& BinMap::operator=(BinMap const& rhs)
{
    if (this != &rhs)
        copy(rhs);
    return *this;
}
inline void BinMap::copy(BinMap const& rhs)
{
    m_vBin = rhs.m_vBin;
    m_dimension[kX] = rhs.m_dimension[kX];
    m_dimension[kY] = rhs.m_dimension[kY];
}

typedef BinMap::BinMap1DIterator BinMap1DIterator;
typedef BinMap::BinMap1DConstIterator BinMap1DConstIterator;

DREAMPLACE_END_NAMESPACE

#endif
