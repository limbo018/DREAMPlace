/*************************************************************************
    > File Name: SubRowNodeMap.h
    > Author: Yibo Lin
    > Mail: yibolin@utexas.edu
    > Created Time: Sun 14 Feb 2016 10:27:34 AM CST
 ************************************************************************/

#ifndef DREAMPLACE_SUBROWNODEMAP_H
#define DREAMPLACE_SUBROWNODEMAP_H

#include <iostream>
#include "Interval.h"
#include "Box.h"
#include "Object.h"
#include "Node.h"
#include "NodeMapElement.h"
#include "SubRowNodeMapHelper.h"

// IntervalHashMap is slightly faster than rtree for large benchmarks 
//#define USE_RTREE 
#define USE_INTERVALHASHMAP 

DREAMPLACE_BEGIN_NAMESPACE

/// forward declaration of data 
class AlgoDB;

#ifdef USE_RTREE
namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;
namespace bgm = boost::geometry::model;
#endif

/// a query map for nodes stored in sub rows, indexed with rtree 
/// Node pointer is directly used for rtree indexing, 
/// which means you have to be careful when update the position of nodes 
///
/// it must be initialized after SubRowMap and BinRowMap are ready 

/// query map 
class SubRowNodeMap 
{
    public:
        typedef Object::coordinate_type coordinate_type;
        typedef Object::index_type index_type;
        typedef Interval<coordinate_type> interval_type;
        typedef NodeMapElement map_element_type; 

#ifdef USE_RTREE
        /// use 1D point in boost to adapt rtree 
        typedef bgi::rtree<map_element_type, bgi::rstar<16> > map_type;
        typedef map_type::const_query_iterator map_const_iterator_type;
#elif defined(USE_INTERVALHASHMAP)
        typedef IntervalHashMap<BookmarkModel<NodeMapElement> > map_type; 
        typedef map_type::const_iterator_type map_const_iterator_type; 
#endif

        /// constructor 
        SubRowNodeMap(AlgoDB const* db = NULL);
        /// copy constructor 
        SubRowNodeMap(SubRowNodeMap const& rhs);
        /// assignment
        SubRowNodeMap& operator=(SubRowNodeMap const& rhs);
        /// destructor
        ~SubRowNodeMap();

        /// member functions 
        /// initialize 
        SubRowNodeMap& set(AlgoDB const* db);

        /// distribute nodes into sub rows, must be called before using the map 
        /// can be called multiple times 
        void setSubRowNodes(bool moveDummyFixedCellOnly);

        /// \return true if an element exists in a sub row 
        bool count(index_type idx, Node const& node) const;
        /// erase an element 
        bool erase(index_type idx, Node const& node);
        /// emplace 
        void insert(index_type idx, Node const& node);

        /// query a range in a sub row 
        /// the left element is not the absolute boundary, it is relaxed to make sure all the nodes with possible overlap are included
        /// \param noBoundary, if true, remove boundary intersects 
        std::pair<map_const_iterator_type, map_const_iterator_type> queryRange(index_type idx, coordinate_type xl, coordinate_type xh, bool noBoundary) const;
        /// directly set \param vNode instead of returning iterators 
        void queryRange(index_type idx, coordinate_type xl, coordinate_type xh, bool noBoundary, std::vector<map_element_type>& vNode) const;
        void queryRange(index_type idx, coordinate_type xl, coordinate_type xh, bool noBoundary, map_type& targetMap) const;
        /// query with a region 
        /// might be slow, as we need to go through all the bins and extract sub rows 
        /// if the sub row is known, directly call subRowMap() and then queryRange() is recommended 
        std::vector<index_type> queryRange(Box<coordinate_type> const& box) const;
        std::vector<index_type> queryRange(coordinate_type xl, coordinate_type yl, coordinate_type xh, coordinate_type yh) const;

        /// helper functions for boost box 
        /// it is better to write accessors with map_const_iterator_type because we can change the internal map_type more easily 
        static inline coordinate_type getMapElementLow(map_const_iterator_type const& it) {return it->inv.low();}
        static inline coordinate_type getMapElementLow(map_element_type const& v) {return v.inv.low();}
        static inline coordinate_type getMapElementHigh(map_const_iterator_type const& it) {return it->inv.high();}
        static inline coordinate_type getMapElementHigh(map_element_type const& v) {return v.inv.high();}
        static inline index_type getMapElementId(map_const_iterator_type const& it) {return it->nodeId;}
        static inline index_type getMapElementId(map_element_type const& v) {return v.nodeId;}
        //Node const& getMapElement(map_const_iterator_type const& it) const; 
        //Node const& getMapElement(map_element_type const& v) const; 

        /// for debug 
        /// print cells in a sub row 
        void print(index_type idx) const;
        /// print range of cells in a sub row 
        void printRange(index_type idx, coordinate_type xl, coordinate_type xh) const;
    protected:
        void copy(SubRowNodeMap const& rhs);

        /// forbiden to public due to security 
        /// \return map of a single sub row 
        map_type& subRowMap(index_type idx) {return m_vMap.at(idx);}
        map_type const& subRowMap(index_type idx) const {return m_vMap.at(idx);}

        AlgoDB const* m_algoDB; ///< AlgoDB has some helper functions useful for indexing 
        std::vector<map_type> m_vMap; ///< same number of sub rows 
        coordinate_type m_avgMovableWidth; ///< average width of movable cells, only used in USE_INTERVALHASHMAP
};

/// the overlap predicate in boost only works for box, it does not give the correct solution for segments 
/// the intersects will count the boundaries, which is not preferred 
struct OverlapPredicate
{
    typedef Object::coordinate_type coordinate_type;
    typedef Interval<coordinate_type> interval_type;

    interval_type inv; 

    OverlapPredicate(coordinate_type xl, coordinate_type xh) : inv(xl, xh) {}
    OverlapPredicate(interval_type const& i) : inv(i) {}
    OverlapPredicate(OverlapPredicate const& rhs) : inv(rhs.inv) {}

    template <typename ValueType>
    inline bool operator()(ValueType const& v) const 
    {
        return !(SubRowNodeMap::getMapElementHigh(v) <= inv.low() 
                || SubRowNodeMap::getMapElementLow(v) >= inv.high());
    }
};

/// remove boundary intersection because bgi::intersects allows intersection of boundaries 
/// I found that it is around 1.5x faster than OverlapPredicate if I combine it with bgi::intersects 
struct NoBoundaryPredicate
{
    typedef Object::coordinate_type coordinate_type;

    coordinate_type xl;
    coordinate_type xh; 

    NoBoundaryPredicate(coordinate_type l, coordinate_type h) : xl(l), xh(h) {}
    NoBoundaryPredicate(NoBoundaryPredicate const& rhs) : xl(rhs.xl), xh(rhs.xh) {}

    /// \return true if it is not boundary intersection 
    template <typename ValueType>
    inline bool operator()(ValueType const& v) const 
    {
        return SubRowNodeMap::getMapElementHigh(v) != xl 
            && SubRowNodeMap::getMapElementLow(v) != xh;
    }

};

DREAMPLACE_END_NAMESPACE

#endif
