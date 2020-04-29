/*************************************************************************
    > File Name: Iterators.h
    > Author: Yibo Lin
    > Mail: yibolin@utexas.edu
    > Created Time: Sun 21 Jun 2015 01:27:50 PM CDT
 ************************************************************************/

#ifndef DREAMPLACE_ITERATORS_H
#define DREAMPLACE_ITERATORS_H

#include "PlaceDB.h"
#include "Index.h"
#include "RowMap.h"

DREAMPLACE_BEGIN_NAMESPACE

/// whether T is a const type  
template <typename T> struct is_constant {const static bool value = false;};

template <typename T>
struct is_constant<const T> {const static bool value = true;};

/// if T1 is const type, set T2 as const type
/// otherwise, still use T2
template <typename T1, typename T2>
struct constant_helper
{
    typedef T2& reference_type;
    typedef T2* pointer_type;
};
template <typename T1, typename T2>
struct constant_helper<const T1, T2>
{
    typedef T2 const& reference_type;
    typedef const T2* pointer_type;
};

template <typename PlaceDBType, typename IteratorTagType>
struct IteratorDeref;

template <typename DBType, typename IteratorTagType>
struct IteratorDimension;

/// database iterator, generic class for all 1D iterators
template <typename PlaceDBType, typename IteratorTagType>
class DBIterator
{
    public:
        typedef PlaceDBType placedb_type;
        typedef IteratorTagType iterator_tag_type;
        typedef typename placedb_type::index_type index_type;
        typedef IteratorDeref<placedb_type, iterator_tag_type> iterator_deref_type;
        typedef typename iterator_deref_type::reference_type reference_type;
        typedef typename iterator_deref_type::pointer_type pointer_type;

        /// constructor 
        DBIterator(index_type id = std::numeric_limits<index_type>::max(), 
                index_type idxl = std::numeric_limits<index_type>::max(), 
                index_type idxh = std::numeric_limits<index_type>::max(), 
                PlaceDBType* db = NULL) : m_index (id), m_range(idxl, idxh), m_db(db) {}
        /// copy constructor
        DBIterator(DBIterator const& rhs) {copy(rhs);}
        template <typename SomePlaceDBType>
        DBIterator(DBIterator<SomePlaceDBType, iterator_tag_type> const& rhs) {copy(rhs);}
        /// assignment
        DBIterator& operator=(DBIterator const& rhs) {copy(rhs); return *this;}
        /// destructor
        virtual ~DBIterator() {}

        DBIterator& operator+=(int offset) {m_index += offset; return *this;}
        DBIterator& operator-=(int offset) {return this->operator+=(-offset);}
        DBIterator& operator++() {return this->operator+=(1);}
        DBIterator& operator--() {return this->operator-=(1);}
        DBIterator operator++(int) const
        {
            DBIterator tmp (*this);
            return (++tmp);
        }
        DBIterator operator--(int) const
        {
            DBIterator tmp (*this);
            return (--tmp);
        }
        DBIterator operator+(int offset) const 
        {
            DBIterator tmp (*this);
            return (tmp+=offset);
        }
        DBIterator operator-(int offset) const {return this->operator+(-offset);}

        bool operator==(DBIterator const& rhs) const {return this->compare(rhs);}
        bool operator!=(DBIterator const& rhs) const {return !this->compare(rhs);}

        /// unsafe dereference 
        reference_type deref() const 
        {
            dreamplaceAssertMsg(inRange(), "index = %u out of range [%u, %u]\n", m_index, m_range.low(), m_range.high());
            return s_deref(*m_db, m_index);
        }

        reference_type operator*() const
        {
            return deref();
        }
        pointer_type operator->() const 
        {
            return &deref();
        }
        bool inRange() const {return m_range.low() <= m_index && m_index < m_range.high();}

        Interval<index_type> range() const {return m_range;}
        PlaceDBType& placeDB() const {return *m_db;}
        index_type index() const {return m_index;}

    protected:
        template <typename SomePlaceDBType>
        void copy(DBIterator<SomePlaceDBType, iterator_tag_type> const& rhs) 
        {
            // only allow conversion iterator -> const_iterator
            // failed to use enable_if like specialization 
            // use static assertion instead 
            dreamplaceStaticAssert<is_constant<placedb_type>::value || !is_constant<SomePlaceDBType>::value>("do not allow conversion from const iterator to iterator");
            m_index = rhs.index();
            m_range = rhs.range();
            m_db = &rhs.placeDB();
        }
        /// \return true if indice are same or both out of range 
        bool compare(DBIterator const& rhs) const
        {
            return m_db == rhs.m_db && (m_index == rhs.m_index || (!inRange() && !rhs.inRange()));
        }

        index_type m_index; ///< current index
        Interval<index_type> m_range; ///< range of index 
        PlaceDBType* m_db;

        static iterator_deref_type s_deref; ///< a static member for dereference to avoid frequent construction 
};

template <typename PlaceDBType, typename IteratorTagType>
typename DBIterator<PlaceDBType, IteratorTagType>::iterator_deref_type DBIterator<PlaceDBType, IteratorTagType>::s_deref;

/// iterator for 2D data structures 
/// in such kind of structures, a 2D map is used 
/// the iterator traverses through 2D map and find the corresponding element 
template <typename DBType, typename IteratorTagType>
class DB2DIterator
{
    public:
        typedef DBType db_type;
        typedef IteratorTagType iterator_tag_type;
        typedef typename db_type::index_type index_type;
        typedef Index2D<index_type> index2d_type;
        typedef IteratorDeref<db_type, iterator_tag_type> iterator_deref_type;
        typedef typename iterator_deref_type::reference_type reference_type;
        typedef typename iterator_deref_type::pointer_type pointer_type;
        /// get dimensions of 2D map 
        /// need to provide two functions to get level-1 and level-2 dimensions 
        typedef IteratorDimension<db_type, iterator_tag_type> iterator_dimension_type;

        /// constructor 
        DB2DIterator(index_type ix = std::numeric_limits<index_type>::max(), 
                index_type iy = std::numeric_limits<index_type>::max(), 
                DBType* db = NULL) : m_index(ix, iy), m_db(db) {}
        /// copy constructor
        DB2DIterator(DB2DIterator const& rhs) {copy(rhs);}
        template <typename SomeDBType>
        DB2DIterator(DB2DIterator<SomeDBType, iterator_tag_type> const& rhs) {copy(rhs);}
        /// assignment
        DB2DIterator& operator=(DB2DIterator const& rhs) {copy(rhs); return *this;}
        /// destructor
        virtual ~DB2DIterator() {}

        /// member functions 
        DB2DIterator& operator++() {return this->increment();}
        DB2DIterator operator++(int) const
        {
            DB2DIterator tmp (*this);
            return (++tmp);
        }

        bool operator==(DB2DIterator const& rhs) const {return this->compare(rhs);}
        bool operator!=(DB2DIterator const& rhs) const {return !this->compare(rhs);}

        /// unsafe dereference 
        reference_type deref() const 
        {
            dreamplaceAssertMsg(inRange(), "index = [%u, %u] out of range [%u, %u]\n", m_index.x(), m_index.y());
            return s_deref(*m_db, m_index);
        }

        reference_type operator*() const
        {
            return deref();
        }
        pointer_type operator->() const 
        {
            return &deref();
        }
        bool inRange() const 
        {
            return m_index[kX] < s_dimension(*m_db, kX, m_index) && m_index[kY] < s_dimension(*m_db, kY, m_index);
        }

        index2d_type const& index() const {return m_index;}
        DBType& db() {return *m_db;}

    protected:
        template <typename SomeDBType>
        void copy(DB2DIterator<SomeDBType, iterator_tag_type> const& rhs) 
        {
            // only allow conversion iterator -> const_iterator
            // failed to use enable_if like specialization 
            // use static assertion instead 
            dreamplaceStaticAssert<is_constant<db_type>::value || !is_constant<SomeDBType>::value>("do not allow conversion from const iterator to iterator");
            m_index = rhs.index();
            m_db = &rhs.db();
        }
        /// \return true if indice are same or both out of range 
        bool compare(DB2DIterator const& rhs) const
        {
            return m_db == rhs.m_db && (m_index == rhs.m_index || (!inRange() && !rhs.inRange()));
        }
        /// self-increment 1 
        /// do nothing if out-of-range 
        DB2DIterator& increment()
        {
            if (m_index[iterator_dimension_type::direct1] < s_dimension(*m_db, iterator_dimension_type::direct1, m_index))
            {
                if (m_index[iterator_dimension_type::direct2]+1 < s_dimension(*m_db, iterator_dimension_type::direct2, m_index))
                    m_index[iterator_dimension_type::direct2] += 1;
                else 
                {
                    m_index[iterator_dimension_type::direct1] += 1;
                    m_index[iterator_dimension_type::direct2] = 0;
                }
            }
            return *this;
        }

        index2d_type m_index; ///< 2D index 
        DBType* m_db; ///< data 

        static iterator_deref_type s_deref; ///< a static member for dereference to avoid frequent construction 
        static iterator_dimension_type s_dimension; ///<  a static member to avoid frequent construction
};

template <typename DBType, typename IteratorTagType>
typename DB2DIterator<DBType, IteratorTagType>::iterator_deref_type DB2DIterator<DBType, IteratorTagType>::s_deref;
template <typename DBType, typename IteratorTagType>
typename DB2DIterator<DBType, IteratorTagType>::iterator_dimension_type DB2DIterator<DBType, IteratorTagType>::s_dimension;

struct MovableNodeIteratorTag {};
struct FixedNodeIteratorTag {};
struct PlaceBlockageIteratorTag {};
struct NonCoreNodeIteratorTag {};
struct IOPinNodeIteratorTag {};
struct CellMacroIteratorTag {};
struct IOPinMacroIteratorTag {};
struct SubRowMap2DIteratorTag {};

typedef DBIterator<PlaceDB, MovableNodeIteratorTag> MovableNodeIterator;
typedef DBIterator<PlaceDB, FixedNodeIteratorTag> FixedNodeIterator;
typedef DBIterator<PlaceDB, PlaceBlockageIteratorTag> PlaceBlockageIterator;
typedef DBIterator<PlaceDB, NonCoreNodeIteratorTag> NonCoreNodeIterator;
typedef DBIterator<PlaceDB, IOPinNodeIteratorTag> IOPinNodeIterator;
typedef DBIterator<PlaceDB, CellMacroIteratorTag> CellMacroIterator;
typedef DBIterator<PlaceDB, IOPinMacroIteratorTag> IOPinMacroIterator;
typedef DB2DIterator<SubRowMap, SubRowMap2DIteratorTag> SubRowMap2DIterator;

typedef DBIterator<const PlaceDB, MovableNodeIteratorTag> MovableNodeConstIterator;
typedef DBIterator<const PlaceDB, FixedNodeIteratorTag> FixedNodeConstIterator;
typedef DBIterator<const PlaceDB, PlaceBlockageIteratorTag> PlaceBlockageConstIterator;
typedef DBIterator<const PlaceDB, NonCoreNodeIteratorTag> NonCoreNodeConstIterator;
typedef DBIterator<const PlaceDB, IOPinNodeIteratorTag> IOPinNodeConstIterator;
typedef DBIterator<const PlaceDB, CellMacroIteratorTag> CellMacroConstIterator;
typedef DBIterator<const PlaceDB, IOPinMacroIteratorTag> IOPinMacroConstIterator;
typedef DB2DIterator<const SubRowMap, SubRowMap2DIteratorTag> SubRowMap2DConstIterator;

/// specialization for different iterators 
template <typename PlaceDBType>
struct IteratorDeref<PlaceDBType, MovableNodeIteratorTag>
{
    typedef typename constant_helper<PlaceDBType, Node>::reference_type reference_type;
    typedef typename constant_helper<PlaceDBType, Node>::pointer_type pointer_type;
    typedef DBIterator<PlaceDBType, MovableNodeIteratorTag> iterator_type;
    typedef typename iterator_type::placedb_type placedb_type; 
    typedef typename iterator_type::index_type index_type;
    inline reference_type operator()(placedb_type& db, index_type index) const 
    {
        return db.nodes().at(db.movableNodeIndices().at(index));
    }
};
template <typename PlaceDBType>
struct IteratorDeref<PlaceDBType, FixedNodeIteratorTag>
{
    typedef typename constant_helper<PlaceDBType, Node>::reference_type reference_type;
    typedef typename constant_helper<PlaceDBType, Node>::pointer_type pointer_type;
    typedef DBIterator<PlaceDBType, FixedNodeIteratorTag> iterator_type;
    typedef typename iterator_type::placedb_type placedb_type; 
    typedef typename iterator_type::index_type index_type;
    inline reference_type operator()(placedb_type& db, index_type index) const 
    {
        return db.nodes().at(db.fixedNodeIndices().at(index));
    }
};
template <typename PlaceDBType>
struct IteratorDeref<PlaceDBType, PlaceBlockageIteratorTag>
{
    typedef typename constant_helper<PlaceDBType, Node>::reference_type reference_type;
    typedef typename constant_helper<PlaceDBType, Node>::pointer_type pointer_type;
    typedef DBIterator<PlaceDBType, PlaceBlockageIteratorTag> iterator_type;
    typedef typename iterator_type::placedb_type placedb_type; 
    typedef typename iterator_type::index_type index_type;
    inline reference_type operator()(placedb_type& db, index_type index) const 
    {
        return db.nodes().at(db.placeBlockageIndices().at(index));
    }
};
template <typename PlaceDBType>
struct IteratorDeref<PlaceDBType, NonCoreNodeIteratorTag>
{
    typedef typename constant_helper<PlaceDBType, Node>::reference_type reference_type;
    typedef typename constant_helper<PlaceDBType, Node>::pointer_type pointer_type;
    typedef DBIterator<PlaceDBType, NonCoreNodeIteratorTag> iterator_type;
    typedef typename iterator_type::placedb_type placedb_type; 
    typedef typename iterator_type::index_type index_type;
    inline reference_type operator()(placedb_type& db, index_type index) const 
    {
        return db.nodes().at(db.nonCoreNodeIndices().at(index));
    }
};
template <typename PlaceDBType>
struct IteratorDeref<PlaceDBType, IOPinNodeIteratorTag>
{
    typedef typename constant_helper<PlaceDBType, Node>::reference_type reference_type;
    typedef typename constant_helper<PlaceDBType, Node>::pointer_type pointer_type;
    typedef DBIterator<PlaceDBType, IOPinNodeIteratorTag> iterator_type;
    typedef typename iterator_type::placedb_type placedb_type; 
    typedef typename iterator_type::index_type index_type;
    inline reference_type operator()(placedb_type& db, index_type index) const 
    {
        return db.nodes().at(index);
    }
};
template <typename PlaceDBType>
struct IteratorDeref<PlaceDBType, CellMacroIteratorTag>
{
    typedef typename constant_helper<PlaceDBType, Node>::reference_type reference_type;
    typedef typename constant_helper<PlaceDBType, Node>::pointer_type pointer_type;
    typedef DBIterator<PlaceDBType, CellMacroIteratorTag> iterator_type;
    typedef typename iterator_type::placedb_type placedb_type; 
    typedef typename iterator_type::index_type index_type;
    inline reference_type operator()(placedb_type& db, index_type index) const 
    {
        return db.macros().at(index);
    }
};
template <typename PlaceDBType>
struct IteratorDeref<PlaceDBType, IOPinMacroIteratorTag>
{
    typedef typename constant_helper<PlaceDBType, Node>::reference_type reference_type;
    typedef typename constant_helper<PlaceDBType, Node>::pointer_type pointer_type;
    typedef DBIterator<PlaceDBType, IOPinMacroIteratorTag> iterator_type;
    typedef typename iterator_type::placedb_type placedb_type; 
    typedef typename iterator_type::index_type index_type;
    inline reference_type operator()(placedb_type& db, index_type index) const 
    {
        return db.macros().at(index);
    }
};
template <typename DBType>
struct IteratorDeref<DBType, SubRowMap2DIteratorTag>
{
    typedef typename constant_helper<DBType, SubRow>::reference_type reference_type;
    typedef typename constant_helper<DBType, SubRow>::pointer_type pointer_type;
    typedef DB2DIterator<DBType, SubRowMap2DIteratorTag> iterator_type;
    typedef typename iterator_type::db_type db_type; 
    typedef typename iterator_type::index2d_type index2d_type;
    inline reference_type operator()(db_type& db, index2d_type const& index) const 
    {
        return db.getSubRow(index[kY], index[kX]);
    }
};

template <typename DBType>
struct IteratorDimension<DBType, SubRowMap2DIteratorTag>
{
    typedef DB2DIterator<DBType, SubRowMap2DIteratorTag> iterator_type;
    typedef typename iterator_type::db_type db_type; 
    typedef typename iterator_type::index_type index_type;
    typedef typename iterator_type::index2d_type index2d_type;
    const static Direction1DType direct1 = kY;
    const static Direction1DType direct2 = kX;
    inline index_type operator()(db_type& db, Direction1DType d, index2d_type const& index) const 
    {
        return (d == kY)? db.getNumRows() : db.getSubRowsByRowIndex(index[kY]).size();
    }
};

DREAMPLACE_END_NAMESPACE

#endif
