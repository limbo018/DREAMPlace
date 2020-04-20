/*************************************************************************
    > File Name: Index.h
    > Author: Yibo Lin
    > Mail: yibolin@utexas.edu
    > Created Time: Sat 27 Jun 2015 04:31:39 PM CDT
 ************************************************************************/

#ifndef DREAMPLACE_INDEX_H
#define DREAMPLACE_INDEX_H

#include "utility/src/msg.h"

DREAMPLACE_BEGIN_NAMESPACE

/// 2D index class in x and y direction
template <typename T>
class Index2D {
 public:
  typedef T index_type;

  ///==== constructors ====
  Index2D(index_type ix = std::numeric_limits<index_type>::max(),
          index_type iy = std::numeric_limits<index_type>::max()) {
    set(ix, iy);
  }
  Index2D(Index2D const& rhs) { copy(rhs); }
  Index2D& operator=(Index2D const& rhs) {
    copy(rhs);
    return *this;
  }

  /// member functions
  Index2D& set(Direction1DType d, index_type v) { m_index[d] = v; }
  Index2D& set(index_type ix, index_type iy) {
    m_index[kX] = ix;
    m_index[kY] = iy;
  }

  index_type get(Direction1DType d) const { return m_index[d]; }
  index_type x() const { return m_index[kX]; }
  index_type y() const { return m_index[kY]; }

  /// compatible to operator []
  index_type const& operator[](Direction1DType d) const { return m_index[d]; }
  index_type& operator[](Direction1DType d) { return m_index[d]; }

 protected:
  void copy(Index2D const& rhs) {
    m_index[kX] = rhs.m_index[kX];
    m_index[kY] = rhs.m_index[kY];
  }

  T m_index[2];
};

DREAMPLACE_END_NAMESPACE

#endif
