/**
 * @file   diamond_search.h
 * @author Yibo Lin
 * @date   Jan 2019
 */
#ifndef _DREAMPLACE_GLOBAL_MOVE_DIAMOND_SEARCH_H
#define _DREAMPLACE_GLOBAL_MOVE_DIAMOND_SEARCH_H

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <type_traits>
#include <vector>
#include "utility/src/msg.h"

DREAMPLACE_BEGIN_NAMESPACE

/// @brief grid index
template <typename T>
struct GridIndex {
  T ir;  ///< row index
  T ic;  ///< column index

  GridIndex()
      : ir(std::numeric_limits<T>::max()), ic(std::numeric_limits<T>::max()) {}

  GridIndex(T r, T c) : ir(r), ic(c) {}

  T manhattan_distance(const GridIndex& rhs) const {
    return fabs(ir - rhs.ir) + fabs(ic - rhs.ic);
  }
  double angle(const GridIndex& rhs) const {
    T dx = ir - rhs.ir;
    T dy = ic - rhs.ic;
    return atan2(dy, dx);
  }
};

/// @brief compare grid (row index, column index) by its manhattan distance to a
/// target grid
template <typename T>
struct CompareGridByDistance2Target {
  GridIndex<T> target;
  CompareGridByDistance2Target(const GridIndex<T>& g) : target(g) {}
  bool operator()(const GridIndex<T>& g1, const GridIndex<T>& g2) const {
    T d1 = g1.manhattan_distance(target);
    double angle1 = g1.angle(target);
    T d2 = g2.manhattan_distance(target);
    double angle2 = g2.angle(target);
    return d1 < d2 || (d1 == d2 && (angle1 < angle2));
  }
};

/// @brief kernel to generate the sequence for diamond search
/// @tparam the template must be a signed integer
/// @param num_rows number of rows
/// @param num_cols number of columns
/// @return the sequence in order from small distance to the center grid to
/// large
template <typename T>
std::vector<GridIndex<T> > diamond_search_sequence_kernel(T num_rows,
                                                          T num_cols) {
  //// 2D grid map in row major
  //// each element is the (row index, column index)
  // std::vector<GridIndex<T> > grid_map (num_rows*num_cols, GridIndex<T>(0,
  // 0));  for (T ir = 0; ir < num_rows; ++ir)
  //{
  //    for (T ic = 0; ic < num_cols; ++ic)
  //    {
  //        grid_map[ir*num_cols+ic] = GridIndex<T>(-(T)num_rows/2+ir,
  //        -(T)num_cols/2+ic);
  //    }
  //}

  //// sort from small distance to large
  // std::sort(grid_map.begin(), grid_map.end(),
  // CompareGridByDistance2Target<T>(GridIndex<T>(0, 0)));

  // directly generate diamond shape grids
  // in clock-wise direction
  // the sequence covers the following shape
  //     1
  //    111
  //   11111
  //    111
  //     1
  std::vector<GridIndex<T> > grid_map;
  grid_map.reserve(num_rows * num_cols / 2);
  T max_sum = std::min(num_rows, num_cols) / 2;
  grid_map.push_back(GridIndex<T>(0, 0));
  for (T sum = 1; sum <= max_sum; ++sum) {
    // y > 0, x [-sum, sum]
    for (T ir = -sum; ir < sum; ++ir) {
      grid_map.push_back(GridIndex<T>(ir, sum - std::abs(ir)));
    }
    // y < 0, x [sum, -sum]
    for (T ir = sum; ir > -sum; --ir) {
      grid_map.push_back(GridIndex<T>(ir, -(sum - std::abs(ir))));
    }
  }

  return grid_map;
}

/// @brief top API to generate the sequence for diamond search
/// @param num_rows number of rows
/// @param num_cols number of columns
/// @return the sequence in order from small distance to the center grid to
/// large
template <typename T>
std::vector<GridIndex<typename std::make_signed<T>::type> >
diamond_search_sequence(T num_rows, T num_cols) {
  return diamond_search_sequence_kernel<typename std::make_signed<T>::type>(
      num_rows, num_cols);
}

template <typename T>
void diamond_search_print(const std::vector<GridIndex<T> >& grid_sequence) {
  unsigned int sum = 0;
  unsigned int count = 0;
  GridIndex<T> target(0, 0);
  printf("[0] ");
  for (typename std::vector<GridIndex<T> >::const_iterator
           it = grid_sequence.begin();
       it != grid_sequence.end(); ++it, ++count) {
    T distance = it->manhattan_distance(target);
    if (sum != distance) {
      sum = distance;
      printf("\n[%u] ", count);
    }
    printf("(%d,%d) ", it->ir, it->ic);
  }
  printf("\n");
}

DREAMPLACE_END_NAMESPACE

#endif
