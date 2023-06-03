/**
 * @file   SwapCandidate.h
 * @author Yibo Lin
 * @date   Mar 2019
 */
#ifndef _DREAMPLACE_GLOBAL_SWAP_SWAPCANDIDATE_H
#define _DREAMPLACE_GLOBAL_SWAP_SWAPCANDIDATE_H

#include "utility/src/utils.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
struct SwapCandidate {
  T cost;
  T node_xl;
  T node_yl;
  T target_node_xl;
  T target_node_yl;
  int node_id;
  int target_node_id;

  /// @brief constructor
  SwapCandidate() {
    cost = std::numeric_limits<T>::max();
    node_xl = std::numeric_limits<T>::max();
    node_yl = std::numeric_limits<T>::max();
    target_node_xl = std::numeric_limits<T>::max();
    target_node_yl = std::numeric_limits<T>::max();
    node_id = std::numeric_limits<int>::max();
    target_node_id = std::numeric_limits<int>::max();
  }
};

DREAMPLACE_END_NAMESPACE

#endif
