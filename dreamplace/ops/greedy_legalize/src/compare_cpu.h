/**
 * @file   compare_cpu.h
 * @author Yibo Lin
 * @date   Oct 2018
 */

#ifndef DREAMPLACE_COMPARE_CPU_H
#define DREAMPLACE_COMPARE_CPU_H

#include "blank.h"
#include "utility/src/utils.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
struct CompareByBlankYLXLCPU {
  CompareByBlankYLXLCPU() {}

  bool operator()(const Blank<T>& blank1, const Blank<T>& blank2) const {
    return blank1.yl < blank2.yl ||
           (blank1.yl == blank2.yl && blank1.xl < blank2.xl);
  }
};

template <typename T>
struct CompareByNodeWidthCPU {
  CompareByNodeWidthCPU(const T* size_x) : node_size_x(size_x) {}

  bool operator()(const int i, const int j) const {
    T wi = node_size_x[i];
    T wj = node_size_x[j];
    return wi < wj || (wi == wj && i < j);
  }

  const T* node_size_x;
};

template <typename T>
struct CompareByNodeCenterXCPU {
  CompareByNodeCenterXCPU(const T* x, const T* size_x)
      : node_x(x), node_size_x(size_x) {}

  bool operator()(const int i, const int j) const {
    T xi = node_x[i] + node_size_x[i] / 2;
    T xj = node_x[j] + node_size_x[j] / 2;
    return xi < xj || (xi == xj && i < j);
  }

  const T* node_x;       ///< xl
  const T* node_size_x;  ///< width
};

template <typename T>
struct CompareByNodeNTUPlaceCostCPU {
  CompareByNodeNTUPlaceCostCPU(const T* x, const T* y, const T* size_x,
                               const T* size_y)
      : node_x(x), node_y(y), node_size_x(size_x), node_size_y(size_y) {}

  bool operator()(const int i, const int j) const {
    T wi = 1000 * (node_x[i] + node_size_x[i] / 2) + node_size_x[i] +
           node_size_y[i];
    T wj = 1000 * (node_x[j] + node_size_x[j] / 2) + node_size_x[j] +
           node_size_y[j];
    return wi < wj || (wi == wj && (node_y[i] < node_y[j] ||
                                    (node_y[i] == node_y[j] && i < j)));
  }

  const T* node_x;
  const T* node_y;
  const T* node_size_x;
  const T* node_size_y;
};

template <typename T>
struct CompareByNodeNTUPlaceCostFromLeftCPU {
  CompareByNodeNTUPlaceCostFromLeftCPU(const T* x, const T* y, const T* size_x,
                                       const T* size_y)
      : node_x(x), node_y(y), node_size_x(size_x), node_size_y(size_y) {}

  bool operator()(const int i, const int j) const {
    T wi = -1000 * (node_x[i] + node_size_x[i] / 2) + node_size_x[i] +
           node_size_y[i];
    T wj = -1000 * (node_x[j] + node_size_x[j] / 2) + node_size_x[j] +
           node_size_y[j];
    return wi < wj || (wi == wj && (node_y[i] > node_y[j] ||
                                    (node_y[i] == node_y[j] && i < j)));
  }

  const T* node_x;
  const T* node_y;
  const T* node_size_x;
  const T* node_size_y;
};

DREAMPLACE_END_NAMESPACE

#endif
