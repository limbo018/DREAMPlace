/**
 * @file   blank.h
 * @author Yibo Lin
 * @date   Oct 2018
 */

#ifndef DREAMPLACE_BLANK_H
#define DREAMPLACE_BLANK_H

#include "utility/src/utils.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
struct Interval {
  T xl;
  T xh;

  Interval(T l, T h) : xl(l), xh(h) {}

  void intersect(T rhs_xl, T rhs_xh) {
    xl = std::max(xl, rhs_xl);
    xh = std::min(xh, rhs_xh);
  }
};

template <typename T>
struct Blank {
  T xl;
  T yl;
  T xh;
  T yh;

  void intersect(const Blank& rhs) {
    xl = std::max(xl, rhs.xl);
    xh = std::min(xh, rhs.xh);
    yl = std::max(yl, rhs.yl);
    yh = std::min(yh, rhs.yh);
  }
};

DREAMPLACE_END_NAMESPACE

#endif
