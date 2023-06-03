/**
 * @file   box.h
 * @author Yibo Lin
 * @date   Jan 2019
 */

#ifndef _DREAMPLACE_UTILITY_BOX_CUH
#define _DREAMPLACE_UTILITY_BOX_CUH

#include <algorithm>
#include "utility/src/limits.h"
#include "utility/src/msg.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
struct Box {
  T xl;
  T yl;
  T xh;
  T yh;

  /// @brief default constructor
  DREAMPLACE_HOST_DEVICE Box() { invalidate(); }

  /// @brief constructor
  /// @param xxl xl
  /// @param yyl yl
  /// @param xxh xh
  /// @param yyh yh
  DREAMPLACE_HOST_DEVICE Box(T xxl, T yyl, T xxh, T yyh)
      : xl(xxl), yl(yyl), xh(xxh), yh(yyh) {}
  /// @brief invalidate the box
  DREAMPLACE_HOST_DEVICE void invalidate() {
    xl = cuda::numeric_limits<T>::max();
    yl = cuda::numeric_limits<T>::max();
    xh = cuda::numeric_limits<T>::lowest();
    yh = cuda::numeric_limits<T>::lowest();
  }
  /// @brief check if the box is valid
  DREAMPLACE_HOST_DEVICE bool valid() const { return (xl <= xh) && (yl <= yh); }
  /// @brief encompass a point
  /// @param x
  /// @param y
  DREAMPLACE_HOST_DEVICE void encompass(T x, T y) {
    xl = DREAMPLACE_STD_NAMESPACE::min(xl, x);
    xh = DREAMPLACE_STD_NAMESPACE::max(xh, x);
    yl = DREAMPLACE_STD_NAMESPACE::min(yl, y);
    yh = DREAMPLACE_STD_NAMESPACE::min(yh, y);
  }
  /// @brief encompass a box
  /// @param xxl xl
  /// @param yyl yl
  /// @param xxh xh
  /// @param yyh yh
  DREAMPLACE_HOST_DEVICE void encompass(T xxl, T yyl, T xxh, T yyh) {
    encompass(xxl, yyl);
    encompass(xxh, yyh);
  }
  /// @brief bloat x direction by 2*dx, and y direction by 2*dy
  /// @param dx
  /// @param dy
  DREAMPLACE_HOST_DEVICE void bloat(T dx, T dy) {
    xl -= dx;
    xh += dx;
    yl -= dy;
    yh += dy;
  }
  /// @brief check if a point is contained by the box
  /// @param x
  /// @param y
  /// @return true if contains
  DREAMPLACE_HOST_DEVICE bool contains(T x, T y) const {
    return xl <= x && x <= xh && yl <= y && y <= yh;
  }
  /// @brief check if a box is contained by the box
  /// @param xxl xl
  /// @param yyl yl
  /// @param xxh xh
  /// @param yyh yh
  /// @return true if contains
  DREAMPLACE_HOST_DEVICE bool contains(T xxl, T yyl, T xxh, T yyh) const {
    return contains(xxl, yyl) && contains(xxh, yyh);
  }
  /// @return width of the box
  DREAMPLACE_HOST_DEVICE T width() const { return xh - xl; }
  /// @return height of the box
  DREAMPLACE_HOST_DEVICE T height() const { return yh - yl; }
  /// @return x coordinate of the center of the box
  DREAMPLACE_HOST_DEVICE T center_x() const { return (xl + xh) / 2; }
  /// @return y coordinate of the center of the box
  DREAMPLACE_HOST_DEVICE T center_y() const { return (yl + yh) / 2; }
  /// @return center manhattan distance to another box
  DREAMPLACE_HOST_DEVICE T center_distance(const Box& rhs) const {
    return fabs(rhs.center_x() - center_x()) +
           fabs(rhs.center_y() - center_y());
  }
  /// @return area of the box
  DREAMPLACE_HOST_DEVICE T area() const { return width() * height(); }
  /// @brief print the box
  DREAMPLACE_HOST_DEVICE void print() const {
    printf("(%g, %g, %g, %g)\n", (double)xl, (double)yl, (double)xh,
           (double)yh);
  }
};

/// @brief simplest box for shared memory
/// Non-empty constructor may result in data race
template <typename T>
struct SharedBox {
  T xl;
  T yl;
  T xh;
  T yh;
};

DREAMPLACE_END_NAMESPACE

#endif
