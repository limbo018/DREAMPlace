/**
 * @file   complex_number.h
 * @author Zixuan Jiang, Jiaqi Gu
 * @date   Aug 2019
 * @brief  Complex number for GPU
 */

#ifndef DREAMPLACE_UTILITY_COMPLEXNUMBER_CUH
#define DREAMPLACE_UTILITY_COMPLEXNUMBER_CUH

#include "utility/src/msg.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
struct ComplexType {
  T x;
  T y;
  DREAMPLACE_HOST_DEVICE ComplexType() {
    x = 0;
    y = 0;
  }

  DREAMPLACE_HOST_DEVICE ComplexType(T real, T imag) {
    x = real;
    y = imag;
  }

  DREAMPLACE_HOST_DEVICE ~ComplexType() {}
};

template <typename T>
inline DREAMPLACE_HOST_DEVICE ComplexType<T> complexMul(
    const ComplexType<T> &x, const ComplexType<T> &y) {
  ComplexType<T> res;
  res.x = x.x * y.x - x.y * y.y;
  res.y = x.x * y.y + x.y * y.x;
  return res;
}

template <typename T>
inline DREAMPLACE_HOST_DEVICE T RealPartOfMul(const ComplexType<T> &x,
                                              const ComplexType<T> &y) {
  return x.x * y.x - x.y * y.y;
}

template <typename T>
inline DREAMPLACE_HOST_DEVICE T ImaginaryPartOfMul(const ComplexType<T> &x,
                                                   const ComplexType<T> &y) {
  return x.x * y.y + x.y * y.x;
}

template <typename T>
inline DREAMPLACE_HOST_DEVICE ComplexType<T> complexAdd(
    const ComplexType<T> &x, const ComplexType<T> &y) {
  ComplexType<T> res;
  res.x = x.x + y.x;
  res.y = x.y + y.y;
  return res;
}

template <typename T>
inline DREAMPLACE_HOST_DEVICE ComplexType<T> complexSubtract(
    const ComplexType<T> &x, const ComplexType<T> &y) {
  ComplexType<T> res;
  res.x = x.x - y.x;
  res.y = x.y - y.y;
  return res;
}

template <typename T>
inline DREAMPLACE_HOST_DEVICE ComplexType<T> complexConj(
    const ComplexType<T> &x) {
  ComplexType<T> res;
  res.x = x.x;
  res.y = -x.y;
  return res;
}

template <typename T>
inline DREAMPLACE_HOST_DEVICE ComplexType<T> complexMulConj(
    const ComplexType<T> &x, const ComplexType<T> &y) {
  ComplexType<T> res;
  res.x = x.x * y.x - x.y * y.y;
  res.y = -(x.x * y.y + x.y * y.x);
  return res;
}

DREAMPLACE_END_NAMESPACE

#endif
