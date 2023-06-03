/**
 * File              : math.h
 * Author            : Yibo Lin <yibolin@pku.edu.cn>
 * Date              : 05.05.2020
 * Last Modified Date: 05.05.2020
 * Last Modified By  : Yibo Lin <yibolin@pku.edu.cn>
 */

#ifndef _DREAMPLACE_UTILITY_MATH_H
#define _DREAMPLACE_UTILITY_MATH_H

#include <cmath>
#include <type_traits>
#include "utility/src/defs.h"

DREAMPLACE_BEGIN_NAMESPACE

/// relative error tolerance for floating point floor/ceil  
template <typename T>
struct NumericTolerance {
  static constexpr T rtol = 0.001; 
};

template <>
struct NumericTolerance<float> {
  static constexpr float rtol = 0.005; 
};

template <typename T, typename V>
inline DREAMPLACE_HOST_DEVICE T div(T a, V b) {
  return a / b;
}

/// @brief template specialization for non-integral types
template <typename T, typename V>
inline DREAMPLACE_HOST_DEVICE typename std::enable_if<!std::is_integral<T>::value, T>::type
floorDiv(T a, V b, T rtol = NumericTolerance<T>::rtol) {
  return floor(div(a + rtol * b, b));
}

/// @brief template specialization for integral types
template <typename T>
inline DREAMPLACE_HOST_DEVICE typename std::enable_if<std::is_integral<T>::value, T>::type
floorDiv(T a, T b) {
  return a / b;
}

/// @brief template specialization for non-integral types
template <typename T, typename V>
inline DREAMPLACE_HOST_DEVICE typename std::enable_if<!std::is_integral<T>::value, T>::type
ceilDiv(T a, V b, T rtol = NumericTolerance<T>::rtol) {
  return ceil(div(a - rtol * b, b));
}

/// @brief template specialization for integral types
template <typename T>
inline DREAMPLACE_HOST_DEVICE typename std::enable_if<std::is_integral<T>::value, T>::type
ceilDiv(T a, T b) {
  return (a + b - 1) / b;
}

/// @brief template specialization for non-integral types
template <typename T, typename V>
inline DREAMPLACE_HOST_DEVICE typename std::enable_if<!std::is_integral<T>::value, T>::type
roundDiv(T a, V b) {
  return round(div(a, b));
}

DREAMPLACE_END_NAMESPACE

#endif
