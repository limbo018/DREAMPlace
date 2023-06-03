/**
 * @file   atomic_ops.h
 * @author Yibo Lin
 * @date   Apr 2020
 */
#include <type_traits>
#include "utility/src/utils.h"

DREAMPLACE_BEGIN_NAMESPACE

/// @brief A class generalized scaled atomic addition for floating point number
/// and integers. For integer, we use it as a fixed point number with the LSB
/// part for fractions.
template <typename T, bool = std::is_integral<T>::value>
struct AtomicAdd {
  typedef T type;

  /// @brief constructor
  AtomicAdd(type = 1) {}

  template <typename V>
  inline void operator()(type* dst, V v) const {
#pragma omp atomic
    *dst += v;
  }
};

/// @brief For atomic addition of fixed point number using integers.
template <typename T>
struct AtomicAdd<T, true> {
  typedef T type;

  type scale_factor;  ///< a scale factor to scale fraction into integer

  /// @brief constructor
  /// @param sf scale factor
  AtomicAdd(type sf = 1) : scale_factor(sf) {}

  template <typename V>
  inline void operator()(type* dst, V v) const {
    type sv = v * scale_factor;
#pragma omp atomic
    *dst += sv;
  }
};

/// @brief Perform a += b * scale_factor
template <typename T, typename V, typename W>
void scaleAdd(T* dst, const V* src, W scale_factor, int n, int num_threads) {
#pragma omp parallel for num_threads(num_threads)
  for (int i = 0; i < n; ++i) {
    dst[i] += src[i] * scale_factor;
  }
}

DREAMPLACE_END_NAMESPACE
