/**
 * @file   dct_lee_cpu.h
 * @author Yibo Lin
 * @date   Oct 2018
 */

#ifndef DREAMPLACE_DCT_LEE_CPU_H
#define DREAMPLACE_DCT_LEE_CPU_H

#include <cmath>
#include <stdexcept>
#include <vector>
#include "utility/src/utils.h"

DREAMPLACE_BEGIN_NAMESPACE

namespace lee {

constexpr double PI = 3.14159265358979323846;

/// Return true if a number is power of 2
template <typename T = unsigned>
inline bool isPowerOf2(T val) {
  return val && (val & (val - 1)) == 0;
}

/// Transpose a row-major matrix with M rows and N columns using block transpose
/// method
template <typename TValue, typename TIndex = unsigned>
inline void transpose(const TValue *in, TValue *out, TIndex M, TIndex N,
                      TIndex blockSize = 16) {
  //#pragma omp parallel for collapse(2) schedule(static)
  for (TIndex j = 0; j < N; j += blockSize) {
    for (TIndex i = 0; i < M; i += blockSize) {
      // Transpose the block beginning at [i, j]
      TIndex xend = std::min(M, i + blockSize);
      TIndex yend = std::min(N, j + blockSize);
      for (TIndex y = j; y < yend; ++y) {
        for (TIndex x = i; x < xend; ++x) {
          out[x + y * M] = in[y + x * N];
        }
      }
    }
  }
}

/// Negate values in odd position of a vector
template <typename TValue, typename TIndex = unsigned>
inline void negateOddEntries(TValue *vec, TIndex N, int num_threads) {
#pragma omp parallel for num_threads(num_threads)
  for (TIndex i = 1; i < N; i += 2) {
    vec[i] = -vec[i];
  }
}

/// Precompute cosine values needed for N-point dct
/// @param  cos  size N - 1 buffer, contains the result after function call
/// @param  N    the length of target dct, must be power of 2
template <typename TValue, typename TIndex = unsigned>
void precompute_dct_cos(TValue *cos, TIndex N) {
  // The input length must be power of 2
  if (!isPowerOf2<TIndex>(N)) {
    throw std::domain_error("Input length is not power of 2.");
  }

  TIndex offset = 0;
  TIndex halfLen = N / 2;
  while (halfLen) {
    TValue phaseStep = 0.5 * PI / halfLen;
    TValue phase = 0.5 * phaseStep;
    for (TIndex i = 0; i < halfLen; ++i) {
      cos[offset + i] = 0.5 / std::cos(phase);
      phase += phaseStep;
    }
    offset += halfLen;
    halfLen /= 2;
  }
}

/// Precompute cosine values needed for N-point idct
/// @param  cos  size N - 1 buffer, contains the result after function call
/// @param  N    the length of target idct, must be power of 2
template <typename TValue, typename TIndex = unsigned>
void precompute_idct_cos(TValue *cos, TIndex N) {
  // The input length must be power of 2
  if (!isPowerOf2<TIndex>(N)) {
    throw std::domain_error("Input length is not power of 2.");
  }

  TIndex offset = 0;
  TIndex halfLen = 1;
  while (halfLen < N) {
    TValue phaseStep = 0.5 * PI / halfLen;
    TValue phase = 0.5 * phaseStep;
    for (TIndex i = 0; i < halfLen; ++i) {
      cos[offset + i] = 0.5 / std::cos(phase);
      phase += phaseStep;
    }
    offset += halfLen;
    halfLen *= 2;
  }
}

/// The implementation of fast Discrete Cosine Transform (DCT) algorithm and its
/// inverse (IDCT) are Lee's algorithms Algorithm reference: A New Algorithm to
/// Compute the Discrete Cosine Transform, by Byeong Gi Lee, 1984
///
/// Lee's algorithm has a recursive structure in nature.
/// Here is a sample recursive implementation:
/// https://www.nayuki.io/page/fast-discrete-cosine-transform-algorithms
///
/// My implementation here is iterative, which is more efficient than the
/// recursive version. Here is a sample iterative implementation:
/// https://www.codeproject.com/Articles/151043/Iterative-Fast-1D-Forvard-DCT

/// Compute y[k] = sum_n=0..N-1 (x[n] * cos((n + 0.5) * k * PI / N)), for k =
/// 0..N-1
///
/// @param  vec   length N sequence to be transformed
/// @param  temp  length 2 * N helping buffer
/// @param  cos   length N - 1, stores cosine values precomputed by function
/// 'precompute_dct_cos'
/// @param  N     length of vec, must be power of 2
template <typename TValue, typename TIndex = unsigned>
inline void dct(TValue *vec, TValue *out, TValue *buf, const TValue *cos,
                TIndex N) {
  // The input length must be power of 2
  if (!isPowerOf2<TIndex>(N)) {
    throw std::domain_error("Input length is not power of 2.");
  }

  // Pointers point to the beginning indices of two adjacent iterations
  TValue *curr = out;
  TValue *next = buf;

  // 'temp' is used to store data of two adjacent iterations
  // Copy 'vec' to the first N element in 'temp'
  std::copy(vec, vec + N, curr);

  // Current bufferfly length and half length
  TIndex len = N;
  TIndex halfLen = len / 2;

  // Iteratively bi-partition sequences into sub-sequences
  TIndex cosOffset = 0;
  while (halfLen) {
    TIndex offset = 0;
    TIndex steps = N / len;
    for (TIndex k = 0; k < steps; ++k) {
      for (TIndex i = 0; i < halfLen; ++i) {
        next[offset + i] = curr[offset + i] + curr[offset + len - i - 1];
        next[offset + halfLen + i] =
            (curr[offset + i] - curr[offset + len - i - 1]) *
            cos[cosOffset + i];
      }
      offset += len;
    }
    std::swap(curr, next);
    cosOffset += halfLen;
    len = halfLen;
    halfLen /= 2;
  }

  // Bottom-up form the final DCT solution
  // Note that the case len = 2 will do nothing, so we start from len = 4
  len = 4;
  halfLen = 2;
  while (halfLen < N) {
    TIndex offset = 0;
    TIndex steps = N / len;
    for (TIndex k = 0; k < steps; ++k) {
      for (TIndex i = 0; i < halfLen - 1; ++i) {
        next[offset + i * 2] = curr[offset + i];
        next[offset + i * 2 + 1] =
            curr[offset + halfLen + i] + curr[offset + halfLen + i + 1];
      }
      next[offset + len - 2] = curr[offset + halfLen - 1];
      next[offset + len - 1] = curr[offset + len - 1];
      offset += len;
    }
    std::swap(curr, next);
    halfLen = len;
    len *= 2;
  }

  // Populate the final results into 'out'
  if (curr != out) {
    std::copy(curr, curr + N, out);
  }
}

/// Compute y[k] = 0.5 * x[0] + sum_n=1..N-1 (x[n] * cos(n * (k + 0.5) * PI /
/// N)), for k = 0..N-1
/// @param  vec   length N sequence to be transformed
/// @param  temp  length 2 * N helping buffer
/// @param  cos   length N - 1, stores cosine values precomputed by function
/// 'precompute_idct_cos'
/// @param  N     length of vec, must be power of 2
template <typename TValue, typename TIndex = unsigned>
inline void idct(TValue *vec, TValue *out, TValue *buf, const TValue *cos,
                 TIndex N) {
  // The input length must be power of 2
  if (!isPowerOf2<TIndex>(N)) {
    throw std::domain_error("Input length is not power of 2.");
  }

  // Pointers point to the beginning indices of two adjacent iterations
  TValue *curr = out;
  TValue *next = buf;

  // This array is used to store date of two adjacent iterations
  // Copy 'vec' to the first N element in 'temp'
  std::copy(vec, vec + N, curr);
  curr[0] /= 2;

  // Current bufferfly length and half length
  TIndex len = N;
  TIndex halfLen = len / 2;

  // Iteratively bi-partition sequences into sub-sequences
  while (halfLen) {
    TIndex offset = 0;
    TIndex steps = N / len;
    for (TIndex k = 0; k < steps; ++k) {
      next[offset] = curr[offset];
      next[offset + halfLen] = curr[offset + 1];
      for (TIndex i = 1; i < halfLen; ++i) {
        next[offset + i] = curr[offset + i * 2];
        next[offset + halfLen + i] =
            curr[offset + i * 2 - 1] + curr[offset + i * 2 + 1];
      }
      offset += len;
    }
    std::swap(curr, next);
    len = halfLen;
    halfLen /= 2;
  }

  // Bottom-up form the final IDCT solution
  len = 2;
  halfLen = 1;
  TIndex cosOffset = 0;
  while (halfLen < N) {
    TIndex offset = 0;
    TIndex steps = N / len;
    for (TIndex k = 0; k < steps; ++k) {
      for (TIndex i = 0; i < halfLen; ++i) {
        TValue g = curr[offset + i];
        TValue h = curr[offset + halfLen + i] * cos[cosOffset + i];
        next[offset + i] = g + h;
        next[offset + len - 1 - i] = g - h;
      }
      offset += len;
    }
    std::swap(curr, next);
    cosOffset += halfLen;
    halfLen = len;
    len *= 2;
  }

  // Populate the final results into 'out'
  if (curr != out) {
    std::copy(curr, curr + N, out);
  }
}

/// Compute batch dct
/// @param  mtx   size M * N row-major matrix to be transformed
/// @param  temp  length 3 * M * N helping buffer, first 2 * M * N is for dct,
/// the last M * N is for matrix transpose
/// @param  cosM  length M - 1, stores cosine values precomputed by function
/// 'precompute_dct_cos' for M-point dct
/// @param  cosN  length N - 1, stores cosine values precomputed by function
/// 'precompute_dct_cos' for N-point dct
/// @param  M     number of rows
/// @param  N     number of columns
template <typename TValue, typename TIndex = unsigned>
inline void dct(TValue *mtx, TValue *out, TValue *buf, const TValue *cos,
                TIndex M, TIndex N, int num_threads) {
#pragma omp parallel for num_threads(num_threads) schedule(static)
  for (TIndex i = 0; i < M; ++i) {
    dct<TValue, TIndex>(mtx + i * N, out + i * N, buf + i * N, cos, N);
  }
}

/// Compute batch idct
/// @param  mtx   size M * N row-major matrix to be transformed
/// @param  temp  length 3 * M * N helping buffer, first 2 * M * N is for dct,
/// the last M * N is for matrix transpose
/// @param  cosM  length M - 1, stores cosine values precomputed by function
/// 'precompute_dct_cos' for M-point dct
/// @param  cosN  length N - 1, stores cosine values precomputed by function
/// 'precompute_dct_cos' for N-point dct
/// @param  M     number of rows
/// @param  N     number of columns
template <typename TValue, typename TIndex = unsigned>
inline void idct(TValue *mtx, TValue *out, TValue *buf, const TValue *cos,
                 TIndex M, TIndex N, int num_threads) {
#pragma omp parallel for num_threads(num_threads) schedule(static)
  for (TIndex i = 0; i < M; ++i) {
    idct<TValue, TIndex>(mtx + i * N, out + i * N, buf + i * N, cos, N);
  }
}

}  // End of namespace lee

DREAMPLACE_END_NAMESPACE

#endif
