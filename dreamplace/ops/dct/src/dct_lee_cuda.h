/**
 * @file   dct_lee_cuda.h
 * @author Yibo Lin
 * @date   Oct 2018
 */

#ifndef DREAMPLACE_DCT_LEE_CUDA_H
#define DREAMPLACE_DCT_LEE_CUDA_H

#include "utility/src/utils.h"

DREAMPLACE_BEGIN_NAMESPACE

namespace lee {

template <typename TValue>
void precompute_dct_cos(TValue *cos, int N);

template <typename TValue>
void precompute_idct_cos(TValue *cos, int N);

template <typename TValue>
void dct(const TValue *vec, TValue *curr, TValue *next, const TValue *cos,
         int M, int N);

template <typename TValue>
void idct(const TValue *vec, TValue *curr, TValue *next, const TValue *cos,
          int M, int N);

}  // End of namespace lee

DREAMPLACE_END_NAMESPACE

#endif
