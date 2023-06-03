/**
 * @file   dct_lee_cuda_kernel.h
 * @author Yibo Lin
 * @date   Oct 2018
 */

#ifndef DREAMPLACE_DCT_LEE_CUDA_KERNEL_H
#define DREAMPLACE_DCT_LEE_CUDA_KERNEL_H

#include "utility/src/utils.cuh"

DREAMPLACE_BEGIN_NAMESPACE

namespace lee {

template <typename TValue, typename TIndex>
__global__ void computeDctForward(const TValue* curr, TValue* next,
                                  const TValue* cos, TIndex M, TIndex N,
                                  TIndex len, TIndex halfLen,
                                  TIndex cosOffset) {
  // for (TIndex thread_id = blockIdx.x * blockDim.x + threadIdx.x; thread_id <
  // M*N/2; thread_id += blockDim.x * gridDim.x)
  //{
  //    TIndex halfN = N/2;
  //    TIndex batch_id = thread_id / halfN;
  //    TIndex rest = thread_id - batch_id*halfN;
  //    TIndex k = rest / halfLen;
  //    TIndex i = rest - k*halfLen;
  //    TIndex batch_offset = batch_id*N;
  //    TIndex offset = batch_offset + k*len;

  //    next[offset + i] = curr[offset + i] + curr[offset + len - i - 1];
  //    next[offset + halfLen + i] = (curr[offset + i] - curr[offset + len -i -
  //    1]) * cos[cosOffset + i];
  //}
  TIndex halfN = (N >> 1);
  TIndex halfMN = M * halfN;
  // for (TIndex thread_id = halfMN_by_gridDim*blockIdx.x + threadIdx.x;
  // thread_id < halfMN_by_gridDim*(blockIdx.x+1); thread_id += blockDim.x)
  for (TIndex thread_id = blockIdx.x * blockDim.x + threadIdx.x;
       thread_id < halfMN; thread_id += blockDim.x * gridDim.x) {
    TIndex rest = thread_id & (halfN - 1);
    TIndex i = rest & (halfLen - 1);
    TIndex offset = (thread_id - i) * 2;

    next[offset + i] = curr[offset + i] + curr[offset + len - i - 1];
    // next[offset + i + halfLen] = (curr[offset + i] - curr[offset + len - i -
    // 1]) * cos[cosOffset + i];
  }
  // for (TIndex thread_id = halfMN_by_gridDim*blockIdx.x + threadIdx.x;
  // thread_id < halfMN_by_gridDim*(blockIdx.x+1); thread_id += blockDim.x)
  for (TIndex thread_id = blockIdx.x * blockDim.x + threadIdx.x;
       thread_id < halfMN; thread_id += blockDim.x * gridDim.x) {
    TIndex rest = thread_id & (halfN - 1);
    TIndex i = rest & (halfLen - 1);
    TIndex offset = (thread_id - i) * 2;

    // next[offset + i] = curr[offset + i] + curr[offset + len - i - 1];
    next[offset + i + halfLen] =
        (curr[offset + i] - curr[offset + len - i - 1]) * cos[cosOffset + i];
  }
}

template <typename TValue, typename TIndex>
__global__ void computeDctBackward(const TValue* curr, TValue* next, TIndex M,
                                   TIndex N, TIndex len, TIndex halfLen) {
  // for (TIndex thread_id = blockIdx.x * blockDim.x + threadIdx.x; thread_id <
  // M*N/2; thread_id += blockDim.x * gridDim.x)
  //{
  //    TIndex halfN = N/2;
  //    TIndex batch_id = thread_id / halfN;
  //    TIndex rest = thread_id - batch_id*halfN;
  //    TIndex k = rest / halfLen;
  //    TIndex i = rest - k*halfLen;
  //    TIndex batch_offset = batch_id*N;
  //    TIndex offset = batch_offset + k*len;

  //    if (i+1 == halfLen)
  //    {
  //        next[offset + len - 2] = curr[offset + halfLen - 1];
  //        next[offset + len - 1] = curr[offset + len - 1];
  //    }
  //    else
  //    {
  //        next[offset + i * 2] = curr[offset + i];
  //        next[offset + i * 2 + 1] = curr[offset + halfLen + i] + curr[offset
  //        + halfLen + i + 1];
  //    }
  //    //next[offset + i] = (i&1)? curr[offset + halfLen + i/2] + curr[offset +
  //    halfLen + i/2 + 1*(i+1 < len)]*(i+1 < len) : curr[offset + i/2];
  //}
  TIndex halfN = (N >> 1);
  TIndex halfMN = M * halfN;
  // TIndex halfMN_by_gridDim = halfMN/gridDim.x;
  // for (TIndex thread_id = halfMN_by_gridDim*blockIdx.x + threadIdx.x;
  // thread_id < halfMN_by_gridDim*(blockIdx.x+1); thread_id += blockDim.x)
  for (TIndex thread_id = blockIdx.x * blockDim.x + threadIdx.x;
       thread_id < halfMN; thread_id += blockDim.x * gridDim.x) {
    TIndex rest = thread_id & (halfN - 1);
    TIndex i = rest & (halfLen - 1);
    TIndex offset = (thread_id - i) * 2;

    next[offset + i * 2] = curr[offset + i];
    next[offset + i * 2 + 1] =
        (i + 1 == halfLen)
            ? curr[offset + len - 1]
            : curr[offset + halfLen + i] + curr[offset + halfLen + i + 1];
  }
}

template <typename TValue, typename TIndex>
__global__ void computeIdctScale0(TValue* curr, TIndex M, TIndex N) {
  for (TIndex thread_id = blockIdx.x * blockDim.x + threadIdx.x; thread_id < M;
       thread_id += blockDim.x * gridDim.x) {
    curr[thread_id * N] *= 0.5;
  }
}

template <typename TValue, typename TIndex>
__global__ void computeIdctForward(const TValue* curr, TValue* next, TIndex M,
                                   TIndex N, TIndex len, TIndex halfLen) {
  for (TIndex thread_id = blockIdx.x * blockDim.x + threadIdx.x;
       thread_id < M * N / 2; thread_id += blockDim.x * gridDim.x) {
    TIndex halfN = N / 2;
    TIndex batch_id = thread_id / halfN;
    TIndex rest = thread_id - batch_id * halfN;
    TIndex k = rest / halfLen;
    TIndex i = rest - k * halfLen;
    TIndex batch_offset = batch_id * N;
    TIndex offset = batch_offset + k * len;

    if (i == 0) {
      next[offset] = curr[offset];
      next[offset + halfLen] = curr[offset + 1];
    } else {
      next[offset + i] = curr[offset + i * 2];
      next[offset + halfLen + i] =
          curr[offset + i * 2 - 1] + curr[offset + i * 2 + 1];
    }
  }
}

template <typename TValue, typename TIndex>
__global__ void ComputeIdctBackward(const TValue* curr, TValue* next,
                                    const TValue* cos, TIndex M, TIndex N,
                                    TIndex len, TIndex halfLen,
                                    TIndex cosOffset) {
  for (TIndex thread_id = blockIdx.x * blockDim.x + threadIdx.x;
       thread_id < M * N / 2; thread_id += blockDim.x * gridDim.x) {
    TIndex halfN = N / 2;
    TIndex batch_id = thread_id / halfN;
    TIndex rest = thread_id - batch_id * halfN;
    TIndex k = rest / halfLen;
    TIndex i = rest - k * halfLen;
    TIndex batch_offset = batch_id * N;
    TIndex offset = batch_offset + k * len;

    TValue g = curr[offset + i];
    TValue h = curr[offset + halfLen + i] * cos[cosOffset + i];
    next[offset + i] = g + h;
    next[offset + len - 1 - i] = g - h;
  }
}

}  // End of namespace lee

DREAMPLACE_END_NAMESPACE

#endif
