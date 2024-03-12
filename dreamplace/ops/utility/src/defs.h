/**
 * @file   defs.h
 * @author Yibo Lin
 * @date   Apr 2020
 */
#ifndef _DREAMPLACE_UTILITY_DEFS_H
#define _DREAMPLACE_UTILITY_DEFS_H

#include "utility/src/namespace.h"

DREAMPLACE_BEGIN_NAMESPACE

#ifndef __NVCC__

/// namespace definition to make functions like
/// min/max general between C++ and CUDA
#define DREAMPLACE_STD_NAMESPACE std
/// namespace definition to make functions
/// general between C++ and CUDA
#define DREAMPLACE_HOST_DEVICE

#else

#define DREAMPLACE_STD_NAMESPACE
#define DREAMPLACE_HOST_DEVICE __host__ __device__

#define allocateCUDA(var, size, type)                               \
  {                                                                 \
    cudaError_t status = cudaMalloc(&(var), (size) * sizeof(type)); \
    if (status != cudaSuccess) {                                    \
      dreamplacePrint(kERROR, "cudaMalloc failed for " #var "\n");  \
    }                                                               \
  }

#define destroyCUDA(var)                                         \
  {                                                              \
    cudaError_t status = cudaFree(var);                          \
    if (status != cudaSuccess) {                                 \
      dreamplacePrint(kERROR, "cudaFree failed for " #var "\n"); \
    }                                                            \
  }

#define checkCUDA(status)                                                  \
  {                                                                        \
    dreamplaceAssertMsg(status == cudaSuccess, "CUDA Runtime Error: %s\n", \
                        cudaGetErrorString(status));                       \
  }

#define allocateCopyCUDA(var, rhs, size)                            \
  {                                                                 \
    allocateCUDA(var, size, decltype(*rhs));                        \
    checkCUDA(cudaMemcpy(var, rhs, sizeof(decltype(*rhs)) * (size), \
                         cudaMemcpyHostToDevice));                  \
  }

#define checkCURAND(x) \
  { dreamplaceAssert(x == CURAND_STATUS_SUCCESS); }

#define allocateCopyCPU(var, rhs, size, T)                           \
  {                                                                  \
    var = (T*)malloc(sizeof(T) * (size));                            \
    checkCUDA(cudaMemcpy((void*)var, (void*)rhs, sizeof(T) * (size), \
                         cudaMemcpyDeviceToHost));                   \
  }

#endif

#define destroyCPU(var) \
  { free((void*)var); }

/// A heuristic to detect movable macros.
/// If a cell has a height larger than how many rows, we regard them as movable
/// macros.
#define DUMMY_FIXED_NUM_ROWS 5

DREAMPLACE_END_NAMESPACE

#endif
