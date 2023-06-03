/**
 * @file   torch.h
 * @author Yibo Lin
 * @date   Mar 2019
 * @brief  Required heads from torch
 */
#ifndef _DREAMPLACE_UTILITY_TORCH_H
#define _DREAMPLACE_UTILITY_TORCH_H

/// As torch may change the header inclusion conventions, it is better to manage
/// it in a consistent way.
#if TORCH_VERSION_MAJOR >= 1
#include <torch/extension.h>

#if TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR < 3

#define DREAMPLACE_TENSOR_DATA_PTR(TENSOR, TYPE) \
  ((TENSOR.defined())? TENSOR.data<TYPE>() : nullptr)
#define DREAMPLACE_TENSOR_SCALARTYPE(TENSOR) TENSOR.type().scalarType()

#else

#define DREAMPLACE_TENSOR_DATA_PTR(TENSOR, TYPE) \
  ((TENSOR.defined())? TENSOR.data_ptr<TYPE>() : nullptr)
#define DREAMPLACE_TENSOR_SCALARTYPE(TENSOR) TENSOR.scalar_type()

#endif

// torch version 1.8 or later 
#if TORCH_VERSION_MAJOR > 1 || (TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR >= 8)

// torch version 1.13 or later 
#if TORCH_VERSION_MAJOR > 1 || (TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR >= 13)

// AT_PRIVATE_CASE_TYPE was recently removed from the public dispatch API (look in the Dispatch.h)
#define AT_PRIVATE_CASE_TYPE(NAME, enum_type, type, ...) \
  case enum_type: {                                      \
    using scalar_t = type;                               \
    return __VA_ARGS__();                                \
  }

#endif

#define DREAMPLACE_PRIVATE_CASE_TYPE(NAME, enum_type, type, ...) \
  AT_PRIVATE_CASE_TYPE(NAME, enum_type, type, __VA_ARGS__)

#else

#define DREAMPLACE_PRIVATE_CASE_TYPE(NAME, enum_type, type, ...) \
  AT_PRIVATE_CASE_TYPE(enum_type, type, __VA_ARGS__)

#endif

#else

#include <torch/torch.h>

#endif

#include <limits>

#define CHECK_CPU(x) AT_ASSERTM(!x.is_cuda(), #x " must be a tensor on CPU")
#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a tensor on CUDA")
#define CHECK_FLAT(x) AT_ASSERTM(x.ndimension() == 1, #x "must be a flat tensor")

#define CHECK_FLAT_CPU(x)                         \
  CHECK_CPU(x);                                   \
  CHECK_FLAT(x); 
#define CHECK_FLAT_CUDA(x)                        \
  CHECK_CUDA(x);                                  \
  CHECK_FLAT(x); 

#define CHECK_EVEN(x) \
  AT_ASSERTM((x.numel() & 1) == 0, #x "must have even number of elements")
#define CHECK_CONTIGUOUS(x) \
  AT_ASSERTM(x.is_contiguous(), #x "must be contiguous")

/// As the API for torch changes, customize a DREAMPlace version to remove
/// warnings

#include "utility/src/torch_fft_api.h"

#define DREAMPLACE_DISPATCH_FLOATING_TYPES(TENSOR, NAME, ...)                         \
  [&] {                                                                               \
    at::ScalarType _st = DREAMPLACE_TENSOR_SCALARTYPE(TENSOR);                        \
    switch (_st) {                                                                    \
      DREAMPLACE_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Double, double, __VA_ARGS__) \
      DREAMPLACE_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Float, float, __VA_ARGS__)   \
      default:                                                                        \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");                \
    }                                                                                 \
  }()

/// I remove the support to Char, since int8_t does not compile for CUDA
/// char does not compile for ATen either
#define DREAMPLACE_DISPATCH_INT_FLOAT_TYPES(TENSOR, NAME, ...)                        \
  [&] {                                                                               \
    at::ScalarType _st = DREAMPLACE_TENSOR_SCALARTYPE(TENSOR);                        \
    switch (_st) {                                                                    \
      DREAMPLACE_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Float, float, __VA_ARGS__)   \
      DREAMPLACE_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Double, double, __VA_ARGS__) \
      DREAMPLACE_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Int, int, __VA_ARGS__)       \
      default:                                                                        \
        AT_ERROR(#NAME, " not implemented for '", at::toString(_st), "'");            \
    }                                                                                 \
  }()

#endif
