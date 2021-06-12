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
#if TORCH_MAJOR_VERSION >= 1
#include <torch/extension.h>

#if TORCH_MINOR_VERSION >= 3
#define DREAMPLACE_TENSOR_DATA_PTR(TENSOR, TYPE) TENSOR.data_ptr<TYPE>()
#else
#define DREAMPLACE_TENSOR_DATA_PTR(TENSOR, TYPE) TENSOR.data<TYPE>()
#endif
#else
#include <torch/torch.h>
#endif
#include <limits>

#define CHECK_CPU(x) AT_ASSERTM(!x.is_cuda(), #x " must be a tensor on CPU")
#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a tensor on CUDA")

#define CHECK_FLAT_CPU(x)                         \
  AT_ASSERTM(!x.is_cuda() && x.ndimension() == 1, \
             #x " must be a flat tensor on CPU")
#define CHECK_FLAT_CUDA(x)                       \
  AT_ASSERTM(x.is_cuda() && x.ndimension() == 1, \
             #x " must be a flat tensor on CUDA")
#define CHECK_EVEN(x) \
  AT_ASSERTM((x.numel() & 1) == 0, #x "must have even number of elements")
#define CHECK_CONTIGUOUS(x) \
  AT_ASSERTM(x.is_contiguous(), #x "must be contiguous")

/// As the API for torch changes, customize a DREAMPlace version to remove
/// warnings
#if TORCH_MAJOR_VERSION > 1 || (TORCH_MAJOR_VERSION == 1 && TORCH_MINOR_VERSION >= 8)
namespace at{
  static inline Tensor rfft(const Tensor & input, int signal_ndim, c10::optional<bool> normalized = false, c10::optional<bool> onesided = true) {
    if (normalized) {                                             \
      if (signal_ndim == 1)
        return fft_rfft(view_as_complex(input), c10::nullopt, -1, "ortho");
      else if (signal_ndim == 2)
        return fft_rfft2(view_as_complex(input), c10::nullopt, {-2, -1}, "ortho");
      else if (signal_ndim == 3)
        return fft_rfftn(view_as_complex(input), c10::nullopt, c10::nullopt, "ortho");
      else
        TORCH_CHECK_VALUE(false, "Ortho-normalized irfft() has illegal number of dimensions ", std::to_string(signal_ndim));
    }
    else {
      if (signal_ndim == 1)
        return fft_rfft(view_as_complex(input), c10::nullopt, -1, "backward");
      else if (signal_ndim == 2)
        return fft_rfft2(view_as_complex(input), c10::nullopt, {-2, -1}, "backward");
      else if (signal_ndim == 3)
        return fft_rfftn(view_as_complex(input), c10::nullopt, c10::nullopt, "backward");
      else
        TORCH_CHECK_VALUE(false, "Backward-normalized rfft() has illegal number of dimensions ", std::to_string(signal_ndim));
    }
  }
  static inline Tensor irfft(const Tensor & input, int signal_ndim, c10::optional<bool> normalized = false, c10::optional<bool> onesided = true, c10::optional<IntArrayRef> signal_sizes = c10::nullopt) {
    if (normalized) {
      if (signal_ndim == 1) {
        if (signal_sizes)
          return fft_irfft(view_as_complex(input), signal_sizes.value()[0], -1, "ortho");
        else
          return fft_irfft(view_as_complex(input), c10::nullopt, -1, "ortho");
      }
      else if (signal_ndim == 2)
        return fft_irfft2(view_as_complex(input), signal_sizes, {-2, -1}, "ortho");
      else if (signal_ndim == 3)
        return fft_irfftn(view_as_complex(input), signal_sizes, c10::nullopt, "ortho");
      else
        TORCH_CHECK_VALUE(false, "Ortho-normalized irfft() has illegal number of dimensions ", std::to_string(signal_ndim));
    }
    else {
      if (signal_ndim == 1) {
        if (signal_sizes)
          return fft_irfft(view_as_complex(input), signal_sizes.value()[0], -1, "backward");
        else
          return fft_irfft(view_as_complex(input), c10::nullopt, -1, "backward");
      }
      else if (signal_ndim == 2)
        return fft_irfft2(view_as_complex(input), signal_sizes, {-2, -1}, "backward");
      else if (signal_ndim == 3)
        return fft_irfftn(view_as_complex(input), signal_sizes, c10::nullopt, "backward");
      else
        TORCH_CHECK_VALUE(false, "Backward-normalized irfft() has illegal number of dimensions ", std::to_string(signal_ndim));
    }
  }
}
#define DREAMPLACE_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...)             \
  [&] {                                                                 \
    const auto& the_type = TYPE;                                        \
    (void)the_type;                                                     \
    at::ScalarType _st = TYPE.scalarType();                             \
    switch (_st) {                                                      \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Double, double, __VA_ARGS__) \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Float, float, __VA_ARGS__)   \
      default:                                                          \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");  \
    }                                                                   \
  }()
#else
#define DREAMPLACE_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...)             \
  [&] {                                                                 \
    const auto& the_type = TYPE;                                        \
    (void)the_type;                                                     \
    at::ScalarType _st = TYPE.scalarType();                             \
    switch (_st) {                                                      \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__) \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)   \
      default:                                                          \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");  \
    }                                                                   \
  }()
#endif

#endif
