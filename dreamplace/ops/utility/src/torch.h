/**
 * @file   torch.h
 * @author Yibo Lin
 * @date   Mar 2019
 * @brief  Required heads from torch 
 */
#ifndef _DREAMPLACE_UTILITY_TORCH_H
#define _DREAMPLACE_UTILITY_TORCH_H

/// As torch may change the header inclusion conventions, it is better to manage it in a consistent way. 
#if TORCH_MAJOR_VERSION >= 1
#include <torch/extension.h>
#else 
#include <torch/torch.h>
#endif
#include <limits>

/// As the API for torch changes, customize a DREAMPlace version to remove warnings 
#define DREAMPLACE_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...)                  \
  [&] {                                                                      \
    const auto& the_type = TYPE;                                             \
    (void)the_type;                                                          \
    at::ScalarType _st = TYPE.scalarType();                                  \
    switch (_st) {                                                           \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__)      \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)        \
      default:                                                               \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");       \
    }                                                                        \
  }()

#endif
