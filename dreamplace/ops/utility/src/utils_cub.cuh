/**
 * File              : utils_cub.cuh
 * Author            : Yibo Lin <yibolin@pku.edu.cn>
 * Date              : 06.25.2021
 * Last Modified Date: 06.25.2021
 * Last Modified By  : Yibo Lin <yibolin@pku.edu.cn>
 */

#ifndef _DREAMPLACE_UTILITY_UTILS_CUB_CUH
#define _DREAMPLACE_UTILITY_UTILS_CUB_CUH

#include "utility/src/namespace.h"

// include cub in a safe manner
// For CUDA 11+, we cannot use CUB_NS_PREFIX/CUB_NS_POSTFIX because
// the newer CUB uses cuda::std namespace which cannot be wrapped
#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 11)
  // CUDA 11+: Use system CUB without namespace wrapping
  #include <cub/cub.cuh>
  // Create an alias in DreamPlace namespace for compatibility
  namespace DREAMPLACE_NAMESPACE {
    namespace cub = ::cub;
  }
#else
  // CUDA 10 and below: Use bundled CUB with namespace wrapping
  #define CUB_NS_PREFIX namespace DREAMPLACE_NAMESPACE {
  #define CUB_NS_POSTFIX }
  #define CUB_NS_QUALIFIER DREAMPLACE_NAMESPACE::cub
  #include "cub/cub.cuh"
  #undef CUB_NS_QUALIFIER
  #undef CUB_NS_POSTFIX
  #undef CUB_NS_PREFIX
#endif

#endif
