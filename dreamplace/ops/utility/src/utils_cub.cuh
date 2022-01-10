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
#define CUB_NS_PREFIX namespace DREAMPLACE_NAMESPACE {
#define CUB_NS_POSTFIX }
#define CUB_NS_QUALIFIER DREAMPLACE_NAMESPACE::cub
#include "cub/cub.cuh"
#undef CUB_NS_QUALIFIER
#undef CUB_NS_POSTFIX
#undef CUB_NS_PREFIX

#endif
