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

#endif
