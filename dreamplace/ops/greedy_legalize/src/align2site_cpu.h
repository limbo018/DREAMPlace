/**
 * @file   align2site_cpu.h
 * @author Yibo Lin
 * @date   Oct 2018
 */
#ifndef GPUPLACE_LEGALIZE_ALIGN2SITE_CPU_H
#define GPUPLACE_LEGALIZE_ALIGN2SITE_CPU_H

#include "utility/src/Msg.h"

DREAMPLACE_BEGIN_NAMESPACE

/// @brief align nodes to sites within the boundary 
template <typename T>
void align2SiteCPU(
        const T* node_size_x, 
        T* x, 
        const T xl, const T xh, 
        const T site_width, 
        const int num_movable_nodes 
        )
{
    for (int i = 0; i < num_movable_nodes; ++i)
    {
        x[i] = std::max(std::min(x[i], xh-node_size_x[i]), xl);
        x[i] = floor((x[i]-xl)/site_width)*site_width+xl; 
    }
}

DREAMPLACE_END_NAMESPACE

#endif
