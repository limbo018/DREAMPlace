/**
 * @file   density_ops.cuh
 * @author Yibo Lin
 * @date   September 2019
 */
#include "utility/src/Msg.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
struct ExactDensity
{
    __device__ __forceinline__ T operator()(T x, T node_size, T bin_center, T bin_size, T l, T h, bool flag) const
    {
        T bin_xl = bin_center-bin_size/2;
        T bin_xh = bin_center+bin_size/2;
        if (!flag) // only for movable nodes 
        {
            // if a node is out of boundary, count in the nearest bin 
            if (bin_xl <= l) // left most bin 
            {
                bin_xl = min(bin_xl, x); 
            }
            if (bin_xh >= h) // right most bin 
            {
                bin_xh = max(bin_xh, x+node_size); 
            }
        }
        return max(T(0.0), min(x+node_size, bin_xh) - max(x, bin_xl));
    }
};

template <typename T>
struct TriangleDensity
{
    __device__ __forceinline__ T operator()(T x, T node_size, T bin_center, T bin_size) const
    {
        return max(T(0.0), min(x+node_size, bin_center+bin_size/2) - max(x, bin_center-bin_size/2));
    }
};

DREAMPLACE_END_NAMESPACE
