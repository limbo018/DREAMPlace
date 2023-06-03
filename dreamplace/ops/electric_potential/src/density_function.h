/**
 * @file   density_function.h
 * @author Yibo Lin
 * @date   Nov 2019
 */

#ifndef DREAMPLACE_ELECTRIC_POTENTIAL_DENSITY_FUNCTION_H
#define DREAMPLACE_ELECTRIC_POTENTIAL_DENSITY_FUNCTION_H

DREAMPLACE_BEGIN_NAMESPACE

#define DEFINE_TRIANGLE_DENSITY_FUNCTION(type) \
    T triangle_density_function(T x, T node_size, T xl, int k, T bin_size) \
    { \
        T bin_xl = xl + k * bin_size; \
        return DREAMPLACE_STD_NAMESPACE::min(x + node_size, bin_xl + bin_size) - DREAMPLACE_STD_NAMESPACE::max(x, bin_xl); \
    } 

#define DEFINE_EXACT_DENSITY_FUNCTION(type) \
    T exact_density_function(T x, T node_size, T bin_center, T bin_size, T l, T h, bool flag) \
    { \
        T bin_xl = bin_center - bin_size / 2; \
        T bin_xh = bin_center + bin_size / 2; \
        if (!flag) \
        { \
            if (bin_xl <= l) \
            { \
                bin_xl = DREAMPLACE_STD_NAMESPACE::min(bin_xl, x); \
            } \
            if (bin_xh >= h) \
            { \
                bin_xh = DREAMPLACE_STD_NAMESPACE::max(bin_xh, x + node_size); \
            } \
        } \
        return DREAMPLACE_STD_NAMESPACE::max(T(0.0), DREAMPLACE_STD_NAMESPACE::min(x + node_size, bin_xh) - DREAMPLACE_STD_NAMESPACE::max(x, bin_xl)); \
    }

DREAMPLACE_END_NAMESPACE

#endif
