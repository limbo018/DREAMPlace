/**
 * @file   scaling_function.h
 * @author Yibo Lin
 * @date   Jan 2020
 */

DREAMPLACE_BEGIN_NAMESPACE

#define SCALING_OP maxScaling 

#define DEFINE_AVERAGE_SCALING_FUNCTION(T) \
    T averageScaling( \
            const T* routing_utilization_map,  \
            T xl, T yl,  \
            T bin_size_x, T bin_size_y,  \
            int num_bins_x, int num_bins_y,  \
            int bin_index_xl,  \
            int bin_index_yl,  \
            int bin_index_xh,  \
            int bin_index_yh,  \
            T x_min, T y_min, T x_max, T y_max \
            ) \
    { \
        T area = 0; \
        for (int x = bin_index_xl; x < bin_index_xh; ++x) \
        { \
            for (int y = bin_index_yl; y < bin_index_yh; ++y) \
            { \
                T bin_xl = xl + x * bin_size_x;  \
                T bin_yl = yl + y * bin_size_y;  \
                T bin_xh = bin_xl + bin_size_x;  \
                T bin_yh = bin_yl + bin_size_y;  \
                T overlap = DREAMPLACE_STD_NAMESPACE::max(DREAMPLACE_STD_NAMESPACE::min(x_max, bin_xh) - DREAMPLACE_STD_NAMESPACE::max(x_min, bin_xl), (T)0) * \
                    DREAMPLACE_STD_NAMESPACE::max(DREAMPLACE_STD_NAMESPACE::min(y_max, bin_yh) - DREAMPLACE_STD_NAMESPACE::max(y_min, bin_yl), (T)0); \
                area += overlap * routing_utilization_map[x * num_bins_y + y]; \
            } \
        } \
        return area;  \
    } 

#define DEFINE_MAX_SCALING_FUNCTION(T) \
    T maxScaling( \
            const T* routing_utilization_map,  \
            T xl, T yl,  \
            T bin_size_x, T bin_size_y,  \
            int num_bins_x, int num_bins_y,  \
            int bin_index_xl,  \
            int bin_index_yl,  \
            int bin_index_xh,  \
            int bin_index_yh,  \
            T x_min, T y_min, T x_max, T y_max \
            ) \
    { \
        T util = 0; \
        for (int x = bin_index_xl; x < bin_index_xh; ++x) \
        { \
            for (int y = bin_index_yl; y < bin_index_yh; ++y) \
            { \
                util = DREAMPLACE_STD_NAMESPACE::max(util, routing_utilization_map[x * num_bins_y + y]);  \
            } \
        } \
        T area = (x_max - x_min) * (y_max - y_min);  \
        return area * util;  \
    } 

DREAMPLACE_END_NAMESPACE
