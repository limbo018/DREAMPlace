/**
 * @file   LegalizationDB.h
 * @author Yibo Lin
 * @date   Nov 2019
 */

#ifndef _DREAMPLACE_UTILITY_LEGALIZATIONDB_H
#define _DREAMPLACE_UTILITY_LEGALIZATIONDB_H

#include <algorithm>
#include "utility/src/Msg.h"
#include "utility/src/utils.h"

DREAMPLACE_BEGIN_NAMESPACE

/// @brief a wrapper class of required data for legalization
template <typename T>
struct LegalizationDB
{
    typedef T type; 

    const T* init_x;
    const T* init_y; 
    const T* node_size_x;
    const T* node_size_y;
    T* x; 
    T* y; 

    T xl;
    T yl; 
    T xh; 
    T yh;

    T site_width;
    T row_height; 
    T bin_size_x; 
    T bin_size_y; 

    int num_bins_x; 
    int num_bins_y; 
    int num_sites_x; 
    int num_sites_y; 

    int num_nodes; 
    int num_movable_nodes; 

    /// @brief check whether a cell is regarded as movable macros in legalization. 
    /// This is mainly because it is painful to handle these cells for legalization. 
    inline bool is_dummy_fixed(int node_id) const
    {
#ifdef DEBUG
        dreamplaceAssert(node_id < db.num_nodes);
#endif
        T height = node_size_y[node_id];
        return (node_id < num_movable_nodes && height > (row_height*DUMMY_FIXED_NUM_ROWS));
    }
    /// @brief align cell to a row 
    inline T align2row(T y, T height) const 
    {
        T yy = std::max(std::min(y, yh-height), yl);
        yy = floor((yy-yl)/row_height)*row_height+yl; 
        return yy; 
    }
    /// @brief align cell to a site 
    inline T align2site(T x, T width) const 
    {
        T xx = std::max(std::min(x, xh-width), xl);
        xx = floor((xx-xl)/site_width)*site_width+xl; 
        return xx; 
    }
};

DREAMPLACE_END_NAMESPACE

#endif
