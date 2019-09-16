/**
 * @file   legality_check.h
 * @author Yibo Lin
 * @date   Oct 2018
 */

#ifndef GPUPLACE_LEGALITY_CHECK_H
#define GPUPLACE_LEGALITY_CHECK_H

#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>
#include "utility/src/Msg.h"

DREAMPLACE_BEGIN_NAMESPACE

/// compare nodes with x center 
/// resolve ambiguity by index 
template <typename T>
struct CompareByNodeXCenter
{
    const T* x; 
    const T* node_size_x; 

    CompareByNodeXCenter(const T* xx, const T* size_x)
        : x(xx)
        , node_size_x(size_x)
    {
    }

    bool operator()(int i, int j) const 
    {
        T xc1 = x[i]+node_size_x[i]/2;
        T xc2 = x[j]+node_size_x[j]/2;
        return (xc1 < xc2) || (xc1 == xc2 && i < j); 
    }
};

template <typename T>
bool legalityCheckKernelCPU(
        const T* init_x, const T* init_y, 
        const T* node_size_x, const T* node_size_y, 
        const T* x, const T* y, 
        T site_width, T row_height, 
        T xl, T yl, T xh, T yh,
        const int num_nodes, 
        const int num_movable_nodes, 
        const int num_filler_nodes
        )
{
    bool legal_flag = true; 
    int num_rows = ceil((yh-yl)/row_height);
    dreamplaceAssert(num_rows > 0); 
    fflush(stdout);
    std::vector<std::vector<int> > row_nodes (num_rows);

    // check node within boundary 
    for (int i = 0; i < num_movable_nodes; ++i)
    {
        T node_xl = x[i]; 
        T node_yl = y[i];
        T node_xh = node_xl+node_size_x[i];
        T node_yh = node_yl+node_size_y[i];
        if (node_xl < xl || node_xh > xh || node_yl < yl || node_yh > yh)
        {
            dreamplacePrint(kDEBUG, "node %d (%g, %g, %g, %g) out of boundary\n", i, node_xl, node_yl, node_xh, node_yh);
            legal_flag = false; 
        }
    }

    // distribute cells to rows 
    for (int i = 0; i < num_nodes-num_filler_nodes; ++i)
    {
        T node_xl = x[i]; 
        T node_yl = y[i];
        //T node_xh = node_xl+node_size_x[i];
        T node_yh = node_yl+node_size_y[i];

        int row_idxl = (node_yl-yl)/row_height; 
        int row_idxh = ceil((node_yh-yl)/row_height)+1;
        row_idxl = std::max(row_idxl, 0); 
        row_idxh = std::min(row_idxh, num_rows); 

        for (int row_id = row_idxl; row_id < row_idxh; ++row_id)
        {
            T row_yl = yl+row_id*row_height; 
            T row_yh = row_yl+row_height; 

            if (node_yl < row_yh && node_yh > row_yl) // overlap with row 
            {
                if (i < num_movable_nodes)
                {
                    if (row_id == row_idxl && node_yl != row_yl) // row alignment failed 
                    {
                        dreamplacePrint(kERROR, "node %d (%g, %g) failed to align to row %d (%g, %g)\n", i, node_xl, node_yl, row_id, row_yl, row_yh);
                        legal_flag = false;
                    }
                    if (floor((node_xl-xl)/site_width)*site_width != node_xl-xl) // site alignment failed
                    {
                        dreamplacePrint(kERROR, "node %d (%g, %g) failed to align to row %d (%g, %g) and site\n", i, node_xl, node_yl, row_id, row_yl, row_yh);
                        legal_flag = false;
                    }
                }
                row_nodes[row_id].push_back(i); 
            }
        }
    }

    // sort cells within rows 
    for (int i = 0; i < num_rows; ++i)
    {
        std::sort(row_nodes[i].begin(), row_nodes[i].end(), CompareByNodeXCenter<T>(x, node_size_x));
    }

    // check overlap 
    for (int i = 0; i < num_rows; ++i)
    {
        for (unsigned int j = 0; j < row_nodes.at(i).size(); ++j)
        {
            if (j > 0)
            {
                int node_id = row_nodes[i][j]; 
                int prev_node_id = row_nodes[i][j-1]; 

                if (node_id < num_movable_nodes || prev_node_id < num_movable_nodes) // ignore two fixed nodes
                {
                    if (x[prev_node_id]+node_size_x[prev_node_id] > x[node_id]) // detect overlap 
                    {
                        dreamplacePrint(kERROR, "row %d, overlap node %d (%g, %g, %g, %g) with node %d (%g, %g, %g, %g)\n", 
                                i, 
                                prev_node_id, x[prev_node_id], y[prev_node_id], x[prev_node_id]+node_size_x[prev_node_id], y[prev_node_id]+node_size_y[prev_node_id], 
                                node_id, x[node_id], y[node_id], x[node_id]+node_size_x[node_id], y[node_id]+node_size_y[node_id]
                              );
                        legal_flag = false; 
                    }
                }
            }
        }
    }

    return legal_flag;
}

template <typename T>
bool legalityCheckSiteMapKernelCPU(
        const T* init_x, const T* init_y, 
        const T* node_size_x, const T* node_size_y, 
        const T* x, const T* y, 
        T site_width, T row_height, 
        T xl, T yl, T xh, T yh,
        const int num_nodes, 
        const int num_movable_nodes, 
        const int num_filler_nodes
        )
{
    int num_rows = ceil((yh-yl))/row_height; 
    int num_sites = ceil((xh-xl)/site_width);
    std::vector<std::vector<unsigned char> > site_map (num_rows, std::vector<unsigned char>(num_sites, 0)); 

    // fixed macros 
    for (int i = num_movable_nodes; i < num_nodes-num_filler_nodes; ++i)
    {
        T node_xl = x[i]; 
        T node_yl = y[i];
        T node_xh = node_xl+node_size_x[i];
        T node_yh = node_yl+node_size_y[i];

        int idxl = (node_xl-xl)/site_width;
        int idxh = ceil((node_xh-xl)/site_width)+1;
        int idyl = (node_yl-yl)/row_height;
        int idyh = ceil((node_yh-yl)/row_height)+1;
        idxl = std::max(idxl, 0); 
        idxh = std::min(idxh, num_sites); 
        idyl = std::max(idyl, 0); 
        idyh = std::min(idyh, num_rows);

        for (int iy = idyl; iy < idyh; ++iy)
        {
            for (int ix = idxl; ix < idxh; ++ix)
            {
                T site_xl = xl+ix*site_width; 
                T site_xh = site_xl+site_width;
                T site_yl = yl+iy*row_height;
                T site_yh = site_yl+row_height;

                if (node_xl < site_xh && node_xh > site_xl 
                        && node_yl < site_yh && node_yh > site_yl) // overlap 
                {
                    site_map[iy][ix] = 255; 
                }
            }
        }
    }

    bool legal_flag = true; 
    // movable cells 
    for (int i = 0; i < num_movable_nodes; ++i)
    {
        T node_xl = x[i]; 
        T node_yl = y[i];
        T node_xh = node_xl+node_size_x[i];
        T node_yh = node_yl+node_size_y[i];

        int idxl = (node_xl-xl)/site_width;
        int idxh = ceil((node_xh-xl)/site_width)+1;
        int idyl = (node_yl-yl)/row_height;
        int idyh = ceil((node_yh-yl)/row_height)+1;
        idxl = std::max(idxl, 0); 
        idxh = std::min(idxh, num_sites); 
        idyl = std::max(idyl, 0); 
        idyh = std::min(idyh, num_rows);

        for (int iy = idyl; iy < idyh; ++iy)
        {
            for (int ix = idxl; ix < idxh; ++ix)
            {
                T site_xl = xl+ix*site_width; 
                T site_xh = site_xl+site_width;
                T site_yl = yl+iy*row_height;
                T site_yh = site_yl+row_height;

                if (node_xl < site_xh && node_xh > site_xl 
                        && node_yl < site_yh && node_yh > site_yl) // overlap 
                {
                    if (site_map[iy][ix])
                    {
                        dreamplacePrint(kERROR, "detect overlap at site (%g, %g, %g, %g) for node %d (%g, %g, %g, %g)\n", 
                                site_xl, site_yl, site_xh, site_yh, 
                                i, 
                                node_xl, node_yl, node_xh, node_yh
                                );
                        legal_flag = false; 
                    }
                    site_map[iy][ix] += 1; 
                }
            }
        }
    }

    return legal_flag; 
}

DREAMPLACE_END_NAMESPACE

#endif
