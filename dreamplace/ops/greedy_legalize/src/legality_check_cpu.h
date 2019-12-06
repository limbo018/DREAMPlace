/**
 * @file   legality_check.h
 * @author Yibo Lin
 * @date   Oct 2018
 */

#ifndef DREAMPLACE_LEGALITY_CHECK_H
#define DREAMPLACE_LEGALITY_CHECK_H

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
        const T* flat_region_boxes, const int* flat_region_boxes_start, const int* node2fence_region_map, 
        const T* x, const T* y, 
        T site_width, T row_height, 
        T xl, T yl, T xh, T yh,
        const int num_nodes, ///< movable and fixed cells 
        const int num_movable_nodes, 
        const int num_regions
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
    for (int i = 0; i < num_nodes; ++i)
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
        auto& nodes_in_row = row_nodes.at(i);
        // using left edge 
        std::sort(nodes_in_row.begin(), nodes_in_row.end(), 
                [&](int node_id1, int node_id2){
                    T x1 = x[node_id1];
                    T x2 = x[node_id2];
                    return x1 < x2 || (x1 == x2 && (node_id1 < node_id2));
                });
        // After sorting by left edge, 
        // there is a special case for fixed cells where 
        // one fixed cell is completely within another in a row. 
        // This will cause failure to detect some overlaps. 
        // We need to remove the "small" fixed cell that is inside another. 
        if (!nodes_in_row.empty())
        {
            std::vector<int> tmp_nodes; 
            tmp_nodes.reserve(nodes_in_row.size());
            tmp_nodes.push_back(nodes_in_row.front()); 
            for (int j = 1, je = nodes_in_row.size(); j < je; ++j)
            {
                int node_id1 = nodes_in_row.at(j-1);
                int node_id2 = nodes_in_row.at(j);
                // two fixed cells 
                if (node_id1 >= num_movable_nodes && node_id2 >= num_movable_nodes)
                {
                    T xl1 = x[node_id1]; 
                    T xl2 = x[node_id2];
                    T width1 = node_size_x[node_id1]; 
                    T width2 = node_size_x[node_id2]; 
                    T xh1 = xl1 + width1; 
                    T xh2 = xl2 + width2; 
                    if (xh1 < xh2)
                    {
                        tmp_nodes.push_back(node_id2);
                    }
                }
                else 
                {
                    tmp_nodes.push_back(node_id2);
                }
            }
            nodes_in_row.swap(tmp_nodes);
        }
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

    // check fence regions 
    for (int i = 0; i < num_movable_nodes; ++i)
    {
        T node_xl = x[i]; 
        T node_yl = y[i];
        T node_xh = node_xl + node_size_x[i];
        T node_yh = node_yl + node_size_y[i];

        int region_id = node2fence_region_map[i]; 
        if (region_id < num_regions)
        {
            int box_bgn = flat_region_boxes_start[region_id];
            int box_end = flat_region_boxes_start[region_id + 1];
            T node_area = (node_xh - node_xl) * (node_yh - node_yl);
            // I assume there is no overlap between boxes of a region 
            // otherwise, preprocessing is required 
            for (int box_id = box_bgn; box_id < box_end; ++box_id)
            {
                int box_offset = box_id*4; 
                T box_xl = flat_region_boxes[box_offset];
                T box_yl = flat_region_boxes[box_offset + 1];
                T box_xh = flat_region_boxes[box_offset + 2];
                T box_yh = flat_region_boxes[box_offset + 3];

                T dx = std::max(std::min(node_xh, box_xh) - std::max(node_xl, box_xl), (T)0); 
                T dy = std::max(std::min(node_yh, box_yh) - std::max(node_yl, box_yl), (T)0); 
                T overlap = dx*dy; 
                if (overlap > 0)
                {
                    node_area -= overlap; 
                }
            }
            if (node_area > 0) // not consumed by boxes within a region 
            {
                dreamplacePrint(kERROR, "node %d (%g, %g, %g, %g), out of fence region %d", 
                        i, node_xl, node_yl, node_xh, node_yh, region_id);
                for (int box_id = box_bgn; box_id < box_end; ++box_id)
                {
                    int box_offset = box_id*4; 
                    T box_xl = flat_region_boxes[box_offset];
                    T box_yl = flat_region_boxes[box_offset + 1];
                    T box_xh = flat_region_boxes[box_offset + 2];
                    T box_yh = flat_region_boxes[box_offset + 3];

                    dreamplacePrint(kNONE, " (%g, %g, %g, %g)", box_xl, box_yl, box_xh, box_yh);
                }
                dreamplacePrint(kNONE, "\n");
                legal_flag = false; 
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
        const int num_movable_nodes
        )
{
    int num_rows = ceil((yh-yl))/row_height; 
    int num_sites = ceil((xh-xl)/site_width);
    std::vector<std::vector<unsigned char> > site_map (num_rows, std::vector<unsigned char>(num_sites, 0)); 

    // fixed macros 
    for (int i = num_movable_nodes; i < num_nodes; ++i)
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
