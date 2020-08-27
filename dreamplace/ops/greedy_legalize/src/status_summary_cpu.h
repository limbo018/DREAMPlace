/**
 * @file   node_status_summary.h
 * @author Yibo Lin
 * @date   Oct 2018
 */
#ifndef DREAMPLACE_NODE_STATUS_SUMMARY_H
#define DREAMPLACE_NODE_STATUS_SUMMARY_H

#include <vector>
#include <limits>
#include "utility/src/utils.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
void minNodeSizeCPU(
        const std::vector<std::vector<int> >& bin_cells, 
        const T* node_size_x, const T* node_size_y, 
        T site_width, T row_height, 
        int num_bins_x, int num_bins_y, 
        int* min_node_size_x
        )
{
    for (int i = 0; i < num_bins_x*num_bins_y; i += 1) 
    {
        const std::vector<int>& cells = bin_cells.at(i); 
        T min_size_x = std::numeric_limits<int>::max(); 
        for (unsigned int k = 0; k < cells.size(); ++k)
        {
            int node_id = cells.at(k);
            min_size_x = std::min(min_size_x, node_size_x[node_id]);
        }
        if (min_size_x != std::numeric_limits<int>::max())
        {
            *min_node_size_x = std::min(*min_node_size_x, (int)ceilDiv(min_size_x, site_width));
        }
    }
}

DREAMPLACE_END_NAMESPACE

#endif
