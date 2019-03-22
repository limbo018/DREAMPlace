/**
 * @file   merge_bin_cpu.cpp
 * @author Yibo Lin
 * @date   Oct 2018
 */

#include "function_cpu.h"

DREAMPLACE_BEGIN_NAMESPACE

void mergeBinCellsCPU(
        const std::vector<std::vector<int> >& src_bin_cells, 
        int src_num_bins_x, int src_num_bins_y, // dimensions for the src
        std::vector<std::vector<int> >& dst_bin_cells, 
        int dst_num_bins_x, int dst_num_bins_y, // dimensions for the dst
        int scale_ratio_x, int scale_ratio_y // roughly src_num_bins_x/dst_num_bins_x, but may not be exactly the same due to even/odd numbers
        )
{
    for (int i = 0; i < dst_num_bins_x*dst_num_bins_y; i += 1) 
    {
        int dst_bin_id_x = i/dst_num_bins_y; 
        int dst_bin_id_y = i-dst_bin_id_x*dst_num_bins_y; 

        int src_bin_id_x_bgn = dst_bin_id_x*scale_ratio_x; 
        int src_bin_id_y_bgn = dst_bin_id_y*scale_ratio_y; 
        int src_bin_id_x_end = std::min(src_bin_id_x_bgn+scale_ratio_x, src_num_bins_x); 
        int src_bin_id_y_end = std::min(src_bin_id_y_bgn+scale_ratio_y, src_num_bins_y); 

        for (int ix = src_bin_id_x_bgn; ix < src_bin_id_x_end; ++ix)
        {
            for (int iy = src_bin_id_y_bgn; iy < src_bin_id_y_end; ++iy)
            {
                int src_bin_id = ix*src_num_bins_y + iy; 

                const std::vector<int>& src_bin_cell = src_bin_cells.at(src_bin_id);

                dst_bin_cells.at(i).insert(dst_bin_cells.at(i).end(), src_bin_cell.begin(), src_bin_cell.end());
            }
        }
    }
}

DREAMPLACE_END_NAMESPACE
