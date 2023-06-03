/**
 * @file   greedy_legalize_cpu.cpp
 * @author Yibo Lin
 * @date   Oct 2018
 */
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include "function_cpu.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
int greedyLegalizationCPU(
        const LegalizationDB<T>& db, 
        const T* init_x, const T* init_y, 
        const T* node_size_x, const T* node_size_y, 
        T* x, T* y, 
        const T xl, const T yl, const T xh, const T yh, 
        const T site_width, const T row_height, 
        int num_bins_x, int num_bins_y, 
        const int num_nodes, 
        const int num_movable_nodes
        )
{
    float milliseconds = 0; 

    // first from right to left 
    // then from left to right 
    for (int i = 0; i < 2; ++i)
    {
        num_bins_x = 1; 
        num_bins_y = 1;
        // adjust bin sizes 
        T bin_size_x = div((xh-xl), num_bins_x); 
        //bin_size_x = std::max(floorDiv(bin_size_x, site_width)*site_width, site_width); 
        T bin_size_y = div((yh-yl), num_bins_y); 
        bin_size_y = std::max((T)(ceilDiv(bin_size_y, row_height)*row_height), (T)row_height);

        //num_bins_x = ceilDiv((xh-xl), bin_size_x);
        num_bins_y = ceilDiv((yh-yl), bin_size_y);

        // bin dimension in y direction for blanks is different from that for cells 
        T blank_bin_size_y = row_height; 
        int blank_num_bins_y = floorDiv((yh-yl), blank_bin_size_y); 
        dreamplacePrint(kDEBUG, "%s blank_num_bins_y = %d\n", "Standard cell legalization", blank_num_bins_y);

        // allocate bin cells 
        std::vector<std::vector<int> > bin_cells (num_bins_x*num_bins_y); 
        std::vector<std::vector<int> > bin_cells_copy (num_bins_x*num_bins_y); 

        // distribute cells to bins 
        distributeCells2BinsCPU(
                db, 
                x, y, 
                node_size_x, node_size_y, 
                bin_size_x, bin_size_y, 
                xl, yl, xh, yh, 
                num_bins_x, num_bins_y, 
                num_nodes, num_movable_nodes, 
                bin_cells
                );


        // allocate bin fixed cells 
        std::vector<std::vector<int> > bin_fixed_cells (num_bins_x*num_bins_y); 

        // distribute fixed cells to bins 
        distributeFixedCells2BinsCPU(
                db, 
                init_x, init_y, 
                node_size_x, node_size_y, 
                bin_size_x, bin_size_y, 
                xl, yl, xh, yh, 
                num_bins_x, num_bins_y, 
                num_nodes, num_movable_nodes, 
                bin_fixed_cells
                ); 

        // allocate bin blanks 
        std::vector<std::vector<Blank<T> > > bin_blanks (num_bins_x*blank_num_bins_y); 
        std::vector<std::vector<Blank<T> > > bin_blanks_copy (num_bins_x*blank_num_bins_y); 

        // distribute blanks to bins 
        distributeBlanks2BinsCPU(
                init_x, init_y, 
                node_size_x, node_size_y, 
                bin_fixed_cells, 
                bin_size_x, bin_size_y, blank_bin_size_y, 
                xl, yl, xh, yh, 
                site_width, row_height, 
                num_bins_x, num_bins_y, blank_num_bins_y, 
                bin_blanks
                ); 

        int num_unplaced_cells_host;
        // minimum width in sites 
        int min_unplaced_node_size_x_host;
        int num_iters = floor(log((T)std::min(num_bins_x, num_bins_y))/log(2.0))+1;
        for (int iter = 0; iter < num_iters; ++iter)
        {
            dreamplacePrint(kDEBUG, "%s iteration %d with %dx%d bins\n", "Standard cell legalization", iter, num_bins_x, num_bins_y);
            num_unplaced_cells_host = 0; 
            //countBinObjects(bin_cells);
            dreamplacePrint(kDEBUG, "%s #bin_blanks\n", "Standard cell legalization");
            countBinObjects(bin_blanks);

            milliseconds = clock(); 
            legalizeBinCPU<T>(
                    init_x, init_y, 
                    node_size_x, node_size_y, 
                    bin_blanks, // blanks in each bin, sorted from low to high, left to right 
                    bin_cells, // unplaced cells in each bin 
                    x, y, 
                    num_bins_x, num_bins_y, blank_num_bins_y, 
                    bin_size_x, bin_size_y, blank_bin_size_y, 
                    site_width, row_height, 
                    xl, yl, xh, yh,
                    0.5, 
                    4.0, 
                    i%2,  
                    &num_unplaced_cells_host
                    );
            milliseconds = (clock()-milliseconds)/CLOCKS_PER_SEC*1000; 
            dreamplacePrint(kINFO, "%s legalizeBin takes %.3f ms\n", "Standard cell legalization", milliseconds);

            dreamplacePrint(kDEBUG, "%s num_unplaced_cells = %d\n", "Standard cell legalization", num_unplaced_cells_host); 
            //countBinObjects(bin_cells);
            //countBinObjects(bin_blanks);

            if (num_unplaced_cells_host == 0 || iter+1 == num_iters)
            {
                break; 
            }

            // compute minimum size of unplaced cells 
            milliseconds = clock(); 
            min_unplaced_node_size_x_host = floorDiv((xh-xl), site_width);
            minNodeSizeCPU(
                    bin_cells, 
                    node_size_x, node_size_y, 
                    site_width, row_height, 
                    num_bins_x, num_bins_y, 
                    &min_unplaced_node_size_x_host
                    );
            milliseconds = (clock()-milliseconds)/CLOCKS_PER_SEC*1000; 
            dreamplacePrint(kINFO, "%s minNodeSize takes %.3f ms\n", "Standard cell legalization", milliseconds);
            dreamplacePrint(kDEBUG, "%s minimum unplaced node_size_x %d sites\n", "Standard cell legalization", min_unplaced_node_size_x_host);

            // ceil(num_bins_x/2), ceil(num_bins_y/2)
            int dst_num_bins_x = (num_bins_x>>1)+(num_bins_x&1); 
            int dst_num_bins_y = (num_bins_y>>1)+(num_bins_y&1); 
            int scale_ratio_x = (num_bins_x == dst_num_bins_x)? 1 : num_bins_x/dst_num_bins_x; 
            int scale_ratio_y = (num_bins_y == dst_num_bins_y)? 1 : num_bins_y/dst_num_bins_y; 

            milliseconds = clock(); 
            resizeBinObjectsCPU(
                    bin_cells_copy, 
                    dst_num_bins_x, dst_num_bins_y
                    );
            mergeBinCellsCPU(
                    bin_cells, 
                    num_bins_x, num_bins_y, // dimensions for the src
                    bin_cells_copy, // ceil(src_num_bins_x/2) * ceil(src_num_bins_y/2)
                    dst_num_bins_x, dst_num_bins_y, 
                    scale_ratio_x, scale_ratio_y
                    );
            milliseconds = (clock()-milliseconds)/CLOCKS_PER_SEC*1000; 
            dreamplacePrint(kDEBUG, "%s mergeBinCells takes %.3f ms\n", "Standard cell legalization", milliseconds);
            milliseconds = clock(); 
            resizeBinObjectsCPU(
                    bin_blanks_copy, 
                    dst_num_bins_x, blank_num_bins_y
                    );
            mergeBinBlanksCPU(
                    bin_blanks, 
                    num_bins_x, blank_num_bins_y, // dimensions for the src
                    bin_blanks_copy, // ceil(src_num_bins_x/2) * ceil(src_num_bins_y/2)
                    dst_num_bins_x, blank_num_bins_y, 
                    scale_ratio_x, 
                    min_unplaced_node_size_x_host*site_width
                    );
            milliseconds = (clock()-milliseconds)/CLOCKS_PER_SEC*1000; 
            dreamplacePrint(kDEBUG, "%s mergeBinBlanks takes %.3f ms\n", "Standard cell legalization", milliseconds);

            // update bin dimensions
            num_bins_x = dst_num_bins_x; 
            num_bins_y = dst_num_bins_y; 

            bin_size_x = bin_size_x*2;
            bin_size_y = bin_size_y*2;

            std::swap(bin_cells, bin_cells_copy); 
            std::swap(bin_blanks, bin_blanks_copy); 
        }
    }

    return 0; 
}

int instantiateGreedyLegalizationCPU(
        const LegalizationDB<float>& db, 
        const float* init_x, const float* init_y, 
        const float* node_size_x, const float* node_size_y, 
        float* x, float* y, 
        const float xl, const float yl, const float xh, const float yh, 
        const float site_width, const float row_height, 
        int num_bins_x, int num_bins_y, 
        const int num_nodes, 
        const int num_movable_nodes
        )
{
    return greedyLegalizationCPU(
            db, 
            init_x, init_y, 
            node_size_x, node_size_y, 
            x, y, 
            xl, yl, xh, yh, 
            site_width, row_height, 
            num_bins_x, num_bins_y, 
            num_nodes, 
            num_movable_nodes
            );
}

int instantiateGreedyLegalizationCPU(
        const LegalizationDB<double>& db, 
        const double* init_x, const double* init_y, 
        const double* node_size_x, const double* node_size_y, 
        double* x, double* y, 
        const double xl, const double yl, const double xh, const double yh, 
        const double site_width, const double row_height, 
        int num_bins_x, int num_bins_y, 
        const int num_nodes, 
        const int num_movable_nodes
        )
{
    return greedyLegalizationCPU(
            db, 
            init_x, init_y, 
            node_size_x, node_size_y, 
            x, y, 
            xl, yl, xh, yh, 
            site_width, row_height, 
            num_bins_x, num_bins_y, 
            num_nodes, 
            num_movable_nodes
            );
}

DREAMPLACE_END_NAMESPACE
