/**
 * @file   abacus_legalize_cpu.h
 * @author Yibo Lin
 * @date   Oct 2018
 */
#ifndef GPUPLACE_ABACUS_LEGALIZE_CPU_H
#define GPUPLACE_ABACUS_LEGALIZE_CPU_H

#include <stdio.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <limits.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include "compare_cpu.h"
#include "abacus_place_row_cpu.h"
#include "align2site_cpu.h"

template <typename T>
void abacusLegalizationCPU(
        const T* init_x, const T* init_y, 
        const T* node_size_x, const T* node_size_y, 
        T* x, T* y, 
        const T xl, const T yl, const T xh, const T yh, 
        const T site_width, const T row_height, 
        int num_bins_x, int num_bins_y, 
        const int num_nodes, 
        const int num_movable_nodes, 
        const int num_filler_nodes
        )
{
    // adjust bin sizes 
    T bin_size_x = (xh-xl)/num_bins_x; 
    T bin_size_y = row_height; 
    //num_bins_x = ceil((xh-xl)/bin_size_x);
    num_bins_y = ceil((yh-yl)/bin_size_y);

    // include both movable and fixed nodes 
    std::vector<std::vector<int> > bin_cells (num_bins_x*num_bins_y); 
    // distribute cells to bins 
    distributeMovableAndFixedCells2BinsCPU(
            x, y, 
            node_size_x, node_size_y, 
            bin_size_x, bin_size_y, 
            xl, yl, xh, yh, 
            num_bins_x, num_bins_y, 
            num_nodes, num_movable_nodes, num_filler_nodes, 
            bin_cells
            );
    std::vector<std::vector<AbacusCluster<T> > > bin_clusters (num_bins_x*num_bins_y);
    for (unsigned int i = 0; i < bin_cells.size(); ++i)
    {
        bin_clusters[i].resize(bin_cells[i].size()); 
    }

    abacusLegalizeRowCPU(
            init_x, 
            node_size_x, node_size_y, 
            x, 
            xl, xh, 
            bin_size_x, bin_size_y, 
            num_bins_x, num_bins_y,
            num_nodes, 
            num_movable_nodes, 
            num_filler_nodes, 
            bin_cells, 
            bin_clusters
            );
    // need to align nodes to sites 
    align2SiteCPU(
            node_size_x, 
            x, 
            xl, xh, 
            site_width, 
            num_movable_nodes 
            );
}

#endif
