/**
 * @file   bin_assignment_cpu.cpp
 * @author Yibo Lin
 * @date   Oct 2018
 */

#include "function_cpu.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
void assignCells2BinsCPU(
        const int* ordered_nodes, 
        const T* init_x, const T* init_y, 
        const T* node_size_x, const T* node_size_y, 
        T bin_size_x, T bin_size_y, 
        T xl, T yl, T xh, T yh, 
        T site_width, T row_height, 
        int num_bins_x, int num_bins_y, 
        int num_nodes, int num_movable_nodes, int num_filler_nodes, 
        int* bin_capacities, // bin capacity in number of sites
        T* x, T* y
        )
{
    std::vector<int> bin_demands (num_bins_x*num_bins_y, 0);
    for (int i = 0; i < num_movable_nodes; i += 1) 
    {
        int node_id = ordered_nodes[num_movable_nodes-i-1]; 
        int bin_id_x = (init_x[node_id]+node_size_x[node_id]/2-xl)/bin_size_x; 
        int bin_id_y = (init_y[node_id]+node_size_y[node_id]/2-yl)/bin_size_y;

        bin_id_x = std::min(std::max(bin_id_x, 0), num_bins_x-1);
        bin_id_y = std::min(std::max(bin_id_y, 0), num_bins_y-1);

        int node_size = ceil(node_size_x[node_id]*node_size_y[node_id]/(site_width*row_height));

        int bin_id_dist_x = std::max(bin_id_x+1, num_bins_x-bin_id_x); 
        int bin_id_dist_y = std::max(bin_id_y+1, num_bins_y-bin_id_y); 
        bool search_flag = true; 
        T best_cost = std::numeric_limits<T>::max();
        int best_bin_ix = -1; 
        int best_bin_iy = -1; 
        for (int bin_id_offset_x = 0; search_flag && std::abs(bin_id_offset_x) < bin_id_dist_x; bin_id_offset_x = (bin_id_offset_x > 0)? -bin_id_offset_x : -(bin_id_offset_x-1))
        {
            int ix = bin_id_x+bin_id_offset_x; 
            if (ix < 0 || ix >= num_bins_x)
            {
                continue; 
            }
            for (int bin_id_offset_y = 0; std::abs(bin_id_offset_y) < bin_id_dist_y; bin_id_offset_y = (bin_id_offset_y > 0)? -bin_id_offset_y : -(bin_id_offset_y-1))
            {
                int iy = bin_id_y+bin_id_offset_y; 
                if (iy < 0 || iy >= num_bins_y)
                {
                    continue; 
                }

                int bin_id = ix*num_bins_y + iy; 

                int& capacity = bin_capacities[bin_id]; 
                int& demand = bin_demands[bin_id];

                // use two atomic operations to perform 
                // capacity >= size? capacity-size : capacity 
                // there will be no wait for this one, but some cells may not be able to find this bin if another cell occupies it. 
                // the result is not deterministic either 
                if (demand+node_size <= capacity+1)
                {
                    T bin_xl = xl+bin_size_x*ix;
                    T bin_yl = yl+bin_size_y*iy;
                    T dx = 0; 
                    T dy = 0; 
                    T init_xl = init_x[node_id];
                    T init_yl = init_y[node_id];
                    if (init_xl+node_size_x[node_id]/2 < bin_xl)
                    {
                        dx = bin_xl-node_size_x[node_id]/2 - init_xl; 
                    }
                    else if (init_xl+node_size_x[node_id]/2 > bin_xl+bin_size_x)
                    {
                        dx = init_xl - (bin_xl+bin_size_x-node_size_x[node_id]/2); 
                    }
                    if (init_yl < bin_yl)
                    {
                        dy = bin_yl - init_yl; 
                    }
                    else if (init_yl+node_size_y[node_id] > bin_yl+bin_size_y)
                    {
                        dy = init_yl - (bin_yl+bin_size_y-node_size_y[node_id]); 
                    }
                    dreamplaceAssert(dx >= 0 && dy >= 0);

                    T cost = dx + dy; 
                    T ratio = (demand+1.0e-3)/(capacity+1.0e-3);
                    cost *= (1.0 + ratio);
                    if (best_cost > cost)
                    {
                        best_cost = cost; 
                        best_bin_ix = ix; 
                        best_bin_iy = iy; 
                    }
                    else if (dy > best_cost+bin_size_y) // early exit 
                    {
                        break; 
                    }
                    else if (dx > best_cost+bin_size_x)
                    {
                        search_flag = false;
                    }
                }
            }
        }
        if (best_cost != std::numeric_limits<T>::max())
        {
            int bin_id = best_bin_ix*num_bins_y + best_bin_iy; 
            int& demand = bin_demands[bin_id];
            demand += node_size;
            T bin_xl = xl+bin_size_x*best_bin_ix;
            T bin_yl = yl+bin_size_y*best_bin_iy;
            if (init_x[node_id]+node_size_x[node_id]/2 < bin_xl)
            {
                x[node_id] = bin_xl-node_size_x[node_id]/2+1; 
            }
            else if (init_x[node_id]+node_size_x[node_id]/2 > bin_xl+bin_size_x)
            {
                x[node_id] = bin_xl+bin_size_x-node_size_x[node_id]/2-1; 
            }
            else 
            {
                x[node_id] = init_x[node_id]; 
            }
            if (init_y[node_id] < bin_yl)
            {
                y[node_id] = bin_yl; 
            }
            else if (init_y[node_id]+node_size_y[node_id] > bin_yl+bin_size_y)
            {
                y[node_id] = bin_yl+bin_size_y-node_size_y[node_id]; 
            }
            else 
            {
                y[node_id] = init_y[node_id]; 
            }
        }
    }
}

template <typename T>
void binAssignmentCPU(
        const int* ordered_nodes, 
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
    T bin_size_x = (xh-xl)/num_bins_x; 
    T bin_size_y = (yh-yl)/num_bins_y; 

    std::vector<std::vector<int> > bin_fixed_cells (num_bins_x*num_bins_y); 
    std::vector<int> bin_capacities(num_bins_x*num_bins_y);

    // distribute fixed cells to bins 
    distributeFixedCells2BinsCPU(
            init_x, init_y, 
            node_size_x, node_size_y, 
            bin_size_x, bin_size_y, 
            xl, yl, xh, yh, 
            num_bins_x, num_bins_y, 
            num_nodes, num_movable_nodes, num_filler_nodes, 
            bin_fixed_cells
            ); 

    // compute bin capacity 
    computeBinCapacityCPU(
            init_x, init_y, 
            node_size_x, node_size_y, 
            bin_fixed_cells,
            bin_size_x, bin_size_y, 
            xl, yl, xh, yh, 
            site_width, row_height, 
            num_bins_x, num_bins_y, 
            bin_capacities.data()
            );

    // assing cells to bins 
    // the location of cells are adjusted 

    // on CPU 
    assignCells2BinsCPU<T>(
            ordered_nodes, 
            init_x, init_y, 
            node_size_x, node_size_y, 
            bin_size_x, bin_size_y, 
            xl, yl, xh, yh, 
            site_width, row_height, 
            num_bins_x, num_bins_y, 
            num_nodes, num_movable_nodes, num_filler_nodes, 
            bin_capacities.data(),
            //bin_cells
            x, y
            );
}

void instantiateBinAssignmentCPU(
        const int* ordered_nodes, 
        const float* init_x, const float* init_y, 
        const float* node_size_x, const float* node_size_y, 
        float* x, float* y, 
        const float xl, const float yl, const float xh, const float yh, 
        const float site_width, const float row_height, 
        int num_bins_x, int num_bins_y, 
        const int num_nodes, 
        const int num_movable_nodes, 
        const int num_filler_nodes
        )
{
    binAssignmentCPU(
            ordered_nodes, 
            init_x, init_y, 
            node_size_x, node_size_y, 
            x, y, 
            xl, yl, xh, yh, 
            site_width, row_height, 
            num_bins_x, num_bins_y, 
            num_nodes, 
            num_movable_nodes, 
            num_filler_nodes
            );
}

void instantiateBinAssignmentCPU(
        const int* ordered_nodes, 
        const double* init_x, const double* init_y, 
        const double* node_size_x, const double* node_size_y, 
        double* x, double* y, 
        const double xl, const double yl, const double xh, const double yh, 
        const double site_width, const double row_height, 
        int num_bins_x, int num_bins_y, 
        const int num_nodes, 
        const int num_movable_nodes, 
        const int num_filler_nodes
        )
{
    binAssignmentCPU(
            ordered_nodes, 
            init_x, init_y, 
            node_size_x, node_size_y, 
            x, y, 
            xl, yl, xh, yh, 
            site_width, row_height, 
            num_bins_x, num_bins_y, 
            num_nodes, 
            num_movable_nodes, 
            num_filler_nodes
            );
}

DREAMPLACE_END_NAMESPACE
