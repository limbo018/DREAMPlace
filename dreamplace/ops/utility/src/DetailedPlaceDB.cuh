/**
 * @file   DetailedPlaceDB.cuh
 * @author Yibo Lin
 * @date   Jan 2019
 */

#ifndef _DREAMPLACE_UTILITY_DETAILEDPLACEDB_CUH
#define _DREAMPLACE_UTILITY_DETAILEDPLACEDB_CUH

#include "utility/src/Msg.h"
#include "utility/src/Box.cuh"
#include "utility/src/utils.cuh"
#include "greedy_legalize/src/legality_check_cpu.h"
#include "draw_place/src/draw_place.h"
//#include <thrust/host_vector.h>
//#include <thrust/device_vector.h>
//#include <thrust/swap.h>
//#include <thrust/reduce.h>
//#include <thrust/functional.h>
#include <cub/cub.cuh>

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
struct Space 
{
    T xl; 
    T xh; 
};

struct BinMapIndex
{
    int bin_id; 
    int sub_id; 
};

struct RowMapIndex
{
    int row_id; 
    int sub_id; 
};

/// @brief a wrapper class of required data for detailed placement 
template <typename T>
struct DetailedPlaceDB
{
    typedef T type; 

    const T* init_x; 
    const T* init_y; 
    const T* node_size_x; 
    const T* node_size_y; 
    T* x; 
    T* y; 
    const int* flat_net2pin_map; 
    const int* flat_net2pin_start_map; 
    const int* pin2net_map; 
    const int* flat_node2pin_map; 
    const int* flat_node2pin_start_map; 
    const int* pin2node_map; 
    const T* pin_offset_x; 
    const T* pin_offset_y; 
    const unsigned char* net_mask; 
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
    int num_nets;
    int num_pins; 

    inline __device__ int pos2site_x(T xx) const 
    {
        return min(max((int)CUDADiv((xx-xl), site_width), 0), num_sites_x-1); 
    }
    inline __device__ int pos2site_y(T yy) const 
    {
        return min(max((int)CUDADiv((yy-yl), row_height), 0), num_sites_y-1); 
    }
    /// @brief site index as an upper bound 
    inline __device__ int pos2site_ub_x(T xx) const  
    {
        return min(max(CUDACeilDiv((xx-xl), site_width), 1), num_sites_x); 
    }
    /// @brief site index as an upper bound 
    inline __device__ int pos2site_ub_y(T yy) const  
    {
        return min(max(CUDACeilDiv((yy-yl), row_height), 1), num_sites_y); 
    }
    inline __device__ int pos2bin_x(T xx) const  
    {
        int bx = CUDADiv((xx-xl), bin_size_x); 
        bx = max(bx, 0); 
        bx = min(bx, num_bins_x-1); 
        return bx; 
    }
    inline __device__ int pos2bin_y(T yy) const  
    {
        int by = CUDADiv((yy-yl), bin_size_y); 
        by = max(by, 0); 
        by = min(by, num_bins_y-1); 
        return by; 
    }
    inline __device__ void shift_box_to_layout(Box<T>& box) const  
    {
        box.xl = max(box.xl, xl);
        box.xl = min(box.xl, xh);
        box.xh = max(box.xh, xl);
        box.xh = min(box.xh, xh);
        box.yl = max(box.yl, yl);
        box.yl = min(box.yl, yh);
        box.yh = max(box.yh, yl);
        box.yh = min(box.yh, yh);
    }
    inline __device__ Box<int> box2sitebox(const Box<T>& box) const  
    {
        // xh, yh are exclusive 
        Box<int> sitebox (
                pos2site_x(box.xl), 
                pos2site_y(box.yl), 
                pos2site_ub_x(box.xh), 
                pos2site_ub_y(box.yh)
                ); 

        return sitebox; 
    }
    inline __device__ Box<int> box2binbox(const Box<T>& box) const
    {
        Box<int> binbox (
                pos2bin_x(box.xl), 
                pos2bin_y(box.yl),  
                pos2bin_x(box.xh), 
                pos2bin_y(box.yh) 
                );

        return binbox; 
    }
    /// @brief align x coordinate to site 
    inline __device__ T align2site(T xx) const 
    {
        return (int)CUDADiv((xx-xl), site_width)*site_width+xl; 
    }
    /// @brief compute optimal region for a cell 
    /// The method to compute optimal region ignores the pin offsets of the target cell. 
    /// If we want to consider the pin offsets, there may not be feasible box for the optimal region. 
    /// Thus, this is just an approximate optimal region. 
    /// When using the optimal region, one needs to refer to the center of the cell to the region, or the region completely covers the entire cell. 
    __device__ Box<T> compute_optimal_region(int node_id, const T* xx, const T* yy) const
    {
        Box<T> box (
                xh, // some large number 
                yh, // some large number 
                xl, // some small number 
                yl  // some small number 
                ); 
        for (int node2pin_id = flat_node2pin_start_map[node_id]; node2pin_id < flat_node2pin_start_map[node_id+1]; ++node2pin_id)
        {
            int node_pin_id = flat_node2pin_map[node2pin_id];
            int net_id = pin2net_map[node_pin_id];
            if (net_mask[net_id])
            {
                for (int net2pin_id = flat_net2pin_start_map[net_id]; net2pin_id < flat_net2pin_start_map[net_id+1]; ++net2pin_id)
                {
                    int net_pin_id = flat_net2pin_map[net2pin_id];
                    int other_node_id = pin2node_map[net_pin_id];
                    if (node_id != other_node_id)
                    {
                        box.xl = min(box.xl, xx[other_node_id]+pin_offset_x[net_pin_id]);
                        box.xh = max(box.xh, xx[other_node_id]+pin_offset_x[net_pin_id]);
                        box.yl = min(box.yl, yy[other_node_id]+pin_offset_y[net_pin_id]);
                        box.yh = max(box.yh, yy[other_node_id]+pin_offset_y[net_pin_id]);
                    }
                }
            }
        }
        shift_box_to_layout(box);

        return box; 
    }
    /// @brief compute HPWL for a net 
    __device__ T compute_net_hpwl(int net_id, const T* xx, const T* yy) const
    {
        Box<T> box (
                xh, // some large number 
                yh, // some large number 
                xl, // some small number 
                yl  // some small number 
                ); 
        for (int net2pin_id = flat_net2pin_start_map[net_id]; net2pin_id < flat_net2pin_start_map[net_id+1]; ++net2pin_id)
        {
            int net_pin_id = flat_net2pin_map[net2pin_id];
            int other_node_id = pin2node_map[net_pin_id];
            box.xl = min(box.xl, xx[other_node_id]+pin_offset_x[net_pin_id]);
            box.xh = max(box.xh, xx[other_node_id]+pin_offset_x[net_pin_id]);
            box.yl = min(box.yl, yy[other_node_id]+pin_offset_y[net_pin_id]);
            box.yh = max(box.yh, yy[other_node_id]+pin_offset_y[net_pin_id]);
        }
        if (box.xl == xh || box.yl == yh) // use xh/yh as some large number 
        {
            return (T)0; 
        }
        return (box.xh-box.xl) + (box.yh-box.yl);
    }
    /// @brief compute HPWL for all nets 
    __device__ T compute_total_hpwl() const
    {
        //printf("[D] start compute_total_hpwl\n");
        T total_hpwl = 0; 
        for (int net_id = 0; net_id < num_nets; ++net_id)
        {
            //if (net_mask[net_id])
            {
                total_hpwl += compute_net_hpwl(net_id, x, y);
            }
        }
        //printf("[D] end compute_total_hpwl\n");
        return total_hpwl; 
    }
    /// @brief distribute cells to rows 
    __host__ void make_row2node_map(const std::vector<T>& host_x, const std::vector<T>& host_y, 
            const std::vector<T>& host_node_size_x, const std::vector<T>& host_node_size_y, 
            std::vector<std::vector<int> >& row2node_map, std::vector<RowMapIndex>& node2row_map, std::vector<Space<T> >& spaces) const 
    {
        // distribute cells to rows 
        for (int i = 0; i < num_nodes+2; ++i)
        {
            T node_yl = host_y[i];
            T node_yh = node_yl+host_node_size_y[i];

            int row_idxl = CPUDiv((node_yl-yl), row_height); 
            int row_idxh = CPUCeilDiv((node_yh-yl), row_height)+1;
            row_idxl = max(row_idxl, 0); 
            row_idxh = min(row_idxh, num_sites_y); 

            for (int row_id = row_idxl; row_id < row_idxh; ++row_id)
            {
                T row_yl = yl+row_id*row_height; 
                T row_yh = row_yl+row_height; 

                if (node_yl < row_yh && node_yh > row_yl) // overlap with row 
                {
                    row2node_map[row_id].push_back(i); 
                }
            }
        }

        // sort cells within rows 
        auto comp = [&] (int node_id1, int node_id2) {
            if (node_id1 == num_nodes || node_id2 == num_nodes)
            {
                return host_x[node_id1]+host_node_size_x[node_id1] < host_x[node_id2]+host_node_size_x[node_id2]; 
            }
            return host_x[node_id1]+host_node_size_x[node_id1]/2 < host_x[node_id2]+host_node_size_x[node_id2]/2;
        };
        for (int i = 0; i < num_sites_y; ++i)
        {
            std::sort(row2node_map[i].begin(), row2node_map[i].end(), comp);
        }

        // construct node2row_map 
        for (int i = 0; i < num_sites_y; ++i)
        {
            for (unsigned int j = 0; j < row2node_map[i].size(); ++j)
            {
                int node_id = row2node_map[i][j];
                if (node_id < num_movable_nodes)
                {
                    RowMapIndex& row_id = node2row_map[node_id];
                    row_id.row_id = i; 
                    row_id.sub_id = j; 
                }
            }
        }

        // construct spaces 
        for (int i = 0; i < num_sites_y; ++i)
        {
            for (unsigned int j = 0; j < row2node_map[i].size(); ++j)
            {
                int node_id = row2node_map[i][j];
                if (node_id < num_movable_nodes)
                {
                    assert(j); 
                    int left_node_id = row2node_map[i][j-1];
                    spaces[node_id].xl = host_x[left_node_id] + host_node_size_x[left_node_id]; 
                    assert(j+1 < row2node_map[i].size());
                    int right_node_id = row2node_map[i][j+1]; 
                    spaces[node_id].xh = host_x[right_node_id]; 
                }
            }
        }
    }
    /// @brief distribute movable cells to bins according to cell (xl, yl) 
    /// @param bin2node_map flatten bin map, column-major 
    /// @param node2bin_index_map the index of cell in bin2node_map
    __host__ void make_bin2node_map(const std::vector<T>& host_x, const std::vector<T>& host_y, 
            const std::vector<T>& host_node_size_x, const std::vector<T>& host_node_size_y, 
            std::vector<std::vector<int> >& bin2node_map, std::vector<BinMapIndex>& node2bin_map) const 
    {
        // construct bin2node_map 
        for (int i = 0; i < num_movable_nodes; ++i)
        {
            int node_id = i; 
            T node_x = host_x[node_id] + host_node_size_x[node_id]/2; 
            T node_y = host_y[node_id] + host_node_size_y[node_id]/2;

            int bx = min(max((int)CPUDiv(node_x-xl, bin_size_x), 0), num_bins_x-1);
            int by = min(max((int)CPUDiv(node_y-yl, bin_size_y), 0), num_bins_y-1);
            int bin_id = bx*num_bins_y+by; 
            int sub_id = bin2node_map.at(bin_id).size(); 
            bin2node_map.at(bin_id).push_back(node_id); 
        }
        // sort cells within bins  
        //auto comp = [&] (int node_id1, int node_id2) {
        //    return host_x[node_id1] < host_x[node_id2] || (host_x[node_id1] == host_x[node_id2] && host_y[node_id1] < host_y[node_id2]);
        //};
        //for (auto& bin2nodes : bin2node_map)
        //{
        //    std::sort(bin2nodes.begin(), bin2nodes.end(), comp);
        //}
        // construct node2bin_map 
        for (int bin_id = 0; bin_id < bin2node_map.size(); ++bin_id)
        {
            for (int sub_id = 0; sub_id < bin2node_map[bin_id].size(); ++sub_id)
            {
                int node_id = bin2node_map[bin_id][sub_id];
                BinMapIndex& bm_idx = node2bin_map.at(node_id); 
                bm_idx.bin_id = bin_id; 
                bm_idx.sub_id = sub_id; 
            }
        }
#ifdef DEBUG
        int max_num_nodes_per_bin = 0; 
        for (int i = 0; i < bin2node_map.size(); ++i)
        {
            max_num_nodes_per_bin = max(max_num_nodes_per_bin, (int)bin2node_map[i].size());
        }
        printf("[D] max_num_nodes_per_bin = %d\n", max_num_nodes_per_bin);
#endif
    }
    /// @brief check whether placement is legal 
    bool check_legality(const T* host_x, const T* host_y, const T* host_node_size_x, const T* host_node_size_y) const 
    {
        T* host_init_x = nullptr; 
        T* host_init_y = nullptr; 
        allocateCUDA(host_init_x, num_nodes, T);
        allocateCUDA(host_init_y, num_nodes, T);
        checkCUDA(cudaMemcpy(host_init_x, init_x, sizeof(T)*num_nodes, cudaMemcpyDeviceToHost));
        checkCUDA(cudaMemcpy(host_init_y, init_y, sizeof(T)*num_nodes, cudaMemcpyDeviceToHost));
        return legalityCheckKernelCPU(
                host_init_x, host_init_y, 
                host_node_size_x, host_node_size_y, 
                host_x, host_y, 
                site_width, row_height, 
                xl, yl, xh, yh,
                num_nodes, 
                num_movable_nodes, 
                0
                );
    }
    /// @brief draw placement 
    void draw_place(const char* filename) const 
    {
        drawPlaceLauncher<T>(
                x, y, 
                node_size_x, node_size_y, 
                pin_offset_x, pin_offset_y, 
                pin2node_map, 
                num_nodes, 
                num_movable_nodes, 
                0, 
                flat_net2pin_start_map[num_nets], 
                xl, yl, xh, yh, 
                site_width, row_height, 
                bin_size_x, bin_size_y, 
                filename
                );
    }
};

/// @brief compute total HPWL 
/// This function is mainly for evaluation, so the performance is not highly tuned. 
/// Consistency is more important. 
/// Thus integer is adopted. 
template <typename T, typename V, int scale=1000>
__global__ void compute_total_hpwl_kernel(DetailedPlaceDB<T> db, const T* xx, const T* yy, V* net_hpwls)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < db.num_nets; i += blockDim.x * gridDim.x)
    {
        net_hpwls[i] = V(db.compute_net_hpwl(i, xx, yy)*(T)scale); 
        //if (db.net_mask[i])
        //{
        //    net_hpwls[i] = V(db.compute_net_hpwl(i, xx, yy)*(T)scale); 
        //}
        //else 
        //{
        //    net_hpwls[i] = 0; 
        //}
    }
}

template <typename T, typename V, int scale=1000>
T compute_total_hpwl(const DetailedPlaceDB<T>& db, const T* xx, const T* yy, V* net_hpwls)
{
    compute_total_hpwl_kernel<T, V, scale><<<CPUCeilDiv(db.num_nets, 512), 512>>>(db, xx, yy, net_hpwls); 
    //auto hpwl = thrust::reduce(thrust::device, net_hpwls, net_hpwls+db.num_nets);

    V* d_out = NULL; 
    // Determine temporary device storage requirements
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, net_hpwls, d_out, db.num_nets);
    // Allocate temporary storage
    checkCUDA(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    checkCUDA(cudaMalloc(&d_out, sizeof(V))); 
    // Run sum-reduction
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, net_hpwls, d_out, db.num_nets);
    // copy d_out to hpwl  
    V hpwl = 0; 
    checkCUDA(cudaMemcpy(&hpwl, d_out, sizeof(V), cudaMemcpyDeviceToHost)); 
    destroyCUDA(d_temp_storage); 
    destroyCUDA(d_out); 

    return T(hpwl)/scale;
}

DREAMPLACE_END_NAMESPACE

#endif
