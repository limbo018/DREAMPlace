/**
 * @file   density_overflow_cuda_by_node_kernel.cu
 * @author Yibo Lin
 * @date   Jun 2018
 * @brief  Compute density map on CUDA  
 */
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "cuda_runtime.h"
#include "utility/src/print.h"

template <typename T>
__global__ void computeDensityMapByNode(
        const T* x_tensor, const T* y_tensor, 
        const T* node_size_x_tensor, const T* node_size_y_tensor, 
        const T* bin_center_x_tensor, const T* bin_center_y_tensor, 
        const int num_nodes, 
        const int num_bins_x, const int num_bins_y, 
        const T xl, const T yl, const T xh, const T yh, 
        const T bin_size_x, const T bin_size_y, 
        T* density_map_tensor) 
{
    // rank-one update density map 
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_nodes; i += blockDim.x * gridDim.x) 
    {
        // density overflow function 
        auto computeDensityOverflowFunc = [](T x, T node_size, T bin_center, T bin_size){
            return max(T(0.0), min(x+node_size, bin_center+bin_size/2) - max(x, bin_center-bin_size/2));
        };
        int node_id = i; 
        // x direction 
        int bin_index_xl = int((x_tensor[node_id]-xl)/bin_size_x);
        bin_index_xl = max(bin_index_xl, 0);
        int bin_index_xh = int((x_tensor[node_id]+node_size_x_tensor[node_id]-xl)/bin_size_x)+1;
        bin_index_xh = min(bin_index_xh, num_bins_x);

        // y direction 
        int bin_index_yl = int((y_tensor[node_id]-yl)/bin_size_y);
        bin_index_yl = max(bin_index_yl, 0);
        int bin_index_yh = int((y_tensor[node_id]+node_size_y_tensor[node_id]-yl)/bin_size_y)+1;
        bin_index_yh = min(bin_index_yh, num_bins_y);

        for (int k = bin_index_xl; k < bin_index_xh; ++k)
        {
            for (int h = bin_index_yl; h < bin_index_yh; ++h)
            {
                T px = computeDensityOverflowFunc(x_tensor[node_id], node_size_x_tensor[node_id], bin_center_x_tensor[k], bin_size_x);
                T py = computeDensityOverflowFunc(y_tensor[node_id], node_size_y_tensor[node_id], bin_center_y_tensor[h], bin_size_y);
                // still area 
                atomicAdd(&density_map_tensor[k*num_bins_y+h], px*py); 
            }
        }
    }
}

template <typename T>
int computeDensityOverflowMapCudaByNodeLauncher(
        const T* x_tensor, const T* y_tensor, 
        const T* node_size_x_tensor, const T* node_size_y_tensor, 
        const T* bin_center_x_tensor, const T* bin_center_y_tensor, 
        const int num_nodes, 
        const int num_bins_x, const int num_bins_y, 
        const T xl, const T yl, const T xh, const T yh, 
        const T bin_size_x, const T bin_size_y, 
        T* density_map_tensor
        )
{
    int block_count = 32; 
    int thread_count = 1024; 

    computeDensityMapByNode<<<block_count, thread_count>>>(
            x_tensor, y_tensor, 
            node_size_x_tensor, node_size_y_tensor, 
            bin_center_x_tensor, bin_center_y_tensor, 
            num_nodes, 
            num_bins_x, num_bins_y, 
            xl, yl, xh, yh, 
            bin_size_x, bin_size_y, 
            density_map_tensor);

    return 0; 
}

#define REGISTER_KERNEL_LAUNCHER(T) \
    int instantiateComputeDensityOverflowMapByNodeLauncher(\
            const T* x_tensor, const T* y_tensor, \
            const T* node_size_x_tensor, const T* node_size_y_tensor, \
            const T* bin_center_x_tensor, const T* bin_center_y_tensor, \
            const int num_nodes, \
            const int num_bins_x, const int num_bins_y, \
            const T xl, const T yl, const T xh, const T yh, \
            const T bin_size_x, const T bin_size_y, \
            T* density_map_tensor\
            )\
    { \
        return computeDensityOverflowMapCudaByNodeLauncher(\
                x_tensor, y_tensor, \
                node_size_x_tensor, node_size_y_tensor, \
                bin_center_x_tensor, bin_center_y_tensor, \
                num_nodes, \
                num_bins_x, num_bins_y, \
                xl, yl, xh, yh, \
                bin_size_x, bin_size_y, \
                density_map_tensor\
                );\
    }
REGISTER_KERNEL_LAUNCHER(float);
REGISTER_KERNEL_LAUNCHER(double);
