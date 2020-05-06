/**
 * @file   density_map_cuda_kernel.cu
 * @author Yibo Lin
 * @date   Dec 2019
 */

#include <stdio.h>
#include <math.h>
#include <float.h>
#include "cuda_runtime.h"
#include "utility/src/utils.cuh"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
inline __device__ void distributeBox2Bin(
        const T* bin_center_x_tensor, const T* bin_center_y_tensor, 
        const int num_bins_x, const int num_bins_y, 
        const T xl, const T yl, const T xh, const T yh, 
        const T bin_size_x, const T bin_size_y, 
        T bxl, T byl, T bxh, T byh, 
        T* buf_map
        )
{
    // density overflow function 
    auto computeDensityFunc = [](T x, T node_size, T bin_center, T bin_size){
        return DREAMPLACE_STD_NAMESPACE::max(T(0.0), DREAMPLACE_STD_NAMESPACE::min(x+node_size, bin_center+bin_size/2) - DREAMPLACE_STD_NAMESPACE::max(x, bin_center-bin_size/2));
    };
    // x direction 
    int bin_index_xl = int((bxl-xl)/bin_size_x);
    int bin_index_xh = int(ceil((bxh-xl)/bin_size_x))+1; // exclusive 
    bin_index_xl = DREAMPLACE_STD_NAMESPACE::max(bin_index_xl, 0); 
    bin_index_xh = DREAMPLACE_STD_NAMESPACE::min(bin_index_xh, num_bins_x);

    // y direction 
    int bin_index_yl = int((byl-yl-2*bin_size_y)/bin_size_y);
    int bin_index_yh = int(ceil((byh-yl+2*bin_size_y)/bin_size_y))+1; // exclusive 
    bin_index_yl = DREAMPLACE_STD_NAMESPACE::max(bin_index_yl, 0); 
    bin_index_yh = DREAMPLACE_STD_NAMESPACE::min(bin_index_yh, num_bins_y);

    for (int k = bin_index_xl; k < bin_index_xh; ++k)
    {
        T px = computeDensityFunc(bxl, bxh - bxl, bin_center_x_tensor[k], bin_size_x);
        for (int h = bin_index_yl; h < bin_index_yh; ++h)
        {
            T py = computeDensityFunc(byl, byh - byl, bin_center_y_tensor[h], bin_size_y);

            // still area 
            atomicAdd(&buf_map[k*num_bins_y+h], px * py);
        }
    }
};

template <typename T>
__global__ void computeDensityMap(
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
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_nodes)
    {
        T bxl = x_tensor[i]; 
        T byl = y_tensor[i]; 
        T bxh = bxl + node_size_x_tensor[i]; 
        T byh = byl + node_size_y_tensor[i]; 
        distributeBox2Bin(
                bin_center_x_tensor, bin_center_y_tensor, 
                num_bins_x, num_bins_y, 
                xl, yl, xh, yh, 
                bin_size_x, bin_size_y, 
                bxl, byl, bxh, byh, 
                density_map_tensor
                );
    }
}

/// @brief compute density map 
/// @param x_tensor cell x locations
/// @param y_tensor cell y locations 
/// @param node_size_x_tensor cell width array
/// @param node_size_y_tensor cell height array 
/// @param bin_center_x_tensor bin center x locations 
/// @param bin_center_y_tensor bin center y locations 
/// @param num_nodes number of cells 
/// @param num_bins_x number of bins in horizontal bins 
/// @param num_bins_y number of bins in vertical bins 
/// @param xl left boundary 
/// @param yl bottom boundary 
/// @param xh right boundary 
/// @param yh top boundary 
/// @param bin_size_x bin width 
/// @param bin_size_y bin height 
/// @param density_map_tensor 2D density map in column-major to write 
template <typename T>
int computeDensityMapCudaLauncher(
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
    int thread_count = 256; 
    int block_count = ceilDiv(num_nodes, thread_count);

    computeDensityMap<<<block_count, thread_count>>>(
            x_tensor, y_tensor, 
            node_size_x_tensor, node_size_y_tensor, 
            bin_center_x_tensor, bin_center_y_tensor, 
            num_nodes, 
            num_bins_x, num_bins_y, 
            xl, yl, xh, yh, 
            bin_size_x, bin_size_y, 
            density_map_tensor
            );

    return 0; 
}

#define REGISTER_KERNEL_LAUNCHER(T) \
    template int computeDensityMapCudaLauncher<T>(\
            const T* x_tensor, const T* y_tensor, \
            const T* node_size_x_tensor, const T* node_size_y_tensor, \
            const T* bin_center_x_tensor, const T* bin_center_y_tensor, \
            const int num_nodes, \
            const int num_bins_x, const int num_bins_y, \
            const T xl, const T yl, const T xh, const T yh, \
            const T bin_size_x, const T bin_size_y, \
            T* density_map_tensor\
            );

REGISTER_KERNEL_LAUNCHER(float);
REGISTER_KERNEL_LAUNCHER(double);

DREAMPLACE_END_NAMESPACE
