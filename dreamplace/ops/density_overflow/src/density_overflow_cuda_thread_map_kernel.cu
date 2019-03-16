/**
 * @file   density_overflow_cuda_thread_map_kernel.cu
 * @author Yibo Lin
 * @date   Jun 2018
 * @brief  Compute density map on CUDA with cell2bin parallelization 
 */
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "cuda_runtime.h"
#include "utility/src/print.h"
#include "utility/src/Msg.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
__global__ void computeDensityMapWithThreadMap(
        const T* x_tensor, const T* y_tensor, 
        const T* node_size_x_tensor, const T* node_size_y_tensor, 
        const T* bin_center_x_tensor, const T* bin_center_y_tensor, 
        const int* thread2node_map, const int* thread2bin_x_map, const int* thread2bin_y_map, 
        const int num_threads, 
        const int num_nodes, 
        const int num_bins_x, const int num_bins_y, 
        const T xl, const T yl, const T xh, const T yh, 
        const T bin_size_x, const T bin_size_y, 
        T* density_map_tensor) 
{
    // rank-one update density map 
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_threads; i += blockDim.x * gridDim.x) 
    {
        // density overflow function 
        auto computeDensityOverflowFunc = [](T x, T node_size, T bin_center, T bin_size){
            return max(T(0.0), min(x+node_size, bin_center+bin_size/2) - max(x, bin_center-bin_size/2));
        };
        int node_id = thread2node_map[i];
        int bin_offset_x = thread2bin_x_map[i]; 
        int bin_offset_y = thread2bin_y_map[i];
        // x direction 
        int bin_index_xl = int((x_tensor[node_id]-xl)/bin_size_x);
        bin_index_xl = max(bin_index_xl, 0);
        int k = bin_index_xl+bin_offset_x; 
        if (k+1 > num_bins_x)
        {
            continue; 
        }
        // y direction 
        int bin_index_yl = int((y_tensor[node_id]-yl)/bin_size_y);
        bin_index_yl = max(bin_index_yl, 0);
        int h = bin_index_yl+bin_offset_y; 
        if (h+1 > num_bins_y)
        {
            continue; 
        }

        T px = computeDensityOverflowFunc(x_tensor[node_id], node_size_x_tensor[node_id], bin_center_x_tensor[k], bin_size_x);
        T py = computeDensityOverflowFunc(y_tensor[node_id], node_size_y_tensor[node_id], bin_center_y_tensor[h], bin_size_y);
        // still area 
        atomicAdd(&density_map_tensor[k*num_bins_y+h], px*py); 
        __syncthreads();
    }
}

template <typename T>
int computeDensityOverflowMapCudaThreadMapLauncher(
        const T* x_tensor, const T* y_tensor, 
        const T* node_size_x_tensor, const T* node_size_y_tensor, 
        const T* bin_center_x_tensor, const T* bin_center_y_tensor, 
        const int* thread2node_map, const int* thread2bin_x_map, const int* thread2bin_y_map, 
        const int num_threads, 
        const int num_nodes, 
        const int num_bins_x, const int num_bins_y, 
        const T xl, const T yl, const T xh, const T yh, 
        const T bin_size_x, const T bin_size_y, 
        T* density_map_tensor
        )
{
    int block_count = 32; 
    int thread_count = 1024; 

    computeDensityMapWithThreadMap<<<block_count, thread_count>>>(
            x_tensor, y_tensor, 
            node_size_x_tensor, node_size_y_tensor, 
            bin_center_x_tensor, bin_center_y_tensor, 
            thread2node_map, thread2bin_x_map, thread2bin_y_map, 
            num_threads, 
            num_nodes, 
            num_bins_x, num_bins_y, 
            xl, yl, xh, yh, 
            bin_size_x, bin_size_y, 
            density_map_tensor);

    return 0; 
}

#define REGISTER_KERNEL_LAUNCHER(T) \
    int instantiateComputeDensityOverflowMapLauncher(\
            const T* x_tensor, const T* y_tensor, \
            const T* node_size_x_tensor, const T* node_size_y_tensor, \
            const T* bin_center_x_tensor, const T* bin_center_y_tensor, \
            const int* thread2node_map, const int* thread2bin_x_map, const int* thread2bin_y_map, \
            const int num_threads, \
            const int num_nodes, \
            const int num_bins_x, const int num_bins_y, \
            const T xl, const T yl, const T xh, const T yh, \
            const T bin_size_x, const T bin_size_y, \
            T* density_map_tensor\
            )\
    { \
        return computeDensityOverflowMapCudaThreadMapLauncher(\
                x_tensor, y_tensor, \
                node_size_x_tensor, node_size_y_tensor, \
                bin_center_x_tensor, bin_center_y_tensor, \
                thread2node_map, thread2bin_x_map, thread2bin_y_map, \
                num_threads, \
                num_nodes, \
                num_bins_x, num_bins_y, \
                xl, yl, xh, yh, \
                bin_size_x, bin_size_y, \
                density_map_tensor\
                );\
    }
REGISTER_KERNEL_LAUNCHER(float);
REGISTER_KERNEL_LAUNCHER(double);

DREAMPLACE_END_NAMESPACE
