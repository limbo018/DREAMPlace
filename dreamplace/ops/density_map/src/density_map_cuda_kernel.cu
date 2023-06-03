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

template <typename T, typename AtomicOp>
inline __device__ void distributeBox2Bin(
        const int num_bins_x, const int num_bins_y, 
        const T xl, const T yl, const T xh, const T yh, 
        const T bin_size_x, const T bin_size_y, 
        T bxl, T byl, T bxh, T byh, AtomicOp atomic_add_op, 
        typename AtomicOp::type* buf_map
        )
{
    // density overflow function 
    auto computeDensityFunc = [](T node_xl, T node_xh, T bin_xl, T bin_xh) {
      return DREAMPLACE_STD_NAMESPACE::max(
          T(0.0),
          DREAMPLACE_STD_NAMESPACE::min(node_xh, bin_xh) -
          DREAMPLACE_STD_NAMESPACE::max(node_xl, bin_xl));
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
        T bin_xl = xl + bin_size_x * k;
        T bin_xh = DREAMPLACE_STD_NAMESPACE::min(bin_xl + bin_size_x, xh); 
        // special treatment for rightmost bins 
        if (k + 1 == num_bins_x) {
          bin_xh = bxh; 
        }
        T px = computeDensityFunc(bxl, bxh, bin_xl, bin_xh);
        for (int h = bin_index_yl; h < bin_index_yh; ++h)
        {
            T bin_yl = yl + bin_size_y * h; 
            T bin_yh = DREAMPLACE_STD_NAMESPACE::min(bin_yl + bin_size_y, yh); 
            // special treatment for upmost bins
            if (h + 1 == num_bins_y) {
              bin_yh = byh; 
            }
            T py = computeDensityFunc(byl, byh, bin_yl, bin_yh);

            // still area 
            atomic_add_op(&buf_map[k * num_bins_y + h], px * py);
        }
    }
}

template <typename T, typename AtomicOp>
__global__ void computeDensityMap(
    const T* x_tensor, const T* y_tensor,
    const T* node_size_x_tensor,
    const T* node_size_y_tensor,
    const int num_nodes,
    const int num_bins_x, const int num_bins_y,
    const T xl, const T yl, const T xh, const T yh,
    AtomicOp atomic_add_op,
    typename AtomicOp::type* buf_map
    )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    T bin_size_x = (xh - xl) / num_bins_x; 
    T bin_size_y = (yh - yl) / num_bins_y; 

    if (i < num_nodes)
    {
        T bxl = x_tensor[i]; 
        T byl = y_tensor[i]; 
        T bxh = bxl + node_size_x_tensor[i]; 
        T byh = byl + node_size_y_tensor[i]; 

        distributeBox2Bin(
                num_bins_x, num_bins_y, 
                xl, yl, xh, yh, 
                bin_size_x, bin_size_y, 
                bxl, byl, bxh, byh, 
                atomic_add_op, buf_map
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
    const int num_nodes, 
    const int num_bins_x, const int num_bins_y, 
    const T xl, const T yl, const T xh, const T yh,
    bool deterministic_flag, 
    T* density_map_tensor
    )
{
    if (deterministic_flag) {
        // total die area
        double diearea = (xh - xl) * (yh - yl);
        int integer_bits = max((int)ceil(log2(diearea)) + 1, 32);
        int fraction_bits = max(64 - integer_bits, 0);
        unsigned long long int scale_factor = (1UL << fraction_bits);
        int num_bins = num_bins_x * num_bins_y;
        unsigned long long int *buf_map = NULL;
        allocateCUDA(buf_map, num_bins, unsigned long long int);

        AtomicAddCUDA<unsigned long long int> atomic_add_op(scale_factor);

        int thread_count = 512;
        int block_count = ceilDiv(num_bins, thread_count);
        copyScaleArray<<<block_count, thread_count>>>(
            buf_map, density_map_tensor, scale_factor, num_bins);

        block_count = ceilDiv(num_nodes, thread_count);
        computeDensityMap<<<block_count, thread_count>>>(
            x_tensor, y_tensor, 
            node_size_x_tensor, node_size_y_tensor, 
            num_nodes, 
            num_bins_x, num_bins_y, 
            xl, yl, xh, yh, 
            atomic_add_op, buf_map
            );

        block_count = ceilDiv(num_bins, thread_count);
        copyScaleArray<<<block_count, thread_count>>>(
            density_map_tensor, buf_map, T(1.0 / scale_factor), num_bins);

        destroyCUDA(buf_map);
    } else {
        AtomicAddCUDA<T> atomic_add_op;
        int thread_count = 512;
        int block_count = ceilDiv(num_nodes, thread_count);
        computeDensityMap<<<block_count, thread_count>>>(
            x_tensor, y_tensor, 
            node_size_x_tensor, node_size_y_tensor, 
            num_nodes, 
            num_bins_x, num_bins_y, 
            xl, yl, xh, yh, 
            atomic_add_op, density_map_tensor
            );
    }

    return 0; 
}

#define REGISTER_KERNEL_LAUNCHER(T) \
    template int computeDensityMapCudaLauncher<T>(\
            const T* x_tensor, const T* y_tensor, \
            const T* node_size_x_tensor, const T* node_size_y_tensor, \
            const int num_nodes, \
            const int num_bins_x, const int num_bins_y, \
            const T xl, const T yl, const T xh, const T yh, \
            bool deterministic_flag, \
            T* density_map_tensor \
            );

REGISTER_KERNEL_LAUNCHER(float);
REGISTER_KERNEL_LAUNCHER(double);

DREAMPLACE_END_NAMESPACE
