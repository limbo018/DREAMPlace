/**
 * @file   electric_density_map_cuda_kernel.cu
 * @author Yibo Lin
 * @date   Aug 2018
 */
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "cuda_runtime.h"
#include "utility/src/print.h"
#include "utility/src/Msg.h"
#include "utility/src/utils.cuh"
#include "electric_potential/src/density_function.h"

DREAMPLACE_BEGIN_NAMESPACE

/// define triangle_density_function
template <typename T> 
inline __device__ 
DEFINE_TRIANGLE_DENSITY_FUNCTION(T);
/// define exact_density_function
template <typename T> 
inline __device__ 
DEFINE_EXACT_DENSITY_FUNCTION(T);

template <typename T>
__global__ void __launch_bounds__(1024, 8) computeTriangleDensityMap(
    const T *x_tensor, const T *y_tensor,
    const T *node_size_x_clamped_tensor, const T *node_size_y_clamped_tensor,
    const T *offset_x_tensor, const T *offset_y_tensor,
    const T *ratio_tensor,
    const T *bin_center_x_tensor, const T *bin_center_y_tensor,
    const int num_nodes,
    const int num_bins_x, const int num_bins_y,
    const T xl, const T yl, const T xh, const T yh,
    const T half_bin_size_x, const T half_bin_size_y,
    const T bin_size_x, const T bin_size_y,
    const T inv_bin_size_x, const T inv_bin_size_y,
    T *density_map_tensor,
    const int *sorted_node_map ///< can be NULL if not sorted 
    )
{
    int index = blockIdx.x * blockDim.z + threadIdx.z;
    if (index < num_nodes)
    {
        int i = (sorted_node_map)? sorted_node_map[index] : index;

        // use stretched node size 
        T node_size_x = node_size_x_clamped_tensor[i];
        T node_size_y = node_size_y_clamped_tensor[i];
        T node_x = x_tensor[i] + offset_x_tensor[i];
        T node_y = y_tensor[i] + offset_y_tensor[i];
        T ratio = ratio_tensor[i];

        int bin_index_xl = int((node_x - xl) * inv_bin_size_x);
        int bin_index_xh = int(((node_x + node_size_x - xl) * inv_bin_size_x)) + 1; // exclusive
        bin_index_xl = DREAMPLACE_STD_NAMESPACE::max(bin_index_xl, 0);
        bin_index_xh = DREAMPLACE_STD_NAMESPACE::min(bin_index_xh, num_bins_x);

        int bin_index_yl = int((node_y - yl) * inv_bin_size_y);
        int bin_index_yh = int(((node_y + node_size_y - yl) * inv_bin_size_y)) + 1; // exclusive
        bin_index_yl = DREAMPLACE_STD_NAMESPACE::max(bin_index_yl, 0);
        bin_index_yh = DREAMPLACE_STD_NAMESPACE::min(bin_index_yh, num_bins_y);

        // update density potential map
        for (int k = bin_index_xl + threadIdx.y; k < bin_index_xh; k += blockDim.y)
        {
            T px = triangle_density_function(node_x, node_size_x, xl, k, bin_size_x);
            T px_by_ratio = px * ratio;

            for (int h = bin_index_yl + threadIdx.x; h < bin_index_yh; h += blockDim.x)
            {
                T py = triangle_density_function(node_y, node_size_y, yl, h, bin_size_y);
                T area = px_by_ratio * py;
                atomicAdd(&density_map_tensor[k * num_bins_y + h], area);
            }
        }
    }
}

/// @brief An unrolled way to compute the density map. 
/// Currently it is not as efficient as computeTriangleDensityMap, 
/// it has the potential to be better. 
/// It is not used for now. 
template <typename T>
__global__ void computeTriangleDensityMapUnroll(
    const T *x_tensor, const T *y_tensor,
    const T *node_size_x_clamped_tensor, const T *node_size_y_clamped_tensor,
    const T *offset_x_tensor, const T *offset_y_tensor,
    const T *ratio_tensor,
    const T *bin_center_x_tensor, const T *bin_center_y_tensor,
    const int num_nodes,
    const int num_bins_x, const int num_bins_y,
    const T xl, const T yl, const T xh, const T yh,
    const T half_bin_size_x, const T half_bin_size_y,
    const T bin_size_x, const T bin_size_y,
    const T inv_bin_size_x, const T inv_bin_size_y,
    T *density_map_tensor,
    const int *sorted_node_map ///< can be NULL if not sorted 
    )
{
    int index = blockIdx.x * blockDim.y + threadIdx.y;
    if (index < num_nodes)
    {
        int i = (sorted_node_map)? sorted_node_map[index] : index;

        T node_size_x = node_size_x_clamped_tensor[i];
        T node_size_y = node_size_y_clamped_tensor[i];
        T node_x = x_tensor[i] + offset_x_tensor[i];
        T node_y = y_tensor[i] + offset_y_tensor[i];
        T ratio = ratio_tensor[i];

        int bin_index_xl = int((node_x - xl) * inv_bin_size_x);
        int bin_index_xh = int(((node_x + node_size_x - xl) * inv_bin_size_x)); // inclusive
        bin_index_xl = DREAMPLACE_STD_NAMESPACE::max(bin_index_xl, 0);
        bin_index_xh = DREAMPLACE_STD_NAMESPACE::min(bin_index_xh, num_bins_x - 1);

        int bin_index_yl = int((node_y - yl) * inv_bin_size_y);
        int bin_index_yh = int(((node_y + node_size_y - yl) * inv_bin_size_y)); // inclusive
        bin_index_yl = DREAMPLACE_STD_NAMESPACE::max(bin_index_yl, 0);
        bin_index_yh = DREAMPLACE_STD_NAMESPACE::min(bin_index_yh, num_bins_y - 1);

        // update density potential map
        int k, h;

        int cond = ((bin_index_xl == bin_index_xh) << 1) | (bin_index_yl == bin_index_yh);
        switch (cond)
        {
        case 0:
        {
            T px_c = bin_size_x;

            T py_l = yl + bin_index_yl * bin_size_y + bin_size_y - node_y;
            T py_c = bin_size_y;
            T py_h = node_y + node_size_y - (bin_index_yh * bin_size_y + yl);

            T area_xc_yl = px_c * py_l * ratio;
            T area_xc_yc = px_c * py_c * ratio;
            T area_xc_yh = px_c * py_h * ratio;

            k = bin_index_xl;

            if (threadIdx.x == 0)
            {
                T px_l = xl + bin_index_xl * bin_size_x + bin_size_x - node_x;
                T area_xl_yl = px_l * py_l * ratio;
                T area_xl_yc = px_l * py_c * ratio;
                T area_xl_yh = px_l * py_h * ratio;
                h = bin_index_yl;
                atomicAdd(&density_map_tensor[k * num_bins_y + h], area_xl_yl);
                for (++h; h < bin_index_yh; ++h)
                {
                    atomicAdd(&density_map_tensor[k * num_bins_y + h], area_xl_yc);
                }
                atomicAdd(&density_map_tensor[k * num_bins_y + h], area_xl_yh);
                k += blockDim.x;
            }

            for (k += threadIdx.x; k < bin_index_xh; k += blockDim.x)
            {
                h = bin_index_yl;
                atomicAdd(&density_map_tensor[k * num_bins_y + h], area_xc_yl);
                for (++h; h < bin_index_yh; ++h)
                {
                    atomicAdd(&density_map_tensor[k * num_bins_y + h], area_xc_yc);
                }
                atomicAdd(&density_map_tensor[k * num_bins_y + h], area_xc_yh);
            }

            if (k == bin_index_xh)
            {
                T px_h = node_x + node_size_x - (bin_index_xh * bin_size_x + xl);
                T area_xh_yl = px_h * py_l * ratio;
                T area_xh_yc = px_h * py_c * ratio;
                T area_xh_yh = px_h * py_h * ratio;
                h = bin_index_yl;
                atomicAdd(&density_map_tensor[k * num_bins_y + h], area_xh_yl);
                for (++h; h < bin_index_yh; ++h)
                {
                    atomicAdd(&density_map_tensor[k * num_bins_y + h], area_xh_yc);
                }
                atomicAdd(&density_map_tensor[k * num_bins_y + h], area_xh_yh);
            }

            return;
        }
        case 1:
        {
            T py = yl + bin_index_yl * bin_size_y + bin_size_y - node_y;
            h = bin_index_yl;
            k = bin_index_xl;

            if (threadIdx.x == 0)
            {
                T px_l = xl + bin_index_xl * bin_size_x + bin_size_x - node_x;
                T area_xl = px_l * py * ratio;
                atomicAdd(&density_map_tensor[k * num_bins_y + h], area_xl);
                k += blockDim.x;
            }

            T px_c = bin_size_x;
            T area_xc = px_c * py * ratio;
            for (k += threadIdx.x; k < bin_index_xh; k += blockDim.x)
            {
                atomicAdd(&density_map_tensor[k * num_bins_y + h], area_xc);
            }

            if (k == bin_index_xh)
            {
                T px_h = node_x + node_size_x - (bin_index_xh * bin_size_x + xl);
                T area_xh = px_h * py * ratio;
                atomicAdd(&density_map_tensor[k * num_bins_y + h], area_xh);
            }

            return;
        }
        case 2:
        {
            T px = xl + bin_index_xl * bin_size_x + bin_size_x - node_x;
            k = bin_index_xl;
            h = bin_index_yl;

            if (threadIdx.x == 0)
            {
                T py_l = yl + bin_index_yl * bin_size_y + bin_size_y - node_y;
                T area_yl = px * py_l * ratio;
                atomicAdd(&density_map_tensor[k * num_bins_y + h], area_yl);
                h += blockDim.x;
            }

            T py_c = bin_size_y;
            T area_yc = px * py_c * ratio;
            for (h += threadIdx.x; h < bin_index_yh; h += blockDim.x)
            {
                atomicAdd(&density_map_tensor[k * num_bins_y + h], area_yc);
            }

            if (h == bin_index_yh)
            {
                T py_h = node_y + node_size_y - (bin_index_yh * bin_size_y + yl);
                T area_yh = px * py_h * ratio;
                atomicAdd(&density_map_tensor[k * num_bins_y + h], area_yh);
            }

            return;
        }
        case 3:
        {
            if (threadIdx.x == 0)
            {
                T px = xl + bin_index_xl * bin_size_x + bin_size_x - node_x;
                T py = yl + bin_index_yl * bin_size_y + bin_size_y - node_y;
                T area = px * py * ratio;

                k = bin_index_xl;
                h = bin_index_yl;
                atomicAdd(&density_map_tensor[k * num_bins_y + h], area);
            }
            return;
        }
        default:
            assert(0);
        }
    }
}

/// @brief Compute exact density map using cell-to-bin parallelization strategy
template <typename T>
__global__ void computeExactDensityMap(
        const T* flat_fixed_node_boxes, 
        const T* bin_center_x_tensor, const T* bin_center_y_tensor,
        const int num_boxes,
        const int num_bins_x, const int num_bins_y,
        const T xl, const T yl, const T xh, const T yh,
        const T bin_size_x, const T bin_size_y,
        const int num_impacted_bins_x, const int num_impacted_bins_y,
        bool fixed_node_flag,
        T *density_map_tensor
    )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // rank-one update density map
    if (i < num_boxes * num_impacted_bins_x * num_impacted_bins_y)
    {
        int box_id = i / (num_impacted_bins_x * num_impacted_bins_y);
        int residual_index = i - box_id * num_impacted_bins_x * num_impacted_bins_y;
        auto box = flat_fixed_node_boxes + i * 4; 
        T bxl = box[0]; 
        T byl = box[1];
        T bxh = box[2]; 
        T byh = box[3];
        // x direction
        int bin_index_xl = int((bxl - xl) / bin_size_x);
        bin_index_xl = DREAMPLACE_STD_NAMESPACE::max(bin_index_xl, 0);
        int k = bin_index_xl + int(residual_index / num_impacted_bins_y);
        if (k + 1 > num_bins_x)
        {
            return;
        }
        // y direction
        int bin_index_yl = int((byl - yl) / bin_size_y);
        bin_index_yl = DREAMPLACE_STD_NAMESPACE::max(bin_index_yl, 0);
        int h = bin_index_yl + (residual_index % num_impacted_bins_y);
        if (h + 1 > num_bins_y)
        {
            return;
        }

        T px = exact_density_function(bxl, bxh - bxl, bin_center_x_tensor[k], bin_size_x, xl, xh, fixed_node_flag);
        T py = exact_density_function(byl, byh - byl, bin_center_y_tensor[h], bin_size_y, yl, yh, fixed_node_flag);

        // still area 
        atomicAdd(&density_map_tensor[k*num_bins_y+h], px*py); 
    }
}

/// @brief Compute exact density map using cell-by-cell parallelization strategy
template <typename T>
__global__ void computeExactDensityMapCellByCell(
        const T* flat_fixed_node_boxes, 
        const T* bin_center_x_tensor, const T* bin_center_y_tensor,
        const int num_boxes,
        const int num_bins_x, const int num_bins_y,
        const T xl, const T yl, const T xh, const T yh,
        const T bin_size_x, const T bin_size_y,
        bool fixed_node_flag,
        T* density_map_tensor
        )
{
    auto box2bin = [&](T bxl, T byl, T bxh, T byh, T* buf_map){
        // x direction
        int bin_index_xl = int((bxl-xl)/bin_size_x);
        int bin_index_xh = int(ceil((bxh-xl)/bin_size_x))+1; // exclusive
        bin_index_xl = DREAMPLACE_STD_NAMESPACE::max(bin_index_xl, 0);
        bin_index_xh = DREAMPLACE_STD_NAMESPACE::min(bin_index_xh, num_bins_x);

        // y direction
        int bin_index_yl = int((byl-yl)/bin_size_y);
        int bin_index_yh = int(ceil((byh-yl)/bin_size_y))+1; // exclusive
        bin_index_yl = DREAMPLACE_STD_NAMESPACE::max(bin_index_yl, 0);
        bin_index_yh = DREAMPLACE_STD_NAMESPACE::min(bin_index_yh, num_bins_y);

        for (int k = bin_index_xl; k < bin_index_xh; ++k)
        {
            T px = exact_density_function(bxl, bxh-bxl, bin_center_x_tensor[k], bin_size_x, xl, xh, fixed_node_flag);
            for (int h = bin_index_yl; h < bin_index_yh; ++h)
            {
                T py = exact_density_function(byl, byh-byl, bin_center_y_tensor[h], bin_size_y, yl, yh, fixed_node_flag);

                // still area
                atomicAdd(&buf_map[k*num_bins_y+h], px * py);
            }
        }
    };

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_boxes)
    {
        auto box = flat_fixed_node_boxes + i * 4; 
        box2bin(box[0], box[1], box[2], box[3], density_map_tensor);
    }
}

template <typename T>
int computeTriangleDensityMapCudaLauncher(
    const T *x_tensor, const T *y_tensor,
    const T *node_size_x_clamped_tensor, const T *node_size_y_clamped_tensor,
    const T *offset_x_tensor, const T *offset_y_tensor,
    const T *ratio_tensor,
    const T *bin_center_x_tensor, const T *bin_center_y_tensor,
    int num_nodes, 
    const int num_bins_x, const int num_bins_y,
    int num_impacted_bins_x, int num_impacted_bins_y,
    const T xl, const T yl, const T xh, const T yh,
    const T bin_size_x, const T bin_size_y,
    T *density_map_tensor,
    const int *sorted_node_map)
{
    int thread_count = 64;
    // dim3 blockSize(4, thread_count, 1);
    dim3 blockSize(2, 2, thread_count);

    int block_count = (num_nodes - 1 + thread_count) / thread_count;
    computeTriangleDensityMap<<<block_count, blockSize>>>(
        x_tensor, y_tensor,
        node_size_x_clamped_tensor, node_size_y_clamped_tensor,
        offset_x_tensor, offset_y_tensor,
        ratio_tensor,
        bin_center_x_tensor, bin_center_y_tensor,
        num_nodes,
        num_bins_x, num_bins_y,
        xl, yl, xh, yh,
        bin_size_x / 2, bin_size_y / 2,
        bin_size_x, bin_size_y,
        1 / bin_size_x, 1 / bin_size_y,
        density_map_tensor,
        sorted_node_map
        );

    return 0;
}

/// @brief Compute exact density map using cell-to-bin parallelization strategy
template <typename T>
int computeExactDensityMapCudaLauncher(
        const T* flat_fixed_node_boxes, 
        const T* bin_center_x_tensor, const T* bin_center_y_tensor,
        const int num_boxes,
        const int num_bins_x, const int num_bins_y,
        const int num_impacted_bins_x, const int num_impacted_bins_y,
        const T xl, const T yl, const T xh, const T yh,
        const T bin_size_x, const T bin_size_y,
        bool fixed_node_flag,
        T *density_map_tensor
    )
{
    int thread_count = 512;
    int block_count = (num_boxes * num_impacted_bins_x * num_impacted_bins_y - 1 + thread_count) / thread_count;

    computeExactDensityMap<<<block_count, thread_count>>>(
            flat_fixed_node_boxes, 
            bin_center_x_tensor, bin_center_y_tensor,
            num_boxes,
            num_bins_x, num_bins_y,
            xl, yl, xh, yh,
            bin_size_x, bin_size_y,
            num_impacted_bins_x, num_impacted_bins_y,
            fixed_node_flag,
            density_map_tensor
        );

    return 0;
}

/// @brief Compute exact density map using cell-by-cell parallelization strategy
template <typename T>
int computeExactDensityMapCellByCellCudaLauncher(
        const T* flat_fixed_node_boxes, 
        const T* bin_center_x_tensor, const T* bin_center_y_tensor,
        const int num_boxes,
        const int num_bins_x, const int num_bins_y,
        const int num_impacted_bins_x, const int num_impacted_bins_y,
        const T xl, const T yl, const T xh, const T yh,
        const T bin_size_x, const T bin_size_y,
        bool fixed_node_flag,
        T *density_map_tensor
        )
{
    int thread_count = 256;
    int block_count = (num_boxes - 1 + thread_count) / thread_count;

    computeExactDensityMapCellByCell<<<block_count, thread_count>>>(
            flat_fixed_node_boxes, 
            bin_center_x_tensor, bin_center_y_tensor,
            num_boxes,
            num_bins_x, num_bins_y,
            xl, yl, xh, yh,
            bin_size_x, bin_size_y,
            fixed_node_flag,
            density_map_tensor
        );

    return 0;
}

#define REGISTER_KERNEL_LAUNCHER(T)                                                   \
    int instantiateComputeTriangleDensityMapLauncher(                                 \
        const T *x_tensor, const T *y_tensor,                                         \
        const T *node_size_x_clamped_tensor, const T *node_size_y_clamped_tensor,     \
        const T *offset_x_tensor, const T *offset_y_tensor,                           \
        const T *ratio_tensor,                                                        \
        const T *bin_center_x_tensor, const T *bin_center_y_tensor,                   \
        const int num_nodes,                                                          \
        const int num_bins_x, const int num_bins_y,                                   \
        const int num_impacted_bins_x, const int num_impacted_bins_y,                 \
        const T xl, const T yl, const T xh, const T yh,                               \
        const T bin_size_x, const T bin_size_y,                                       \
        T *density_map_tensor,                                                        \
        const int *sorted_node_map)                                                   \
    {                                                                                 \
        return computeTriangleDensityMapCudaLauncher(                                 \
            x_tensor, y_tensor,                                                       \
            node_size_x_clamped_tensor, node_size_y_clamped_tensor,                   \
            offset_x_tensor, offset_y_tensor,                                         \
            ratio_tensor,                                                             \
            bin_center_x_tensor, bin_center_y_tensor,                                 \
            num_nodes,                                                                \
            num_bins_x, num_bins_y,                                                   \
            num_impacted_bins_x, num_impacted_bins_y,                                 \
            xl, yl, xh, yh,                                                           \
            bin_size_x, bin_size_y,                                                   \
            density_map_tensor,                                                       \
            sorted_node_map);                                                         \
    }                                                                                 \
                                                                                      \
    int instantiateComputeExactDensityMapLauncher(                                    \
        const T* flat_fixed_node_boxes,                                               \
        const T *bin_center_x_tensor, const T *bin_center_y_tensor,                   \
        const int num_boxes,                                                          \
        const int num_bins_x, const int num_bins_y,                                   \
        const int num_impacted_bins_x, const int num_impacted_bins_y,                 \
        const T xl, const T yl, const T xh, const T yh,                               \
        const T bin_size_x, const T bin_size_y,                                       \
        bool fixed_node_flag,                                                         \
        T *density_map_tensor)                                                        \
    {                                                                                 \
        int a = computeExactDensityMapCudaLauncher(                                   \
            flat_fixed_node_boxes,                                                    \
            bin_center_x_tensor, bin_center_y_tensor,                                 \
            num_boxes,                                                                \
            num_bins_x, num_bins_y,                                                   \
            num_impacted_bins_x, num_impacted_bins_y,                                 \
            xl, yl, xh, yh,                                                           \
            bin_size_x, bin_size_y,                                                   \
            fixed_node_flag,                                                          \
            density_map_tensor);                                                      \
        int b = computeExactDensityMapCellByCellCudaLauncher(                         \
            flat_fixed_node_boxes,                                                    \
            bin_center_x_tensor, bin_center_y_tensor,                                 \
            num_boxes,                                                                \
            num_bins_x, num_bins_y,                                                   \
            num_impacted_bins_x, num_impacted_bins_y,                                 \
            xl, yl, xh, yh,                                                           \
            bin_size_x, bin_size_y,                                                   \
            fixed_node_flag,                                                          \
            density_map_tensor);                                                      \
        return a && b;                                                                \
    }

REGISTER_KERNEL_LAUNCHER(float);
REGISTER_KERNEL_LAUNCHER(double);

DREAMPLACE_END_NAMESPACE
