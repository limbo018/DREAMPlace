/**
 * @file   electric_force_cuda_kernel.cu
 * @author Yibo Lin
 * @date   Aug 2018
 */
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "cuda_runtime.h"
#include "utility/src/print.h"
#include "utility/src/Msg.h"

DREAMPLACE_BEGIN_NAMESPACE

#define SQRT2 1.4142135623730950488016887242096980785696718753769480731766797379907324784621

template <typename T>
__global__ void computeElectricForceAtomic(
    int num_bins_x, int num_bins_y,
    int num_impacted_bins_x, int num_impacted_bins_y,
    const T *field_map_x_tensor, const T *field_map_y_tensor,
    const T *x_tensor, const T *y_tensor,
    const T *node_size_x_tensor, const T *node_size_y_tensor,
    const T *bin_center_x_tensor, const T *bin_center_y_tensor,
    T xl, T yl, T xh, T yh,
    T bin_size_x, T bin_size_y,
    int num_nodes,
    T *grad_x_tensor, T *grad_y_tensor)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_nodes * num_impacted_bins_x * num_impacted_bins_y)
    {
        // density overflow function
        auto computeDensityFunc = [](T x, T node_size, T bin_center, T bin_size) {
            //return max(T(0.0), min(x+node_size, bin_center+bin_size/2) - max(x, bin_center-bin_size/2));
            // Yibo: cannot understand why negative overlap is allowed in RePlAce
            return min(x + node_size, bin_center + bin_size / 2) - max(x, bin_center - bin_size / 2);
        };
        int node_id = i / (num_impacted_bins_x * num_impacted_bins_y);
        int residual_index = i - node_id * num_impacted_bins_x * num_impacted_bins_y;

        // stretch node size to bin size
        T node_size_x = max(bin_size_x * SQRT2, node_size_x_tensor[node_id]);
        T node_size_y = max(bin_size_y * SQRT2, node_size_y_tensor[node_id]);
        T node_x = x_tensor[node_id] + node_size_x_tensor[node_id] / 2 - node_size_x / 2;
        T node_y = y_tensor[node_id] + node_size_y_tensor[node_id] / 2 - node_size_y / 2;
        // boundary condition
        //node_x = max(node_x, xl);
        //node_x = min(node_x, xh-node_size_x);
        //node_y = max(node_y, yl);
        //node_y = min(node_y, yh-node_size_y);

        // x direction
        int bin_index_xl = int((node_x - xl) / bin_size_x);
        bin_index_xl = max(bin_index_xl, 0);
        int k = bin_index_xl + int(residual_index / num_impacted_bins_y);
        if (k + 1 > num_bins_x || node_x + node_size_x <= bin_center_x_tensor[k] - bin_size_x / 2)
        {
            return;
        }
        // y direction
        int bin_index_yl = int((node_y - yl) / bin_size_y);
        bin_index_yl = max(bin_index_yl, 0);
        int h = bin_index_yl + (residual_index % num_impacted_bins_y);
        if (h + 1 > num_bins_y || node_y + node_size_y <= bin_center_y_tensor[h] - bin_size_y / 2)
        {
            return;
        }

        T px = computeDensityFunc(node_x, node_size_x, bin_center_x_tensor[k], bin_size_x);
        T py = computeDensityFunc(node_y, node_size_y, bin_center_y_tensor[h], bin_size_y);
        // scale the total area back to node area
        T area = px * py * (node_size_x_tensor[node_id] * node_size_y_tensor[node_id] / (node_size_x * node_size_y));

        // still area 
        atomicAdd(&grad_x_tensor[node_id], area*field_map_x_tensor[k*num_bins_y+h]);
        atomicAdd(&grad_y_tensor[node_id], area*field_map_y_tensor[k*num_bins_y+h]);
    }
}

template <typename T>
inline __device__ T computeDensityFunc(T x, T node_size, T bin_center, T half_bin_size)
{
    // Yibo: cannot understand why negative overlap is allowed in RePlAce
    return min(x + node_size, bin_center + half_bin_size) - max(x, bin_center - half_bin_size);
};

template <typename T>
inline __device__ T computeDensityFunc(T x, T node_size, T xl, int k, T bin_size)
{
    T bin_xl = xl + k * bin_size;
    return min(x + node_size, bin_xl + bin_size) - max(x, bin_xl);
};

template <typename T>
__global__ void __launch_bounds__(1024, 8) computeElectricForce(
    int num_bins_x, int num_bins_y,
    const T *field_map_x_tensor, const T *field_map_y_tensor,
    const T *x_tensor, const T *y_tensor,
    const T *node_size_x_clamped_tensor, const T *node_size_y_clamped_tensor,
    const T *offset_x_tensor, const T *offset_y_tensor,
    const T *ratio_tensor,
    const T *bin_center_x_tensor, const T *bin_center_y_tensor,
    T xl, T yl, T xh, T yh,
    const T half_bin_size_x, const T half_bin_size_y,
    const T bin_size_x, const T bin_size_y,
    const T inv_bin_size_x, const T inv_bin_size_y,
    int num_nodes,
    T *grad_x_tensor, T *grad_y_tensor)
{
    int i = blockIdx.x * blockDim.z + threadIdx.z;
    if (i < num_nodes)
    {
        // stretch node size to bin size
        T node_size_x = node_size_x_clamped_tensor[i];
        T node_size_y = node_size_y_clamped_tensor[i];
        T node_x = x_tensor[i] + offset_x_tensor[i];
        T node_y = y_tensor[i] + offset_y_tensor[i];
        T ratio = ratio_tensor[i];

        // Yibo: looks very weird implementation, but this is how RePlAce implements it
        // Zixuan and Jiaqi: use the common practice of floor
        int bin_index_xl = int((node_x - xl) * inv_bin_size_x);
        int bin_index_xh = int(((node_x + node_size_x - xl) * inv_bin_size_x)) + 1; // exclusive
        bin_index_xl = (bin_index_xl > 0) * bin_index_xl;                           // max(bin_index_xl, 0);
        bin_index_xh = min(bin_index_xh, num_bins_x);

        int bin_index_yl = int((node_y - yl) * inv_bin_size_y);
        int bin_index_yh = int(((node_y + node_size_y - yl) * inv_bin_size_y)) + 1; // exclusive
        bin_index_yl = (bin_index_yl > 0) * bin_index_yl;                           // max(bin_index_yl, 0);
        bin_index_yh = min(bin_index_yh, num_bins_y);

        // blockDim.x * blockDim.y threads will be used to update one node
        // shared memory is used to privatize the atomic memory access to thread block
        extern __shared__ unsigned char s_xy[];
        T *s_x = (T *)s_xy;
        T *s_y = s_x + blockDim.z;
        if (threadIdx.x == 0 && threadIdx.y == 0)
        {
            s_x[threadIdx.z] = s_y[threadIdx.z] = 0;
        }
        __syncthreads();

        T tmp_x, tmp_y;
        tmp_x = 0;
        tmp_y = 0;

        // update density potential map
        for (int k = bin_index_xl + threadIdx.y; k < bin_index_xh; k += blockDim.y)
        {
            T px = computeDensityFunc(node_x, node_size_x, xl, k, bin_size_x);

            for (int h = bin_index_yl + threadIdx.x; h < bin_index_yh; h += blockDim.x)
            {
                T py = computeDensityFunc(node_y, node_size_y, yl, h, bin_size_y);
                T area = px * py;

                tmp_x += area * field_map_x_tensor[k * num_bins_y + h];
                tmp_y += area * field_map_y_tensor[k * num_bins_y + h];
            }
        }

        atomicAdd(&s_x[threadIdx.z], tmp_x * ratio);
        atomicAdd(&s_y[threadIdx.z], tmp_y * ratio);
        __syncthreads();

        if (threadIdx.x == 0 && threadIdx.y == 0)
        {
            grad_x_tensor[i] = s_x[threadIdx.z];
            grad_y_tensor[i] = s_y[threadIdx.z];
        }
    }
}

template <typename T>
__global__ void __launch_bounds__(1024, 8) computeElectricForce(
    int num_bins_x, int num_bins_y,
    const T *field_map_x_tensor, const T *field_map_y_tensor,
    const T *x_tensor, const T *y_tensor,
    const T *node_size_x_clamped_tensor, const T *node_size_y_clamped_tensor,
    const T *offset_x_tensor, const T *offset_y_tensor,
    const T *ratio_tensor,
    const T *bin_center_x_tensor, const T *bin_center_y_tensor,
    T xl, T yl, T xh, T yh,
    const T half_bin_size_x, const T half_bin_size_y,
    const T bin_size_x, const T bin_size_y,
    const T inv_bin_size_x, const T inv_bin_size_y,
    int num_nodes,
    T *grad_x_tensor, T *grad_y_tensor,
    const int *sorted_node_map)
{
    int index = blockIdx.x * blockDim.z + threadIdx.z;
    if (index < num_nodes)
    {
        int i = sorted_node_map[index];

        // stretch node size to bin size
        T node_size_x = node_size_x_clamped_tensor[i];
        T node_size_y = node_size_y_clamped_tensor[i];
        T node_x = x_tensor[i] + offset_x_tensor[i];
        T node_y = y_tensor[i] + offset_y_tensor[i];
        T ratio = ratio_tensor[i];

        // Yibo: looks very weird implementation, but this is how RePlAce implements it
        // Zixuan and Jiaqi: use the common practice of floor
        int bin_index_xl = int((node_x - xl) * inv_bin_size_x);
        int bin_index_xh = int(((node_x + node_size_x - xl) * inv_bin_size_x)) + 1; // exclusive
        bin_index_xl = (bin_index_xl > 0) * bin_index_xl;                           // max(bin_index_xl, 0);
        bin_index_xh = min(bin_index_xh, num_bins_x);

        int bin_index_yl = int((node_y - yl) * inv_bin_size_y);
        int bin_index_yh = int(((node_y + node_size_y - yl) * inv_bin_size_y)) + 1; // exclusive
        bin_index_yl = (bin_index_yl > 0) * bin_index_yl;                           // max(bin_index_yl, 0);
        bin_index_yh = min(bin_index_yh, num_bins_y);

        // blockDim.x * blockDim.y threads will be used to update one node
        // shared memory is used to privatize the atomic memory access to thread block
        extern __shared__ unsigned char s_xy[];
        T *s_x = (T *)s_xy;
        T *s_y = s_x + blockDim.z;
        if (threadIdx.x == 0 && threadIdx.y == 0)
        {
            s_x[threadIdx.z] = s_y[threadIdx.z] = 0;
        }
        __syncthreads();

        T tmp_x, tmp_y;
        tmp_x = 0;
        tmp_y = 0;

        // update density potential map
        for (int k = bin_index_xl + threadIdx.y; k < bin_index_xh; k += blockDim.y)
        {
            T px = computeDensityFunc(node_x, node_size_x, xl, k, bin_size_x);

            for (int h = bin_index_yl + threadIdx.x; h < bin_index_yh; h += blockDim.x)
            {
                T py = computeDensityFunc(node_y, node_size_y, yl, h, bin_size_y);
                T area = px * py;

                tmp_x += area * field_map_x_tensor[k * num_bins_y + h];
                tmp_y += area * field_map_y_tensor[k * num_bins_y + h];
            }
        }

        atomicAdd(&s_x[threadIdx.z], tmp_x * ratio);
        atomicAdd(&s_y[threadIdx.z], tmp_y * ratio);
        __syncthreads();

        if (threadIdx.x == 0 && threadIdx.y == 0)
        {
            grad_x_tensor[i] = s_x[threadIdx.z];
            grad_y_tensor[i] = s_y[threadIdx.z];
        }
    }
}

template <typename T>
__global__ void computeElectricForceUnroll(
    int num_bins_x, int num_bins_y,
    const T *field_map_x_tensor, const T *field_map_y_tensor,
    const T *x_tensor, const T *y_tensor,
    const T *node_size_x_clamped_tensor, const T *node_size_y_clamped_tensor,
    const T *offset_x_tensor, const T *offset_y_tensor,
    const T *ratio_tensor,
    const T *bin_center_x_tensor, const T *bin_center_y_tensor,
    T xl, T yl, T xh, T yh,
    const T half_bin_size_x, const T half_bin_size_y,
    const T bin_size_x, const T bin_size_y,
    const T inv_bin_size_x, const T inv_bin_size_y,
    int num_nodes,
    T *grad_x_tensor, T *grad_y_tensor)
{
    int i = blockIdx.x * blockDim.y + threadIdx.y;
    if (i < num_nodes)
    {
        // stretch node size to bin size
        T node_size_x = node_size_x_clamped_tensor[i];
        T node_size_y = node_size_y_clamped_tensor[i];
        T node_x = x_tensor[i] + offset_x_tensor[i];
        T node_y = y_tensor[i] + offset_y_tensor[i];
        T ratio = ratio_tensor[i];

        // Yibo: looks very weird implementation, but this is how RePlAce implements it
        // Zixuan and Jiaqi: use the common practice of floor
        int bin_index_xl = int((node_x - xl) * inv_bin_size_x);
        int bin_index_xh = int(((node_x + node_size_x - xl) * inv_bin_size_x)); // inclusive
        bin_index_xl = (bin_index_xl > 0) * bin_index_xl;                       // max(bin_index_xl, 0);
        bin_index_xh = min(bin_index_xh, num_bins_x - 1);

        int bin_index_yl = int((node_y - yl) * inv_bin_size_y);
        int bin_index_yh = int(((node_y + node_size_y - yl) * inv_bin_size_y)); // inclusive
        bin_index_yl = (bin_index_yl > 0) * bin_index_yl;                       // max(bin_index_yl, 0);
        bin_index_yh = min(bin_index_yh, num_bins_y - 1);

        int k, h;

        int cond = ((bin_index_xl == bin_index_xh) << 1) | (bin_index_yl == bin_index_yh);
        switch (cond)
        {
        case 0:
        {
            // blockDim.x threads will be used to update one node
            // shared memory is used to privatize the atomic memory access to thread block
            extern __shared__ unsigned char shared_memory[];
            T *s_x = (T *)shared_memory;
            T *s_y = s_x + blockDim.y;
            if (threadIdx.x == 0)
            {
                s_x[threadIdx.y] = s_y[threadIdx.y] = 0;
            }
            __syncthreads();

            T tmp_x = 0;
            T tmp_y = 0;

            T px_c = bin_size_x;

            T py_l = yl + bin_index_yl * bin_size_y + bin_size_y - node_y;
            T py_c = bin_size_y;
            T py_h = node_y + node_size_y - (bin_index_yh * bin_size_y + yl);

            T area_xc_yl = px_c * py_l;
            T area_xc_yc = px_c * py_c;
            T area_xc_yh = px_c * py_h;

            k = bin_index_xl;
            if (threadIdx.x == 0)
            {
                T px_l = xl + bin_index_xl * bin_size_x + bin_size_x - node_x;
                T area_xl_yl = px_l * py_l;
                T area_xl_yc = px_l * py_c;
                T area_xl_yh = px_l * py_h;

                h = bin_index_yl;
                tmp_x = area_xl_yl * field_map_x_tensor[k * num_bins_y + h];
                tmp_y = area_xl_yl * field_map_y_tensor[k * num_bins_y + h];
                for (++h; h < bin_index_yh; ++h)
                {
                    tmp_x += area_xl_yc * field_map_x_tensor[k * num_bins_y + h];
                    tmp_y += area_xl_yc * field_map_y_tensor[k * num_bins_y + h];
                }
                tmp_x += area_xl_yh * field_map_x_tensor[k * num_bins_y + h];
                tmp_y += area_xl_yh * field_map_y_tensor[k * num_bins_y + h];
                k += blockDim.x;
            }

            for (k += threadIdx.x; k < bin_index_xh; k += blockDim.x)
            {
                h = bin_index_yl;
                tmp_x += area_xc_yl * field_map_x_tensor[k * num_bins_y + h];
                tmp_y += area_xc_yl * field_map_y_tensor[k * num_bins_y + h];
                for (++h; h < bin_index_yh; ++h)
                {
                    tmp_x += area_xc_yc * field_map_x_tensor[k * num_bins_y + h];
                    tmp_y += area_xc_yc * field_map_y_tensor[k * num_bins_y + h];
                }
                tmp_x += area_xc_yh * field_map_x_tensor[k * num_bins_y + h];
                tmp_y += area_xc_yh * field_map_y_tensor[k * num_bins_y + h];
            }

            if (k == bin_index_xh)
            {
                T px_h = node_x + node_size_x - (bin_index_xh * bin_size_x + xl);
                T area_xh_yl = px_h * py_l;
                T area_xh_yc = px_h * py_c;
                T area_xh_yh = px_h * py_h;

                h = bin_index_yl;
                tmp_x += area_xh_yl * field_map_x_tensor[k * num_bins_y + h];
                tmp_y += area_xh_yl * field_map_y_tensor[k * num_bins_y + h];
                for (++h; h < bin_index_yh; ++h)
                {
                    tmp_x += area_xh_yc * field_map_x_tensor[k * num_bins_y + h];
                    tmp_y += area_xh_yc * field_map_y_tensor[k * num_bins_y + h];
                }
                tmp_x += area_xh_yh * field_map_x_tensor[k * num_bins_y + h];
                tmp_y += area_xh_yh * field_map_y_tensor[k * num_bins_y + h];
            }

            atomicAdd(&s_x[threadIdx.y], tmp_x * ratio);
            atomicAdd(&s_y[threadIdx.y], tmp_y * ratio);
            __syncthreads();

            if (threadIdx.x == 0)
            {
                grad_x_tensor[i] = s_x[threadIdx.y];
                grad_y_tensor[i] = s_y[threadIdx.y];
            }

            return;
        }
        case 1:
        {
            extern __shared__ unsigned char shared_memory[];
            T *s_x = (T *)shared_memory;
            T *s_y = s_x + blockDim.y;
            if (threadIdx.x == 0)
            {
                s_x[threadIdx.y] = s_y[threadIdx.y] = 0;
            }
            __syncthreads();

            T tmp_x = 0;
            T tmp_y = 0;

            T py = yl + bin_index_yl * bin_size_y + bin_size_y - node_y;
            h = bin_index_yl;

            k = bin_index_xl;
            if (threadIdx.x == 0)
            {
                T px_l = xl + bin_index_xl * bin_size_x + bin_size_x - node_x;
                T area_xl = px_l * py;
                tmp_x = area_xl * field_map_x_tensor[k * num_bins_y + h];
                tmp_y = area_xl * field_map_y_tensor[k * num_bins_y + h];
                k += blockDim.x;
            }

            T px_c = bin_size_x;
            T area_xc = px_c * py;
            for (k += threadIdx.x; k < bin_index_xh; k += blockDim.x)
            {
                tmp_x += area_xc * field_map_x_tensor[k * num_bins_y + h];
                tmp_y += area_xc * field_map_y_tensor[k * num_bins_y + h];
            }

            if (k == bin_index_xh)
            {
                T px_h = node_x + node_size_x - (bin_index_xh * bin_size_x + xl);
                T area_xh = px_h * py;
                tmp_x += area_xh * field_map_x_tensor[k * num_bins_y + h];
                tmp_y += area_xh * field_map_y_tensor[k * num_bins_y + h];
            }

            atomicAdd(&s_x[threadIdx.y], tmp_x * ratio);
            atomicAdd(&s_y[threadIdx.y], tmp_y * ratio);
            __syncthreads();

            if (threadIdx.x == 0)
            {
                grad_x_tensor[i] = s_x[threadIdx.y];
                grad_y_tensor[i] = s_y[threadIdx.y];
            }

            return;
        }
        case 2:
        {
            extern __shared__ unsigned char shared_memory[];
            T *s_x = (T *)shared_memory;
            T *s_y = s_x + blockDim.y;
            if (threadIdx.x == 0)
            {
                s_x[threadIdx.y] = s_y[threadIdx.y] = 0;
            }
            __syncthreads();

            T tmp_x = 0;
            T tmp_y = 0;

            T px = xl + bin_index_xl * bin_size_x + bin_size_x - node_x;
            k = bin_index_xl;

            h = bin_index_yl;
            if (threadIdx.x == 0)
            {
                T py_l = yl + bin_index_yl * bin_size_y + bin_size_y - node_y;
                T area_yl = px * py_l;
                tmp_x = area_yl * field_map_x_tensor[k * num_bins_y + h];
                tmp_y = area_yl * field_map_y_tensor[k * num_bins_y + h];
                h += blockDim.x;
            }

            T py_c = bin_size_y;
            T area_yc = px * py_c;
            for (h += threadIdx.x; h < bin_index_yh; h += blockDim.x)
            {
                tmp_x += area_yc * field_map_x_tensor[k * num_bins_y + h];
                tmp_y += area_yc * field_map_y_tensor[k * num_bins_y + h];
            }

            if (h == bin_index_yh)
            {
                T py_h = node_y + node_size_y - (bin_index_yh * bin_size_y + yl);
                T area_yh = px * py_h;
                tmp_x += area_yh * field_map_x_tensor[k * num_bins_y + h];
                tmp_y += area_yh * field_map_y_tensor[k * num_bins_y + h];
            }

            atomicAdd(&s_x[threadIdx.y], tmp_x * ratio);
            atomicAdd(&s_y[threadIdx.y], tmp_y * ratio);
            __syncthreads();

            if (threadIdx.x == 0)
            {
                grad_x_tensor[i] = s_x[threadIdx.y];
                grad_y_tensor[i] = s_y[threadIdx.y];
            }

            return;
        }
        case 3:
        {
            if (threadIdx.x == 0)
            {
                T px = xl + bin_index_xl * bin_size_x + bin_size_x - node_x;
                T py = yl + bin_index_yl * bin_size_y + bin_size_y - node_y;
                T area_by_ratio = px * py * ratio;

                k = bin_index_xl;
                h = bin_index_yl;

                grad_x_tensor[i] = area_by_ratio * field_map_x_tensor[k * num_bins_y + h];
                grad_y_tensor[i] = area_by_ratio * field_map_y_tensor[k * num_bins_y + h];
            }

            return;
        }
        default:
            assert(0);
        }
    }
}

template <typename T>
__global__ void computeElectricForceUnroll(
    int num_bins_x, int num_bins_y,
    const T *field_map_x_tensor, const T *field_map_y_tensor,
    const T *x_tensor, const T *y_tensor,
    const T *node_size_x_clamped_tensor, const T *node_size_y_clamped_tensor,
    const T *offset_x_tensor, const T *offset_y_tensor,
    const T *ratio_tensor,
    const T *bin_center_x_tensor, const T *bin_center_y_tensor,
    T xl, T yl, T xh, T yh,
    const T half_bin_size_x, const T half_bin_size_y,
    const T bin_size_x, const T bin_size_y,
    const T inv_bin_size_x, const T inv_bin_size_y,
    int num_nodes,
    T *grad_x_tensor, T *grad_y_tensor,
    const int *sorted_node_map)
{
    int index = blockIdx.x * blockDim.y + threadIdx.y;
    if (index < num_nodes)
    {
        int i = sorted_node_map[index];

        // stretch node size to bin size
        T node_size_x = node_size_x_clamped_tensor[i];
        T node_size_y = node_size_y_clamped_tensor[i];
        T node_x = x_tensor[i] + offset_x_tensor[i];
        T node_y = y_tensor[i] + offset_y_tensor[i];
        T ratio = ratio_tensor[i];

        // Yibo: looks very weird implementation, but this is how RePlAce implements it
        // Zixuan and Jiaqi: use the common practice of floor
        int bin_index_xl = int((node_x - xl) * inv_bin_size_x);
        int bin_index_xh = int(((node_x + node_size_x - xl) * inv_bin_size_x)); // inclusive
        bin_index_xl = (bin_index_xl > 0) * bin_index_xl;                       // max(bin_index_xl, 0);
        bin_index_xh = min(bin_index_xh, num_bins_x - 1);

        int bin_index_yl = int((node_y - yl) * inv_bin_size_y);
        int bin_index_yh = int(((node_y + node_size_y - yl) * inv_bin_size_y)); // inclusive
        bin_index_yl = (bin_index_yl > 0) * bin_index_yl;                       // max(bin_index_yl, 0);
        bin_index_yh = min(bin_index_yh, num_bins_y - 1);

        int k, h;

        int cond = ((bin_index_xl == bin_index_xh) << 1) | (bin_index_yl == bin_index_yh);
        switch (cond)
        {
        case 0:
        {
            // blockDim.x threads will be used to update one node
            // shared memory is used to privatize the atomic memory access to thread block
            extern __shared__ unsigned char shared_memory[];
            T *s_x = (T *)shared_memory;
            T *s_y = s_x + blockDim.y;
            if (threadIdx.x == 0)
            {
                s_x[threadIdx.y] = s_y[threadIdx.y] = 0;
            }
            __syncthreads();

            T tmp_x = 0;
            T tmp_y = 0;

            T px_c = bin_size_x;

            T py_l = yl + bin_index_yl * bin_size_y + bin_size_y - node_y;
            T py_c = bin_size_y;
            T py_h = node_y + node_size_y - (bin_index_yh * bin_size_y + yl);

            T area_xc_yl = px_c * py_l;
            T area_xc_yc = px_c * py_c;
            T area_xc_yh = px_c * py_h;

            k = bin_index_xl;
            if (threadIdx.x == 0)
            {
                T px_l = xl + bin_index_xl * bin_size_x + bin_size_x - node_x;
                T area_xl_yl = px_l * py_l;
                T area_xl_yc = px_l * py_c;
                T area_xl_yh = px_l * py_h;

                h = bin_index_yl;
                tmp_x = area_xl_yl * field_map_x_tensor[k * num_bins_y + h];
                tmp_y = area_xl_yl * field_map_y_tensor[k * num_bins_y + h];
                for (++h; h < bin_index_yh; ++h)
                {
                    tmp_x += area_xl_yc * field_map_x_tensor[k * num_bins_y + h];
                    tmp_y += area_xl_yc * field_map_y_tensor[k * num_bins_y + h];
                }
                tmp_x += area_xl_yh * field_map_x_tensor[k * num_bins_y + h];
                tmp_y += area_xl_yh * field_map_y_tensor[k * num_bins_y + h];
                k += blockDim.x;
            }

            for (k += threadIdx.x; k < bin_index_xh; k += blockDim.x)
            {
                h = bin_index_yl;
                tmp_x += area_xc_yl * field_map_x_tensor[k * num_bins_y + h];
                tmp_y += area_xc_yl * field_map_y_tensor[k * num_bins_y + h];
                for (++h; h < bin_index_yh; ++h)
                {
                    tmp_x += area_xc_yc * field_map_x_tensor[k * num_bins_y + h];
                    tmp_y += area_xc_yc * field_map_y_tensor[k * num_bins_y + h];
                }
                tmp_x += area_xc_yh * field_map_x_tensor[k * num_bins_y + h];
                tmp_y += area_xc_yh * field_map_y_tensor[k * num_bins_y + h];
            }

            if (k == bin_index_xh)
            {
                T px_h = node_x + node_size_x - (bin_index_xh * bin_size_x + xl);
                T area_xh_yl = px_h * py_l;
                T area_xh_yc = px_h * py_c;
                T area_xh_yh = px_h * py_h;

                h = bin_index_yl;
                tmp_x += area_xh_yl * field_map_x_tensor[k * num_bins_y + h];
                tmp_y += area_xh_yl * field_map_y_tensor[k * num_bins_y + h];
                for (++h; h < bin_index_yh; ++h)
                {
                    tmp_x += area_xh_yc * field_map_x_tensor[k * num_bins_y + h];
                    tmp_y += area_xh_yc * field_map_y_tensor[k * num_bins_y + h];
                }
                tmp_x += area_xh_yh * field_map_x_tensor[k * num_bins_y + h];
                tmp_y += area_xh_yh * field_map_y_tensor[k * num_bins_y + h];
            }

            atomicAdd(&s_x[threadIdx.y], tmp_x * ratio);
            atomicAdd(&s_y[threadIdx.y], tmp_y * ratio);
            __syncthreads();

            if (threadIdx.x == 0)
            {
                grad_x_tensor[i] = s_x[threadIdx.y];
                grad_y_tensor[i] = s_y[threadIdx.y];
            }

            return;
        }
        case 1:
        {
            extern __shared__ unsigned char shared_memory[];
            T *s_x = (T *)shared_memory;
            T *s_y = s_x + blockDim.y;
            if (threadIdx.x == 0)
            {
                s_x[threadIdx.y] = s_y[threadIdx.y] = 0;
            }
            __syncthreads();

            T tmp_x = 0;
            T tmp_y = 0;

            T py = yl + bin_index_yl * bin_size_y + bin_size_y - node_y;
            h = bin_index_yl;

            k = bin_index_xl;
            if (threadIdx.x == 0)
            {
                T px_l = xl + bin_index_xl * bin_size_x + bin_size_x - node_x;
                T area_xl = px_l * py;
                tmp_x = area_xl * field_map_x_tensor[k * num_bins_y + h];
                tmp_y = area_xl * field_map_y_tensor[k * num_bins_y + h];
                k += blockDim.x;
            }

            T px_c = bin_size_x;
            T area_xc = px_c * py;
            for (k += threadIdx.x; k < bin_index_xh; k += blockDim.x)
            {
                tmp_x += area_xc * field_map_x_tensor[k * num_bins_y + h];
                tmp_y += area_xc * field_map_y_tensor[k * num_bins_y + h];
            }

            if (k == bin_index_xh)
            {
                T px_h = node_x + node_size_x - (bin_index_xh * bin_size_x + xl);
                T area_xh = px_h * py;
                tmp_x += area_xh * field_map_x_tensor[k * num_bins_y + h];
                tmp_y += area_xh * field_map_y_tensor[k * num_bins_y + h];
            }

            atomicAdd(&s_x[threadIdx.y], tmp_x * ratio);
            atomicAdd(&s_y[threadIdx.y], tmp_y * ratio);
            __syncthreads();

            if (threadIdx.x == 0)
            {
                grad_x_tensor[i] = s_x[threadIdx.y];
                grad_y_tensor[i] = s_y[threadIdx.y];
            }

            return;
        }
        case 2:
        {
            extern __shared__ unsigned char shared_memory[];
            T *s_x = (T *)shared_memory;
            T *s_y = s_x + blockDim.y;
            if (threadIdx.x == 0)
            {
                s_x[threadIdx.y] = s_y[threadIdx.y] = 0;
            }
            __syncthreads();

            T tmp_x = 0;
            T tmp_y = 0;

            T px = xl + bin_index_xl * bin_size_x + bin_size_x - node_x;
            k = bin_index_xl;

            h = bin_index_yl;
            if (threadIdx.x == 0)
            {
                T py_l = yl + bin_index_yl * bin_size_y + bin_size_y - node_y;
                T area_yl = px * py_l;
                tmp_x = area_yl * field_map_x_tensor[k * num_bins_y + h];
                tmp_y = area_yl * field_map_y_tensor[k * num_bins_y + h];
                h += blockDim.x;
            }

            T py_c = bin_size_y;
            T area_yc = px * py_c;
            for (h += threadIdx.x; h < bin_index_yh; h += blockDim.x)
            {
                tmp_x += area_yc * field_map_x_tensor[k * num_bins_y + h];
                tmp_y += area_yc * field_map_y_tensor[k * num_bins_y + h];
            }

            if (h == bin_index_yh)
            {
                T py_h = node_y + node_size_y - (bin_index_yh * bin_size_y + yl);
                T area_yh = px * py_h;
                tmp_x += area_yh * field_map_x_tensor[k * num_bins_y + h];
                tmp_y += area_yh * field_map_y_tensor[k * num_bins_y + h];
            }

            atomicAdd(&s_x[threadIdx.y], tmp_x * ratio);
            atomicAdd(&s_y[threadIdx.y], tmp_y * ratio);
            __syncthreads();

            if (threadIdx.x == 0)
            {
                grad_x_tensor[i] = s_x[threadIdx.y];
                grad_y_tensor[i] = s_y[threadIdx.y];
            }

            return;
        }
        case 3:
        {
            if (threadIdx.x == 0)
            {
                T px = xl + bin_index_xl * bin_size_x + bin_size_x - node_x;
                T py = yl + bin_index_yl * bin_size_y + bin_size_y - node_y;
                T area_by_ratio = px * py * ratio;

                k = bin_index_xl;
                h = bin_index_yl;

                grad_x_tensor[i] = area_by_ratio * field_map_x_tensor[k * num_bins_y + h];
                grad_y_tensor[i] = area_by_ratio * field_map_y_tensor[k * num_bins_y + h];
            }

            return;
        }
        default:
            assert(0);
        }
    }
}

template <typename T>
int computeElectricForceCudaLauncher(
    int num_bins_x, int num_bins_y,
    int num_movable_impacted_bins_x, int num_movable_impacted_bins_y,
    int num_filler_impacted_bins_x, int num_filler_impacted_bins_y,
    const T *field_map_x_tensor, const T *field_map_y_tensor,
    const T *x_tensor, const T *y_tensor,
    const T *node_size_x_clamped_tensor, const T *node_size_y_clamped_tensor,
    const T *offset_x_tensor, const T *offset_y_tensor,
    const T *ratio_tensor,
    const T *bin_center_x_tensor, const T *bin_center_y_tensor,
    T xl, T yl, T xh, T yh,
    T bin_size_x, T bin_size_y,
    int num_nodes, int num_movable_nodes, int num_filler_nodes,
    T *grad_x_tensor, T *grad_y_tensor,
    const int *sorted_node_map)
{
    int thread_count = 64;
    // dim3 blockSize(4, thread_count, 1);
    dim3 blockSize(2, 2, thread_count);
    size_t shared_mem_size = sizeof(T) * thread_count * 2;

    int block_count_nodes = (num_movable_nodes + thread_count - 1) / thread_count;
    computeElectricForce<<<block_count_nodes, blockSize, shared_mem_size>>>(
        num_bins_x, num_bins_y,
        field_map_x_tensor, field_map_y_tensor,
        x_tensor, y_tensor,
        node_size_x_clamped_tensor, node_size_y_clamped_tensor,
        offset_x_tensor, offset_y_tensor,
        ratio_tensor,
        bin_center_x_tensor, bin_center_y_tensor,
        xl, yl, xh, yh,
        bin_size_x / 2, bin_size_y / 2,
        bin_size_x, bin_size_y,
        1 / bin_size_x, 1 / bin_size_y,
        num_movable_nodes,
        grad_x_tensor, grad_y_tensor,
        sorted_node_map);

    if (num_filler_nodes)
    {
        cudaError_t status;
        cudaStream_t stream_filler;

        status = cudaStreamCreate(&stream_filler);
        if (status != cudaSuccess)
        {
            printf("cudaStreamCreate failed for stream_filler\n");
            fflush(stdout);
            return 1;
        }

        block_count_nodes = (num_filler_nodes + thread_count - 1) / thread_count;
        int num_physical_nodes = num_nodes - num_filler_nodes;
        computeElectricForce<<<block_count_nodes, blockSize, shared_mem_size, stream_filler>>>(
            num_bins_x, num_bins_y,
            field_map_x_tensor, field_map_y_tensor,
            x_tensor + num_physical_nodes, y_tensor + num_physical_nodes,
            node_size_x_clamped_tensor + num_physical_nodes, node_size_y_clamped_tensor + num_physical_nodes,
            offset_x_tensor + num_physical_nodes, offset_y_tensor + num_physical_nodes,
            ratio_tensor + num_physical_nodes,
            bin_center_x_tensor, bin_center_y_tensor,
            xl, yl, xh, yh,
            bin_size_x / 2, bin_size_y / 2,
            bin_size_x, bin_size_y,
            1 / bin_size_x, 1 / bin_size_y,
            num_filler_nodes,
            grad_x_tensor + num_physical_nodes, grad_y_tensor + num_physical_nodes);

        status = cudaStreamDestroy(stream_filler);
        if (status != cudaSuccess)
        {
            printf("stream_filler destroy failed\n");
            fflush(stdout);
            return 1;
        }
    }

    return 0;
}

#define REGISTER_KERNEL_LAUNCHER(T)                                               \
    int instantiateComputeElectricForceLauncher(                                  \
        int num_bins_x, int num_bins_y,                                           \
        int num_movable_impacted_bins_x, int num_movable_impacted_bins_y,         \
        int num_filler_impacted_bins_x, int num_filler_impacted_bins_y,           \
        const T *field_map_x_tensor, const T *field_map_y_tensor,                 \
        const T *x_tensor, const T *y_tensor,                                     \
        const T *node_size_x_clamped_tensor, const T *node_size_y_clamped_tensor, \
        const T *offset_x_tensor, const T *offset_y_tensor,                       \
        const T *ratio_tensor,                                                    \
        const T *bin_center_x_tensor, const T *bin_center_y_tensor,               \
        T xl, T yl, T xh, T yh,                                                   \
        T bin_size_x, T bin_size_y,                                               \
        int num_nodes, int num_movable_nodes, int num_filler_nodes,               \
        T *grad_x_tensor, T *grad_y_tensor,                                       \
        const int *sorted_node_map)                                               \
    {                                                                             \
        return computeElectricForceCudaLauncher(                                  \
            num_bins_x, num_bins_y,                                               \
            num_movable_impacted_bins_x, num_movable_impacted_bins_y,             \
            num_filler_impacted_bins_x, num_filler_impacted_bins_y,               \
            field_map_x_tensor, field_map_y_tensor,                               \
            x_tensor, y_tensor,                                                   \
            node_size_x_clamped_tensor, node_size_y_clamped_tensor,               \
            offset_x_tensor, offset_y_tensor,                                     \
            ratio_tensor,                                                         \
            bin_center_x_tensor, bin_center_y_tensor,                             \
            xl, yl, xh, yh,                                                       \
            bin_size_x, bin_size_y,                                               \
            num_nodes, num_movable_nodes, num_filler_nodes,                       \
            grad_x_tensor, grad_y_tensor,                                         \
            sorted_node_map);                                                     \
    }

REGISTER_KERNEL_LAUNCHER(float);
REGISTER_KERNEL_LAUNCHER(double);

DREAMPLACE_END_NAMESPACE
