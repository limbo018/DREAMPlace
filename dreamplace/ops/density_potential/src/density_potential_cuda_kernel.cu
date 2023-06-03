/**
 * @file   density_potential_cuda_kernel.cu
 * @author Yibo Lin
 * @date   Jun 2018
 * @brief  Compute density potential according to NTUPlace3 (https://doi.org/10.1109/TCAD.2008.923063).
 *          This is for movable and filler cells.
 */
#include <stdio.h>
#include <float.h>
#include <cstdint>
#include "cuda_runtime.h"
#include "utility/src/utils.cuh"

DREAMPLACE_BEGIN_NAMESPACE

#if 0
template <typename T>
__global__ void computePaddingDensityMap(
        const int num_bins_x, const int num_bins_y,
        const int padding,
        T* density_map_tensor)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_nodes*num_impacted_bins_x*num_impacted_bins_y; i += blockDim.x * gridDim.x)
    {
        int ix = i/num_bins_y;
        int iy = i-ix*num_bins_y;

        if (ix < padding)
        {
            density_map_tensor[i] = density_map_tensor[padding*num_bins_y+iy];
        }
    }
}
#endif

template <typename T>
__global__ void computeDensityMap(
        const T* x_tensor, const T* y_tensor,
        const T* node_size_x_tensor, const T* node_size_y_tensor,
        const T* ax_tensor, const T* bx_tensor, const T* cx_tensor,
        const T* ay_tensor, const T* by_tensor, const T* cy_tensor,
        const T* bin_center_x_tensor, const T* bin_center_y_tensor,
        const int num_nodes,
        const int num_bins_x, const int num_bins_y,
        const T xl, const T yl, const T xh, const T yh,
        const T bin_size_x, const T bin_size_y,
        const int num_impacted_bins_x, const int num_impacted_bins_y,
        T* density_map_tensor)
{
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t bound = int64_t(num_nodes)*num_impacted_bins_x*num_impacted_bins_y;
    // rank-one update density map
    if (i < bound)
    {
        // density potential function
        auto computeDensityPotentialFunc = [](T x, T node_size, T bin_center, T bin_size, T a, T b, T c){
            // from origin to center
            x += node_size/2;
            //printf("x = %g, bin_center = %g\n", x, bin_center);
            T dist = fabs(x-bin_center);
            //printf("dist = %g\n", dist);
            T partition1 = node_size/2+bin_size;
            //printf("partition1 = %g\n", partition1);
            T partition2 = partition1+bin_size;
            //printf("partition2 = %g\n", partition2);
            //printf("a = %g, b = %g, c = %g\n", a, b, c);
            if (dist < partition1)
            {
                return c*(1-a*dist*dist);
            }
            else if (dist < partition2)
            {
                return c*(b*(dist-partition2)*(dist-partition2));
            }
            else
            {
                return T(0.0);
            }
        };
        int node_id = i/(num_impacted_bins_x*num_impacted_bins_y);
        int residual_index = i-node_id*num_impacted_bins_x*num_impacted_bins_y;
        // x direction
        int bin_index_xl = int((x_tensor[node_id]-xl-2*bin_size_x)/bin_size_x);
        bin_index_xl = max(bin_index_xl, 0);
        int k = bin_index_xl+int(residual_index / num_impacted_bins_y);
        if (k+1 > num_bins_x)
        {
            return;
        }
        // y direction
        int bin_index_yl = int((y_tensor[node_id]-yl-2*bin_size_y)/bin_size_y);
        bin_index_yl = max(bin_index_yl, 0);
        int h = bin_index_yl+(residual_index % num_impacted_bins_y);
        if (h+1 > num_bins_y)
        {
            return;
        }

        T px = computeDensityPotentialFunc(x_tensor[node_id], node_size_x_tensor[node_id], bin_center_x_tensor[k], bin_size_x, ax_tensor[node_id], bx_tensor[node_id], cx_tensor[node_id]);
        T py = computeDensityPotentialFunc(y_tensor[node_id], node_size_y_tensor[node_id], bin_center_y_tensor[h], bin_size_y, ay_tensor[node_id], by_tensor[node_id], cy_tensor[node_id]);
        //printf("px[%d, %d] = %g, py[%d, %d] = %g\n", k, h, px, k, h, py);
        // still area 
        atomicAdd(&density_map_tensor[k*num_bins_y+h], px*py); 
    }
}

template <typename T>
__global__ void computeDensityGradient(
        const T* x_tensor, const T* y_tensor,
        const T* node_size_x_tensor, const T* node_size_y_tensor,
        const T* ax_tensor, const T* bx_tensor, const T* cx_tensor,
        const T* ay_tensor, const T* by_tensor, const T* cy_tensor,
        const T* bin_center_x_tensor, const T* bin_center_y_tensor,
        const int num_nodes,
        const int num_bins_x, const int num_bins_y,
        const T xl, const T yl, const T xh, const T yh,
        const T bin_size_x, const T bin_size_y,
        const int num_impacted_bins_x, const int num_impacted_bins_y,
        const T* grad_tensor, const T target_area,
        const T* density_map_tensor,
        T* grad_x_tensor, T* grad_y_tensor
        )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // rank-one update density map
    if (i < num_nodes)
    {
        // density potential function
        auto computeDensityPotentialFunc = [](T x, T node_size, T bin_center, T bin_size, T a, T b, T c){
            // from origin to center
            x += node_size/2;
            //printf("x = %g, bin_center = %g\n", x, bin_center);
            T dist = fabs(x-bin_center);
            //printf("dist = %g\n", dist);
            T partition1 = node_size/2+bin_size;
            //printf("partition1 = %g\n", partition1);
            T partition2 = partition1+bin_size;
            //printf("partition2 = %g\n", partition2);
            //printf("a = %g, b = %g, c = %g\n", a, b, c);
            if (dist < partition1)
            {
                return c*(1-a*dist*dist);
            }
            else if (dist < partition2)
            {
                return c*(b*(dist-partition2)*(dist-partition2));
            }
            else
            {
                return T(0.0);
            }
        };
        // density potential gradient function
        auto computeDensityPotentialGradFunc = [](T x, T node_size, T bin_center, T bin_size, T a, T b, T c){
            // from origin to center
            x += node_size/2;
            T dist = fabs(x-bin_center);
            T partition1 = node_size/2+bin_size;
            T partition2 = partition1+bin_size;
            if (dist < partition1)
            {
                return -2*c*a*(x-bin_center);
            }
            else if (dist < partition2)
            {
                T sign = (x < bin_center)? -1.0 : 1.0;
                return 2*c*b*(dist-partition2)*sign;
            }
            else
            {
                return T(0.0);
            }
        };
        int bin_index_xl = int((x_tensor[i]-xl-2*bin_size_x)/bin_size_x);
        int bin_index_xh = int(ceil((x_tensor[i]-xl+node_size_x_tensor[i]+2*bin_size_x)/bin_size_x))+1; // exclusive
        bin_index_xl = max(bin_index_xl, 0);
        // be careful about the bin_index_xl and bin_index_xh here
        // the assumption is that num_bins_x >= num_impacted_bins_x
        // each row of the px matrix should be filled with num_impacted_bins_x columns
        bin_index_xl = min(bin_index_xl, num_bins_x-num_impacted_bins_x);
        bin_index_xh = min(bin_index_xh, num_bins_x);
        //int bin_index_xh = bin_index_xl+num_impacted_bins_x;

        int bin_index_yl = int((y_tensor[i]-yl-2*bin_size_y)/bin_size_y);
        int bin_index_yh = int(ceil((y_tensor[i]-yl+node_size_y_tensor[i]+2*bin_size_y)/bin_size_y))+1; // exclusive
        bin_index_yl = max(bin_index_yl, 0);
        // be careful about the bin_index_yl and bin_index_yh here
        // the assumption is that num_bins_y >= num_impacted_bins_y
        // each row of the py matrix should be filled with num_impacted_bins_y columns
        bin_index_yl = min(bin_index_yl, num_bins_y-num_impacted_bins_y);
        bin_index_yh = min(bin_index_yh, num_bins_y);
        //int bin_index_yh = bin_index_yl+num_impacted_bins_y;

        grad_x_tensor[i] = 0;
        grad_y_tensor[i] = 0;
        // update density potential map
        for (int k = bin_index_xl; k < bin_index_xh; ++k)
        {
            T px = computeDensityPotentialFunc(x_tensor[i], node_size_x_tensor[i], bin_center_x_tensor[k], bin_size_x, ax_tensor[i], bx_tensor[i], cx_tensor[i]);
            T gradx = computeDensityPotentialGradFunc(x_tensor[i], node_size_x_tensor[i], bin_center_x_tensor[k], bin_size_x, ax_tensor[i], bx_tensor[i], cx_tensor[i]);
            for (int h = bin_index_yl; h < bin_index_yh; ++h)
            {
                T py = computeDensityPotentialFunc(y_tensor[i], node_size_y_tensor[i], bin_center_y_tensor[h], bin_size_y, ay_tensor[i], by_tensor[i], cy_tensor[i]);
                T grady = computeDensityPotentialGradFunc(y_tensor[i], node_size_y_tensor[i], bin_center_y_tensor[h], bin_size_y, ay_tensor[i], by_tensor[i], cy_tensor[i]);

                T delta = density_map_tensor[k*num_bins_y+h]-target_area;
                //delta = max(delta, (T)0);

                grad_x_tensor[i] += 2*delta*py*gradx;
                grad_y_tensor[i] += 2*delta*px*grady;

            }
        }

        grad_x_tensor[i] *= *grad_tensor;
        grad_y_tensor[i] *= *grad_tensor;
    }
}

template <typename T>
int computeDensityPotentialMapCudaLauncher(
        const T* x_tensor, const T* y_tensor,
        const T* node_size_x_tensor, const T* node_size_y_tensor,
        const T* ax_tensor, const T* bx_tensor, const T* cx_tensor,
        const T* ay_tensor, const T* by_tensor, const T* cy_tensor,
        const T* bin_center_x_tensor, const T* bin_center_y_tensor,
        const int num_impacted_bins_x, const int num_impacted_bins_y,
        const int mat_size_x, const int mat_size_y,
        const int num_nodes,
        const int num_bins_x, const int num_bins_y, const int padding,
        const T xl, const T yl, const T xh, const T yh,
        const T bin_size_x, const T bin_size_y,
        const T target_area,
        T* density_map_tensor,
        const T* grad_tensor,
        T* grad_x_tensor, T* grad_y_tensor
        )
{
    int64_t block_count;
    int64_t thread_count = 512;

    // compute gradient
    if (grad_tensor)
    {
        block_count = (num_nodes - 1 + thread_count) / thread_count;

        computeDensityGradient<<<block_count, thread_count>>>(
                x_tensor, y_tensor,
                node_size_x_tensor, node_size_y_tensor,
                ax_tensor, bx_tensor, cx_tensor,
                ay_tensor, by_tensor, cy_tensor,
                bin_center_x_tensor, bin_center_y_tensor,
                num_nodes,
                num_bins_x, num_bins_y,
                xl, yl, xh, yh,
                bin_size_x, bin_size_y,
                num_impacted_bins_x, num_impacted_bins_y,
                grad_tensor, target_area,
                density_map_tensor,
                grad_x_tensor, grad_y_tensor
                );

        // print gradient
        //printArray(grad_x_tensor, 10, "grad_x_tensor");
        //printArray(grad_y_tensor, 10, "grad_y_tensor");
    }
    else
    {
        block_count = (int64_t(num_nodes)*num_impacted_bins_x*num_impacted_bins_y - 1 + thread_count) / thread_count;

        computeDensityMap<<<block_count, thread_count>>>(
                x_tensor, y_tensor,
                node_size_x_tensor, node_size_y_tensor,
                ax_tensor, bx_tensor, cx_tensor,
                ay_tensor, by_tensor, cy_tensor,
                bin_center_x_tensor, bin_center_y_tensor,
                num_nodes,
                num_bins_x, num_bins_y,
                xl, yl, xh, yh,
                bin_size_x, bin_size_y,
                num_impacted_bins_x, num_impacted_bins_y,
                density_map_tensor);

        // print density map
        //print2DArray(density_map_tensor, num_bins_x, num_bins_y, "potential density_map_tensor");
        //printScalar(bin_size_x, "bin_size_x");
        //printScalar(bin_size_y, "bin_size_y");
        //printScalar(target_area, "target_area");

    }

    return 0;
}

#define REGISTER_KERNEL_LAUNCHER(T) \
    template int computeDensityPotentialMapCudaLauncher<T>(\
            const T* x_tensor, const T* y_tensor, \
            const T* node_size_x_tensor, const T* node_size_y_tensor, \
            const T* ax_tensor, const T* bx_tensor, const T* cx_tensor, \
            const T* ay_tensor, const T* by_tensor, const T* cy_tensor, \
            const T* bin_center_x_tensor, const T* bin_center_y_tensor, \
            const int num_impacted_bins_x, const int num_impacted_bins_y, \
            const int mat_size_x, const int mat_size_y, \
            const int num_nodes, \
            const int num_bins_x, const int num_bins_y, const int padding, \
            const T xl, const T yl, const T xh, const T yh, \
            const T bin_size_x, const T bin_size_y, \
            const T target_area, \
            T* density_map_tensor, \
            const T* grad_tensor, \
            T* grad_x_tensor, T* grad_y_tensor \
            ); 

REGISTER_KERNEL_LAUNCHER(float);
REGISTER_KERNEL_LAUNCHER(double);

DREAMPLACE_END_NAMESPACE
