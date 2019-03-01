/**
 * @file   electric_density_map_cuda_kernel.cu
 * @author Yibo Lin
 * @date   Aug 2018
 */
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "cuda_runtime.h"
#include "print.h"

#define SQRT2 1.4142135623730950488016887242096980785696718753769480731766797379907324784621

template <typename T>
__global__ void computeTriangleDensityMapAtomic(
        const T* x_tensor, const T* y_tensor, 
        const T* node_size_x_tensor, const T* node_size_y_tensor, 
        const T* bin_center_x_tensor, const T* bin_center_y_tensor, 
        const int num_nodes, 
        const int num_bins_x, const int num_bins_y, 
        const T xl, const T yl, const T xh, const T yh, 
        const T bin_size_x, const T bin_size_y, 
        const int num_impacted_bins_x, const int num_impacted_bins_y, 
        T* density_map_tensor) 
{
    // rank-one update density map 
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_nodes*num_impacted_bins_x*num_impacted_bins_y; i += blockDim.x * gridDim.x) 
    {
        // density overflow function 
        auto computeDensityFunc = [](T x, T node_size, T bin_center, T bin_size){
            return max(T(0.0), min(x+node_size, bin_center+bin_size/2) - max(x, bin_center-bin_size/2));
        };
        int node_id = i/(num_impacted_bins_x*num_impacted_bins_y);
        int residual_index = i-node_id*num_impacted_bins_x*num_impacted_bins_y;

        // stretch node size to bin size 
        T node_size_x = max(bin_size_x*SQRT2, node_size_x_tensor[node_id]); 
        T node_size_y = max(bin_size_y*SQRT2, node_size_y_tensor[node_id]); 
        T node_x = x_tensor[node_id]+node_size_x_tensor[node_id]/2-node_size_x/2;
        T node_y = y_tensor[node_id]+node_size_y_tensor[node_id]/2-node_size_y/2;
        // boundary condition 
        //node_x = max(node_x, xl);
        //node_x = min(node_x, xh-node_size_x);
        //node_y = max(node_y, yl);
        //node_y = min(node_y, yh-node_size_y);

        // x direction 
        int bin_index_xl = int((node_x-xl)/bin_size_x);
        bin_index_xl = max(bin_index_xl, 0);
        int k = bin_index_xl+int(residual_index / num_impacted_bins_y); 
        if (k+1 > num_bins_x || node_x+node_size_x <= bin_center_x_tensor[k]-bin_size_x/2)
        {
            continue; 
        }
        // y direction 
        int bin_index_yl = int((node_y-yl)/bin_size_y);
        bin_index_yl = max(bin_index_yl, 0);
        int h = bin_index_yl+(residual_index % num_impacted_bins_y); 
        if (h+1 > num_bins_y || node_y+node_size_y <= bin_center_y_tensor[h]-bin_size_y/2)
        {
            continue; 
        }

        T px = computeDensityFunc(node_x, node_size_x, bin_center_x_tensor[k], bin_size_x);
        T py = computeDensityFunc(node_y, node_size_y, bin_center_y_tensor[h], bin_size_y);
        // scale the total area back to node area 
        T area = px*py*(node_size_x_tensor[node_id]*node_size_y_tensor[node_id]/(node_size_x*node_size_y));

        // debug 
        //if (k == 1 && h == 3)
        //{
        //    printf("bin[%d, %d] (%g, %g, %g, %g) add cell %d den (%g, %g, %g, %g) area %g, orig (%g, %g, %g, %g)\n", 
        //            k, h, 
        //            bin_center_x_tensor[k]-bin_size_x/2, bin_center_y_tensor[h]-bin_size_y/2, bin_center_x_tensor[k]+bin_size_x/2, bin_center_y_tensor[h]+bin_size_y/2, 
        //            node_id, 
        //            node_x, node_y, node_x+node_size_x, node_y+node_size_y, 
        //            area, 
        //            x_tensor[node_id], y_tensor[node_id], x_tensor[node_id]+node_size_x_tensor[node_id], y_tensor[node_id]+node_size_y_tensor[node_id]
        //            );
        //}
        // still area 
        atomicAdd(&density_map_tensor[k*num_bins_y+h], area); 
        //__syncthreads();
    }
}

template <typename T>
__global__ void computeTriangleDensityMap(
        const T* x_tensor, const T* y_tensor, 
        const T* node_size_x_tensor, const T* node_size_y_tensor, 
        const T* bin_center_x_tensor, const T* bin_center_y_tensor, 
        const int num_nodes, 
        const int num_bins_x, const int num_bins_y, 
        const T xl, const T yl, const T xh, const T yh, 
        const T bin_size_x, const T bin_size_y, 
        const int num_impacted_bins_x, const int num_impacted_bins_y, 
        T* density_map_tensor) 
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_nodes; i += blockDim.x * gridDim.x) 
    {
        auto computeDensityFunc = [](T x, T node_size, T bin_center, T bin_size){
            return max(T(0.0), min(x+node_size, bin_center+bin_size/2) - max(x, bin_center-bin_size/2));
        };
        // stretch node size to bin size 
        T node_size_x = max(bin_size_x*SQRT2, node_size_x_tensor[i]); 
        T node_size_y = max(bin_size_y*SQRT2, node_size_y_tensor[i]); 
        T node_x = x_tensor[i]+node_size_x_tensor[i]/2-node_size_x/2;
        T node_y = y_tensor[i]+node_size_y_tensor[i]/2-node_size_y/2;
        T ratio = (node_size_x_tensor[i]*node_size_y_tensor[i]/(node_size_x*node_size_y));

        int bin_index_xl = int((node_x-xl)/bin_size_x);
        int bin_index_xh = int(((node_x+node_size_x-xl)/bin_size_x))+1; // exclusive 
        bin_index_xl = max(bin_index_xl, 0); 
        bin_index_xh = min(bin_index_xh, num_bins_x);
        //int bin_index_xh = bin_index_xl+num_impacted_bins_x; 

        int bin_index_yl = int((node_y-yl)/bin_size_y);
        int bin_index_yh = int(((node_y+node_size_y-yl)/bin_size_y))+1; // exclusive 
        bin_index_yl = max(bin_index_yl, 0); 
        bin_index_yh = min(bin_index_yh, num_bins_y);
        //int bin_index_yh = bin_index_yl+num_impacted_bins_y; 

        // update density potential map 
        for (int k = bin_index_xl; k < bin_index_xh; ++k)
        {
            T px = computeDensityFunc(node_x, node_size_x, bin_center_x_tensor[k], bin_size_x);
            for (int h = bin_index_yl; h < bin_index_yh; ++h)
            {
                T py = computeDensityFunc(node_y, node_size_y, bin_center_y_tensor[h], bin_size_y);

                T area = px*py*ratio; 

                atomicAdd(&density_map_tensor[k*num_bins_y+h], area); 
            }
        }
    }
}

template <typename T>
__global__ void computeExactDensityMap(
        const T* x_tensor, const T* y_tensor, 
        const T* node_size_x_tensor, const T* node_size_y_tensor, 
        const T* bin_center_x_tensor, const T* bin_center_y_tensor, 
        const int num_nodes, 
        const int num_bins_x, const int num_bins_y, 
        const T xl, const T yl, const T xh, const T yh, 
        const T bin_size_x, const T bin_size_y, 
        const int num_impacted_bins_x, const int num_impacted_bins_y, 
        bool fixed_node_flag, 
        T* density_map_tensor) 
{
    // rank-one update density map 
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_nodes*num_impacted_bins_x*num_impacted_bins_y; i += blockDim.x * gridDim.x) 
    {
        // density overflow function 
        auto computeDensityFunc = [](T x, T node_size, T bin_center, T bin_size, T l, T h, bool flag){
            T bin_xl = bin_center-bin_size/2;
            T bin_xh = bin_center+bin_size/2;
            if (!flag) // only for movable nodes 
            {
                // if a node is out of boundary, count in the nearest bin 
                if (bin_xl <= l) // left most bin 
                {
                    bin_xl = min(bin_xl, x); 
                }
                if (bin_xh >= h) // right most bin 
                {
                    bin_xh = max(bin_xh, x+node_size); 
                }
            }
            return max(T(0.0), min(x+node_size, bin_xh) - max(x, bin_xl));
        };
        int node_id = i/(num_impacted_bins_x*num_impacted_bins_y);
        int residual_index = i-node_id*num_impacted_bins_x*num_impacted_bins_y;
        // x direction 
        int bin_index_xl = int((x_tensor[node_id]-xl)/bin_size_x);
        bin_index_xl = max(bin_index_xl, 0);
        int k = bin_index_xl+int(residual_index / num_impacted_bins_y); 
        if (k+1 > num_bins_x)
        {
            continue; 
        }
        // y direction 
        int bin_index_yl = int((y_tensor[node_id]-yl)/bin_size_y);
        bin_index_yl = max(bin_index_yl, 0);
        int h = bin_index_yl+(residual_index % num_impacted_bins_y); 
        if (h+1 > num_bins_y)
        {
            continue; 
        }

        T px = computeDensityFunc(x_tensor[node_id], node_size_x_tensor[node_id], bin_center_x_tensor[k], bin_size_x, xl, xh, fixed_node_flag);
        T py = computeDensityFunc(y_tensor[node_id], node_size_y_tensor[node_id], bin_center_y_tensor[h], bin_size_y, yl, yh, fixed_node_flag);
        // still area 
        atomicAdd(&density_map_tensor[k*num_bins_y+h], px*py); 
        //__syncthreads();
    }
}


template <typename T>
int computeTriangleDensityMapCudaLauncher(
        const T* x_tensor, const T* y_tensor, 
        const T* node_size_x_tensor, const T* node_size_y_tensor, 
        const T* bin_center_x_tensor, const T* bin_center_y_tensor, 
        int num_nodes, int num_movable_nodes, int num_filler_nodes, 
        const int num_bins_x, const int num_bins_y, 
        int num_movable_impacted_bins_x, int num_movable_impacted_bins_y, 
        int num_filler_impacted_bins_x, int num_filler_impacted_bins_y, 
        const T xl, const T yl, const T xh, const T yh, 
        const T bin_size_x, const T bin_size_y, 
        T* density_map_tensor
        )
{
    int block_count = 32; 
    int thread_count = 1024; 

    cudaError_t status; 
    cudaStream_t stream_movable; 
    cudaStream_t stream_filler; 
    status = cudaStreamCreate(&stream_movable);
    if (status != cudaSuccess)
    {
        printf("cudaStreamCreate failed for stream_movable\n");
        fflush(stdout);
        return 1; 
    }

    computeTriangleDensityMap<<<block_count, thread_count, 0, stream_movable>>>(
            x_tensor, y_tensor, 
            node_size_x_tensor, node_size_y_tensor, 
            bin_center_x_tensor, bin_center_y_tensor, 
            num_movable_nodes, 
            num_bins_x, num_bins_y, 
            xl, yl, xh, yh, 
            bin_size_x, bin_size_y, 
            num_movable_impacted_bins_x, num_movable_impacted_bins_y, 
            density_map_tensor);

    if (num_filler_nodes)
    {
        status = cudaStreamCreate(&stream_filler);
        if (status != cudaSuccess)
        {
            printf("cudaStreamCreate failed for stream_filler\n");
            fflush(stdout);
            return 1; 
        }

        computeTriangleDensityMap<<<block_count, thread_count, 0, stream_filler>>>(
                x_tensor+num_nodes-num_filler_nodes, y_tensor+num_nodes-num_filler_nodes, 
                node_size_x_tensor+num_nodes-num_filler_nodes, node_size_y_tensor+num_nodes-num_filler_nodes, 
                bin_center_x_tensor, bin_center_y_tensor, 
                num_filler_nodes, 
                num_bins_x, num_bins_y, 
                xl, yl, xh, yh, 
                bin_size_x, bin_size_y, 
                num_filler_impacted_bins_x, num_filler_impacted_bins_y, 
                density_map_tensor);

        status = cudaStreamDestroy(stream_filler); 
        stream_filler = 0; 
        if (status != cudaSuccess) 
        {
            printf("stream_filler destroy failed\n");
            fflush(stdout);
            return 1;
        }   
    }

    /* destroy stream */
    status = cudaStreamDestroy(stream_movable); 
    stream_movable = 0;
    if (status != cudaSuccess) 
    {
        printf("stream_movable destroy failed\n");
        fflush(stdout);
        return 1;
    }   
    return 0; 
}

template <typename T>
int computeExactDensityMapCudaLauncher(
        const T* x_tensor, const T* y_tensor, 
        const T* node_size_x_tensor, const T* node_size_y_tensor, 
        const T* bin_center_x_tensor, const T* bin_center_y_tensor, 
        const int num_nodes, 
        const int num_bins_x, const int num_bins_y, 
        const int num_impacted_bins_x, const int num_impacted_bins_y, 
        const T xl, const T yl, const T xh, const T yh, 
        const T bin_size_x, const T bin_size_y, 
        bool fixed_node_flag, 
        T* density_map_tensor
        )
{
    int block_count = 32; 
    int thread_count = 1024; 

    computeExactDensityMap<<<block_count, thread_count>>>(
            x_tensor, y_tensor, 
            node_size_x_tensor, node_size_y_tensor, 
            bin_center_x_tensor, bin_center_y_tensor, 
            num_nodes, 
            num_bins_x, num_bins_y, 
            xl, yl, xh, yh, 
            bin_size_x, bin_size_y, 
            num_impacted_bins_x, num_impacted_bins_y, 
            fixed_node_flag, 
            density_map_tensor);

    return 0; 
}

#define REGISTER_KERNEL_LAUNCHER(T) \
    int instantiateComputeTriangleDensityMapLauncher(\
            const T* x_tensor, const T* y_tensor, \
            const T* node_size_x_tensor, const T* node_size_y_tensor, \
            const T* bin_center_x_tensor, const T* bin_center_y_tensor, \
            const int num_nodes, const int num_movable_nodes, const int num_filler_nodes, \
            const int num_bins_x, const int num_bins_y, \
            const int num_movable_impacted_bins_x, const int num_movable_impacted_bins_y, \
            const int num_filler_impacted_bins_x, const int num_filler_impacted_bins_y, \
            const T xl, const T yl, const T xh, const T yh, \
            const T bin_size_x, const T bin_size_y, \
            T* density_map_tensor\
            )\
    { \
        return computeTriangleDensityMapCudaLauncher(\
                x_tensor, y_tensor, \
                node_size_x_tensor, node_size_y_tensor, \
                bin_center_x_tensor, bin_center_y_tensor, \
                num_nodes, num_movable_nodes, num_filler_nodes, \
                num_bins_x, num_bins_y, \
                num_movable_impacted_bins_x, num_movable_impacted_bins_y, \
                num_filler_impacted_bins_x, num_filler_impacted_bins_y, \
                xl, yl, xh, yh, \
                bin_size_x, bin_size_y, \
                density_map_tensor\
                );\
    } \
    \
    int instantiateComputeExactDensityMapLauncher(\
            const T* x_tensor, const T* y_tensor, \
            const T* node_size_x_tensor, const T* node_size_y_tensor, \
            const T* bin_center_x_tensor, const T* bin_center_y_tensor, \
            const int num_nodes, \
            const int num_bins_x, const int num_bins_y, \
            const int num_impacted_bins_x, const int num_impacted_bins_y, \
            const T xl, const T yl, const T xh, const T yh, \
            const T bin_size_x, const T bin_size_y, \
            bool fixed_node_flag, \
            T* density_map_tensor\
            )\
    { \
        return computeExactDensityMapCudaLauncher(\
                x_tensor, y_tensor, \
                node_size_x_tensor, node_size_y_tensor, \
                bin_center_x_tensor, bin_center_y_tensor, \
                num_nodes, \
                num_bins_x, num_bins_y, \
                num_impacted_bins_x, num_impacted_bins_y, \
                xl, yl, xh, yh, \
                bin_size_x, bin_size_y, \
                fixed_node_flag, \
                density_map_tensor\
                );\
    }

REGISTER_KERNEL_LAUNCHER(float);
REGISTER_KERNEL_LAUNCHER(double);

