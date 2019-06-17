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
        const T* field_map_x_tensor, const T* field_map_y_tensor, 
        const T* x_tensor, const T* y_tensor, 
        const T* node_size_x_tensor, const T* node_size_y_tensor, 
        const T* bin_center_x_tensor, const T* bin_center_y_tensor, 
        T xl, T yl, T xh, T yh, 
        T bin_size_x, T bin_size_y, 
        int num_nodes, 
        T* grad_x_tensor, T* grad_y_tensor
        ) 
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_nodes*num_impacted_bins_x*num_impacted_bins_y; i += blockDim.x * gridDim.x) 
    {
        // density overflow function 
        auto computeDensityFunc = [](T x, T node_size, T bin_center, T bin_size){
            //return max(T(0.0), min(x+node_size, bin_center+bin_size/2) - max(x, bin_center-bin_size/2));
            // Yibo: cannot understand why negative overlap is allowed in RePlAce 
            return min(x+node_size, bin_center+bin_size/2) - max(x, bin_center-bin_size/2);
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

        // still area 
        atomicAdd(&grad_x_tensor[node_id], area*field_map_x_tensor[k*num_bins_y+h]);
        atomicAdd(&grad_y_tensor[node_id], area*field_map_y_tensor[k*num_bins_y+h]);
    }
}

template <typename T>
__global__ void computeElectricForce(
        int num_bins_x, int num_bins_y, 
        int num_impacted_bins_x, int num_impacted_bins_y, 
        const T* field_map_x_tensor, const T* field_map_y_tensor, 
        const T* x_tensor, const T* y_tensor, 
        const T* node_size_x_tensor, const T* node_size_y_tensor, 
        const T* bin_center_x_tensor, const T* bin_center_y_tensor, 
        T xl, T yl, T xh, T yh, 
        T bin_size_x, T bin_size_y, 
        int num_nodes, 
        T* grad_x_tensor, T* grad_y_tensor
        ) 
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_nodes; i += blockDim.x * gridDim.x) 
    {
        auto computeDensityFunc = [](T x, T node_size, T bin_center, T bin_size){
            //return max(T(0.0), min(x+node_size, bin_center+bin_size/2) - max(x, bin_center-bin_size/2));
            // Yibo: cannot understand why negative overlap is allowed in RePlAce 
            return min(x+node_size, bin_center+bin_size/2) - max(x, bin_center-bin_size/2);
        };
        // stretch node size to bin size 
        T node_size_x = max(bin_size_x*SQRT2, node_size_x_tensor[i]); 
        T node_size_y = max(bin_size_y*SQRT2, node_size_y_tensor[i]); 
        T node_x = x_tensor[i]+node_size_x_tensor[i]/2-node_size_x/2;
        T node_y = y_tensor[i]+node_size_y_tensor[i]/2-node_size_y/2;
        T ratio = (node_size_x_tensor[i]*node_size_y_tensor[i]/(node_size_x*node_size_y));

        // Yibo: looks very weird implementation, but this is how RePlAce implements it 
        // the common practice should be floor 
        int bin_index_xl = round((node_x-xl)/bin_size_x);
        int bin_index_xh = round(((node_x+node_size_x-xl)/bin_size_x))+1; // exclusive 
        bin_index_xl = max(bin_index_xl, 0); 
        bin_index_xh = min(bin_index_xh, num_bins_x);
        //int bin_index_xh = bin_index_xl+num_impacted_bins_x; 

        // Yibo: looks very weird implementation, but this is how RePlAce implements it 
        // the common practice should be floor 
        int bin_index_yl = round((node_y-yl)/bin_size_y);
        int bin_index_yh = round(((node_y+node_size_y-yl)/bin_size_y))+1; // exclusive 
        bin_index_yl = max(bin_index_yl, 0); 
        bin_index_yh = min(bin_index_yh, num_bins_y);
        //int bin_index_yh = bin_index_yl+num_impacted_bins_y; 

        grad_x_tensor[i] = 0; 
        grad_y_tensor[i] = 0; 
        // update density potential map 
        for (int k = bin_index_xl; k < bin_index_xh; ++k)
        {
            T px = computeDensityFunc(node_x, node_size_x, bin_center_x_tensor[k], bin_size_x);
            for (int h = bin_index_yl; h < bin_index_yh; ++h)
            {
                T py = computeDensityFunc(node_y, node_size_y, bin_center_y_tensor[h], bin_size_y);

                T area = px*py*ratio; 

                grad_x_tensor[i] += area*field_map_x_tensor[k*num_bins_y+h];
                grad_y_tensor[i] += area*field_map_y_tensor[k*num_bins_y+h];

#if 0
                if (i == 0)
                {
                    printf("bin[%d, %d] (%g, %g, %g, %g) add cell %d den (%g, %g, %g, %g) %g, x %g, y %g orig (%g, %g, %g, %g)\n", 
                            k, h, 
                            bin_center_x_tensor[k]-bin_size_x/2, bin_center_y_tensor[h]-bin_size_y/2, bin_center_x_tensor[k]+bin_size_x/2, bin_center_y_tensor[h]+bin_size_y/2, 
                            i, 
                            node_x, node_y, node_x+node_size_x, node_y+node_size_y, 
                            area, field_map_x_tensor[k*num_bins_y+h], field_map_y_tensor[k*num_bins_y+h], 
                            x_tensor[i], y_tensor[i], x_tensor[i]+node_size_x_tensor[i], y_tensor[i]+node_size_y_tensor[i]
                          );
                }
#endif

            }
        }
    }
}

template <typename T>
int computeElectricForceCudaLauncher(
        int num_bins_x, int num_bins_y, 
        int num_movable_impacted_bins_x, int num_movable_impacted_bins_y, 
        int num_filler_impacted_bins_x, int num_filler_impacted_bins_y, 
        const T* field_map_x_tensor, const T* field_map_y_tensor, 
        const T* x_tensor, const T* y_tensor, 
        const T* node_size_x_tensor, const T* node_size_y_tensor, 
        const T* bin_center_x_tensor, const T* bin_center_y_tensor, 
        T xl, T yl, T xh, T yh, 
        T bin_size_x, T bin_size_y, 
        int num_nodes, int num_movable_nodes, int num_filler_nodes, 
        T* grad_x_tensor, T* grad_y_tensor
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

    computeElectricForce<<<block_count, thread_count, 0, stream_movable>>>(
            num_bins_x, num_bins_y, 
            num_movable_impacted_bins_x, num_movable_impacted_bins_y, 
            field_map_x_tensor, field_map_y_tensor, 
            x_tensor, y_tensor, 
            node_size_x_tensor, node_size_y_tensor, 
            bin_center_x_tensor, bin_center_y_tensor, 
            xl, yl, xh, yh, 
            bin_size_x, bin_size_y, 
            num_movable_nodes, 
            grad_x_tensor, grad_y_tensor
            );

    if (num_filler_nodes)
    {
        status = cudaStreamCreate(&stream_filler);
        if (status != cudaSuccess)
        {
            printf("cudaStreamCreate failed for stream_filler\n");
            fflush(stdout);
            return 1; 
        }

        computeElectricForce<<<block_count, thread_count, 0, stream_filler>>>(
                num_bins_x, num_bins_y, 
                num_filler_impacted_bins_x, num_filler_impacted_bins_y, 
                field_map_x_tensor, field_map_y_tensor, 
                x_tensor+num_nodes-num_filler_nodes, y_tensor+num_nodes-num_filler_nodes, 
                node_size_x_tensor+num_nodes-num_filler_nodes, node_size_y_tensor+num_nodes-num_filler_nodes, 
                bin_center_x_tensor, bin_center_y_tensor, 
                xl, yl, xh, yh, 
                bin_size_x, bin_size_y, 
                num_filler_nodes, 
                grad_x_tensor+num_nodes-num_filler_nodes, grad_y_tensor+num_nodes-num_filler_nodes
                );

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

#define REGISTER_KERNEL_LAUNCHER(T) \
    int instantiateComputeElectricForceLauncher(\
            int num_bins_x, int num_bins_y, \
            int num_movable_impacted_bins_x, int num_movable_impacted_bins_y, \
            int num_filler_impacted_bins_x, int num_filler_impacted_bins_y, \
            const T* field_map_x_tensor, const T* field_map_y_tensor, \
            const T* x_tensor, const T* y_tensor, \
            const T* node_size_x_tensor, const T* node_size_y_tensor, \
            const T* bin_center_x_tensor, const T* bin_center_y_tensor, \
            T xl, T yl, T xh, T yh, \
            T bin_size_x, T bin_size_y, \
            int num_nodes, int num_movable_nodes, int num_filler_nodes, \
            T* grad_x_tensor, T* grad_y_tensor\
            )\
    { \
        return computeElectricForceCudaLauncher(\
                num_bins_x, num_bins_y, \
                num_movable_impacted_bins_x, num_movable_impacted_bins_y, \
                num_filler_impacted_bins_x, num_filler_impacted_bins_y, \
                field_map_x_tensor, field_map_y_tensor, \
                x_tensor, y_tensor, \
                node_size_x_tensor, node_size_y_tensor, \
                bin_center_x_tensor, bin_center_y_tensor, \
                xl, yl, xh, yh, \
                bin_size_x, bin_size_y, \
                num_nodes, num_movable_nodes, num_filler_nodes, \
                grad_x_tensor, grad_y_tensor\
                );\
    }

REGISTER_KERNEL_LAUNCHER(float);
REGISTER_KERNEL_LAUNCHER(double);

DREAMPLACE_END_NAMESPACE
