/*
 * @file   weighted_average_wirelength_cuda_reduce_kernel.cu
 * @author Xiaohan Gao
 * @date   Aug 2019
 */

#include <cfloat>
#include <stdio.h>
#include "assert.h"
#include "cuda_runtime.h"
#include "utility/src/print.h"
#include "utility/src/Msg.h"
#include "weighted_average_wirelength/src/functional_cuda.h"
#include <cub/cub.cuh>

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
__global__ void multiply(const T* a, const T* b, int n, T* c)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) 
    {
        c[i] = a[i]*b[i]; 
    }
}

template <typename T>
void sortByNet(const int* sum_keys, int* keys_sorted, const T* sum_unsorted, T* sum_sorted, int num_nets, cudaStream_t stream)
{
        void *d_temp_storage = NULL;
        size_t temp_storage_bytes = 0;
        cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, 
                sum_keys, keys_sorted, sum_unsorted, sum_sorted, num_nets, 0, sizeof(int)*8, stream);
        
        cudaMalloc(&d_temp_storage, temp_storage_bytes);

        cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                sum_keys, keys_sorted, sum_unsorted, sum_sorted, num_nets, 0, sizeof(int)*8, stream);
        
        cudaFree(d_temp_storage);
}

#if 1
template <typename T>
void computeExpSum(
        const T* exp_x,
        const int* pin2net_map,
        const unsigned char* net_mask,
        int num_nets,
        int num_pins,
        T* exp_x_sum,
        cudaStream_t stream)
{
    cudaError_t status;
    int *d_unique_out;
    cudaMalloc((void**)&d_unique_out, num_nets*sizeof(int));
    int *d_num_runs_out;
    cudaMalloc((void**)&d_num_runs_out, sizeof(int));
    auto reduction_op = cub::Sum();
    // T *exp_x_sorted;
    // cudaMalloc((void**)&exp_x_sorted, num_pins*sizeof(T));
    // int *pin2net_sorted;
    // cudaMalloc((void**)&pin2net_sorted, num_pins*sizeof(int));

    // sortByNet(pin2net_map, pin2net_sorted, exp_x, exp_x_sorted, num_pins, stream);

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    // cub::DeviceReduce::ReduceByKey(d_temp_storage, temp_storage_bytes, pin2net_sorted, d_unique_out,
    // exp_x_sorted, exp_x_sum, d_num_runs_out, reduction_op, num_pins, stream);
    cub::DeviceReduce::ReduceByKey(d_temp_storage, temp_storage_bytes, pin2net_map, d_unique_out,
            exp_x, exp_x_sum, d_num_runs_out, reduction_op, num_pins, stream);

    status = cudaMalloc(&d_temp_storage, temp_storage_bytes);
    if(status != cudaSuccess)
    {
        printf("cudaMalloc failed for ReduceByKey temp storage\n");
    }
    // cub::DeviceReduce::ReduceByKey(d_temp_storage, temp_storage_bytes, pin2net_sorted, d_unique_out,
    // exp_x_sorted, exp_x_sum, d_num_runs_out, reduction_op, num_pins, stream);
    cub::DeviceReduce::ReduceByKey(d_temp_storage, temp_storage_bytes, pin2net_map, d_unique_out,
            exp_x, exp_x_sum, d_num_runs_out, reduction_op, num_pins, stream);

    cudaFree(d_unique_out);
    cudaFree(d_num_runs_out);
    // cudaFree(exp_x_sorted);
    // cudaFree(pin2net_sorted);
    cudaFree(d_temp_storage);
}

#else

template <typename T>
__global__ void computeExpSum(
        const T* exp_x, 
        const int* pin2net_map, 
        const unsigned char* net_mask, 
        int num_nets,
        int num_pins, 
        T* exp_x_sum
        )
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_pins; i += blockDim.x * gridDim.x) 
    {
        int net_id = pin2net_map[i]; 
        if (net_id >= 0 || net_mask[net_id])
        {
            atomicAdd(&exp_x_sum[net_id], exp_x[i]); 
        }
    }
}
#endif

#if 0
template <typename T>
void computeXExpSum(
        const T* x, 
        const T* exp_x, 
        const int* pin2net_map, 
        const unsigned char* net_mask, 
        int num_nets,
        int num_pins, 
        T* xexp_x_sum,
        cudaStream_t stream)
{
    T *xexp_x = NULL;
    cudaMalloc((void**)&xexp_x, num_pins*sizeof(T));

    int block_count = 32;
    int thread_count = 1024;
    multiply<<<block_count, thread_count, 0, stream>>>(x, exp_x, num_pins, xexp_x);

    computeExpSum(xexp_x, pin2net_map, net_mask, num_nets, num_pins, xexp_x_sum, stream);

    cudaFree(xexp_x);
}

#else 

template <typename T>
__global__ void computeXExpSum(
        const T* x, 
        const T* exp_x, 
        const int* pin2net_map, 
        const unsigned char* net_mask, 
        int num_nets,
        int num_pins, 
        T* xexp_x_sum
        )
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_pins; i += blockDim.x * gridDim.x) 
    {
        int net_id = pin2net_map[i]; 
        if (net_id >= 0 || net_mask[net_id])
        {
            atomicAdd(&xexp_x_sum[net_id], x[i]*exp_x[i]); 
        }
    }
}

#endif

template <typename T, typename V>
int computeWeightedAverageWirelengthCudaReduceLauncher(
        const T* x, const T* y, 
        const int* pin2net_map, 
        const unsigned char* net_mask, 
        int num_nets,
        int num_pins, 
        const T* gamma, 
        T* exp_xy, T* exp_nxy, 
        T* exp_xy_sum, T* exp_nxy_sum, 
        T* xyexp_xy_sum, T* xyexp_nxy_sum, 
        V* xy_max, V* xy_min, 
        T* partial_wl, // wirelength of each net 
        const T* grad_tensor, 
        T* grad_x_tensor, T* grad_y_tensor // the gradient is partial total wirelength to partial pin position  
        )
{
    int thread_count = 1024; 
    int block_count = 32; // separate x and y

    cudaError_t status; 
    cudaStream_t stream_x_exp; 
    cudaStream_t stream_nx_exp; 
    cudaStream_t stream_y_exp; 
    cudaStream_t stream_ny_exp; 
    status = cudaStreamCreate(&stream_x_exp);
    if (status != cudaSuccess)
    {
        printf("cudaStreamCreate failed for stream_x_exp\n");
        fflush(stdout);
        return 1; 
    }
    status = cudaStreamCreate(&stream_y_exp);
    if (status != cudaSuccess)
    {
        printf("cudaStreamCreate failed for stream_y_exp\n");
        fflush(stdout);
        return 1; 
    }

    if (grad_tensor)
    {
        computeWeightedAverageWirelengthGrad<<<block_count, thread_count, 0, stream_x_exp>>>(
                x, 
                exp_xy, exp_nxy, 
                exp_xy_sum, exp_nxy_sum, 
                xyexp_xy_sum, xyexp_nxy_sum, 
                pin2net_map, 
                net_mask, 
                num_nets,
                num_pins, 
                gamma, 
                grad_tensor, 
                grad_x_tensor
                );
        computeWeightedAverageWirelengthGrad<<<block_count, thread_count, 0, stream_y_exp>>>(
                y, 
                exp_xy+num_pins, exp_nxy+num_pins, 
                exp_xy_sum+num_nets, exp_nxy_sum+num_nets, 
                xyexp_xy_sum+num_nets, xyexp_nxy_sum+num_nets, 
                pin2net_map, 
                net_mask, 
                num_nets,
                num_pins, 
                gamma, 
                grad_tensor, 
                grad_y_tensor
                );
    }
    else
    {
        status = cudaStreamCreate(&stream_nx_exp);
        if (status != cudaSuccess)
        {
            printf("cudaStreamCreate failed for stream_nx_exp\n");
            fflush(stdout);
            return 1; 
        }
        status = cudaStreamCreate(&stream_ny_exp);
        if (status != cudaSuccess)
        {
            printf("cudaStreamCreate failed for stream_ny_exp\n");
            fflush(stdout);
            return 1; 
        }

        // compute max/min 
        computeMax<<<block_count, thread_count, 0, stream_x_exp>>>(
                x, 
                pin2net_map, 
                net_mask, 
                num_nets, 
                num_pins, 
                xy_max
                );
        computeMin<<<block_count, thread_count, 0, stream_nx_exp>>>(
                x, 
                pin2net_map, 
                net_mask, 
                num_nets, 
                num_pins, 
                xy_min
                );
        computeMax<<<block_count, thread_count, 0, stream_y_exp>>>(
                y, 
                pin2net_map, 
                net_mask, 
                num_nets, 
                num_pins, 
                xy_max+num_nets
                );
        computeMin<<<block_count, thread_count, 0, stream_ny_exp>>>(
                y, 
                pin2net_map, 
                net_mask, 
                num_nets, 
                num_pins, 
                xy_min+num_nets
                );

        // compute exp and negative exp 
        computeExp<<<block_count, thread_count, 0, stream_x_exp>>>(
                x, 
                pin2net_map, 
                net_mask, 
                num_nets,
                num_pins, 
                gamma, 
                xy_max, 
                exp_xy
                );
        computeNegExp<<<block_count, thread_count, 0, stream_nx_exp>>>(
                x, 
                pin2net_map, 
                net_mask, 
                num_nets,
                num_pins, 
                gamma, 
                xy_min, 
                exp_nxy
                );
        computeExp<<<block_count, thread_count, 0, stream_y_exp>>>(
                y, 
                pin2net_map, 
                net_mask, 
                num_nets,
                num_pins, 
                gamma, 
                xy_max+num_nets, 
                exp_xy+num_pins
                );
        computeNegExp<<<block_count, thread_count, 0, stream_ny_exp>>>(
                y, 
                pin2net_map, 
                net_mask, 
                num_nets,
                num_pins, 
                gamma, 
                xy_min+num_nets, 
                exp_nxy+num_pins
                );

        // compute exp sum 
#if 1
        computeExpSum(
                exp_xy, 
                pin2net_map, 
                net_mask, 
                num_nets,
                num_pins, 
                exp_xy_sum,
                stream_x_exp
                );
        computeExpSum(
                exp_nxy, 
                pin2net_map, 
                net_mask, 
                num_nets,
                num_pins, 
                exp_nxy_sum,
                stream_nx_exp
                );
        computeExpSum(
                exp_xy+num_pins, 
                pin2net_map, 
                net_mask, 
                num_nets,
                num_pins, 
                exp_xy_sum+num_nets,
                stream_y_exp
                );
        computeExpSum(
                exp_nxy+num_pins, 
                pin2net_map, 
                net_mask, 
                num_nets,
                num_pins, 
                exp_nxy_sum+num_nets,
                stream_ny_exp
                );
#else 
        computeExpSum<<<block_count, thread_count, 0, stream_x_exp>>>(
                exp_xy, 
                pin2net_map, 
                net_mask, 
                num_nets,
                num_pins, 
                exp_xy_sum
                );
        computeExpSum<<<block_count, thread_count, 0, stream_nx_exp>>>(
                exp_nxy, 
                pin2net_map, 
                net_mask, 
                num_nets,
                num_pins, 
                exp_nxy_sum
                );
        computeExpSum<<<block_count, thread_count, 0, stream_y_exp>>>(
                exp_xy+num_pins, 
                pin2net_map, 
                net_mask, 
                num_nets,
                num_pins, 
                exp_xy_sum+num_nets
                );
        computeExpSum<<<block_count, thread_count, 0, stream_ny_exp>>>(
                exp_nxy+num_pins, 
                pin2net_map, 
                net_mask, 
                num_nets,
                num_pins, 
                exp_nxy_sum+num_nets
                );
#endif

#if 0
        // compute x exp sum 
        computeXExpSum(
                x, 
                exp_xy, 
                pin2net_map, 
                net_mask, 
                num_nets,
                num_pins, 
                xyexp_xy_sum,
                stream_x_exp
                );
        computeXExpSum(
                x, 
                exp_nxy, 
                pin2net_map, 
                net_mask, 
                num_nets,
                num_pins, 
                xyexp_nxy_sum,
                stream_nx_exp
                );
        computeXExpSum(
                y, 
                exp_xy+num_pins, 
                pin2net_map, 
                net_mask, 
                num_nets,
                num_pins, 
                xyexp_xy_sum+num_nets,
                stream_y_exp
                );
        computeXExpSum(
                y, 
                exp_nxy+num_pins, 
                pin2net_map, 
                net_mask, 
                num_nets,
                num_pins, 
                xyexp_nxy_sum+num_nets,
                stream_ny_exp
                );
#else 
        // compute x exp sum 
        computeXExpSum<<<block_count, thread_count, 0, stream_x_exp>>>(
                x, 
                exp_xy, 
                pin2net_map, 
                net_mask, 
                num_nets,
                num_pins, 
                xyexp_xy_sum
                );
        computeXExpSum<<<block_count, thread_count, 0, stream_nx_exp>>>(
                x, 
                exp_nxy, 
                pin2net_map, 
                net_mask, 
                num_nets,
                num_pins, 
                xyexp_nxy_sum
                );
        computeXExpSum<<<block_count, thread_count, 0, stream_y_exp>>>(
                y, 
                exp_xy+num_pins, 
                pin2net_map, 
                net_mask, 
                num_nets,
                num_pins, 
                xyexp_xy_sum+num_nets
                );
        computeXExpSum<<<block_count, thread_count, 0, stream_ny_exp>>>(
                y, 
                exp_nxy+num_pins, 
                pin2net_map, 
                net_mask, 
                num_nets,
                num_pins, 
                xyexp_nxy_sum+num_nets
                );
#endif

        // compute log sum exp 
        computeXExpSumByExpSum<<<block_count, thread_count, 0, stream_x_exp>>>(
                xyexp_xy_sum, 
                exp_xy_sum, 
                pin2net_map, 
                net_mask, 
                num_nets,
                gamma, 
                partial_wl
                );
        computeXNegExpSumByNegExpSum<<<block_count, thread_count, 0, stream_nx_exp>>>(
                xyexp_nxy_sum, 
                exp_nxy_sum, 
                pin2net_map, 
                net_mask, 
                num_nets,
                gamma, 
                partial_wl+num_nets
                );

        computeXExpSumByExpSum<<<block_count, thread_count, 0, stream_y_exp>>>(
                xyexp_xy_sum+num_nets, 
                exp_xy_sum+num_nets, 
                pin2net_map, 
                net_mask, 
                num_nets,
                gamma, 
                partial_wl+2*num_nets
                );
        computeXNegExpSumByNegExpSum<<<block_count, thread_count, 0, stream_ny_exp>>>(
                xyexp_nxy_sum+num_nets, 
                exp_nxy_sum+num_nets, 
                pin2net_map, 
                net_mask, 
                num_nets,
                gamma, 
                partial_wl+3*num_nets
                );

        // I move out the summation to use ATen 
        // significant speedup is observed 
        //sumArray<<<1, 1>>>(partial_wl, 2*num_nets, wl);

        status = cudaStreamDestroy(stream_nx_exp); 
        stream_nx_exp = 0;
        if (status != cudaSuccess) 
        {
            printf("stream_nx_exp destroy failed\n");
            fflush(stdout);
            return 1;
        }   
        status = cudaStreamDestroy(stream_ny_exp); 
        stream_ny_exp = 0; 
        if (status != cudaSuccess) 
        {
            printf("stream_ny_exp destroy failed\n");
            fflush(stdout);
            return 1;
        }   
    }

    /* destroy stream */
    status = cudaStreamDestroy(stream_x_exp); 
    stream_x_exp = 0;
    if (status != cudaSuccess) 
    {
        printf("stream_x_exp destroy failed\n");
        fflush(stdout);
        return 1;
    }   
    status = cudaStreamDestroy(stream_y_exp); 
    stream_y_exp = 0; 
    if (status != cudaSuccess) 
    {
        printf("stream_y_exp destroy failed\n");
        fflush(stdout);
        return 1;
    }   

    return 0; 
}


#define REGISTER_KERNEL_LAUNCHER(T, V) \
    int instantiateComputeWeightedAverageWirelengthReduceLauncher(\
            const T* x, const T* y, \
            const int* pin2net_map, \
            const unsigned char* net_mask, \
            int num_nets, \
            int num_pins, \
            const T* gamma, \
            T* exp_xy, T* exp_nxy, \
            T* exp_xy_sum, T* exp_nxy_sum,\
            T* xyexp_xy_sum, T* xyexp_nxy_sum, \
            V* xy_max, V* xy_min, \
            T* partial_wl, \
            const T* grad_tensor, \
            T* grad_x_tensor, T* grad_y_tensor \
            )\
    {\
        return computeWeightedAverageWirelengthCudaReduceLauncher(\
                x, y, \
                pin2net_map, \
                net_mask, \
                num_nets,\
                num_pins,\
                gamma, \
                exp_xy, exp_nxy, \
                exp_xy_sum, exp_nxy_sum, \
                xyexp_xy_sum, xyexp_nxy_sum, \
                xy_max, xy_min, \
                partial_wl, \
                grad_tensor, \
                grad_x_tensor, grad_y_tensor  \
                );\
    }
REGISTER_KERNEL_LAUNCHER(float, int);
REGISTER_KERNEL_LAUNCHER(double, int);

DREAMPLACE_END_NAMESPACE
