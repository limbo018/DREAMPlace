#include <cfloat>
#include <stdio.h>
#include "assert.h"
#include "cuda_runtime.h"
#include "utility/src/utils.cuh"

DREAMPLACE_BEGIN_NAMESPACE

// V has to be int, or long long int
template <typename T, typename V>
__global__ void computeMax(
        const T* x,
        const int* pin2net_map,
        const unsigned char* net_mask,
        int num_nets,
        int num_pins,
        V* x_max
        )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_pins)
    {
        int net_id = pin2net_map[i];
        if (net_mask[net_id])
        {
            atomicMax(&x_max[net_id], (V)(x[i]));
        }
    }
}

// V has to be int, or long long int
template <typename T, typename V>
__global__ void computeMin(
        const T* x,
        const int* pin2net_map,
        const unsigned char* net_mask,
        int num_nets,
        int num_pins,
        V* x_min
        )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_pins)
    {
        int net_id = pin2net_map[i];
        if (net_mask[net_id])
        {
            atomicMin(&x_min[net_id], (V)(x[i]));
        }
    }
}

template <typename T, typename V>
__global__ void computeExp(
        const T* x,
        const int* pin2net_map,
        const unsigned char* net_mask,
        int num_nets,
        int num_pins,
        const T* gamma,
        V* x_max,
        T* exp_x
        )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_pins)
    {
        int net_id = pin2net_map[i];
        if (net_mask[net_id])
        {
            exp_x[i] = exp((x[i]-x_max[net_id])/(*gamma));
        }
    }
}

template <typename T, typename V>
__global__ void computeNegExp(
        const T* x,
        const int* pin2net_map,
        const unsigned char* net_mask,
        int num_nets,
        int num_pins,
        const T* gamma,
        V* x_min,
        T* exp_nx
        )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_pins)
    {
        int net_id = pin2net_map[i];
        if (net_mask[net_id])
        {
            exp_nx[i] = exp(-(x[i]-x_min[net_id])/(*gamma));
        }
    }
}

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
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_pins)
    {
        int net_id = pin2net_map[i];
        if (net_mask[net_id])
        {
            atomicAdd(&exp_x_sum[net_id], exp_x[i]); 
        }
    }
}

template <typename T, typename V>
__global__ void computeLogSumExp(
        const T* exp_x_sum,
        const V* x_max,
        const int* pin2net_map,
        const unsigned char* net_mask,
        int num_nets,
        const T* gamma,
        T* partial_wl
        )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_nets)
    {
        if (net_mask[i])
        {
            partial_wl[i] = (*gamma)*log(exp_x_sum[i]) + (T)x_max[i];
        }
    }
}

template <typename T, typename V>
__global__ void computeLogSumNegExp(
        const T* exp_nx_sum,
        const V* x_min,
        const int* pin2net_map,
        const unsigned char* net_mask,
        int num_nets,
        const T* gamma,
        T* partial_wl
        )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_nets)
    {
        if (net_mask[i])
        {
            partial_wl[i] = (*gamma)*log(exp_nx_sum[i]) - (T)x_min[i];
        }
    }
}

template <typename T>
__global__ void computeLogSumExpWirelengthGrad(
        const T* exp_x, const T* exp_nx,
        const T* exp_x_sum, const T* exp_nx_sum,
        const int* pin2net_map,
        const unsigned char* net_mask,
        int num_nets,
        int num_pins,
        const T* gamma,
        const T* grad_tensor,
        T* grad_x_tensor
        )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_pins)
    {
        int net_id = pin2net_map[i];
        if (net_mask[net_id])
        {
            grad_x_tensor[i] = (exp_x[i]/exp_x_sum[net_id] - exp_nx[i]/exp_nx_sum[net_id])*(*grad_tensor);
        }
    }
}

template <typename T, typename V>
int computeLogSumExpWirelengthCudaAtomicLauncher(
        const T* x, const T* y,
        const int* pin2net_map,
        const unsigned char* net_mask,
        int num_nets,
        int num_pins,
        const T* gamma,
        T* exp_xy, T* exp_nxy,
        T* exp_xy_sum, T* exp_nxy_sum,
        V* xy_max, V* xy_min,
        T* partial_wl, // wirelength of each net
        const T* grad_tensor,
        T* grad_x_tensor, T* grad_y_tensor // the gradient is partial total wirelength to partial pin position
        )
{
    int thread_count = 512;
    int block_count_pins = (num_pins + thread_count - 1) / thread_count;
    int block_count_nets = (num_nets + thread_count - 1) / thread_count;

    cudaError_t status;
    cudaStream_t stream_nx_exp;
    cudaStream_t stream_y_exp;
    cudaStream_t stream_ny_exp;
    status = cudaStreamCreate(&stream_y_exp);
    if (status != cudaSuccess)
    {
        printf("cudaStreamCreate failed for stream_y_exp\n");
        fflush(stdout);
        return 1;
    }

    if (grad_tensor)
    {
        computeLogSumExpWirelengthGrad<<<block_count_pins, thread_count>>>(
                exp_xy, exp_nxy,
                exp_xy_sum, exp_nxy_sum,
                pin2net_map,
                net_mask,
                num_nets,
                num_pins,
                gamma,
                grad_tensor,
                grad_x_tensor
                );
        computeLogSumExpWirelengthGrad<<<block_count_pins, thread_count, 0, stream_y_exp>>>(
                exp_xy+num_pins, exp_nxy+num_pins,
                exp_xy_sum+num_nets, exp_nxy_sum+num_nets,
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
        computeMax<<<block_count_pins, thread_count>>>(
                x,
                pin2net_map,
                net_mask,
                num_nets,
                num_pins,
                xy_max
                );
        computeMin<<<block_count_pins, thread_count, 0, stream_nx_exp>>>(
                x,
                pin2net_map,
                net_mask,
                num_nets,
                num_pins,
                xy_min
                );
        computeMax<<<block_count_pins, thread_count, 0, stream_y_exp>>>(
                y,
                pin2net_map,
                net_mask,
                num_nets,
                num_pins,
                xy_max+num_nets
                );
        computeMin<<<block_count_pins, thread_count, 0, stream_ny_exp>>>(
                y,
                pin2net_map,
                net_mask,
                num_nets,
                num_pins,
                xy_min+num_nets
                );

        // compute exp and negative exp
        computeExp<<<block_count_pins, thread_count>>>(
                x,
                pin2net_map,
                net_mask,
                num_nets,
                num_pins,
                gamma,
                xy_max,
                exp_xy
                );
        computeNegExp<<<block_count_pins, thread_count, 0, stream_nx_exp>>>(
                x,
                pin2net_map,
                net_mask,
                num_nets,
                num_pins,
                gamma,
                xy_min,
                exp_nxy
                );
        computeExp<<<block_count_pins, thread_count, 0, stream_y_exp>>>(
                y,
                pin2net_map,
                net_mask,
                num_nets,
                num_pins,
                gamma,
                xy_max+num_nets,
                exp_xy+num_pins
                );
        computeNegExp<<<block_count_pins, thread_count, 0, stream_ny_exp>>>(
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
        computeExpSum<<<block_count_pins, thread_count>>>(
                exp_xy,
                pin2net_map,
                net_mask,
                num_nets,
                num_pins,
                exp_xy_sum
                );
        computeExpSum<<<block_count_pins, thread_count, 0, stream_nx_exp>>>(
                exp_nxy,
                pin2net_map,
                net_mask,
                num_nets,
                num_pins,
                exp_nxy_sum
                );
        computeExpSum<<<block_count_pins, thread_count, 0, stream_y_exp>>>(
                exp_xy+num_pins,
                pin2net_map,
                net_mask,
                num_nets,
                num_pins,
                exp_xy_sum+num_nets
                );
        computeExpSum<<<block_count_pins, thread_count, 0, stream_ny_exp>>>(
                exp_nxy+num_pins,
                pin2net_map,
                net_mask,
                num_nets,
                num_pins,
                exp_nxy_sum+num_nets
                );

        // compute log sum exp
        computeLogSumExp<<<block_count_nets, thread_count>>>(
                exp_xy_sum,
                xy_max,
                pin2net_map,
                net_mask,
                num_nets,
                gamma,
                partial_wl
                );
        computeLogSumNegExp<<<block_count_nets, thread_count, 0, stream_nx_exp>>>(
                exp_nxy_sum,
                xy_min,
                pin2net_map,
                net_mask,
                num_nets,
                gamma,
                partial_wl+num_nets
                );

        computeLogSumExp<<<block_count_nets, thread_count, 0, stream_y_exp>>>(
                exp_xy_sum+num_nets,
                xy_max+num_nets,
                pin2net_map,
                net_mask,
                num_nets,
                gamma,
                partial_wl+2*num_nets
                );
        computeLogSumNegExp<<<block_count_nets, thread_count, 0, stream_ny_exp>>>(
                exp_nxy_sum+num_nets,
                xy_min+num_nets,
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
        if (status != cudaSuccess)
        {
            printf("stream_nx_exp destroy failed\n");
            fflush(stdout);
            return 1;
        }
        status = cudaStreamDestroy(stream_ny_exp);
        if (status != cudaSuccess)
        {
            printf("stream_ny_exp destroy failed\n");
            fflush(stdout);
            return 1;
        }
    }

    /* destroy stream */
    status = cudaStreamDestroy(stream_y_exp);
    if (status != cudaSuccess)
    {
        printf("stream_y_exp destroy failed\n");
        fflush(stdout);
        return 1;
    }

    return 0;
}


#define REGISTER_KERNEL_LAUNCHER(T, V) \
    template int computeLogSumExpWirelengthCudaAtomicLauncher<T, V>(\
            const T* x, const T* y, \
            const int* pin2net_map, \
            const unsigned char* net_mask, \
            int num_nets, \
            int num_pins, \
            const T* gamma, \
            T* exp_xy, T* exp_nxy, \
            T* exp_xy_sum, T* exp_nxy_sum,\
            V* xy_max, V* xy_min, \
            T* partial_wl, \
            const T* grad_tensor, \
            T* grad_x_tensor, T* grad_y_tensor \
            ); 

REGISTER_KERNEL_LAUNCHER(float, int);
REGISTER_KERNEL_LAUNCHER(double, int);

DREAMPLACE_END_NAMESPACE
