#include <stdio.h>
#include <math.h>
#include <float.h>
#include "cuda_runtime.h"
#include "utility/src/Msg.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
__global__ void computeHPWLMax(
        const T* x,
        const int* pin2net_map,
        const unsigned char* net_mask,
        int num_pins,
        T* partial_hpwl_x_max
        )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_pins)
    {
        int net_id = pin2net_map[i];
        if (net_mask[net_id])
        {
            atomicMax(&partial_hpwl_x_max[net_id], x[i]);
            __syncthreads();
        }
    }
}

template <typename T>
__global__ void computeHPWLMin(
        const T* x,
        const int* pin2net_map,
        const unsigned char* net_mask,
        int num_pins,
        T* partial_hpwl_x_min
        )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_pins)
    {
        int net_id = pin2net_map[i];
        if (net_mask[net_id])
        {
            atomicMin(&partial_hpwl_x_min[net_id], x[i]);
            __syncthreads();
        }
    }
}

template <typename T>
int computeHPWLCudaAtomicLauncher(
        const T* x, const T* y,
        const int* pin2net_map,
        const unsigned char* net_mask,
        int num_nets,
        int num_pins,
        T* partial_hpwl_max,
        T* partial_hpwl_min
        )
{
    const int thread_count = 512;
    const int block_count_pins = (num_pins + thread_count - 1) / thread_count;

    cudaError_t status;
    cudaStream_t stream_x_min;
    status = cudaStreamCreate(&stream_x_min);
    if (status != cudaSuccess)
    {
        printf("cudaStreamCreate failed for stream_x_min\n");
        fflush(stdout);
        return 1;
    }
    cudaStream_t stream_y_max;
    status = cudaStreamCreate(&stream_y_max);
    if (status != cudaSuccess)
    {
        printf("cudaStreamCreate failed for stream_y_max\n");
        fflush(stdout);
        return 1;
    }
    cudaStream_t stream_y_min;
    status = cudaStreamCreate(&stream_y_min);
    if (status != cudaSuccess)
    {
        printf("cudaStreamCreate failed for stream_y_min\n");
        fflush(stdout);
        return 1;
    }

    computeHPWLMax<<<block_count_pins, thread_count>>>(
            x,
            pin2net_map,
            net_mask,
            num_pins,
            partial_hpwl_max
            );

    computeHPWLMin<<<block_count_pins, thread_count, 0, stream_x_min>>>(
            x,
            pin2net_map,
            net_mask,
            num_pins,
            partial_hpwl_min
            );

    computeHPWLMax<<<block_count_pins, thread_count, 0, stream_y_max>>>(
            y,
            pin2net_map,
            net_mask,
            num_pins,
            partial_hpwl_max+num_nets
            );

    computeHPWLMin<<<block_count_pins, thread_count, 0, stream_y_min>>>(
            y,
            pin2net_map,
            net_mask,
            num_pins,
            partial_hpwl_min+num_nets
            );

    status = cudaStreamDestroy(stream_x_min);
    if (status != cudaSuccess)
    {
        printf("stream_x_min destroy failed\n");
        fflush(stdout);
        return 1;
    }
    status = cudaStreamDestroy(stream_y_max);
    if (status != cudaSuccess)
    {
        printf("stream_y_max destroy failed\n");
        fflush(stdout);
        return 1;
    }
    status = cudaStreamDestroy(stream_y_min);
    if (status != cudaSuccess)
    {
        printf("stream_y_min destroy failed\n");
        fflush(stdout);
        return 1;
    }
    //printArray(partial_hpwl, num_nets, "partial_hpwl");

    // I move out the summation to use ATen
    // significant speedup is observed
    //sumArray<<<1, 1>>>(partial_hpwl, num_nets, hpwl);

    return 0;
}

// manually instantiate the template function
#define REGISTER_KERNEL_LAUNCHER(type) \
    int instantiateComputeHPWLAtomicLauncher(\
        const type* x, const type* y, \
        const int* pin2net_map, \
        const unsigned char* net_mask, \
        int num_nets, \
        int num_pins, \
        type* partial_hpwl_max, \
        type* partial_hpwl_min \
        ) \
    { \
        return computeHPWLCudaAtomicLauncher(x, y, \
                pin2net_map, \
                net_mask, \
                num_nets, \
                num_pins, \
                partial_hpwl_max, \
                partial_hpwl_min \
                ); \
    }

REGISTER_KERNEL_LAUNCHER(int);
REGISTER_KERNEL_LAUNCHER(long long int);

DREAMPLACE_END_NAMESPACE
