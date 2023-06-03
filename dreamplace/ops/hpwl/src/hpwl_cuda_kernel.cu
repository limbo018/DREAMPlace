#include <stdio.h>
#include <math.h>
#include <float.h>
#include "cuda_runtime.h"
#include "utility/src/utils.cuh"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
__global__ void computeHPWL(
        const T* x,
        const int* flat_netpin,
        const int* netpin_start,
        const unsigned char* net_mask,
        int num_nets,
        T* partial_hpwl
        )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_nets)
    {
        T max_x = -FLT_MAX;
        T min_x = FLT_MAX;

        if (net_mask[i])
        {
            for (int j = netpin_start[i]; j < netpin_start[i+1]; ++j)
            {
                min_x = min(min_x, x[flat_netpin[j]]);
                max_x = max(max_x, x[flat_netpin[j]]);
            }
            partial_hpwl[i] = max_x-min_x;
        }
        else
        {
            partial_hpwl[i] = 0;
        }
    }
}

template <typename T>
int computeHPWLCudaLauncher(
        const T* x, const T* y,
        const int* flat_netpin,
        const int* netpin_start,
        const unsigned char* net_mask,
        int num_nets,
        T* partial_hpwl
        )
{
    const int thread_count = 512;
    const int block_count_nets = (num_nets + thread_count - 1) / thread_count;

    cudaError_t status;
    cudaStream_t stream_y;
    status = cudaStreamCreate(&stream_y);
    if (status != cudaSuccess)
    {
        printf("cudaStreamCreate failed for stream_y\n");
        fflush(stdout);
        return 1;
    }

    computeHPWL<<<block_count_nets, thread_count>>>(
            x,
            flat_netpin,
            netpin_start,
            net_mask,
            num_nets,
            partial_hpwl
            );

    computeHPWL<<<block_count_nets, thread_count, 0, stream_y>>>(
            y,
            flat_netpin,
            netpin_start,
            net_mask,
            num_nets,
            partial_hpwl+num_nets
            );

    /* destroy stream */
    status = cudaStreamDestroy(stream_y);
    if (status != cudaSuccess)
    {
        printf("stream_y destroy failed\n");
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
    template int computeHPWLCudaLauncher<type>(\
        const type* x, const type* y, \
        const int* flat_netpin, \
        const int* netpin_start, \
        const unsigned char* net_mask, \
        int num_nets, \
        type* partial_hpwl \
        ); 

REGISTER_KERNEL_LAUNCHER(float);
REGISTER_KERNEL_LAUNCHER(double);

DREAMPLACE_END_NAMESPACE
