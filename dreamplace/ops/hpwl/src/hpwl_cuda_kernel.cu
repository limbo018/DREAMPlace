#include <stdio.h>
#include <math.h>
#include <float.h>
#include "cuda_runtime.h"

template <typename T>
void printArray(const T* x, const int n, const char* str)
{
    printf("%s[%d] = ", str, n); 
    T* host_x = (T*)malloc(n*sizeof(T));
    if (host_x == NULL)
    {
        printf("failed to allocate memory on CPU\n");
        return;
    }
    cudaMemcpy(host_x, x, n*sizeof(T), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; ++i)
    {
        printf("%g ", double(host_x[i]));
    }
    printf("\n");

    free(host_x);
}

template <typename T>
void printScalar(const T& x, const char* str)
{
    printf("%s = ", str); 
    T* host_x = (T*)malloc(sizeof(T));
    if (host_x == NULL)
    {
        printf("failed to allocate memory on CPU\n");
        return;
    }
    cudaMemcpy(host_x, &x, sizeof(T), cudaMemcpyDeviceToHost);
    printf("%g\n", double(*host_x));

    free(host_x);
}

template <typename T>
__global__ void fillArray(T* x, const int n, const T v)
{
    //for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) 
    int i  = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        x[i] = v; 
    }
}

template <typename T>
__global__ void computeHPWL(
        const T* x, 
        const int* flat_netpin, 
        const int* netpin_start, 
        const int ignore_net_degree, 
        int num_nets,
        T* partial_hpwl 
        )
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_nets; i += blockDim.x * gridDim.x)
    {
        T max_x = -FLT_MAX;
        T min_x = FLT_MAX;

        // ignore large degree nets 
        if (netpin_start[i+1]-netpin_start[i] >= ignore_net_degree)
        {
            partial_hpwl[i] = 0; 
            continue; 
        }

        for (int j = netpin_start[i]; j < netpin_start[i+1]; ++j)
        {
            min_x = min(min_x, x[flat_netpin[j]]);
            max_x = max(max_x, x[flat_netpin[j]]);
        }
        partial_hpwl[i] = max_x-min_x; 
    }
}

template <typename T>
int computeHPWLCudaLauncher(
        const T* x, const T* y, 
        const int* flat_netpin, 
        const int* netpin_start, 
        const int ignore_net_degree, 
        int num_nets,
        T* partial_hpwl
        )
{
    const int thread_count = 1024; 
    const int block_count = 32; 

    cudaError_t status; 
    cudaStream_t stream_x; 
    status = cudaStreamCreate(&stream_x);
    if (status != cudaSuccess)
    {
        printf("cudaStreamCreate failed for stream_x\n");
        fflush(stdout);
        return 1; 
    }
    cudaStream_t stream_y; 
    status = cudaStreamCreate(&stream_y);
    if (status != cudaSuccess)
    {
        printf("cudaStreamCreate failed for stream_y\n");
        fflush(stdout);
        return 1; 
    }

    computeHPWL<<<block_count, thread_count, 0, stream_x>>>(
            x, 
            flat_netpin, 
            netpin_start, 
            ignore_net_degree, 
            num_nets,
            partial_hpwl
            );

    computeHPWL<<<block_count, thread_count, 0, stream_y>>>(
            y, 
            flat_netpin, 
            netpin_start, 
            ignore_net_degree, 
            num_nets,
            partial_hpwl+num_nets
            );

    /* destroy stream */
    status = cudaStreamDestroy(stream_x); 
    stream_x = 0;
    if (status != cudaSuccess) 
    {
        printf("stream_x destroy failed\n");
        fflush(stdout);
        return 1;
    }   
    status = cudaStreamDestroy(stream_y); 
    stream_y = 0; 
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
    int instantiateComputeHPWLLauncher(\
        const type* x, const type* y, \
        const int* flat_netpin, \
        const int* netpin_start, \
        const int ignore_net_degree, \
        int num_nets, \
        type* partial_hpwl \
        ) \
    { \
        return computeHPWLCudaLauncher(x, y, \
                flat_netpin, \
                netpin_start, \
                ignore_net_degree, \
                num_nets, \
                partial_hpwl \
                ); \
    }

REGISTER_KERNEL_LAUNCHER(float);
REGISTER_KERNEL_LAUNCHER(double);
