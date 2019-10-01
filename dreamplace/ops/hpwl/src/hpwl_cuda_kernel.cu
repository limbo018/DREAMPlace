#include <stdio.h>
#include <math.h>
#include <float.h>
#include "cuda_runtime.h"
#include "utility/src/Msg.h"

DREAMPLACE_BEGIN_NAMESPACE

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
        const T* x, const T* y, 
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
        T max_y = -FLT_MAX;
        T min_y = FLT_MAX;

        if (net_mask[i])
        {
            for (int j = netpin_start[i]; j < netpin_start[i+1]; ++j)
            {
                int k = flat_netpin[j];
                T xx = x[k]; 
                T yy = y[k]; 
                min_x = min(min_x, xx);
                max_x = max(max_x, xx);
                min_y = min(min_y, yy);
                max_y = max(max_y, yy);
            }
            partial_hpwl[i] = max_x-min_x; 
            partial_hpwl[i+num_nets] = max_y-min_y; 
        }
        else 
        {
            partial_hpwl[i] = 0; 
            partial_hpwl[i+num_nets] = 0; 
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

    computeHPWL<<<(num_nets+thread_count-1) / thread_count, thread_count>>>(
            x, y,
            flat_netpin, 
            netpin_start, 
            net_mask, 
            num_nets,
            partial_hpwl
            );

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
        const unsigned char* net_mask, \
        int num_nets, \
        type* partial_hpwl \
        ) \
    { \
        return computeHPWLCudaLauncher(x, y, \
                flat_netpin, \
                netpin_start, \
                net_mask, \
                num_nets, \
                partial_hpwl \
                ); \
    }

REGISTER_KERNEL_LAUNCHER(float);
REGISTER_KERNEL_LAUNCHER(double);

DREAMPLACE_END_NAMESPACE
