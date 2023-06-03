#include <stdio.h>
#include <math.h>
#include <float.h>
#include "cuda_runtime.h"
#include "utility/src/utils.cuh"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
__global__ void computeHPWLMax(
    const T *x,
    const int *pin2net_map,
    const unsigned char *net_mask,
    int num_pins,
    T *partial_hpwl_x_max)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_pins)
    {
        int net_id = pin2net_map[i];
        atomicMax(&partial_hpwl_x_max[net_id], (T)net_mask[net_id] * x[i]); 
    }
}

template <typename T>
__global__ void computeHPWLMin(
    const T *x,
    const int *pin2net_map,
    const unsigned char *net_mask,
    int num_pins,
    T *partial_hpwl_x_min)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_pins)
    {
        int net_id = pin2net_map[i];
        atomicMin(&partial_hpwl_x_min[net_id], (T)net_mask[net_id] * x[i]); 
    }
}

template <typename T>
__global__ void computeHPWLMaxMin(
    const T *x, const T *y,
    const int *pin2net_map,
    const unsigned char *net_mask,
    int num_pins,
    T *partial_hpwl_x_max, T *partial_hpwl_x_min,
    T *partial_hpwl_y_max, T *partial_hpwl_y_min)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_pins)
    {
        int net_id = pin2net_map[i];

        T xx = (T)net_mask[net_id] * x[i]; 
        atomicMax(&partial_hpwl_x_max[net_id], xx);
        atomicMin(&partial_hpwl_x_min[net_id], xx);

        T yy = (T)net_mask[net_id] * y[i]; 
        atomicMax(&partial_hpwl_y_max[net_id], yy);
        atomicMin(&partial_hpwl_y_min[net_id], yy);
    }
}

template <typename T>
int computeHPWLCudaAtomicLauncher(
    const T *x, const T *y,
    const int *pin2net_map,
    const unsigned char *net_mask,
    int num_nets,
    int num_pins,
    T *partial_hpwl_max,
    T *partial_hpwl_min)
{
    const int thread_count = 64;
    const int block_count_pins = (num_pins + thread_count - 1) / thread_count;

    computeHPWLMaxMin<<<block_count_pins, thread_count>>>(
        x, y,
        pin2net_map,
        net_mask,
        num_pins,
        partial_hpwl_max, partial_hpwl_min,
        partial_hpwl_max + num_nets, partial_hpwl_min + num_nets);

    //printArray(partial_hpwl, num_nets, "partial_hpwl");

    // I move out the summation to use ATen
    // significant speedup is observed
    //sumArray<<<1, 1>>>(partial_hpwl, num_nets, hpwl);

    return 0;
}

// manually instantiate the template function
#define REGISTER_KERNEL_LAUNCHER(type)                          \
    template int computeHPWLCudaAtomicLauncher<type>(           \
        const type *x, const type *y,                           \
        const int *pin2net_map,                                 \
        const unsigned char *net_mask,                          \
        int num_nets,                                           \
        int num_pins,                                           \
        type *partial_hpwl_max,                                 \
        type *partial_hpwl_min);

REGISTER_KERNEL_LAUNCHER(int);
REGISTER_KERNEL_LAUNCHER(long long int);

DREAMPLACE_END_NAMESPACE
