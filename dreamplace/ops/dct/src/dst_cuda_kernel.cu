#include <stdio.h>
#include <math.h>
#include <float.h>
#include "cuda_runtime.h"
#include "utility/src/utils.cuh"

DREAMPLACE_BEGIN_NAMESPACE


template <typename T>
__global__ void computeFlip(
        const T* x, 
        const int M, 
        const int N, 
        T* y
        )
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < M*N; i += blockDim.x * gridDim.x) 
    {
        int ii = i%N; 
        y[i] = x[i+N-ii*2-1];
    }
}

template <typename T>
void computeFlipCudaLauncher(
        const T* x, 
        const int M, 
        const int N, 
        T* y
        )
{
    computeFlip<<<32, 1024>>>(
            x, 
            M, 
            N, 
            y
            );
}

template <typename T>
__global__ void computeFlipAndShift(
        const T* x, 
        const int M, 
        const int N, 
        T* y
        )
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < M*N; i += blockDim.x * gridDim.x) 
    {
        int ii = i%N; 
        y[i] = (ii)? x[i+N-ii*2] : 0;
    }
}

template <typename T>
void computeFlipAndShiftCudaLauncher(
        const T* x, 
        const int M, 
        const int N, 
        T* y
        )
{
    computeFlipAndShift<<<32, 1024>>>(
            x, 
            M, 
            N, 
            y
            );
}

template <typename T>
__global__ void negateOddEntries(
        T* x, 
        const int M, 
        const int N
        )
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < M*(N>>1); i += blockDim.x * gridDim.x) 
    {
        x[i*2+1] = -x[i*2+1]; 
    }
}

template <typename T>
void negateOddEntriesCudaLauncher(
        T* x, 
        const int M, 
        const int N
        )
{
    negateOddEntries<<<32, 1024>>>(
            x, 
            M, 
            N
            );
}

#define REGISTER_FLIP_KERNEL_LAUNCHER(type) \
    template void computeFlipCudaLauncher<type>(\
        const type* x, \
        const int M, \
        const int N, \
        type* y \
        ); 

REGISTER_FLIP_KERNEL_LAUNCHER(float);
REGISTER_FLIP_KERNEL_LAUNCHER(double);

#define REGISTER_FLIPANDSHIFT_KERNEL_LAUNCHER(type) \
    template void computeFlipAndShiftCudaLauncher<type>(\
        const type* x, \
        const int M, \
        const int N, \
        type* y \
        ); 

REGISTER_FLIPANDSHIFT_KERNEL_LAUNCHER(float);
REGISTER_FLIPANDSHIFT_KERNEL_LAUNCHER(double);

#define REGISTER_NEGATE_KERNEL_LAUNCHER(type) \
    template void negateOddEntriesCudaLauncher<type>(\
        type* x, \
        const int M, \
        const int N \
        ); 

REGISTER_NEGATE_KERNEL_LAUNCHER(float);
REGISTER_NEGATE_KERNEL_LAUNCHER(double);

DREAMPLACE_END_NAMESPACE
