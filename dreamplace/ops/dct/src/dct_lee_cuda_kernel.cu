/**
 * @file   dct_lee_cuda_kernel.cu
 * @author Yibo Lin
 * @date   Oct 2018
 */

//#include <stdexcept>
//#include <algorithm>
#include <cassert>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "cuda_runtime.h"

// #include "dct_lee_cuda.h"
#include "dct_lee_cuda_kernel.h"

DREAMPLACE_BEGIN_NAMESPACE

namespace lee
{

constexpr double PI = 3.14159265358979323846;

/// Return true if a number is power of 2
template <typename T>
inline bool isPowerOf2(T val)
{
    return val && (val & (val - 1)) == 0;
}

template <typename T>
inline void swap(T& x, T& y)
{
    T tmp = x; 
    x = y; 
    y = tmp; 
}

/// Precompute cosine values needed for N-point dct
/// @param  cos  size N - 1 buffer on GPU, contains the result after function call
/// @param  N    the length of target dct, must be power of 2
template <typename TValue>
void precompute_dct_cos(TValue *cos, int N)
{
    // The input length must be power of 2
    if (! isPowerOf2<int>(N))
    {
        printf("Input length is not power of 2.\n");
        assert(0); 
    }

    // create the array on host 
    TValue* cos_host = new TValue [N]; 

    int offset = 0;
    int halfLen = N / 2;
    while (halfLen)
    {
        TValue phaseStep = 0.5 * PI / halfLen;
        TValue phase = 0.5 * phaseStep;
        for (int i = 0; i < halfLen; ++i)
        {
            cos_host[offset + i] = 0.5 / std::cos(phase);
            phase += phaseStep;
        }
        offset += halfLen;
        halfLen /= 2;
    }

    // copy to GPU 
    cudaMemcpy(cos, cos_host, N*sizeof(TValue), cudaMemcpyHostToDevice); 

    delete [] cos_host; 
}

/// Precompute cosine values needed for N-point idct
/// @param  cos  size N - 1 buffer on GPU, contains the result after function call
/// @param  N    the length of target idct, must be power of 2
template <typename TValue>
void precompute_idct_cos(TValue *cos, int N)
{
    // The input length must be power of 2
    if (! isPowerOf2<int>(N))
    {
        printf("Input length is not power of 2.\n");
        assert(0); 
    }

    // create the array on host 
    TValue* cos_host = new TValue [N]; 

    int offset = 0;
    int halfLen = 1;
    while(halfLen < N)
    {
        TValue phaseStep = 0.5 * PI / halfLen;
        TValue phase = 0.5 * phaseStep;
        for (int i = 0; i < halfLen; ++i)
        {
            cos_host[offset + i] = 0.5 / std::cos(phase);
            phase += phaseStep;
        }
        offset += halfLen;
        halfLen *= 2;
    }

    // copy to GPU 
    cudaMemcpy(cos, cos_host, N*sizeof(TValue), cudaMemcpyHostToDevice); 

    delete [] cos_host; 
}

/// The implementation of fast Discrete Cosine Transform (DCT) algorithm and its inverse (IDCT) are Lee's algorithms
/// Algorithm reference: A New Algorithm to Compute the Discrete Cosine Transform, by Byeong Gi Lee, 1984
///
/// Lee's algorithm has a recursive structure in nature.
/// Here is a sample recursive implementation: https://www.nayuki.io/page/fast-discrete-cosine-transform-algorithms
///   
/// My implementation here is iterative, which is more efficient than the recursive version.
/// Here is a sample iterative implementation: https://www.codeproject.com/Articles/151043/Iterative-Fast-1D-Forvard-DCT

/// Compute y[k] = sum_n=0..N-1 (x[n] * cos((n + 0.5) * k * PI / N)), for k = 0..N-1
/// 
/// @param  vec   length M * N sequence to be transformed in last dimension
/// @param  out   length M * N helping buffer, which is also the output
/// @param  buf   length M * N helping buffer
/// @param  cos   length N - 1, stores cosine values precomputed by function 'precompute_dct_cos'
/// @param  M     length of dimension 0 of vec  
/// @param  N     length of dimension 1 of vec, must be power of 2
template <typename TValue>
void dct(const TValue *vec, TValue *out, TValue* buf, const TValue *cos, int M, int N)
{
    int block_count = 2048; 
    int thread_count = 512; 

    // The input length must be power of 2
    if (! isPowerOf2<int>(N))
    {
        printf("Input length is not power of 2.\n");
        assert(0); 
    }

    // Pointers point to the beginning indices of two adjacent iterations
    TValue *curr = buf; 
    TValue *next = out; 

    // 'temp' used to store date of two adjacent iterations
    // Copy 'vec' to the first N element in 'temp'
    cudaMemcpy(curr, vec, M*N*sizeof(TValue), cudaMemcpyDeviceToDevice);

    // Current bufferfly length and half length
    int len = N;
    int halfLen = len / 2;

    // Iteratively bi-partition sequences into sub-sequences
    int cosOffset = 0;
    while (halfLen)
    {
        computeDctForward<<<block_count, thread_count>>>(curr, next, cos, M, N, len, halfLen, cosOffset);
        swap(curr, next);
        cosOffset += halfLen;
        len = halfLen;
        halfLen /= 2;
    }

    // Bottom-up form the final DCT solution
    // Note that the case len = 2 will do nothing, so we start from len = 4
    len = 4;
    halfLen = 2;
    while (halfLen < N)
    {
        computeDctBackward<<<block_count, thread_count>>>(curr, next, M, N, len, halfLen);
        swap(curr, next);
        halfLen = len;
        len *= 2;
    }

    // Populate the final results into 'out'
    if (curr != out)
    {
        cudaMemcpy(out, curr, M*N*sizeof(TValue), cudaMemcpyDeviceToDevice);
    }
}

/// Compute y[k] = 0.5 * x[0] + sum_n=1..N-1 (x[n] * cos(n * (k + 0.5) * PI / N)), for k = 0..N-1
/// @param  vec   length M * N sequence to be transformed
/// @param  out   length M * N helping buffer, which is also the output
/// @param  buf   length M * N helping buffer
/// @param  cos   length N - 1, stores cosine values precomputed by function 'precompute_idct_cos'
/// @param  M     length of dimension 0 of vec  
/// @param  N     length of dimension 1 of vec, must be power of 2
template <typename TValue>
void idct(const TValue *vec, TValue *out, TValue *buf, const TValue *cos, int M, int N)
{
    int block_count = 32; 
    int thread_count = 1024; 

    // The input length must be power of 2
    if (! isPowerOf2<int>(N))
    {
        printf("Input length is not power of 2.\n");
        assert(0); 
    }

    // Pointers point to the beginning indices of two adjacent iterations
    TValue *curr = buf; 
    TValue *next = out; 

    // This array is used to store date of two adjacent iterations
    // Copy 'vec' to the first N element in 'temp'
    cudaMemcpy(curr, vec, M*N*sizeof(TValue), cudaMemcpyDeviceToDevice);
    computeIdctScale0<<<block_count, thread_count>>>(curr, M, N);

    // Current bufferfly length and half length
    int len = N;
    int halfLen = len / 2;

    // Iteratively bi-partition sequences into sub-sequences
    while (halfLen)
    {
        computeIdctForward<<<block_count, thread_count>>>(curr, next, M, N, len, halfLen);
        swap(curr, next);
        len = halfLen;
        halfLen /= 2;
    }

    // Bottom-up form the final IDCT solution
    len = 2;
    halfLen = 1;
    int cosOffset = 0;
    while(halfLen < N)
    {
        ComputeIdctBackward<<<block_count, thread_count>>>(curr, next, cos, M, N, len, halfLen, cosOffset);
        swap(curr, next);
        cosOffset += halfLen;
        halfLen = len;
        len *= 2;
    }

    // Populate the final results into 'out'
    if (curr != out)
    {
        cudaMemcpy(out, curr, M*N*sizeof(TValue), cudaMemcpyDeviceToDevice);
    }
}

} // End of namespace lee

#define REGISTER_DCT_PRECOMPUTE_COS_KERNEL_LAUNCHER(type) \
    template void lee::precompute_dct_cos<type>(\
        type* cos, \
        int N \
        ); 

REGISTER_DCT_PRECOMPUTE_COS_KERNEL_LAUNCHER(float);
REGISTER_DCT_PRECOMPUTE_COS_KERNEL_LAUNCHER(double);

#define REGISTER_IDCT_PRECOMPUTE_COS_KERNEL_LAUNCHER(type) \
    template void lee::precompute_idct_cos<type>(\
        type* cos, \
        int N \
        ); 

REGISTER_IDCT_PRECOMPUTE_COS_KERNEL_LAUNCHER(float);
REGISTER_IDCT_PRECOMPUTE_COS_KERNEL_LAUNCHER(double);

#define REGISTER_DCT_KERNEL_LAUNCHER(type) \
    template void lee::dct<type>(\
        const type* vec, \
        type* curr, \
        type* next, \
        const type* cos, \
        int M, \
        int N \
        ); 

REGISTER_DCT_KERNEL_LAUNCHER(float);
REGISTER_DCT_KERNEL_LAUNCHER(double);

#define REGISTER_IDCT_KERNEL_LAUNCHER(type) \
    template void lee::idct<type>(\
        const type* vec, \
        type* curr, \
        type* next, \
        const type* cos, \
        int M, \
        int N \
        ); 

REGISTER_IDCT_KERNEL_LAUNCHER(float);
REGISTER_IDCT_KERNEL_LAUNCHER(double);

DREAMPLACE_END_NAMESPACE
