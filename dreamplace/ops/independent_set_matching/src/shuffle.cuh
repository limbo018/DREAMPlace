/**
 * @file   shuffle.cuh
 * @author Jiaqi Gu, Yibo Lin
 * @date   Mar 2019
 */
#ifndef _DREAMPLACE_INDEPENDENT_SET_MATCHING_SHUFFLE_CUH
#define _DREAMPLACE_INDEPENDENT_SET_MATCHING_SHUFFLE_CUH

#include <curand.h>
#include <curand_kernel.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include "utility/src/utils.cuh"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T, typename V>
__global__ void print_shuffle(const T* values, const V* keys, int n)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        printf("values[%d]\n", n);
        for (int i = 0; i < n; ++i)
        {
            printf("%d ", int(values[i]));
        }
        printf("\n");
        printf("keys[%d]\n", n);
        for (int i = 0; i < n; ++i)
        {
            printf("%d ", int(keys[i]));
        }
        printf("\n");
    }
}

template <typename T, typename V>
void shuffle(curandGenerator_t& gen, T* values, V* keys, int n)
{
    /* Generate n floats on device */
    checkCURAND(curandGenerate(gen, keys, n));
    //print_shuffle<<<1, 1>>>(values, keys, n);
    thrust::sort_by_key(thrust::device, keys, keys+n, values); 
    //print_shuffle<<<1, 1>>>(values, keys, n);
}

DREAMPLACE_END_NAMESPACE

#endif
