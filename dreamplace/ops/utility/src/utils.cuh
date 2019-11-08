/**
 * @file   utils.cuh
 * @author Yibo Lin
 * @date   Jan 2019
 */
#ifndef _DREAMPLACE_UTILITY_UTILS_CUH
#define _DREAMPLACE_UTILITY_UTILS_CUH

#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include "utility/src/utils.h"

#define allocateCUDA(var, size, type) \
{\
    cudaError_t status = cudaMalloc(&(var), (size)*sizeof(type)); \
    if (status != cudaSuccess) \
    { \
        printf("cudaMalloc failed for var##\n"); \
    } \
}

#define destroyCUDA(var) \
{ \
    cudaError_t status = cudaFree(var); \
    if (status != cudaSuccess) \
    { \
        printf("cudaFree failed for var##\n"); \
    } \
}

#define checkCUDA(status) \
{\
	if (status != cudaSuccess) { \
		printf("CUDA Runtime Error: %s\n", \
			cudaGetErrorString(status)); \
		assert(status == cudaSuccess); \
	} \
}

#define allocateCopyCUDA(var, rhs, size) \
{\
    allocateCUDA(var, size, decltype(*rhs)); \
    checkCUDA(cudaMemcpy(var, rhs, sizeof(decltype(*rhs))*(size), cudaMemcpyHostToDevice)); \
}

#define checkCURAND(x) do { if(x!=CURAND_STATUS_SUCCESS) { \
    printf("cuRAND Error at %s:%d\n",__FILE__,__LINE__);\
    assert(x == CURAND_STATUS_SUCCESS);}} while(0)

#define allocateCopyCPU(var, rhs, size, T) \
{ \
    var = (T*)malloc(sizeof(T)*(size));  \
    checkCUDA(cudaMemcpy((void*)var, (void*)rhs, sizeof(T)*(size), cudaMemcpyDeviceToHost)); \
}

#define destroyCPU(var) \
{\
    free((void*)var); \
}

__device__ inline long long int d_get_globaltime(void) 
{
	long long int ret;

	asm volatile ("mov.u64 %0, %%globaltimer;" : "=l"(ret));

	return ret;
}

// Returns the period in miliseconds
__device__ inline double d_get_timer_period(void) 
{
	return 1.0e-6;
}

#define declareCUDAKernel(k)						\
	hr_clock_rep k##_time = 0;						\
	int k##_runs = 0;

#define callCUDAKernel(k, n_blocks, n_threads, shared, ...)	\
{														\
	timer_start = d_get_globaltime();					\
	k <<< n_blocks, n_threads,  shared>>> (__VA_ARGS__);			\
	checkCUDA(cudaDeviceSynchronize());					\
	timer_stop = d_get_globaltime();					\
	k##_time += timer_stop - timer_start;				\
	k##_runs++;											\
}

#define callCUDAKernelAsync(k, n_blocks, n_threads, shared, stream, ...)	\
{														\
	timer_start = d_get_globaltime();					\
	k <<< n_blocks, n_threads,  shared, stream>>> (__VA_ARGS__);			\
	checkCUDA(cudaDeviceSynchronize());					\
	timer_stop = d_get_globaltime();					\
	k##_time += timer_stop - timer_start;				\
	k##_runs++;											\
}

#define reportCUDAKernelStats(k)						\
	printf(#k "\t %g \t %d \t %g\n", d_get_timer_period() * k##_time, k##_runs, d_get_timer_period() * k##_time / k##_runs);

template <typename T>
inline __device__ T CUDADiv(T a, T b) 
{
    return a / b; 
}

template <>
inline __device__ float CUDADiv(float a, float b)
{
    return fdividef(a, b); 
}

template <typename T>
inline __device__ T CUDACeilDiv(T a, T b)
{
    return ceil(CUDADiv(a, b));
}

template <>
inline __device__ int CUDACeilDiv(int a, int b)
{
    return CUDADiv(a+b-1, b); 
}
template <>
inline __device__ unsigned int CUDACeilDiv(unsigned int a, unsigned int b)
{
    return CUDADiv(a+b-1, b); 
}

template <typename T>
inline __host__ T CPUDiv(T a, T b) 
{
    return a / b; 
}

template <typename T>
inline __host__ T CPUCeilDiv(T a, T b)
{
    return ceil(CPUDiv(a, b));
}

template <>
inline __host__ int CPUCeilDiv(int a, int b)
{
    return CPUDiv(a+b-1, b); 
}
template <>
inline __host__ unsigned int CPUCeilDiv(unsigned int a, unsigned int b)
{
    return CPUDiv(a+b-1, b); 
}

template <typename T>
__global__ void iota(T* a, int n)
{
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < n; i += blockDim.x*gridDim.x)
    {
        a[i] = i; 
    }
}

template <typename T>
__global__ void fill_array_kernel(T* array, int n, T v)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        array[i] = v; 
    }
}

template <typename T>
inline void fill_array(T* array, int n, T v)
{
    fill_array_kernel<<<CPUCeilDiv(n, 512), 512>>>(array, n, v);
}

__global__ void reset_element_set_sizes_kernel(int num_sets, int* element_set_sizes)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < num_sets)
    {
        element_set_sizes[i] = 0; 
    }
}

template <typename T>
__global__ void collect_element_sets_kernel(int n, int num_sets, int max_set_size, 
        const T* elements, const int* element2partition_map, 
        T* element_sets, int* element_set_sizes)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        const T& element = elements[i]; 
        int partition_id = element2partition_map[i]; 
        assert(partition_id < num_sets); 
        int& size = element_set_sizes[partition_id]; 
        int index = atomicAdd(&size, 1); 
        if (index < max_set_size)
        {
            element_sets[partition_id*max_set_size + index] = element; 
        }
    }
}

__global__ void correct_element_set_sizes_kernel(int num_sets, int max_set_size, int* element_set_sizes)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < num_sets)
    {
        int& size = element_set_sizes[i]; 
        size = min(size, max_set_size); 
    }
}

/// @brief gather elements into sets according to element2partition_map 
/// For example, elements = {a0, a1, a2, a3, a4, a5}
/// element2partition_map = {1, 0, 0, 1, 2, 1}
/// expected result element_sets = {{a1, a2, null}, {a0, a3, a5}, {a4, null, null}}
/// Current implementation is not deterministic. If introducing sorting, determinism is possible. 
/// @param n length of elements 
/// @param num_sets number of partitions 
/// @param max_set_size maximum number of elements in a partition 
/// @param elements array of elements 
/// @param element2partition_map map element index to partition 
/// @param element_sets output element sets in dimension num_sets x max_set_size
/// @param element_set_sizes size of each set in dimension num_sets x 1
template <typename T>
inline __host__ void gather(int n, int num_sets, int max_set_size, 
        const T* elements, const int* element2partition_map, 
        T* element_sets, int* element_set_sizes)
{
    fill_array(element_sets, num_sets*max_set_size, std::numeric_limits<T>::max());
    reset_element_set_sizes_kernel<<<CPUCeilDiv(num_sets, 512), 512>>>(num_sets, element_set_sizes); 
    collect_element_sets_kernel<<<CPUCeilDiv(n, 512), 512>>>(n, num_sets, max_set_size, elements, element2partition_map, element_sets, element_set_sizes); 
    correct_element_set_sizes_kernel<<<CPUCeilDiv(num_sets, 512), 512>>>(num_sets, max_set_size, element_set_sizes);
}

#endif
