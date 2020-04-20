/**
 * @file   utils.cuh
 * @author Yibo Lin
 * @date   Jan 2019
 */
#ifndef _DREAMPLACE_UTILITY_UTILS_CUH
#define _DREAMPLACE_UTILITY_UTILS_CUH

#include <cuda.h>
#include <cuda_runtime.h>
// namespace
#include "utility/src/namespace.h"
// macro definitions
#include "utility/src/defs.h"
// print utilities
#include "utility/src/msg.h"
#include "utility/src/print.cuh"
// timer utilities
#include "utility/src/timer.cuh"
#include "utility/src/timer.h"
// numeric limits
#include "utility/src/limits.h"
// simple data structures
#include "utility/src/box.h"
#include "utility/src/complex_number.h"
#include "utility/src/diamond_search.h"
#include "utility/src/flat_nested_vector.cuh"
#include "utility/src/pitch_nested_vector.cuh"

// placement database
//#include "utility/src/detailed_place_db.h"
//#include "utility/src/legalization_db.h"
//#include "utility/src/make_placedb.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
inline __device__ T cudaDiv(T a, T b) {
  return a / b;
}

template <>
inline __device__ float cudaDiv(float a, float b) {
  return fdividef(a, b);
}

/// @brief template specialization for non-integral types
template <typename T>
inline __device__ typename std::enable_if<!std::is_integral<T>::value, T>::type
cudaCeilDiv(T a, T b) {
  return ceil(cudaDiv(a, b));
}

/// @brief template specialization for integral types
template <typename T>
inline __device__ typename std::enable_if<std::is_integral<T>::value, T>::type
cudaCeilDiv(T a, T b) {
  return cudaDiv(a + b - 1, b);
}

template <typename T>
inline __host__ T cpuDiv(T a, T b) {
  return a / b;
}

/// @brief template specialization for non-integral types
template <typename T>
inline __host__ typename std::enable_if<!std::is_integral<T>::value, T>::type
cpuCeilDiv(T a, T b) {
  return ceil(cpuDiv(a, b));
}

/// @brief template specialization for integral types
template <typename T>
inline __host__ typename std::enable_if<std::is_integral<T>::value, T>::type
cpuCeilDiv(T a, T b) {
  return (a + b - 1) / b;
}

template <typename T>
__global__ void iota(T* a, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    a[i] = i;
  }
}

template <typename T>
__global__ void fill_array_kernel(T* array, int n, T v) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    array[i] = v;
  }
}

template <typename T>
inline void fill_array(T* array, int n, T v) {
  fill_array_kernel<<<cpuCeilDiv(n, 512), 512>>>(array, n, v);
}

template <typename T>
__global__ void reset_element_set_sizes_kernel(int num_sets,
                                               T* element_set_sizes) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_sets) {
    element_set_sizes[i] = 0;
  }
}

template <typename T>
__global__ void collect_element_sets_kernel(int n, int num_sets,
                                            int max_set_size, const T* elements,
                                            const int* element2partition_map,
                                            T* element_sets,
                                            int* element_set_sizes) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    const T& element = elements[i];
    int partition_id = element2partition_map[i];
    assert(partition_id < num_sets);
    int& size = element_set_sizes[partition_id];
    int index = atomicAdd(&size, 1);
    if (index < max_set_size) {
      element_sets[partition_id * max_set_size + index] = element;
    }
  }
}

template <typename T>
__global__ void correct_element_set_sizes_kernel(int num_sets, T max_set_size,
                                                 T* element_set_sizes) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_sets) {
    T& size = element_set_sizes[i];
    size = min(size, max_set_size);
  }
}

/// @brief gather elements into sets according to element2partition_map
/// For example, elements = {a0, a1, a2, a3, a4, a5}
/// element2partition_map = {1, 0, 0, 1, 2, 1}
/// expected result element_sets = {{a1, a2, null}, {a0, a3, a5}, {a4, null,
/// null}} Current implementation is not deterministic. If introducing sorting,
/// determinism is possible.
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
                            T* element_sets, int* element_set_sizes) {
  fill_array(element_sets, num_sets * max_set_size,
             std::numeric_limits<T>::max());
  reset_element_set_sizes_kernel<<<cpuCeilDiv(num_sets, 512), 512>>>(
      num_sets, element_set_sizes);
  collect_element_sets_kernel<<<cpuCeilDiv(n, 512), 512>>>(
      n, num_sets, max_set_size, elements, element2partition_map, element_sets,
      element_set_sizes);
  correct_element_set_sizes_kernel<<<cpuCeilDiv(num_sets, 512), 512>>>(
      num_sets, max_set_size, element_set_sizes);
}

DREAMPLACE_END_NAMESPACE

#endif
