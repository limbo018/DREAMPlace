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
// math utilities
#include "utility/src/math.h"
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
// atomic ops
#include "utility/src/atomic_ops.cuh"

// placement database
//#include "utility/src/detailed_place_db.h"
//#include "utility/src/legalization_db.h"
//#include "utility/src/make_placedb.h"

DREAMPLACE_BEGIN_NAMESPACE
/// to replace thrust::swap 
template<typename Assignable1, typename Assignable2>
__host__ __device__ inline void host_device_swap(Assignable1 &a, Assignable2 &b) {
  Assignable1 tmp = a; 
  a = b; 
  b = tmp; 
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
  fill_array_kernel<<<ceilDiv(n, 512), 512>>>(array, n, v);
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
  reset_element_set_sizes_kernel<<<ceilDiv(num_sets, 512), 512>>>(
      num_sets, element_set_sizes);
  collect_element_sets_kernel<<<ceilDiv(n, 512), 512>>>(
      n, num_sets, max_set_size, elements, element2partition_map, element_sets,
      element_set_sizes);
  correct_element_set_sizes_kernel<<<ceilDiv(num_sets, 512), 512>>>(
      num_sets, max_set_size, element_set_sizes);
}

DREAMPLACE_END_NAMESPACE

#endif
