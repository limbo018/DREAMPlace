#include <cfloat>
#include <stdio.h>
#include "assert.h"
#include "cuda_runtime.h"
#include "utility/src/utils.cuh"
#include "utility/src/utils_cub.cuh"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
__global__ void permuteGrad(
	const T* grad_out_x,
	const T* grad_out_y,
	const int* flat_node2pin_map,
	const int num_pins,
	T* grad_out_x_perm,
	T* grad_out_y_perm
	)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_pins)
    {
        int pin_id = flat_node2pin_map[i];
		grad_out_x_perm[i] = grad_out_x[pin_id];
		grad_out_y_perm[i] = grad_out_y[pin_id];
    }
}

/// @brief Compute pin position from node position 
template <typename T, typename K>
__global__ void computePinPos(
	const T* x, const T* y,
	const T* pin_offset_x,
	const T* pin_offset_y,
	const K* pin2node_map,
	const int num_pins,
	T* pin_x, T* pin_y
	)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_pins)
	{
		int node_id = pin2node_map[i];
		pin_x[i] = pin_offset_x[i] + x[node_id];
		pin_y[i] = pin_offset_y[i] + y[node_id];
	}
}

template <typename T>
int computePinPosCudaSegmentLauncher(
	const T* x, const T* y,
	const T* pin_offset_x,
	const T* pin_offset_y,
	const long* pin2node_map,
	const int* flat_node2pin_map,
	const int* flat_node2pin_start_map,
	int num_pins,
	T* pin_x, T* pin_y
    )
{
	int thread_count = 512;

	computePinPos<<<(num_pins+thread_count-1) / thread_count, thread_count>>>(x, y, pin_offset_x, pin_offset_y, pin2node_map, num_pins, pin_x, pin_y);

    return 0;
}

template <typename T>
int computePinPosGradCudaSegmentLauncher(
	const T* grad_out_x, const T* grad_out_y,
	const T* x, const T* y,
	const T* pin_offset_x,
	const T* pin_offset_y,
	const long* pin2node_map,
	const int* flat_node2pin_map,
	const int* flat_node2pin_start_map,
	int num_nodes,
	int num_pins,
	T* grad_x, T* grad_y, 
    T* grad_perm_buf ///< 2*num_pins, buffer for store the permutated gradients  
    )
{
    int thread_count = 512;

    T* grad_out_x_perm = grad_perm_buf; 
    T* grad_out_y_perm = grad_perm_buf + num_pins;

    permuteGrad<<<(num_pins+thread_count-1) / thread_count, thread_count>>>(grad_out_x, grad_out_y, flat_node2pin_map, num_pins, grad_out_x_perm, grad_out_y_perm);

    void* d_temp_storage = NULL; 
    size_t temp_storage_bytes = 0; 

    // allocate temp storage 
    cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, grad_out_x_perm, grad_x, 
            num_nodes, flat_node2pin_start_map, flat_node2pin_start_map + 1);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // for x
    cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, grad_out_x_perm, grad_x, 
            num_nodes, flat_node2pin_start_map, flat_node2pin_start_map + 1);
    // for y 
    cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, grad_out_y_perm, grad_y, 
            num_nodes, flat_node2pin_start_map, flat_node2pin_start_map + 1);

    cudaFree(d_temp_storage);

    return 0;	
}


#define REGISTER_KERNEL_LAUNCHER(T) \
    template int computePinPosCudaSegmentLauncher<T>(\
    	    const T* x, const T* y, \
    	    const T* pin_offset_x, \
	        const T* pin_offset_y, \
	        const long* pin2node_map, \
	        const int* flat_node2pin_map, \
	        const int* flat_node2pin_start_map, \
	        int num_pins, \
	        T* pin_x, T* pin_y \
            );\
    \
    template int computePinPosGradCudaSegmentLauncher<T>(\
        	const T* grad_out_x, const T* grad_out_y, \
	        const T* x, const T* y, \
	        const T* pin_offset_x, \
	        const T* pin_offset_y, \
	        const long* pin2node_map, \
	        const int* flat_node2pin_map, \
	        const int* flat_node2pin_start_map, \
	        int num_nodes, \
	        int num_pins, \
	        T* grad_x, T* grad_y, \
            T* grad_perm_buf \
            ); 

REGISTER_KERNEL_LAUNCHER(float);
REGISTER_KERNEL_LAUNCHER(double);

DREAMPLACE_END_NAMESPACE
