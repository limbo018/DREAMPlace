#include <cfloat>
#include <stdio.h>
#include "assert.h"
#include "cuda_runtime.h"
#include "utility/src/utils.cuh"

DREAMPLACE_BEGIN_NAMESPACE

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
int computePinPosCudaLauncher(
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

/// @brief Compute pin position from node position 
template <typename T>
__global__ void computeNodeGrad(
	const T* grad_out_x,
	const T* grad_out_y,
	const int* flat_node2pin_map,
    const int* flat_node2pin_start_map, 
    const int num_nodes, 
	T* grad_x,
	T* grad_y
	)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_nodes)
	{
        T& gx = grad_x[i];
        T& gy = grad_y[i];
        gx = 0; 
        gy = 0; 
        for (int j = flat_node2pin_start_map[i]; j < flat_node2pin_start_map[i+1]; ++j)
        {
            int pin_id = flat_node2pin_map[j]; 
            gx += grad_out_x[pin_id]; 
            gy += grad_out_y[pin_id]; 
        }
	}
}

template <typename T>
int computePinPosGradCudaLauncher(
	const T* grad_out_x, const T* grad_out_y,
	const T* x, const T* y,
	const T* pin_offset_x,
	const T* pin_offset_y,
	const long* pin2node_map,
	const int* flat_node2pin_map,
	const int* flat_node2pin_start_map,
	int num_nodes,
	int num_pins,
	T* grad_x, T* grad_y
    )
{
    int thread_count = 512;

    computeNodeGrad<<<(num_nodes + thread_count - 1) / thread_count, thread_count>>>(
            grad_out_x, 
            grad_out_y, 
            flat_node2pin_map, 
            flat_node2pin_start_map, 
            num_nodes, 
            grad_x, 
            grad_y
            );

    return 0;	
}


#define REGISTER_KERNEL_LAUNCHER(T) \
    template int computePinPosCudaLauncher<T>(\
    	    const T* x, const T* y, \
    	    const T* pin_offset_x, \
	        const T* pin_offset_y, \
	        const long* pin2node_map, \
	        const int* flat_node2pin_map, \
	        const int* flat_node2pin_start_map, \
	        int num_pins, \
	        T* pin_x, T* pin_y \
            ); \
    \
    template int computePinPosGradCudaLauncher<T>(\
        	const T* grad_out_x, const T* grad_out_y, \
	        const T* x, const T* y, \
	        const T* pin_offset_x, \
	        const T* pin_offset_y, \
	        const long* pin2node_map, \
	        const int* flat_node2pin_map, \
	        const int* flat_node2pin_start_map, \
	        int num_nodes, \
	        int num_pins, \
	        T* grad_x, T* grad_y \
            ); 

REGISTER_KERNEL_LAUNCHER(float);
REGISTER_KERNEL_LAUNCHER(double);

DREAMPLACE_END_NAMESPACE
