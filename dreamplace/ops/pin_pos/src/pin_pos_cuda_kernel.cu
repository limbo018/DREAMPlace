#include <cfloat>
#include <stdio.h>
#include "assert.h"
#include "cuda_runtime.h"
#include "utility/src/print.h"
#include "utility/src/Msg.h"
#include <cub/cub.cuh>

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
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_pins; i += blockDim.x * gridDim.x)
    {
        int pin_id = flat_node2pin_map[i];
		grad_out_x_perm[i] = grad_out_x[pin_id];
		grad_out_y_perm[i] = grad_out_y[pin_id];
    }
}

template <typename T, typename K>
__global__ void computeLoca(
	const T* x, const T* y,
	const T* pin_offset_x,
	const T* pin_offset_y,
	const K* pin2node_map,
	const int num_pins,
	T* pin_x, T* pin_y
	)
{
	for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_pins; i += blockDim.x * gridDim.x)
	{
		int node_id = pin2node_map[i];
		pin_x[i] = pin_offset_x[i] + x[node_id];
		pin_y[i] = pin_offset_y[i] + y[node_id];
	}
}

template <typename T>
void segmentSum(
	const T* d_in,
	T* d_out,
	const int* d_offset,
	const int num_seg
	)
{
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, 
	    num_seg, d_offset, d_offset + 1);

    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out,
	    num_seg, d_offset, d_offset + 1);

    cudaFree(d_temp_storage);
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
	int thread_count = 1024;
	int block_count = 32;

	computeLoca<<<block_count, thread_count>>>(x, y, pin_offset_x, pin_offset_y, pin2node_map, num_pins, pin_x, pin_y);

    return 0;
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
    int thread_count = 1024;
    int block_count = 32;

    T *grad_out_x_perm, *grad_out_y_perm;
    cudaMalloc((void**)&grad_out_x_perm, num_pins*sizeof(T));
    cudaMalloc((void**)&grad_out_y_perm, num_pins*sizeof(T));

    permuteGrad<<<block_count, thread_count>>>(grad_out_x, grad_out_y, flat_node2pin_map, num_pins, grad_out_x_perm, grad_out_y_perm);

    segmentSum(grad_out_x_perm, grad_x, flat_node2pin_start_map, num_nodes);
    segmentSum(grad_out_y_perm, grad_y, flat_node2pin_start_map, num_nodes);

    cudaFree(grad_out_x_perm);
    cudaFree(grad_out_y_perm);
    return 0;	
}


#define REGISTER_KERNEL_LAUNCHER(T) \
    int instantiateComputePinPosCudaLauncher(\
    	    const T* x, const T* y, \
    	    const T* pin_offset_x, \
	        const T* pin_offset_y, \
	        const long* pin2node_map, \
	        const int* flat_node2pin_map, \
	        const int* flat_node2pin_start_map, \
	        int num_pins, \
	        T* pin_x, T* pin_y \
            )\
    {\
        return computePinPosCudaLauncher(\
    	        x, y, \
    	        pin_offset_x, \
	            pin_offset_y, \
	            pin2node_map, \
	            flat_node2pin_map, \
	            flat_node2pin_start_map, \
	            num_pins, \
	            pin_x, pin_y \
                );\
    } \
    \
    int instantiateComputePinPosGradCudaLauncher(\
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
            )\
    {\
        return computePinPosGradCudaLauncher(\
        	    grad_out_x, grad_out_y, \
	            x, y, \
	            pin_offset_x, \
	            pin_offset_y, \
	            pin2node_map, \
	            flat_node2pin_map, \
	            flat_node2pin_start_map, \
	            num_nodes, \
	            num_pins, \
	            grad_x, grad_y \
                );\
    }
REGISTER_KERNEL_LAUNCHER(float);
REGISTER_KERNEL_LAUNCHER(double);

DREAMPLACE_END_NAMESPACE