#include "utility/src/utils.cuh"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
__global__ void updatePinOffset(
    const T* node_size_x, const T* node_size_y, 
    const int *flat_node2pin_start_map,
    const int *flat_node2pin_map,
    const T *node_ratios,
    const int num_nodes,
    T *pin_offset_x, T *pin_offset_y
    )
{
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < num_nodes)
    {
        T ratio = (node_ratios[i] - 1) / 2;
        T sx = node_size_x[i]; 
        T sy = node_size_y[i]; 

        int start = flat_node2pin_start_map[i];
        int end = flat_node2pin_start_map[i + 1];
        for (int j = start; j < end; ++j)
        {
            int pin_id = flat_node2pin_map[j];
            pin_offset_x[pin_id] += ratio * sx;
            pin_offset_y[pin_id] += ratio * sy;
        }
    }
}

template <typename T>
void updatePinOffsetCudaLauncher(
    const T* node_size_x, const T* node_size_y, 
    const int *flat_node2pin_start_map,
    const int *flat_node2pin_map,
    const T *node_ratios,
    const int num_nodes,
    T *pin_offset_x, T *pin_offset_y
    )
{
    int thread_count = 512;
    int block_count = ceilDiv(num_nodes, thread_count);
    updatePinOffset<<<block_count, thread_count>>>(
        node_size_x, node_size_y, 
        flat_node2pin_start_map,
        flat_node2pin_map,
        node_ratios,
        num_nodes,
        pin_offset_x, pin_offset_y
        );
}

#define REGISTER_KERNEL_LAUNCHER(T)               \
    template void updatePinOffsetCudaLauncher<T>( \
        const T* node_size_x, const T* node_size_y, \
        const int *flat_node2pin_start_map,       \
        const int *flat_node2pin_map,             \
        const T *node_ratios,             \
        const int num_nodes,                      \
        T *pin_offset_x, T *pin_offset_y)

REGISTER_KERNEL_LAUNCHER(float);
REGISTER_KERNEL_LAUNCHER(double);

DREAMPLACE_END_NAMESPACE
