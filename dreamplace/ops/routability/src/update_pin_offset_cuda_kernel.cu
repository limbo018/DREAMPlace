#include "utility/src/utils.cuh"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
__global__ void updatePinOffset(
    const int num_nodes,
    const int num_movable_nodes,
    const int num_filler_nodes,
    const int *flat_node2pin_start_map,
    const int *flat_node2pin_map,
    const T *movable_nodes_ratio,
    const T filler_nodes_ratio,
    T *pin_offset_x, T *pin_offset_y)
{
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < num_movable_nodes)
    {
        T ratio = movable_nodes_ratio[i];

        int start = flat_node2pin_start_map[i];
        int end = flat_node2pin_start_map[i + 1];
        for (int j = start; j < end; ++j)
        {
            int pin_id = flat_node2pin_map[j];
            pin_offset_x[pin_id] *= ratio;
            pin_offset_y[pin_id] *= ratio;
        }
    }
}

template <typename T>
void updatePinOffsetCudaLauncher(
    const int num_nodes,
    const int num_movable_nodes,
    const int num_filler_nodes,
    const int *flat_node2pin_start_map,
    const int *flat_node2pin_map,
    const T *movable_nodes_ratio,
    const T filler_nodes_ratio,
    T *pin_offset_x, T *pin_offset_y)
{
    int block_count;
    int thread_count = 512;

    block_count = (num_movable_nodes - 1 + thread_count) / thread_count;
    updatePinOffset<<<block_count, thread_count>>>(
        num_nodes,
        num_movable_nodes,
        num_filler_nodes,
        flat_node2pin_start_map,
        flat_node2pin_map,
        movable_nodes_ratio,
        filler_nodes_ratio,
        pin_offset_x, pin_offset_y);
}

DREAMPLACE_END_NAMESPACE
