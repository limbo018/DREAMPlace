#include <stdio.h>
#include <math.h>
#include <float.h>
#include "cuda_runtime.h"
#include "utility/src/utils.cuh"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
__global__ void computePWS(
        const T* net_weights, const int* flat_nodepin,
        const int* nodepin_start, const int* pin2net_map,
        int num_physical_nodes, T* node_weights) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_physical_nodes) {
        // ignore large degree nets
        node_weights[i] = 0;
        for (int j = nodepin_start[i]; j < nodepin_start[i + 1]; ++j) {
            node_weights[i] += net_weights[pin2net_map[flat_nodepin[j]]];
        }
    }
}

template <typename T>
int computePWSCudaLauncher(const T* net_weights, const int* flat_nodepin,
                           const int* nodepin_start, const int* pin2net_map,
                           int num_physical_nodes, T* node_weights) {
    int thread_count = 512;
    int block_count = (num_physical_nodes - 1 + thread_count) / thread_count;
    computePWS<<<block_count, thread_count>>>(
        net_weights, flat_nodepin, nodepin_start,
        pin2net_map, num_physical_nodes, node_weights);
    return 0;
}

// manually instantiate the template function
#define REGISTER_KERNEL_LAUNCHER(type)         \
    template int computePWSCudaLauncher<type>( \
        const type* net_weights,               \
        const int* flat_nodepin,               \
        const int* nodepin_start,              \
        const int* pin2net_map,                \
        int num_physical_nodes,                \
        type* node_weights); 

REGISTER_KERNEL_LAUNCHER(float);
REGISTER_KERNEL_LAUNCHER(double);

DREAMPLACE_END_NAMESPACE
