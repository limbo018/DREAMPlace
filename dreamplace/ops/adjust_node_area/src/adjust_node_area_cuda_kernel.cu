/**
 * @file   adjust_node_area_cuda_kernel.cu
 * @author Zixuan Jiang, Jiaqi Gu, Yibo Lin
 * @date   Dec 2019
 * @brief  Adjust cell area according to congestion map.
 */

#include "utility/src/utils.cuh"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
__global__ void computeInstanceRoutabilityOptimizationMap(
    const T *pos_x, const T *pos_y,
    const T *node_size_x, const T *node_size_y,
    const T *routing_utilization_map,
    T xl, T yl,
    T bin_size_x, T bin_size_y,
    int num_bins_x, int num_bins_y,
    int num_movable_nodes,
    T *instance_route_area)
{
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < num_movable_nodes)
    {
        const T x_max = pos_x[i] + node_size_x[i];
        const T x_min = pos_x[i];
        const T y_max = pos_y[i] + node_size_y[i];
        const T y_min = pos_y[i];

        // compute the bin box that this net will affect
        // We do NOT follow Wuxi's implementation. Instead, we clamp the bounding box.
        int bin_index_xl = int((x_min - xl) / bin_size_x);
        int bin_index_xh = int((x_max - xl) / bin_size_x) + 1;
        bin_index_xl = DREAMPLACE_STD_NAMESPACE::max(bin_index_xl, 0);
        bin_index_xh = DREAMPLACE_STD_NAMESPACE::min(bin_index_xh, num_bins_x);

        int bin_index_yl = int((y_min - yl) / bin_size_y);
        int bin_index_yh = int((y_max - yl) / bin_size_y) + 1;
        bin_index_yl = DREAMPLACE_STD_NAMESPACE::max(bin_index_yl, 0);
        bin_index_yh = DREAMPLACE_STD_NAMESPACE::min(bin_index_yh, num_bins_y);

        T &area = instance_route_area[i];
        area = 0;
        for (int x = bin_index_xl; x < bin_index_xh; ++x)
        {
            for (int y = bin_index_yl; y < bin_index_yh; ++y)
            {
                T overlap = (DREAMPLACE_STD_NAMESPACE::min(x_max, (x + 1) * bin_size_x) - DREAMPLACE_STD_NAMESPACE::max(x_min, x * bin_size_x)) *
                            (DREAMPLACE_STD_NAMESPACE::min(y_max, (y + 1) * bin_size_y) - DREAMPLACE_STD_NAMESPACE::max(y_min, y * bin_size_y));
                area += overlap * routing_utilization_map[x * num_bins_y + y];
            }
        }
    }
}

template <typename T>
int computeInstanceRoutabilityOptimizationMapCudaLauncher(
    const T *pos_x, const T *pos_y,
    const T *node_size_x, const T *node_size_y,
    const T *routing_utilization_map,
    T xl, T yl,
    T bin_size_x, T bin_size_y,
    int num_bins_x, int num_bins_y,
    int num_movable_nodes,
    T *instance_route_area)
{
    int thread_count = 512;
    int block_count = CPUCeilDiv(num_movable_nodes, thread_count);
    computeInstanceRoutabilityOptimizationMap<<<block_count, thread_count>>>(
        pos_x, pos_y,
        node_size_x, node_size_y,
        routing_utilization_map,
        xl, yl,
        bin_size_x, bin_size_y,
        num_bins_x, num_bins_y,
        num_movable_nodes,
        instance_route_area);
    return 0;
}

#define REGISTER_KERNEL_LAUNCHER(T)                                        \
    template int computeInstanceRoutabilityOptimizationMapCudaLauncher<T>( \
        const T *pos_x, const T *pos_y,                                    \
        const T *node_size_x, const T *node_size_y,                        \
        const T *routing_utilization_map,                                  \
        T xl, T yl,                                                        \
        T bin_size_x, T bin_size_y,                                        \
        int num_bins_x, int num_bins_y,                                    \
        int num_movable_nodes,                                             \
        T *instance_route_area)

REGISTER_KERNEL_LAUNCHER(float);
REGISTER_KERNEL_LAUNCHER(double);

DREAMPLACE_END_NAMESPACE
