#include "utility/src/utils.cuh"

DREAMPLACE_BEGIN_NAMESPACE
// fill the demand map net by net
template <typename T>
__global__ void fillDemandMap(T bin_size_x, T bin_size_y,
                              T *node_center_x, T *node_center_y,
                              T *half_node_size_stretch_x, T *half_node_size_stretch_y,
                              T xl, T yl, T xh, T yh,
                              const T *pin_weights,
                              int num_bins_x, int num_bins_y,
                              int num_physical_nodes,
                              T *pin_utilization_map)
{
    const int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i < num_physical_nodes)
    {
        const T x_min = node_center_x[i] - half_node_size_stretch_x[i];
        const T x_max = node_center_x[i] + half_node_size_stretch_x[i];
        int bin_index_xl = int((x_min - xl) / bin_size_x);
        int bin_index_xh = int((x_max - xl) / bin_size_x) + 1;
        bin_index_xl = DREAMPLACE_STD_NAMESPACE::max(bin_index_xl, 0);
        bin_index_xh = DREAMPLACE_STD_NAMESPACE::min(bin_index_xh, num_bins_x);

        const T y_min = node_center_y[i] - half_node_size_stretch_y[i];
        const T y_max = node_center_y[i] + half_node_size_stretch_y[i];
        int bin_index_yl = int((y_min - yl) / bin_size_y);
        int bin_index_yh = int((y_max - yl) / bin_size_y) + 1;
        bin_index_yl = DREAMPLACE_STD_NAMESPACE::max(bin_index_yl, 0);
        bin_index_yh = DREAMPLACE_STD_NAMESPACE::min(bin_index_yh, num_bins_y);

        T density = pin_weights[i] / (half_node_size_stretch_x[i] * half_node_size_stretch_y[i] * 4);
        for (int x = bin_index_xl; x < bin_index_xh; ++x)
        {
            for (int y = bin_index_yl; y < bin_index_yh; ++y)
            {
                T overlap = (DREAMPLACE_STD_NAMESPACE::min(x_max, (x + 1) * bin_size_x) - DREAMPLACE_STD_NAMESPACE::max(x_min, x * bin_size_x)) *
                            (DREAMPLACE_STD_NAMESPACE::min(y_max, (y + 1) * bin_size_y) - DREAMPLACE_STD_NAMESPACE::max(y_min, y * bin_size_y));
                pin_utilization_map[x * num_bins_y + y] += overlap * density;
            }
        }
    }
}

template <typename T>
__global__ void computeInstancePinOptimizationMap(
    T *pos_x, T *pos_y,
    T *node_size_x, T *node_size_y,
    T xl, T yl,
    T bin_size_x, T bin_size_y,
    int num_bins_x, int num_bins_y,
    int num_movable_nodes,
    T unit_pin_capacity,
    T *pin_utilization_map,
    T *pin_weights,
    T *instance_pin_area)
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

        for (int x = bin_index_xl; x < bin_index_xh; ++x)
        {
            for (int y = bin_index_yl; y < bin_index_yh; ++y)
            {
                T overlap = (DREAMPLACE_STD_NAMESPACE::min(x_max, (x + 1) * bin_size_x) - DREAMPLACE_STD_NAMESPACE::max(x_min, x * bin_size_x)) *
                            (DREAMPLACE_STD_NAMESPACE::min(y_max, (y + 1) * bin_size_y) - DREAMPLACE_STD_NAMESPACE::max(y_min, y * bin_size_y));
                instance_pin_area[i] += overlap * pin_utilization_map[x * num_bins_y + y];
            }
        }
        instance_pin_area[i] *= pin_weights[i] / (node_size_x[i] * node_size_y[i] * unit_pin_capacity);
    }
}

// fill the demand map net by net
template <typename T>
int fillDemandMapCudaLauncher(T bin_size_x, T bin_size_y,
                              T *node_center_x, T *node_center_y,
                              T *half_node_size_stretch_x, T *half_node_size_stretch_y,
                              T xl, T yl, T xh, T yh,
                              const T *pin_weights,
                              int num_bins_x, int num_bins_y,
                              int num_physical_nodes,
                              T *pin_utilization_map)
{
    int thread_count = 512;
    int block_count = (num_physical_nodes - 1 + thread_count) / thread_count;
    fillDemandMap<<<block_count, thread_count>>>(
        bin_size_x, bin_size_y,
        node_center_x, node_center_y,
        half_node_size_stretch_x, half_node_size_stretch_y,
        xl, yl, xh, yh,
        pin_weights,
        num_bins_x, num_bins_y,
        num_physical_nodes,
        pin_utilization_map);
    return 0;
}

template <typename T>
int computeInstancePinOptimizationMapCudaLauncher(
    T *pos_x, T *pos_y,
    T *node_size_x, T *node_size_y,
    T xl, T yl,
    T bin_size_x, T bin_size_y,
    int num_bins_x, int num_bins_y,
    int num_movable_nodes,
    T unit_pin_capacity,
    T *pin_utilization_map,
    T *pin_weights,
    T *instance_pin_area)
{
    int thread_count = 512;
    int block_count = (num_movable_nodes - 1 + thread_count) / thread_count;
    computeInstancePinOptimizationMap<<<block_count, thread_count>>>(
        pos_x, pos_y,
        node_size_x, node_size_y,
        xl, yl,
        bin_size_x, bin_size_y,
        num_bins_x, num_bins_y,
        num_movable_nodes,
        unit_pin_capacity,
        pin_utilization_map,
        pin_weights,
        instance_pin_area);
    return 0;
}

#define REGISTER_KERNEL_LAUNCHER(T)                                                                       \
    template int fillDemandMapCudaLauncher<T>(T bin_size_x, T bin_size_y,                                 \
                                              T * node_center_x, T * node_center_y,                       \
                                              T * half_node_size_stretch_x, T * half_node_size_stretch_y, \
                                              T xl, T yl, T xh, T yh,                                     \
                                              const T *pin_weights,                                       \
                                              int num_bins_x, int num_bins_y,                             \
                                              int num_physical_nodes,                                     \
                                              T *pin_utilization_map);                                    \
                                                                                                          \
    template int computeInstancePinOptimizationMapCudaLauncher<T>(                                        \
        T * pos_x, T * pos_y,                                                                             \
        T * node_size_x, T * node_size_y,                                                                 \
        T xl, T yl,                                                                                       \
        T bin_size_x, T bin_size_y,                                                                       \
        int num_bins_x, int num_bins_y,                                                                   \
        int num_movable_nodes,                                                                            \
        T unit_pin_capacity,                                                                              \
        T *pin_utilization_map,                                                                           \
        T *pin_weights,                                                                                   \
        T *instance_pin_area)

REGISTER_KERNEL_LAUNCHER(float);
REGISTER_KERNEL_LAUNCHER(double);

DREAMPLACE_END_NAMESPACE
