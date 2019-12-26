#include "utility/src/utils.cuh"
#include "routability/src/parameters.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
__global__ void fillDemandMap(const T *pin_pos_x,
                              const T *pin_pos_y,
                              const int *netpin_start,
                              const int *flat_netpin,
                              T bin_size_x, T bin_size_y,
                              T xl, T yl, T xh, T yh,

                              const bool exist_net_weights,
                              const T *net_weights,

                              int num_bins_x, int num_bins_y,
                              int num_nets,
                              T *routing_utilization_map_x,
                              T *routing_utilization_map_y)
{
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < num_nets)
    {
        const int start = netpin_start[i];
        const int end = netpin_start[i + 1];

        T x_max = pin_pos_x[flat_netpin[start]];
        T x_min = x_max;
        T y_max = pin_pos_y[flat_netpin[start]];
        T y_min = y_max;

        for (int j = start + 1; j < end; ++j)
        {
            const T xx = pin_pos_x[flat_netpin[j]];
            x_max = DREAMPLACE_STD_NAMESPACE::max(xx, x_max);
            x_min = DREAMPLACE_STD_NAMESPACE::min(xx, x_min);
            const T yy = pin_pos_y[flat_netpin[j]];
            y_max = DREAMPLACE_STD_NAMESPACE::max(yy, y_max);
            y_min = DREAMPLACE_STD_NAMESPACE::min(yy, y_min);
        }

        // compute the bin box that this net will affect
        int bin_index_xl = int((x_min - xl) / bin_size_x);
        int bin_index_xh = int((x_max - xl) / bin_size_x) + 1;
        bin_index_xl = DREAMPLACE_STD_NAMESPACE::max(bin_index_xl, 0);
        bin_index_xh = DREAMPLACE_STD_NAMESPACE::min(bin_index_xh, num_bins_x);

        int bin_index_yl = int((y_min - yl) / bin_size_y);
        int bin_index_yh = int((y_max - yl) / bin_size_y) + 1;
        bin_index_yl = DREAMPLACE_STD_NAMESPACE::max(bin_index_yl, 0);
        bin_index_yh = DREAMPLACE_STD_NAMESPACE::min(bin_index_yh, num_bins_y);

        T wt;
        if (exist_net_weights)
        {
            wt = net_weights[i] * netWiringDistributionMapWeight<T>(netpin_start[i + 1] - netpin_start[i]);
        }
        else
        {
            wt = netWiringDistributionMapWeight<T>(netpin_start[i + 1] - netpin_start[i]);
        }

        for (int x = bin_index_xl; x < bin_index_xh; ++x)
        {
            for (int y = bin_index_yl; y < bin_index_yh; ++y)
            {
                int index = x * num_bins_y + y;
                T overlap = wt * (DREAMPLACE_STD_NAMESPACE::min(x_max, (x + 1) * bin_size_x) - DREAMPLACE_STD_NAMESPACE::max(x_min, x * bin_size_x)) *
                            (DREAMPLACE_STD_NAMESPACE::min(y_max, (y + 1) * bin_size_y) - DREAMPLACE_STD_NAMESPACE::max(y_min, y * bin_size_y));
                routing_utilization_map_x[index] += overlap / (y_max - y_min);
                routing_utilization_map_y[index] += overlap / (x_max - x_min);
            }
        }
    }
}

// fill the demand map net by net
template <typename T>
int fillDemandMapCudaLauncher(const T *pin_pos_x,
                              const T *pin_pos_y,
                              const int *netpin_start,
                              const int *flat_netpin,
                              T bin_size_x, T bin_size_y,
                              T xl, T yl, T xh, T yh,

                              const bool exist_net_weights,
                              const T *net_weights,

                              int num_bins_x, int num_bins_y,
                              int num_nets,
                              T *routing_utilization_map_x,
                              T *routing_utilization_map_y)
{
    int block_count;
    int thread_count = 512;

    block_count = (num_nets - 1 + thread_count) / thread_count;
    fillDemandMap<<<block_count, thread_count>>>(pin_pos_x,
                                                 pin_pos_y,
                                                 netpin_start,
                                                 flat_netpin,
                                                 bin_size_x, bin_size_y,
                                                 xl, yl, xh, yh,
                                                 exist_net_weights,
                                                 net_weights,
                                                 num_bins_x, num_bins_y,
                                                 num_nets,
                                                 routing_utilization_map_x,
                                                 routing_utilization_map_y);
    return 0;
}

template <typename T>
__global__ void computeInstanceRoutabilityOptimizationMap(
    T *pos_x, T *pos_y,
    T *pin_pos_x, T *pin_pos_y,
    T *node_size_x, T *node_size_y,
    T xl, T yl,
    T bin_size_x, T bin_size_y,
    int num_bins_x, int num_bins_y,
    int num_nets,
    int num_nodes,
    int num_movable_nodes,
    T *routing_utilization_map,
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

        for (int x = bin_index_xl; x < bin_index_xh; ++x)
        {
            for (int y = bin_index_yl; y < bin_index_yh; ++y)
            {
                T overlap = (DREAMPLACE_STD_NAMESPACE::min(x_max, (x + 1) * bin_size_x) - DREAMPLACE_STD_NAMESPACE::max(x_min, x * bin_size_x)) *
                            (DREAMPLACE_STD_NAMESPACE::min(y_max, (y + 1) * bin_size_y) - DREAMPLACE_STD_NAMESPACE::max(y_min, y * bin_size_y));
                instance_route_area[i] += overlap * routing_utilization_map[x * num_bins_y + y];
            }
        }
    }

    return 0;
}

template <typename T>
int computeInstanceRoutabilityOptimizationMapCudaLauncher(
    T *pos_x, T *pos_y,
    T *pin_pos_x, T *pin_pos_y,
    T *node_size_x, T *node_size_y,
    T xl, T yl,
    T bin_size_x, T bin_size_y,
    int num_bins_x, int num_bins_y,
    int num_nets,
    int num_nodes,
    int num_movable_nodes,
    T *routing_utilization_map,
    T *instance_route_area)
{
    int block_count;
    int thread_count = 512;

    block_count = (num_movable_nodes - 1 + thread_count) / thread_count;
    computeInstanceRoutabilityOptimizationMap<<<block_count, thread_count>>>(
        pos_x, pos_y,
        pin_pos_x, pin_pos_y,
        node_size_x, node_size_y,
        xl, yl,
        bin_size_x, bin_size_y,
        num_bins_x, num_bins_y,
        num_nets,
        num_nodes,
        num_movable_nodes,
        routing_utilization_map,
        instance_route_area);
    return 0;
}

DREAMPLACE_END_NAMESPACE
