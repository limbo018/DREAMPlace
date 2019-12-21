#include "utility/src/torch.h"
#include "utility/src/Msg.h"
#include "utility/src/utils.h"
#include "routability/src/parameters.h"

DREAMPLACE_BEGIN_NAMESPACE

#define CHECK_FLAT(x) AT_ASSERTM(!x.is_cuda() && x.ndimension() == 1, #x "must be a flat tensor on CPU")
#define CHECK_EVEN(x) AT_ASSERTM((x.numel() & 1) == 0, #x "must have even number of elements")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x "must be contiguous")

// fill the demand map net by net
template <typename T>
int fillDemandMapLauncher(const T *pin_pos_x,
                          const T *pin_pos_y,
                          const int *netpin_start,
                          const int *flat_netpin,
                          T bin_size_x, T bin_size_y,
                          T xl, T yl, T xh, T yh,
                          
                          const bool exist_net_weights, 
                          const T* net_weights,

                          int num_bins_x, int num_bins_y,
                          int num_nets,
                          int num_threads,
                          T *routing_utilization_map_x,
                          T *routing_utilization_map_y
                          )
{
    const T inv_bin_size_x = 1.0 / bin_size_x;
    const T inv_bin_size_y = 1.0 / bin_size_y;

    int chunk_size = DREAMPLACE_STD_NAMESPACE::max(int(num_nets / num_threads / 16), 1);
    #pragma omp parallel for num_threads(num_threads) schedule(dynamic, chunk_size)
    for (int i = 0; i < num_nets; ++i)
    {
        T x_max = -std::numeric_limits<T>::max();
        T x_min = std::numeric_limits<T>::max();
        T y_max = -std::numeric_limits<T>::max();
        T y_min = std::numeric_limits<T>::max();

        for (int j = netpin_start[i]; j < netpin_start[i + 1]; ++j)
        {
            const T xx = pin_pos_x[flat_netpin[j]];
            x_max = DREAMPLACE_STD_NAMESPACE::max(xx, x_max);
            x_min = DREAMPLACE_STD_NAMESPACE::min(xx, x_min);
            const T yy = pin_pos_y[flat_netpin[j]];
            y_max = DREAMPLACE_STD_NAMESPACE::max(yy, y_max);
            y_min = DREAMPLACE_STD_NAMESPACE::min(yy, y_min);
        }

        // compute the bin box that this net will affect
        int bin_index_xl = int((x_min - xl) * inv_bin_size_x);
        int bin_index_xh = int((x_max - xl) * inv_bin_size_x) + 1;
        bin_index_xl = DREAMPLACE_STD_NAMESPACE::max(bin_index_xl, 0);
        bin_index_xh = DREAMPLACE_STD_NAMESPACE::min(bin_index_xh, num_bins_x);

        int bin_index_yl = int((y_min - yl) * inv_bin_size_y);
        int bin_index_yh = int((y_max - yl) * inv_bin_size_y) + 1;
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
    return 0;
}

template <typename T>
int computeInstanceRoutabilityOptimizationMapLauncher(
    T *pos_x, T *pos_y,
    T *pin_pos_x, T *pin_pos_y,
    T *node_size_x, T *node_size_y,
    T xl, T yl,
    T bin_size_x, T bin_size_y,
    int num_bins_x, int num_bins_y,
    int num_nets,
    int num_nodes,
    int num_movable_nodes,
    int num_threads,
    T *routing_utilization_map,
    T *instance_route_area)
{
    const T inv_bin_size_x = 1.0 / bin_size_x;
    const T inv_bin_size_y = 1.0 / bin_size_y;

    int chunk_size = DREAMPLACE_STD_NAMESPACE::max(int(num_movable_nodes / num_threads / 16), 1);
    #pragma omp parallel for num_threads(num_threads) schedule(dynamic, chunk_size)
    for (int i = 0; i < num_movable_nodes; ++i) // for each movable node
    {
        const T x_max = pos_x[i] + node_size_x[i];
        const T x_min = pos_x[i];
        const T y_max = pos_y[i] + node_size_y[i];
        const T y_min = pos_y[i];

        // compute the bin box that this net will affect
        // We do NOT follow Wuxi's implementation. Instead, we clamp the bounding box.
        int bin_index_xl = int((x_min - xl) * inv_bin_size_x);
        int bin_index_xh = int((x_max - xl) * inv_bin_size_x) + 1;
        bin_index_xl = DREAMPLACE_STD_NAMESPACE::max(bin_index_xl, 0);
        bin_index_xh = DREAMPLACE_STD_NAMESPACE::min(bin_index_xh, num_bins_x);

        int bin_index_yl = int((y_min - yl) * inv_bin_size_y);
        int bin_index_yh = int((y_max - yl) * inv_bin_size_y) + 1;
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

void instance_route_optimization_area(
    at::Tensor instance_route_area,
    at::Tensor pos,
    at::Tensor pin_pos,
    at::Tensor netpin_start,
    at::Tensor flat_netpin,
    int num_bins_x,
    int num_bins_y,
    double bin_size_x,
    double bin_size_y,
    at::Tensor node_size_x,
    at::Tensor node_size_y,
    at::Tensor net_weights,
    double xl,
    double yl,
    double xh,
    double yh,
    int num_nets,
    int num_nodes,
    int num_movable_nodes,
    double unit_horizontal_routing_capacity,
    double unit_vertical_routing_capacity,
    double max_route_opt_adjust_rate,
    double min_route_opt_adjust_rate,
    int num_threads)
{
    CHECK_FLAT(instance_route_area);
    CHECK_CONTIGUOUS(instance_route_area);

    CHECK_FLAT(pos);
    CHECK_EVEN(pos);
    CHECK_CONTIGUOUS(pos);

    CHECK_FLAT(pin_pos);
    CHECK_EVEN(pin_pos);
    CHECK_CONTIGUOUS(pin_pos);

    CHECK_FLAT(netpin_start);
    CHECK_CONTIGUOUS(netpin_start);

    CHECK_FLAT(flat_netpin);
    CHECK_CONTIGUOUS(flat_netpin);

    CHECK_FLAT(node_size_x);
    CHECK_CONTIGUOUS(node_size_x);

    CHECK_FLAT(node_size_y);
    CHECK_CONTIGUOUS(node_size_y);

    CHECK_FLAT(net_weights);
    CHECK_CONTIGUOUS(net_weights);

    int num_pins = pin_pos.numel() / 2;
    at::Tensor routing_utilization_map_x = at::zeros(num_bins_x * num_bins_y, pin_pos.options());
    at::Tensor routing_utilization_map_y = at::zeros(num_bins_x * num_bins_y, pin_pos.options());

    const bool exist_net_weights = net_weights.numel();
    // Call the cpp kernel launcher
    DREAMPLACE_DISPATCH_FLOATING_TYPES(pin_pos.type(), "fillDemandMapLauncher", [&] {
        fillDemandMapLauncher<scalar_t>(
            pin_pos.data<scalar_t>(), pin_pos.data<scalar_t>() + num_pins,
            netpin_start.data<int>(),
            flat_netpin.data<int>(),
            bin_size_x, bin_size_y,
            xl, yl, xh, yh,
            
            exist_net_weights,
            net_weights.data<scalar_t>(),
            
            num_bins_x, num_bins_y,
            num_nets,
            num_threads,
            routing_utilization_map_x.data<scalar_t>(),
            routing_utilization_map_y.data<scalar_t>());
    });

    // convert demand to utilization in each bin
    routing_utilization_map_x.mul_(1 / (bin_size_x * bin_size_y * unit_horizontal_routing_capacity));
    routing_utilization_map_y.mul_(1 / (bin_size_x * bin_size_y * unit_vertical_routing_capacity));
    // infinity norm
    at::Tensor routing_utilization_map = at::empty_like(routing_utilization_map_x);
    routing_utilization_map = at::max(routing_utilization_map_x.abs(), routing_utilization_map_y.abs());
    // clamp the routing square of routing utilization map
    routing_utilization_map = at::clamp(routing_utilization_map * routing_utilization_map, min_route_opt_adjust_rate, max_route_opt_adjust_rate);

    // compute routability and density optimziation instance area
    DREAMPLACE_DISPATCH_FLOATING_TYPES(pin_pos.type(), "computeInstanceRoutabilityOptimizationMapLauncher", [&] {
        computeInstanceRoutabilityOptimizationMapLauncher<scalar_t>(
            pos.data<scalar_t>(), pos.data<scalar_t>() + num_nodes,
            pin_pos.data<scalar_t>(), pin_pos.data<scalar_t>() + num_pins,
            node_size_x.data<scalar_t>(), node_size_y.data<scalar_t>(),
            xl, yl,
            bin_size_x, bin_size_y,
            num_bins_x, num_bins_y,
            num_nets,
            num_nodes,
            num_movable_nodes,
            num_threads,
            routing_utilization_map.data<scalar_t>(),
            instance_route_area.data<scalar_t>());
    });
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("instance_route_optimization_area", &DREAMPLACE_NAMESPACE::instance_route_optimization_area, "compute routability optimized area");
}
