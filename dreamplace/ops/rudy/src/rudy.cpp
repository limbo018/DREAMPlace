/**
 * @file   rudy.cpp
 * @author Zixuan Jiang, Jiaqi Gu, Yibo Lin
 * @date   Dec 2019
 * @brief  Compute the RUDY/RISA map for routing demand. 
 *         A routing/pin utilization estimator based on the following two papers
 *         "Fast and Accurate Routing Demand Estimation for efficient Routability-driven Placement", by Peter Spindler, DATE'07
 *         "RISA: Accurate and Efficient Placement Routability Modeling", by Chih-liang Eric Cheng, ICCAD'94
 */

#include "utility/src/torch.h"
#include "utility/src/Msg.h"
#include "utility/src/utils.h"
#include "rudy/src/parameters.h"

DREAMPLACE_BEGIN_NAMESPACE

#define CHECK_FLAT(x) AT_ASSERTM(!x.is_cuda() && x.ndimension() == 1, #x "must be a flat tensor on CPU")
#define CHECK_EVEN(x) AT_ASSERTM((x.numel() & 1) == 0, #x "must have even number of elements")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x "must be contiguous")

template <typename T>
inline DEFINE_NET_WIRING_DISTRIBUTION_MAP_WEIGHT;

// fill the demand map net by net
template <typename T>
int rudyLauncher(const T *pin_pos_x,
                          const T *pin_pos_y,
                          const int *netpin_start,
                          const int *flat_netpin,
                          const T *net_weights,
                          const T bin_size_x, const T bin_size_y,
                          T xl, T yl, T xh, T yh,
                          int num_bins_x, int num_bins_y,
                          int num_nets,
                          int num_threads,
                          T *routing_utilization_map_x,
                          T *routing_utilization_map_y)
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

        // Following Wuxi's implementation, a tolerance is added to avoid 0-size bounding box
        x_max += TOLERANCE;
        y_max += TOLERANCE;

        // compute the bin box that this net will affect
        int bin_index_xl = int((x_min - xl) * inv_bin_size_x);
        int bin_index_xh = int((x_max - xl) * inv_bin_size_x) + 1;
        bin_index_xl = DREAMPLACE_STD_NAMESPACE::max(bin_index_xl, 0);
        bin_index_xh = DREAMPLACE_STD_NAMESPACE::min(bin_index_xh, num_bins_x);

        int bin_index_yl = int((y_min - yl) * inv_bin_size_y);
        int bin_index_yh = int((y_max - yl) * inv_bin_size_y) + 1;
        bin_index_yl = DREAMPLACE_STD_NAMESPACE::max(bin_index_yl, 0);
        bin_index_yh = DREAMPLACE_STD_NAMESPACE::min(bin_index_yh, num_bins_y);

        T wt = netWiringDistributionMapWeight<T>(netpin_start[i + 1] - netpin_start[i]);
        if (net_weights)
        {
            wt *= net_weights[i];
        }

        for (int x = bin_index_xl; x < bin_index_xh; ++x)
        {
            for (int y = bin_index_yl; y < bin_index_yh; ++y)
            {
                T overlap = wt * (DREAMPLACE_STD_NAMESPACE::min(x_max, (x + 1) * bin_size_x) - DREAMPLACE_STD_NAMESPACE::max(x_min, x * bin_size_x)) *
                            (DREAMPLACE_STD_NAMESPACE::min(y_max, (y + 1) * bin_size_y) - DREAMPLACE_STD_NAMESPACE::max(y_min, y * bin_size_y));
                int index = x * num_bins_y + y;
                #pragma omp atomic update
                routing_utilization_map_x[index] += overlap / (y_max - y_min);
                #pragma omp atomic update
                routing_utilization_map_y[index] += overlap / (x_max - x_min);
            }
        }
    }
    return 0;
}

at::Tensor rudy_forward(
    at::Tensor pin_pos,
    at::Tensor netpin_start,
    at::Tensor flat_netpin,
    at::Tensor net_weights,
    double bin_size_x,
    double bin_size_y,
    double xl,
    double yl,
    double xh,
    double yh,
    double unit_horizontal_routing_capacity,
    double unit_vertical_routing_capacity,
    double max_route_opt_adjust_rate,
    double min_route_opt_adjust_rate,
    int num_bins_x,
    int num_bins_y,
    int num_threads
    )
{
    CHECK_FLAT(pin_pos);
    CHECK_EVEN(pin_pos);
    CHECK_CONTIGUOUS(pin_pos);

    CHECK_FLAT(netpin_start);
    CHECK_CONTIGUOUS(netpin_start);

    CHECK_FLAT(flat_netpin);
    CHECK_CONTIGUOUS(flat_netpin);

    CHECK_FLAT(net_weights);
    CHECK_CONTIGUOUS(net_weights);

    int num_nets = netpin_start.numel() - 1; 
    int num_pins = pin_pos.numel() / 2;
    at::Tensor routing_utilization_map_x = at::zeros({num_bins_x, num_bins_y}, pin_pos.options());
    at::Tensor routing_utilization_map_y = at::zeros({num_bins_x, num_bins_y}, pin_pos.options());

    DREAMPLACE_DISPATCH_FLOATING_TYPES(pin_pos.type(), "rudyLauncher", [&] {
        rudyLauncher<scalar_t>(
            pin_pos.data<scalar_t>(), pin_pos.data<scalar_t>() + num_pins,
            netpin_start.data<int>(),
            flat_netpin.data<int>(),
            (net_weights.numel())? net_weights.data<scalar_t>() : nullptr,
            bin_size_x, bin_size_y,
            xl, yl, xh, yh,

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
    at::Tensor routing_utilization_map = at::max(routing_utilization_map_x.abs(), routing_utilization_map_y.abs());
    // clamp the routing square of routing utilization map
    routing_utilization_map.pow_(2).clamp_(min_route_opt_adjust_rate, max_route_opt_adjust_rate);

    return routing_utilization_map;
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &DREAMPLACE_NAMESPACE::rudy_forward, "compute RUDY map");
}
