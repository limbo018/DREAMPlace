#include "utility/src/torch.h"
#include "utility/src/Msg.h"

DREAMPLACE_BEGIN_NAMESPACE

#define CHECK_FLAT(x) AT_ASSERTM(x.is_cuda() && x.ndimension() == 1, #x "must be a flat tensor on GPU")
#define CHECK_EVEN(x) AT_ASSERTM((x.numel() & 1) == 0, #x "must have even number of elements")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x "must be contiguous")

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
                              T *routing_utilization_map_y);

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
    T *instance_route_area);

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
    double min_route_opt_adjust_rate)
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
    // Call the cuda kernel launcher
    DREAMPLACE_DISPATCH_FLOATING_TYPES(pin_pos.type(), "fillDemandMapCudaLauncher", [&] {
        fillDemandMapCudaLauncher<scalar_t>(
            pin_pos.data<scalar_t>(), pin_pos.data<scalar_t>() + num_pins,
            netpin_start.data<int>(),
            flat_netpin.data<int>(),
            bin_size_x, bin_size_y,
            xl, yl, xh, yh,

            exist_net_weights,
            net_weights.data<scalar_t>(),

            num_bins_x, num_bins_y,
            num_nets,
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
    DREAMPLACE_DISPATCH_FLOATING_TYPES(pin_pos.type(), "computeInstanceRoutabilityOptimizationMapCudaLauncher", [&] {
        computeInstanceRoutabilityOptimizationMapCudaLauncher<scalar_t>(
            pos.data<scalar_t>(), pos.data<scalar_t>() + num_nodes,
            pin_pos.data<scalar_t>(), pin_pos.data<scalar_t>() + num_pins,
            node_size_x.data<scalar_t>(), node_size_y.data<scalar_t>(),
            xl, yl,
            bin_size_x, bin_size_y,
            num_bins_x, num_bins_y,
            num_nets,
            num_nodes,
            num_movable_nodes,
            routing_utilization_map.data<scalar_t>(),
            instance_route_area.data<scalar_t>());
    });
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("instance_route_optimization_area", &DREAMPLACE_NAMESPACE::instance_route_optimization_area, "Fill Demand Map");
}
