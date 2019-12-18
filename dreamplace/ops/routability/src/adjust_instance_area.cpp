#include "utility/src/torch.h"
#include "utility/src/Msg.h"
#include "utility/src/utils.h"

DREAMPLACE_BEGIN_NAMESPACE

#define CHECK_FLAT(x) AT_ASSERTM(!x.is_cuda() && x.ndimension() == 1, #x "must be a flat tensor on CPU")
#define CHECK_EVEN(x) AT_ASSERTM((x.numel() & 1) == 0, #x "must have even number of elements")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x "must be contiguous")

#define route_area_adjust_stop_ratio (0.01)
template <typename T>
int f()
{
}

bool adjust_instance_area(
    at::Tensor pos,
    at::Tensor pin_pos,
    at::Tensor node_size_x,
    at::Tensor node_size_y,
    at::Tensor netpin_start,
    at::Tensor flat_netpin,
    int num_nodes,
    int num_movable_nodes,
    int num_filler_nodes,
    at::Tensor instance_route_area,
    at::Tensor max_total_area,
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

    CHECK_FLAT(max_total_area);
    CHECK_CONTIGUOUS(max_total_area);

    // compute final areas
    at::Tensor movable_base_area = node_size_x[:num_movable_nodes].mul(node_size_y[:num_movable_nodes]);
    at::Tensor prev_total_base_area = movable_base_area.sum();

    at::Tensor final_movable_area = at::relu(instance_route_area - movable_base_area);
    at::Tensor total_extra_area = final_movable_area.sum();

    // check whether the total area is larger than the max area requirement
    // If yes, scale the extra area to meet the requirement
    // We assume the total base area is no greater than the max area requirement
    at::Tensor scale_factor = at::clamp((max_total_area - prev_total_base_area) / total_extra_area, 0, 1);

    // set the areas of movable instance as base_area + scaled extra area
    final_movable_area = movable_base_area + scale_factor.mul(final_movable_area);

    // scale the filler instance areas to make the total area meets the max area requirement
    at::Tensor filler_area = at::relu((max_total_area - prev_total_base_area - scale_factor * total_extra_area) / num_filler_nodes);
    at::Tensor final_filler_area = at::empty(num_filler_nodes, pos.options()).fill_(filler_area);

    // compute the adjusted area increment
    at::Tensor movable_area_increment = at::relu(final_movable_area - movable_base_area);
    at::Tensor route_area_increment = at::relu(instance_route_area - movable_base_area);

    adjust_route_area &= (route_area_increment.sum() / prev_total_base_area > route_area_adjust_stop_ratio);
    adjust_area &= (monvable_area_increment.sum() / prev_total_base_area > area_adjust_stop_ratio);
    adjust_area &= adjust_route_area;
    if (!adjust_area)
    {
        return false;
    }

    at::Tensor movable_nodes_ratio = at::sqrt(final_movable_area / movable_base_area);
    // adjust the size of nodes
    node_size_x.mul_(movable_nodes_ratio);
    node_size_y.mul_(movable_nodes_ratio);

    // Call the cpp kernel launcher
    DREAMPLACE_DISPATCH_FLOATING_TYPES(pin_pos.type(), "fillDemandMapLauncher", [&] {
        fillDemandMapLauncher<scalar_t>(
            pin_pos.data<scalar_t>(), pin_pos.data<scalar_t>() + num_pins,
            netpin_start.data<int>(),
            flat_netpin.data<int>(),
            bin_size_x, bin_size_y,
            xl, yl, xh, yh,
            num_bins_x, num_bins_y,
            num_nets,
            num_threads,
            routing_utilization_map_x.data<scalar_t>(),
            routing_utilization_map_y.data<scalar_t>());
    });
}

DREAMPLACE_END_NAMESPACE
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("adjust_instance_area", &DREAMPLACE_NAMESPACE::adjust_instance_area, "adjust instance area");
}
