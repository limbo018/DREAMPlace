#include "utility/src/torch.h"
#include "utility/src/Msg.h"
#include "utility/src/utils.h"

DREAMPLACE_BEGIN_NAMESPACE

#define CHECK_FLAT(x) AT_ASSERTM(!x.is_cuda() && x.ndimension() == 1, #x "must be a flat tensor on CPU")
#define CHECK_EVEN(x) AT_ASSERTM((x.numel() & 1) == 0, #x "must have even number of elements")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x "must be contiguous")

at::Tensor instance_route_optimization_area_cuda(
    at::Tensor instance_route_area_array,
    at::Tensor pos,
    at::Tensor pin_pos,
    at::Tensor netpin_start,
    at::Tensor flat_netpin,
    double num_bins_x,
    double num_bins_y,
    double bin_size_x,
    double bin_size_y,
    at::Tensor node_size_x,
    at::Tensor node_size_y,
    double xl,
    double yl,
    double xh,
    double yh,
    int num_nets,
    int num_nodes,
    int num_movable_nodes,
    int num_filler_nodes,
    double unit_horizontal_routing_capacity,
    double unit_vertical_routing_capacity,
    double max_route_opt_adjust_rate,
    double min_route_opt_adjust_rate,
    int num_threads)
{
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("instance_route_optimization_area_cuda", &DREAMPLACE_NAMESPACE::instance_route_optimization_area_cuda, "Fill Demand Map");
}
