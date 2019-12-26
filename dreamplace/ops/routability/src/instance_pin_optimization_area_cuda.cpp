#include "utility/src/torch.h"
#include "utility/src/Msg.h"
#include "utility/src/utils.h"

DREAMPLACE_BEGIN_NAMESPACE

#define CHECK_FLAT(x) AT_ASSERTM(x.is_cuda() && x.ndimension() == 1, #x "must be a flat tensor on GPU")
#define CHECK_EVEN(x) AT_ASSERTM((x.numel() & 1) == 0, #x "must have even number of elements")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x "must be contiguous")

// fill the demand map pin by pin
template <typename T>
int fillDemandMapCudaLauncher(T bin_size_x, T bin_size_y,
                              T *node_center_x, T *node_center_y,
                              T *half_node_size_stretch_x, T *half_node_size_stretch_y,
                              T xl, T yl, T xh, T yh,
                              const T *pin_weights,
                              int num_bins_x, int num_bins_y,
                              int num_physical_nodes,
                              T *pin_utilization_map);

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
    T *instance_pin_area);

void instance_pin_optimization_area(
    at::Tensor pos,
    at::Tensor node_center_x,
    at::Tensor node_center_y,
    at::Tensor half_node_size_stretch_x,
    at::Tensor half_node_size_stretch_y,
    at::Tensor pin_weights,
    int num_bins_x,
    int num_bins_y,
    double bin_size_x,
    double bin_size_y,
    at::Tensor node_size_x,
    at::Tensor node_size_y,
    double xl,
    double yl,
    double xh,
    double yh,
    int num_nodes,
    int num_movable_nodes,
    int num_physical_nodes,
    double unit_pin_capacity,
    double max_pin_opt_adjust_rate,
    double min_pin_opt_adjust_rate,
    at::Tensor instance_pin_area)
{
    CHECK_FLAT(pos);
    CHECK_EVEN(pos);
    CHECK_CONTIGUOUS(pos);

    CHECK_FLAT(node_center_x);
    CHECK_CONTIGUOUS(node_center_x);

    CHECK_FLAT(node_center_y);
    CHECK_CONTIGUOUS(node_center_y);

    CHECK_FLAT(half_node_size_stretch_x);
    CHECK_CONTIGUOUS(half_node_size_stretch_x);

    CHECK_FLAT(half_node_size_stretch_y);
    CHECK_CONTIGUOUS(half_node_size_stretch_y);

    CHECK_FLAT(pin_weights);
    CHECK_CONTIGUOUS(pin_weights);

    CHECK_FLAT(node_size_x);
    CHECK_CONTIGUOUS(node_size_x);

    CHECK_FLAT(node_size_y);
    CHECK_CONTIGUOUS(node_size_y);

    CHECK_FLAT(instance_pin_area);
    CHECK_CONTIGUOUS(instance_pin_area);

    at::Tensor pin_utilization_map = at::zeros(num_bins_x * num_bins_y, pos.options());

    DREAMPLACE_DISPATCH_FLOATING_TYPES(pos.type(), "fillDemandMapCudaLauncher", [&] {
        fillDemandMapCudaLauncher<scalar_t>(
            bin_size_x, bin_size_y,
            node_center_x.data<scalar_t>(), node_center_y.data<scalar_t>(),
            half_node_size_stretch_x.data<scalar_t>(), half_node_size_stretch_y.data<scalar_t>(),
            xl, yl, xh, yh,
            pin_weights.data<scalar_t>(),
            num_bins_x, num_bins_y,
            num_physical_nodes,
            pin_utilization_map.data<scalar_t>());
    });

    // convert demand to utilization in each bin
    pin_utilization_map.mul_(1 / (bin_size_x * bin_size_y * unit_pin_capacity));
    pin_utilization_map.clamp_(min_pin_opt_adjust_rate, max_pin_opt_adjust_rate);

    // compute pin density optimization instance area
    DREAMPLACE_DISPATCH_FLOATING_TYPES(pos.type(), "computeInstancePinOptimizationMapCudaLauncher", [&] {
        computeInstancePinOptimizationMapCudaLauncher<scalar_t>(
            pos.data<scalar_t>(), pos.data<scalar_t>() + num_nodes,
            node_size_x.data<scalar_t>(), node_size_y.data<scalar_t>(),
            xl, yl,
            bin_size_x, bin_size_y,
            num_bins_x, num_bins_y,
            num_movable_nodes,
            unit_pin_capacity,
            pin_utilization_map.data<scalar_t>(),
            pin_weights.data<scalar_t>(),
            instance_pin_area.data<scalar_t>());
    });
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("instance_pin_optimization_area", &DREAMPLACE_NAMESPACE::instance_pin_optimization_area, "compute pin density optimized area");
}
