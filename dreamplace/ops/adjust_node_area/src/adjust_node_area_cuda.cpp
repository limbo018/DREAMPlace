/**
 * @file   adjust_node_area_cuda.cpp
 * @author Zixuan Jiang, Jiaqi Gu, Yibo Lin
 * @date   Dec 2019
 * @brief  Adjust cell area according to congestion map. 
 */

#include "utility/src/torch.h"
#include "utility/src/Msg.h"

DREAMPLACE_BEGIN_NAMESPACE

#define CHECK_FLAT(x) AT_ASSERTM(x.is_cuda() && x.ndimension() == 1, #x "must be a flat tensor on GPU")
#define CHECK_EVEN(x) AT_ASSERTM((x.numel() & 1) == 0, #x "must have even number of elements")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x "must be contiguous")

template <typename T>
int computeInstanceRoutabilityOptimizationMapCudaLauncher(
    const T *pos_x, const T *pos_y,
    const T *node_size_x, const T *node_size_y,
    const T *routing_utilization_map,
    T xl, T yl,
    T bin_size_x, T bin_size_y,
    int num_bins_x, int num_bins_y,
    int num_movable_nodes,
    T *instance_route_area
    );

at::Tensor adjust_node_area_forward(
    at::Tensor pos,
    at::Tensor node_size_x,
    at::Tensor node_size_y,
    at::Tensor routing_utilization_map,
    double bin_size_x,
    double bin_size_y,
    double xl,
    double yl,
    double xh,
    double yh,
    int num_movable_nodes,
    int num_bins_x,
    int num_bins_y
    )
{
    CHECK_FLAT(pos);
    CHECK_EVEN(pos);
    CHECK_CONTIGUOUS(pos);

    CHECK_FLAT(node_size_x);
    CHECK_CONTIGUOUS(node_size_x);

    CHECK_FLAT(node_size_y);
    CHECK_CONTIGUOUS(node_size_y);

    int num_nodes = pos.numel() / 2; 
    at::Tensor instance_route_area = at::zeros({num_movable_nodes}, pos.options());

    // compute routability and density optimziation instance area
    DREAMPLACE_DISPATCH_FLOATING_TYPES(pos.type(), "computeInstanceRoutabilityOptimizationMapCudaLauncher", [&] {
        computeInstanceRoutabilityOptimizationMapCudaLauncher<scalar_t>(
            pos.data<scalar_t>(), pos.data<scalar_t>() + num_nodes,
            node_size_x.data<scalar_t>(), node_size_y.data<scalar_t>(),
            routing_utilization_map.data<scalar_t>(),
            xl, yl,
            bin_size_x, bin_size_y,
            num_bins_x, num_bins_y,
            num_movable_nodes,
            instance_route_area.data<scalar_t>());
    });

    return instance_route_area; 
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &DREAMPLACE_NAMESPACE::adjust_node_area_forward, "Compute adjusted area for routability optimization (CUDA)");
}
