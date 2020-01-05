/**
 * @file   pin_utilization_map_cuda.cpp
 * @author Zixuan Jiang, Jiaqi Gu, Yibo Lin
 * @date   Dec 2019
 * @brief  Compute the RUDY/RISA map for routing demand. 
 *         A routing/pin utilization estimator based on the following two papers
 *         "Fast and Accurate Routing Demand Estimation for efficient Routability-driven Placement", by Peter Spindler, DATE'07
 *         "RISA: Accurate and Efficient Placement Routability Modeling", by Chih-liang Eric Cheng, ICCAD'94
 */

#include "utility/src/torch.h"
#include "utility/src/Msg.h"

DREAMPLACE_BEGIN_NAMESPACE

#define CHECK_FLAT(x) AT_ASSERTM(x.is_cuda() && x.ndimension() == 1, #x "must be a flat tensor on GPU")
#define CHECK_EVEN(x) AT_ASSERTM((x.numel() & 1) == 0, #x "must have even number of elements")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x "must be contiguous")

// fill the demand map pin by pin
template <typename T>
int pinDemandMapCudaLauncher(const T *node_x, const T *node_y,
                          const T *node_size_x, const T *node_size_y,
                          const T *half_node_size_stretch_x, const T *half_node_size_stretch_y,
                          const T *pin_weights,
                          T xl, T yl, T xh, T yh,
                          T bin_size_x, T bin_size_y,
                          int num_bins_x, int num_bins_y,
                          int num_nodes,
                          T *pin_utilization_map 
                          );

at::Tensor pin_utilization_map_forward(
    at::Tensor pos,
    at::Tensor node_size_x,
    at::Tensor node_size_y,
    at::Tensor half_node_size_stretch_x,
    at::Tensor half_node_size_stretch_y,
    at::Tensor pin_weights,
    double xl,
    double yl,
    double xh,
    double yh,
    double bin_size_x,
    double bin_size_y,
    int num_physical_nodes,
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

    at::Tensor pin_utilization_map = at::zeros({num_bins_x, num_bins_y}, pos.options());
    auto num_nodes = pos.numel()/2;

    DREAMPLACE_DISPATCH_FLOATING_TYPES(pos.type(), "pinDemandMapCudaLauncher", [&] {
            pinDemandMapCudaLauncher<scalar_t>(
                    pos.data<scalar_t>(), pos.data<scalar_t>() + num_nodes, 
                    node_size_x.data<scalar_t>(), node_size_y.data<scalar_t>(), 
                    half_node_size_stretch_x.data<scalar_t>(), half_node_size_stretch_y.data<scalar_t>(),
                    pin_weights.data<scalar_t>(),
                    xl, yl, xh, yh,
                    bin_size_x, bin_size_y,
                    num_bins_x, num_bins_y,
                    num_physical_nodes,
                    pin_utilization_map.data<scalar_t>()
                    );
    });

    return pin_utilization_map;
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &DREAMPLACE_NAMESPACE::pin_utilization_map_forward, "compute pin utilization map (CUDA)");
}
