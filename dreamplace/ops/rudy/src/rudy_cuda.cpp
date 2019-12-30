/**
 * @file   rudy_cuda.cpp
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

// fill the demand map net by net
template <typename T>
int rudyCudaLauncher(const T *pin_pos_x,
                              const T *pin_pos_y,
                              const int *netpin_start,
                              const int *flat_netpin,
                              const T *net_weights,
                              T bin_size_x, T bin_size_y,
                              T xl, T yl, T xh, T yh,

                              int num_bins_x, int num_bins_y,
                              int num_nets,
                              T *horizontal_utilization_map,
                              T *vertical_utilization_map);

void rudy_forward(
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
    int num_bins_x,
    int num_bins_y,
    at::Tensor horizontal_utilization_map, 
    at::Tensor vertical_utilization_map 
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

    // Call the cuda kernel launcher
    DREAMPLACE_DISPATCH_FLOATING_TYPES(pin_pos.type(), "rudyCudaLauncher", [&] {
        rudyCudaLauncher<scalar_t>(
            pin_pos.data<scalar_t>(), pin_pos.data<scalar_t>() + num_pins,
            netpin_start.data<int>(),
            flat_netpin.data<int>(),
            (net_weights.numel())? net_weights.data<scalar_t>() : nullptr,
            bin_size_x, bin_size_y,
            xl, yl, xh, yh,

            num_bins_x, num_bins_y,
            num_nets,
            horizontal_utilization_map.data<scalar_t>(),
            vertical_utilization_map.data<scalar_t>());
    });
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &DREAMPLACE_NAMESPACE::rudy_forward, "compute RUDY map (CUDA)");
}
