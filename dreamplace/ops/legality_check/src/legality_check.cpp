/**
 * @file   legality_check.cpp
 * @author Yibo Lin
 * @date   Jan 2020
 */
#include "utility/src/torch.h"
#include "legality_check/src/legality_check.h"

DREAMPLACE_BEGIN_NAMESPACE

#define CHECK_FLAT(x) AT_ASSERTM(!x.is_cuda() && x.ndimension() == 1, #x "must be a flat tensor on CPU")
#define CHECK_EVEN(x) AT_ASSERTM((x.numel()&1) == 0, #x "must have even number of elements")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x "must be contiguous")

/// @brief check legality 
bool legality_check_forward(
        at::Tensor pos, 
        at::Tensor node_size_x, at::Tensor node_size_y, 
        at::Tensor flat_region_boxes, at::Tensor flat_region_boxes_start, at::Tensor node2fence_region_map, 
        double site_width, double row_height, 
        double xl, double yl, double xh, double yh,
        const int num_fixed_nodes, 
        const int num_movable_nodes, 
        const int num_regions
        )
{
    CHECK_FLAT(pos); 
    CHECK_EVEN(pos);
    CHECK_CONTIGUOUS(pos);

    int num_nodes = pos.numel() / 2; 
    bool legal_flag = true; 

    hr_clock_rep timer_start, timer_stop; 
    timer_start = get_globaltime(); 
    // Call the cuda kernel launcher
    DREAMPLACE_DISPATCH_FLOATING_TYPES(pos.type(), "legalityCheckKernelCPU", [&] {
            legal_flag = legalityCheckKernelCPU<scalar_t>(
                    pos.data<scalar_t>(), pos.data<scalar_t>() + num_nodes, 
                    node_size_x.data<scalar_t>(), node_size_y.data<scalar_t>(), 
                    flat_region_boxes.data<scalar_t>(), flat_region_boxes_start.data<int>(), node2fence_region_map.data<int>(), 
                    site_width, row_height, 
                    xl, yl, xh, yh,
                    num_movable_nodes + num_fixed_nodes, ///< movable and fixed cells 
                    num_movable_nodes, 
                    num_regions
                    );
            });
    timer_stop = get_globaltime(); 
    dreamplacePrint(kINFO, "Legality check takes %g ms\n", (timer_stop-timer_start)*get_timer_period());

    return legal_flag; 
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &DREAMPLACE_NAMESPACE::legality_check_forward, "Legality check forward");
}
