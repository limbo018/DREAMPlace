/**
 * @file   legality_check.cpp
 * @author Yibo Lin
 * @date   Jan 2020
 */
#include "legality_check/src/legality_check.h"
#include "utility/src/torch.h"

DREAMPLACE_BEGIN_NAMESPACE

/// @brief check legality
bool legality_check_forward(
    at::Tensor pos, at::Tensor node_size_x, at::Tensor node_size_y,
    at::Tensor flat_region_boxes, at::Tensor flat_region_boxes_start,
    at::Tensor node2fence_region_map, double xl, double yl, double xh,
    double yh, double site_width, double row_height, double scale_factor,
    const int num_fixed_nodes, const int num_movable_nodes) {
  CHECK_FLAT_CPU(pos);
  CHECK_EVEN(pos);
  CHECK_CONTIGUOUS(pos);

  int num_nodes = pos.numel() / 2;
  bool legal_flag = true;

  CPUTimer::hr_clock_rep timer_start, timer_stop;
  timer_start = CPUTimer::getGlobaltime();
  // Call the cuda kernel launcher
  DREAMPLACE_DISPATCH_FLOATING_TYPES(pos, "legalityCheckKernelCPU", [&] {
    legal_flag = legalityCheckKernelCPU<scalar_t>(
        DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + num_nodes,
        DREAMPLACE_TENSOR_DATA_PTR(node_size_x, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(node_size_y, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(flat_region_boxes, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(flat_region_boxes_start, int),
        DREAMPLACE_TENSOR_DATA_PTR(node2fence_region_map, int), xl, yl, xh, yh,
        site_width, row_height, scale_factor,
        num_movable_nodes + num_fixed_nodes,  ///< movable and fixed cells
        num_movable_nodes, flat_region_boxes_start.numel() - 1);
  });
  timer_stop = CPUTimer::getGlobaltime();
  dreamplacePrint(kINFO, "Legality check takes %g ms\n",
                  (timer_stop - timer_start) * CPUTimer::getTimerPeriod());

  return legal_flag;
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &DREAMPLACE_NAMESPACE::legality_check_forward,
        "Legality check forward");
}
