/**
 * @file   pin_utilization_map_cuda.cpp
 * @author Zixuan Jiang, Jiaqi Gu, Yibo Lin
 * @date   Dec 2019
 * @brief  Compute the RUDY/RISA map for routing demand.
 *         A routing/pin utilization estimator based on the following two papers
 *         "Fast and Accurate Routing Demand Estimation for efficient
 * Routability-driven Placement", by Peter Spindler, DATE'07 "RISA: Accurate and
 * Efficient Placement Routability Modeling", by Chih-liang Eric Cheng,
 * ICCAD'94
 */

#include "utility/src/torch.h"
#include "utility/src/utils.h"

DREAMPLACE_BEGIN_NAMESPACE

// fill the demand map pin by pin
template <typename T>
int pinDemandMapCudaLauncher(const T *node_x, const T *node_y,
                             const T *node_size_x, const T *node_size_y,
                             const T *half_node_size_stretch_x,
                             const T *half_node_size_stretch_y,
                             const T *pin_weights, T xl, T yl, T xh, T yh,
                             int num_bins_x,
                             int num_bins_y, int num_nodes,
                             bool deterministic_flag, 
                             T *pin_utilization_map);

at::Tensor pin_utilization_map_forward(
    at::Tensor pos, at::Tensor node_size_x, at::Tensor node_size_y,
    at::Tensor half_node_size_stretch_x, at::Tensor half_node_size_stretch_y,
    at::Tensor pin_weights, double xl, double yl, double xh, double yh,
    int num_physical_nodes,
    int num_bins_x, int num_bins_y, int deterministic_flag) {
  CHECK_FLAT_CUDA(pos);
  CHECK_EVEN(pos);
  CHECK_CONTIGUOUS(pos);

  CHECK_FLAT_CUDA(node_size_x);
  CHECK_CONTIGUOUS(node_size_x);

  CHECK_FLAT_CUDA(node_size_y);
  CHECK_CONTIGUOUS(node_size_y);

  CHECK_FLAT_CUDA(half_node_size_stretch_x);
  CHECK_CONTIGUOUS(half_node_size_stretch_x);

  CHECK_FLAT_CUDA(half_node_size_stretch_y);
  CHECK_CONTIGUOUS(half_node_size_stretch_y);

  CHECK_FLAT_CUDA(pin_weights);
  CHECK_CONTIGUOUS(pin_weights);

  CHECK_FLAT_CUDA(node_size_x);
  CHECK_CONTIGUOUS(node_size_x);

  CHECK_FLAT_CUDA(node_size_y);
  CHECK_CONTIGUOUS(node_size_y);

  at::Tensor pin_utilization_map =
      at::zeros({num_bins_x, num_bins_y}, pos.options());
  auto num_nodes = pos.numel() / 2;

  DREAMPLACE_DISPATCH_FLOATING_TYPES(
      pos, "pinDemandMapCudaLauncher", [&] {
        pinDemandMapCudaLauncher<scalar_t>(
            DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + num_nodes,
            DREAMPLACE_TENSOR_DATA_PTR(node_size_x, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(node_size_y, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(half_node_size_stretch_x, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(half_node_size_stretch_y, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(pin_weights, scalar_t), xl, yl, xh, yh,
            num_bins_x, num_bins_y, num_physical_nodes,
            (bool)deterministic_flag,
            DREAMPLACE_TENSOR_DATA_PTR(pin_utilization_map, scalar_t));
      });

  return pin_utilization_map;
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &DREAMPLACE_NAMESPACE::pin_utilization_map_forward,
        "compute pin utilization map (CUDA)");
}
