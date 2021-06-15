/**
 * @file   move_boundary.cpp
 * @author Yibo Lin
 * @date   Jun 2018
 * @brief  Move out-of-bound cells back to inside placement region
 */
#include "utility/src/torch.h"
#include "utility/src/utils.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
int computeMoveBoundaryMapLauncher(
    T* x_tensor, T* y_tensor, const T* node_size_x_tensor,
    const T* node_size_y_tensor, const T xl, const T yl, const T xh, const T yh,
    const int num_nodes, const int num_movable_nodes,
    const int num_filler_nodes, const int num_threads);

at::Tensor move_boundary_forward(at::Tensor pos, at::Tensor node_size_x,
                                 at::Tensor node_size_y, double xl, double yl,
                                 double xh, double yh, int num_movable_nodes,
                                 int num_filler_nodes) {
  CHECK_FLAT_CPU(pos);
  CHECK_EVEN(pos);
  CHECK_CONTIGUOUS(pos);

  // Call the cuda kernel launcher
  DREAMPLACE_DISPATCH_FLOATING_TYPES(
      pos, "computeMoveBoundaryMapLauncher", [&] {
        computeMoveBoundaryMapLauncher<scalar_t>(
            DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + pos.numel() / 2,
            DREAMPLACE_TENSOR_DATA_PTR(node_size_x, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(node_size_y, scalar_t), xl, yl, xh, yh,
            pos.numel() / 2, num_movable_nodes, num_filler_nodes,
            at::get_num_threads());
      });

  return pos;
}

template <typename T>
int computeMoveBoundaryMapLauncher(
    T* x_tensor, T* y_tensor, const T* node_size_x_tensor,
    const T* node_size_y_tensor, const T xl, const T yl, const T xh, const T yh,
    const int num_nodes, const int num_movable_nodes,
    const int num_filler_nodes, const int num_threads) {
#pragma omp parallel for num_threads(num_threads)
  for (int i = 0; i < num_nodes; ++i) {
    if (i < num_movable_nodes || i >= num_nodes - num_filler_nodes) {
      x_tensor[i] = std::max(x_tensor[i], xl);
      x_tensor[i] = std::min(x_tensor[i], xh - node_size_x_tensor[i]);

      y_tensor[i] = std::max(y_tensor[i], yl);
      y_tensor[i] = std::min(y_tensor[i], yh - node_size_y_tensor[i]);
    }
  }

  return 0;
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &DREAMPLACE_NAMESPACE::move_boundary_forward,
        "MoveBoundary forward");
  // m.def("backward", &DREAMPLACE_NAMESPACE::move_boundary_backward,
  // "MoveBoundary backward");
}
