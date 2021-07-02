/**
 * @file   electric_force.cpp
 * @author Yibo Lin
 * @date   Aug 2018
 * @brief  Compute electric force according to e-place
 */
#include "utility/src/torch.h"
#include "utility/src/utils.h"
// local dependency
#include "electric_potential/src/density_function.h"

DREAMPLACE_BEGIN_NAMESPACE

/// define triangle_density_function
template <typename T>
DEFINE_TRIANGLE_DENSITY_FUNCTION(T);

template <typename T>
int computeElectricForceLauncher(
    int num_bins_x, int num_bins_y, int num_impacted_bins_x,
    int num_impacted_bins_y, const T* field_map_x_tensor,
    const T* field_map_y_tensor, const T* x_tensor, const T* y_tensor,
    const T* node_size_x_clamped_tensor, const T* node_size_y_clamped_tensor,
    const T* offset_x_tensor, const T* offset_y_tensor, const T* ratio_tensor,
    const T* bin_center_x_tensor, const T* bin_center_y_tensor, T xl, T yl,
    T xh, T yh, T bin_size_x, T bin_size_y, int num_nodes, int num_threads,
    T* grad_x_tensor, T* grad_y_tensor);

#define CALL_LAUNCHER(begin, end)                                         \
  computeElectricForceLauncher<scalar_t>(                                 \
      num_bins_x, num_bins_y, num_filler_impacted_bins_x,                 \
      num_filler_impacted_bins_y,                                         \
      DREAMPLACE_TENSOR_DATA_PTR(field_map_x, scalar_t),                  \
      DREAMPLACE_TENSOR_DATA_PTR(field_map_y, scalar_t),                  \
      DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + begin,                  \
      DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + num_nodes + begin,      \
      DREAMPLACE_TENSOR_DATA_PTR(node_size_x_clamped, scalar_t) + begin,  \
      DREAMPLACE_TENSOR_DATA_PTR(node_size_y_clamped, scalar_t) + begin,  \
      DREAMPLACE_TENSOR_DATA_PTR(offset_x, scalar_t) + begin,             \
      DREAMPLACE_TENSOR_DATA_PTR(offset_y, scalar_t) + begin,             \
      DREAMPLACE_TENSOR_DATA_PTR(ratio, scalar_t) + begin,                \
      DREAMPLACE_TENSOR_DATA_PTR(bin_center_x, scalar_t),                 \
      DREAMPLACE_TENSOR_DATA_PTR(bin_center_y, scalar_t), xl, yl, xh, yh, \
      bin_size_x, bin_size_y, end - (begin), at::get_num_threads(),       \
      DREAMPLACE_TENSOR_DATA_PTR(grad_out, scalar_t) + begin,             \
      DREAMPLACE_TENSOR_DATA_PTR(grad_out, scalar_t) + num_nodes + begin)

/// @brief compute electric force for movable and filler cells
/// @param grad_pos input gradient from backward propagation
/// @param num_bins_x number of bins in horizontal bins
/// @param num_bins_y number of bins in vertical bins
/// @param num_movable_impacted_bins_x number of impacted bins for any movable
/// cell in x direction
/// @param num_movable_impacted_bins_y number of impacted bins for any movable
/// cell in y direction
/// @param num_filler_impacted_bins_x number of impacted bins for any filler
/// cell in x direction
/// @param num_filler_impacted_bins_y number of impacted bins for any filler
/// cell in y direction
/// @param field_map_x electric field map in x direction
/// @param field_map_y electric field map in y direction
/// @param pos cell locations. The array consists of all x locations and then y
/// locations.
/// @param node_size_x cell width array
/// @param node_size_y cell height array
/// @param bin_center_x bin center x locations
/// @param bin_center_y bin center y locations
/// @param xl left boundary
/// @param yl bottom boundary
/// @param xh right boundary
/// @param yh top boundary
/// @param bin_size_x bin width
/// @param bin_size_y bin height
/// @param num_movable_nodes number of movable cells
/// @param num_filler_nodes number of filler cells
at::Tensor electric_force(
    at::Tensor grad_pos, int num_bins_x, int num_bins_y,
    int num_movable_impacted_bins_x, int num_movable_impacted_bins_y,
    int num_filler_impacted_bins_x, int num_filler_impacted_bins_y,
    at::Tensor field_map_x, at::Tensor field_map_y, at::Tensor pos,
    at::Tensor node_size_x_clamped, at::Tensor node_size_y_clamped,
    at::Tensor offset_x, at::Tensor offset_y, at::Tensor ratio,
    at::Tensor bin_center_x, at::Tensor bin_center_y, double xl, double yl,
    double xh, double yh, double bin_size_x, double bin_size_y,
    int num_movable_nodes, int num_filler_nodes) {
  CHECK_FLAT_CPU(pos);
  CHECK_EVEN(pos);
  CHECK_CONTIGUOUS(pos);

  at::Tensor grad_out = at::zeros_like(pos);
  int num_nodes = pos.numel() / 2;

  DREAMPLACE_DISPATCH_FLOATING_TYPES(
      pos, "computeElectricForceLauncher", [&] {
        CALL_LAUNCHER(0, num_movable_nodes);
        if (num_filler_nodes) {
          int num_physical_nodes = num_nodes - num_filler_nodes;
          CALL_LAUNCHER(num_physical_nodes, num_nodes);
        }
      });

  return grad_out.mul_(grad_pos);
}

template <typename T>
int computeElectricForceLauncher(
    int num_bins_x, int num_bins_y, int num_impacted_bins_x,
    int num_impacted_bins_y, const T* field_map_x_tensor,
    const T* field_map_y_tensor, const T* x_tensor, const T* y_tensor,
    const T* node_size_x_clamped_tensor, const T* node_size_y_clamped_tensor,
    const T* offset_x_tensor, const T* offset_y_tensor, const T* ratio_tensor,
    const T* bin_center_x_tensor, const T* bin_center_y_tensor, T xl, T yl,
    T xh, T yh, T bin_size_x, T bin_size_y, int num_nodes, int num_threads,
    T* grad_x_tensor, T* grad_y_tensor) {
  // density_map_tensor should be initialized outside

  T inv_bin_size_x = 1.0 / bin_size_x;
  T inv_bin_size_y = 1.0 / bin_size_y;
  int chunk_size =
      DREAMPLACE_STD_NAMESPACE::max(int(num_nodes / num_threads / 16), 1);
#pragma omp parallel for num_threads(num_threads) schedule(dynamic, chunk_size)
  for (int i = 0; i < num_nodes; ++i) {
    // use stretched node size
    T node_size_x = node_size_x_clamped_tensor[i];
    T node_size_y = node_size_y_clamped_tensor[i];
    T node_x = x_tensor[i] + offset_x_tensor[i];
    T node_y = y_tensor[i] + offset_y_tensor[i];
    T ratio = ratio_tensor[i];

    // Yibo: looks very weird implementation, but this is how RePlAce implements
    // it the common practice should be floor Zixuan and Jiaqi: use the common
    // practice of floor
    int bin_index_xl = int((node_x - xl) * inv_bin_size_x);
    int bin_index_xh =
        int(((node_x + node_size_x - xl) * inv_bin_size_x)) + 1;  // exclusive
    bin_index_xl = DREAMPLACE_STD_NAMESPACE::max(bin_index_xl, 0);
    bin_index_xh = DREAMPLACE_STD_NAMESPACE::min(bin_index_xh, num_bins_x);
    // int bin_index_xh = bin_index_xl+num_impacted_bins_x;

    // Yibo: looks very weird implementation, but this is how RePlAce implements
    // it the common practice should be floor Zixuan and Jiaqi: use the common
    // practice of floor
    int bin_index_yl = int((node_y - yl) * inv_bin_size_y);
    int bin_index_yh =
        int(((node_y + node_size_y - yl) * inv_bin_size_y)) + 1;  // exclusive
    bin_index_yl = DREAMPLACE_STD_NAMESPACE::max(bin_index_yl, 0);
    bin_index_yh = DREAMPLACE_STD_NAMESPACE::min(bin_index_yh, num_bins_y);
    // int bin_index_yh = bin_index_yl+num_impacted_bins_y;

    T& gx = grad_x_tensor[i];
    T& gy = grad_y_tensor[i];
    gx = 0;
    gy = 0;
    // update density potential map
    for (int k = bin_index_xl; k < bin_index_xh; ++k) {
      T px = triangle_density_function(node_x, node_size_x, xl, k, bin_size_x);
      for (int h = bin_index_yl; h < bin_index_yh; ++h) {
        T py =
            triangle_density_function(node_y, node_size_y, yl, h, bin_size_y);
        T area = px * py;

        int idx = k * num_bins_y + h;
        gx += area * field_map_x_tensor[idx];
        gy += area * field_map_y_tensor[idx];
      }
    }
    gx *= ratio;
    gy *= ratio;
  }

  return 0;
}

DREAMPLACE_END_NAMESPACE
