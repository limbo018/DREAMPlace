/**
 * @file   electric_force_cuda.cpp
 * @author Yibo Lin
 * @date   Aug 2018
 * @brief  Compute electric force according to e-place
 */
#include "utility/src/torch.h"
#include "utility/src/utils.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
int computeElectricForceCudaLauncher(
    int num_bins_x, int num_bins_y, int num_impacted_bins_x,
    int num_impacted_bins_y, const T* field_map_x_tensor,
    const T* field_map_y_tensor, const T* x_tensor, const T* y_tensor,
    const T* node_size_x_clamped_tensor, const T* node_size_y_clamped_tensor,
    const T* offset_x_tensor, const T* offset_y_tensor, const T* ratio_tensor,
    const T* bin_center_x_tensor, const T* bin_center_y_tensor, T xl, T yl,
    T xh, T yh, T bin_size_x, T bin_size_y, int num_nodes, bool deterministic_flag, 
    T* grad_x_tensor, T* grad_y_tensor, const int* sorted_node_map);

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
/// @param node_size_x_clamped cell width array clamp(min = sqrt2 * bin_size_x)
/// @param node_size_y_clamped cell height array clamp(min = sqrt2 * bin_size_y)
/// @param offset_x (node_size_x - node_size_x_clamped)/2
/// @param offset_y (node_size_y - node_size_y_clamped)/2
/// @param ratio (node_size_x * node_size_y)  / (node_size_x_clamped *
/// node_size_y_clamped)
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
/// @param sorted_node_map the indices of the movable node map
at::Tensor electric_force(
    at::Tensor grad_pos, int num_bins_x, int num_bins_y,
    int num_movable_impacted_bins_x, int num_movable_impacted_bins_y,
    int num_filler_impacted_bins_x, int num_filler_impacted_bins_y,
    at::Tensor field_map_x, at::Tensor field_map_y, at::Tensor pos,
    at::Tensor node_size_x_clamped, at::Tensor node_size_y_clamped,
    at::Tensor offset_x, at::Tensor offset_y, at::Tensor ratio,
    at::Tensor bin_center_x, at::Tensor bin_center_y, double xl, double yl,
    double xh, double yh, double bin_size_x, double bin_size_y,
    int num_movable_nodes, int num_filler_nodes, int deterministic_flag, at::Tensor sorted_node_map) {
  CHECK_FLAT_CUDA(pos);
  CHECK_EVEN(pos);
  CHECK_CONTIGUOUS(pos);
  CHECK_FLAT_CUDA(field_map_x);
  CHECK_CONTIGUOUS(field_map_x);
  CHECK_FLAT_CUDA(field_map_y);
  CHECK_CONTIGUOUS(field_map_y);

  at::Tensor grad_out = at::zeros_like(pos);
  int num_nodes = pos.numel() / 2;

  DREAMPLACE_DISPATCH_FLOATING_TYPES(
      pos, "computeElectricForceCudaLauncher", [&] {
        computeElectricForceCudaLauncher<scalar_t>(
            num_bins_x, num_bins_y, num_movable_impacted_bins_x,
            num_movable_impacted_bins_y,
            DREAMPLACE_TENSOR_DATA_PTR(field_map_x, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(field_map_y, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + num_nodes,
            DREAMPLACE_TENSOR_DATA_PTR(node_size_x_clamped, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(node_size_y_clamped, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(offset_x, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(offset_y, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(ratio, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(bin_center_x, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(bin_center_y, scalar_t), xl, yl, xh, yh,
            bin_size_x, bin_size_y, num_movable_nodes, (bool)deterministic_flag,
            DREAMPLACE_TENSOR_DATA_PTR(grad_out, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(grad_out, scalar_t) + num_nodes,
            DREAMPLACE_TENSOR_DATA_PTR(sorted_node_map, int));
      });

  if (num_filler_nodes) {
    int num_physical_nodes = num_nodes - num_filler_nodes;
    DREAMPLACE_DISPATCH_FLOATING_TYPES(
        pos, "computeElectricForceCudaLauncher", [&] {
          computeElectricForceCudaLauncher<scalar_t>(
              num_bins_x, num_bins_y, num_filler_impacted_bins_x,
              num_filler_impacted_bins_y,
              DREAMPLACE_TENSOR_DATA_PTR(field_map_x, scalar_t),
              DREAMPLACE_TENSOR_DATA_PTR(field_map_y, scalar_t),
              DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + num_physical_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + num_nodes + num_physical_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(node_size_x_clamped, scalar_t) + num_physical_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(node_size_y_clamped, scalar_t) + num_physical_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(offset_x, scalar_t) + num_physical_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(offset_y, scalar_t) + num_physical_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(ratio, scalar_t) + num_physical_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(bin_center_x, scalar_t),
              DREAMPLACE_TENSOR_DATA_PTR(bin_center_y, scalar_t), xl, yl, xh,
              yh, bin_size_x, bin_size_y, num_filler_nodes, (bool)deterministic_flag,
              DREAMPLACE_TENSOR_DATA_PTR(grad_out, scalar_t) + num_physical_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(grad_out, scalar_t) + num_nodes + num_physical_nodes,
              NULL);
        });
  }

  return grad_out.mul_(grad_pos);
}

DREAMPLACE_END_NAMESPACE
