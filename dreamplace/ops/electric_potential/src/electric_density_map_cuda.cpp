/**
 * @file   density_map_cuda.cpp
 * @author Yibo Lin
 * @date   Aug 2018
 * @brief  Compute density map according to e-place
 * (http://cseweb.ucsd.edu/~jlu/papers/eplace-todaes14/paper.pdf)
 */
#include "utility/src/torch.h"
#include "utility/src/utils.h"

DREAMPLACE_BEGIN_NAMESPACE

// The triangular density model from e-place
// The impact of a cell to bins is extended to two neighboring bins
template <typename T>
int computeTriangleDensityMapCudaLauncher(
    const T* x_tensor, const T* y_tensor, const T* node_size_x_clamped_tensor,
    const T* node_size_y_clamped_tensor, const T* offset_x_tensor,
    const T* offset_y_tensor, const T* ratio_tensor,
    const T* bin_center_x_tensor, const T* bin_center_y_tensor, int num_nodes,
    const int num_bins_x, const int num_bins_y, int num_impacted_bins_x,
    int num_impacted_bins_y, const T xl, const T yl, const T xh, const T yh,
    const T bin_size_x, const T bin_size_y, bool deterministic_flag,
    T* density_map_tensor, const int* sorted_node_map);

// The exact density model
// Compute the exact overlap area for density
template <typename T>
int computeExactDensityMapCudaLauncher(
    const T* x_tensor, const T* y_tensor, const T* node_size_x_tensor,
    const T* node_size_y_tensor, const T* bin_center_x_tensor,
    const T* bin_center_y_tensor, const int num_nodes, const int num_bins_x,
    const int num_bins_y, const int num_impacted_bins_x,
    const int num_impacted_bins_y, const T xl, const T yl, const T xh,
    const T yh, const T bin_size_x, const T bin_size_y, bool fixed_node_flag,
    bool deterministic_flag, T* density_map_tensor);

/// @brief compute density map for movable and filler cells
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
/// @param initial_density_map initial density map for fixed cells
/// @param target_density target density
/// @param xl left boundary
/// @param yl bottom boundary
/// @param xh right boundary
/// @param yh top boundary
/// @param bin_size_x bin width
/// @param bin_size_y bin height
/// @param num_movable_nodes number of movable cells
/// @param num_filler_nodes number of filler cells
/// @param padding bin padding to boundary of placement region
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
/// @param sorted_node_map the indices of the movable node map
at::Tensor density_map(
    at::Tensor pos, at::Tensor node_size_x_clamped,
    at::Tensor node_size_y_clamped, at::Tensor offset_x, at::Tensor offset_y,
    at::Tensor ratio, at::Tensor bin_center_x, at::Tensor bin_center_y,
    at::Tensor initial_density_map, double target_density, double xl, double yl,
    double xh, double yh, double bin_size_x, double bin_size_y,
    int num_movable_nodes, int num_filler_nodes, int padding, int num_bins_x,
    int num_bins_y, int num_movable_impacted_bins_x,
    int num_movable_impacted_bins_y, int num_filler_impacted_bins_x,
    int num_filler_impacted_bins_y, int deterministic_flag,
    at::Tensor sorted_node_map) {
  CHECK_FLAT_CUDA(pos);
  CHECK_EVEN(pos);
  CHECK_CONTIGUOUS(pos);

  at::Tensor density_map = initial_density_map.clone();
  int num_nodes = pos.numel() / 2;

  // Call the cuda kernel launcher
  DREAMPLACE_DISPATCH_FLOATING_TYPES(
      pos, "computeTriangleDensityMapCudaLauncher", [&] {
        computeTriangleDensityMapCudaLauncher<scalar_t>(
            DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + num_nodes,
            DREAMPLACE_TENSOR_DATA_PTR(node_size_x_clamped, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(node_size_y_clamped, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(offset_x, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(offset_y, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(ratio, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(bin_center_x, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(bin_center_y, scalar_t),
            num_movable_nodes, num_bins_x, num_bins_y,
            num_movable_impacted_bins_x, num_movable_impacted_bins_y, xl, yl,
            xh, yh, bin_size_x, bin_size_y, (bool)deterministic_flag,
            DREAMPLACE_TENSOR_DATA_PTR(density_map, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(sorted_node_map, int));
      });

  if (num_filler_nodes) {
    int num_physical_nodes = num_nodes - num_filler_nodes;
    DREAMPLACE_DISPATCH_FLOATING_TYPES(
        pos, "computeTriangleDensityMapCudaLauncher", [&] {
          computeTriangleDensityMapCudaLauncher<scalar_t>(
              DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + num_physical_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + num_nodes +
                  num_physical_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(node_size_x_clamped, scalar_t) +
                  num_physical_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(node_size_y_clamped, scalar_t) +
                  num_physical_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(offset_x, scalar_t) +
                  num_physical_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(offset_y, scalar_t) +
                  num_physical_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(ratio, scalar_t) + num_physical_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(bin_center_x, scalar_t),
              DREAMPLACE_TENSOR_DATA_PTR(bin_center_y, scalar_t),
              num_filler_nodes, num_bins_x, num_bins_y,
              num_filler_impacted_bins_x, num_filler_impacted_bins_y, xl, yl,
              xh, yh, bin_size_x, bin_size_y, (bool)deterministic_flag,
              DREAMPLACE_TENSOR_DATA_PTR(density_map, scalar_t), NULL);
        });
  }

  return density_map;
}

/// @brief compute density map for fixed cells
at::Tensor fixed_density_map(at::Tensor pos, at::Tensor node_size_x,
                             at::Tensor node_size_y, at::Tensor bin_center_x,
                             at::Tensor bin_center_y, double xl, double yl,
                             double xh, double yh, double bin_size_x,
                             double bin_size_y, int num_movable_nodes,
                             int num_terminals, int num_bins_x, int num_bins_y,
                             int num_fixed_impacted_bins_x,
                             int num_fixed_impacted_bins_y,
                             int deterministic_flag) {
  CHECK_FLAT_CUDA(pos);
  CHECK_EVEN(pos);
  CHECK_CONTIGUOUS(pos);

  at::Tensor density_map = at::zeros({num_bins_x, num_bins_y}, pos.options());

  int num_nodes = pos.numel() / 2;

  // Call the cuda kernel launcher
  if (num_terminals) {
    DREAMPLACE_DISPATCH_FLOATING_TYPES(
        pos, "computeExactDensityMapCudaLauncher", [&] {
          computeExactDensityMapCudaLauncher<scalar_t>(
              DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + num_movable_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + num_nodes +
                  num_movable_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(node_size_x, scalar_t) +
                  num_movable_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(node_size_y, scalar_t) +
                  num_movable_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(bin_center_x, scalar_t),
              DREAMPLACE_TENSOR_DATA_PTR(bin_center_y, scalar_t), num_terminals,
              num_bins_x, num_bins_y, num_fixed_impacted_bins_x,
              num_fixed_impacted_bins_y, xl, yl, xh, yh, bin_size_x, bin_size_y,
              true, (bool)deterministic_flag,
              DREAMPLACE_TENSOR_DATA_PTR(density_map, scalar_t));
        });

    // Fixed cells may have overlaps. We should not over-compute the density
    // map. This is just an approximate fix. It does not guarantee the exact
    // value in each bin.
    density_map.clamp_max_(bin_size_x * bin_size_y);
  }

  return density_map;
}

/// @brief Compute electric force for movable and filler cells
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
at::Tensor electric_force(
    at::Tensor grad_pos, int num_bins_x, int num_bins_y,
    int num_movable_impacted_bins_x, int num_movable_impacted_bins_y,
    int num_filler_impacted_bins_x, int num_filler_impacted_bins_y,
    at::Tensor field_map_x, at::Tensor field_map_y, at::Tensor pos,
    at::Tensor node_size_x_clamped, at::Tensor node_size_y_clamped,
    at::Tensor offset_x, at::Tensor offset_y, at::Tensor ratio,
    at::Tensor bin_center_x, at::Tensor bin_center_y, double xl, double yl,
    double xh, double yh, double bin_size_x, double bin_size_y,
    int num_movable_nodes, int num_filler_nodes, int deterministic_flag, at::Tensor sorted_node_map);

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("density_map", &DREAMPLACE_NAMESPACE::density_map,
        "ElectricPotential Density Map (CUDA)");
  m.def("fixed_density_map", &DREAMPLACE_NAMESPACE::fixed_density_map,
        "ElectricPotential Density Map for Fixed Cells (CUDA)");
  m.def("electric_force", &DREAMPLACE_NAMESPACE::electric_force,
        "ElectricPotential Electric Force (CUDA)");
}
