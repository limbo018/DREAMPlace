/**
 * @file   density_map.cpp
 * @author Yibo Lin
 * @date   Jun 2018
 * @brief  Compute density map on GPU
 */
#include "utility/src/torch.h"
#include "utility/src/utils.h"

DREAMPLACE_BEGIN_NAMESPACE

/// @brief compute density map
/// @param x_tensor cell x locations
/// @param y_tensor cell y locations
/// @param node_size_x_tensor cell width array
/// @param node_size_y_tensor cell height array
/// @param num_nodes number of cells
/// @param num_bins_x number of bins in horizontal bins
/// @param num_bins_y number of bins in vertical bins
/// @param xl left boundary
/// @param yl bottom boundary
/// @param xh right boundary
/// @param yh top boundary
/// @param deterministic_flag whether ensure run-to-run determinism
/// @param density_map_tensor 2D density map in column-major to write
template <typename T>
int computeDensityMapCudaLauncher(
    const T* x_tensor, const T* y_tensor, 
    const T* node_size_x_tensor, const T* node_size_y_tensor, 
    const int num_nodes, 
    const int num_bins_x, const int num_bins_y, 
    const T xl, const T yl, const T xh, const T yh,
    bool deterministic_flag, 
    T* density_map_tensor);

/// @brief Compute density map.
/// @param pos cell locations, array of x locations and then y locations
/// @param node_size_x_tensor cell width array
/// @param node_size_y_tensor cell height array
/// @param initial_density_map initial density map
/// @param xl left boundary
/// @param yl bottom boundary
/// @param xh right boundary
/// @param yh top boundary
/// @param range_begin begin index of the range [range_begin, range_end)
/// @param range_end end index of the range [range_begin, range_end)
/// @return density map
at::Tensor density_map_forward(at::Tensor pos, at::Tensor node_size_x,
                               at::Tensor node_size_y, at::Tensor initial_density_map, 
                               double xl, double yl, double xh, double yh,
                               int num_bins_x, int num_bins_y, 
                               int range_begin, int range_end, 
                               int deterministic_flag) {
  CHECK_FLAT_CUDA(pos);
  CHECK_EVEN(pos);
  CHECK_CONTIGUOUS(pos);

  at::Tensor density_map = initial_density_map.clone();

  DREAMPLACE_DISPATCH_FLOATING_TYPES(
      pos, "computeDensityMapCudaLauncher", [&] {
        computeDensityMapCudaLauncher<scalar_t>(
            DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + range_begin,
            DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + pos.numel() / 2 + range_begin,
            DREAMPLACE_TENSOR_DATA_PTR(node_size_x, scalar_t) + range_begin,
            DREAMPLACE_TENSOR_DATA_PTR(node_size_y, scalar_t) + range_begin,
            range_end - range_begin, 
            num_bins_x, num_bins_y, 
            xl, yl, xh, yh, 
            (bool)deterministic_flag, 
            DREAMPLACE_TENSOR_DATA_PTR(density_map, scalar_t));
      });

  return density_map;
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &DREAMPLACE_NAMESPACE::density_map_forward,
        "DensityMap forward (CUDA)");
}
