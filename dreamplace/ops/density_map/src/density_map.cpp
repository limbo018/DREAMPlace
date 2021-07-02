/**
 * @file   density_map.cpp
 * @author Yibo Lin
 * @date   Jun 2018
 * @brief  Compute density map on CPU
 */
#include "utility/src/torch.h"
#include "utility/src/utils.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
void distributeBox2Bin(const T* bin_center_x_tensor,
                       const T* bin_center_y_tensor, const int num_bins_x,
                       const int num_bins_y, const T xl, const T yl, const T xh,
                       const T yh, const T bin_size_x, const T bin_size_y,
                       T bxl, T byl, T bxh, T byh, T* buf_map) {
  // density overflow function
  auto computeDensityFunc = [](T x, T node_size, T bin_center, T bin_size) {
    return DREAMPLACE_STD_NAMESPACE::max(
        T(0.0),
        DREAMPLACE_STD_NAMESPACE::min(x + node_size,
                                      bin_center + bin_size / 2) -
            DREAMPLACE_STD_NAMESPACE::max(x, bin_center - bin_size / 2));
  };
  // x direction
  int bin_index_xl = int((bxl - xl) / bin_size_x);
  int bin_index_xh = int(ceil((bxh - xl) / bin_size_x)) + 1;  // exclusive
  bin_index_xl = DREAMPLACE_STD_NAMESPACE::max(bin_index_xl, 0);
  bin_index_xh = DREAMPLACE_STD_NAMESPACE::min(bin_index_xh, num_bins_x);

  // y direction
  int bin_index_yl = int((byl - yl - 2 * bin_size_y) / bin_size_y);
  int bin_index_yh =
      int(ceil((byh - yl + 2 * bin_size_y) / bin_size_y)) + 1;  // exclusive
  bin_index_yl = DREAMPLACE_STD_NAMESPACE::max(bin_index_yl, 0);
  bin_index_yh = DREAMPLACE_STD_NAMESPACE::min(bin_index_yh, num_bins_y);

  for (int k = bin_index_xl; k < bin_index_xh; ++k) {
    T px =
        computeDensityFunc(bxl, bxh - bxl, bin_center_x_tensor[k], bin_size_x);
    for (int h = bin_index_yl; h < bin_index_yh; ++h) {
      T py = computeDensityFunc(byl, byh - byl, bin_center_y_tensor[h],
                                bin_size_y);

      // still area
      T& density = buf_map[k * num_bins_y + h];
#pragma omp atomic
      density += px * py;
    }
  }
};

/// @brief compute density map
/// @param x_tensor cell x locations
/// @param y_tensor cell y locations
/// @param node_size_x_tensor cell width array
/// @param node_size_y_tensor cell height array
/// @param bin_center_x_tensor bin center x locations
/// @param bin_center_y_tensor bin center y locations
/// @param num_nodes number of cells
/// @param num_bins_x number of bins in horizontal bins
/// @param num_bins_y number of bins in vertical bins
/// @param xl left boundary
/// @param yl bottom boundary
/// @param xh right boundary
/// @param yh top boundary
/// @param bin_size_x bin width
/// @param bin_size_y bin height
/// @param num_threads number of threads
/// @param density_map_tensor 2D density map in column-major to write
template <typename T>
int computeDensityMapLauncher(const T* x_tensor, const T* y_tensor,
                              const T* node_size_x_tensor,
                              const T* node_size_y_tensor,
                              const T* bin_center_x_tensor,
                              const T* bin_center_y_tensor, const int num_nodes,
                              const int num_bins_x, const int num_bins_y,
                              const T xl, const T yl, const T xh, const T yh,
                              const T bin_size_x, const T bin_size_y,
                              int num_threads, T* density_map_tensor) {
// density_map_tensor should be initialized outside

#pragma omp parallel for num_threads(num_threads)
  for (int i = 0; i < num_nodes; ++i) {
    T bxl = x_tensor[i];
    T byl = y_tensor[i];
    T bxh = bxl + node_size_x_tensor[i];
    T byh = byl + node_size_y_tensor[i];
    distributeBox2Bin(bin_center_x_tensor, bin_center_y_tensor, num_bins_x,
                      num_bins_y, xl, yl, xh, yh, bin_size_x, bin_size_y, bxl,
                      byl, bxh, byh, density_map_tensor);
  }

  return 0;
}

/// @brief Compute density map.
/// @param pos cell locations, array of x locations and then y locations
/// @param node_size_x_tensor cell width array
/// @param node_size_y_tensor cell height array
/// @param bin_center_x_tensor bin center x locations
/// @param bin_center_y_tensor bin center y locations
/// @param initial_density_map initial density map
/// @param xl left boundary
/// @param yl bottom boundary
/// @param xh right boundary
/// @param yh top boundary
/// @param bin_size_x bin width
/// @param bin_size_y bin height
/// @param num_movable_nodes number of movable cells
/// @param num_filler_nodes number of filler cells
/// @return density map
at::Tensor density_map_forward(at::Tensor pos, at::Tensor node_size_x,
                               at::Tensor node_size_y, at::Tensor bin_center_x,
                               at::Tensor bin_center_y,
                               at::Tensor initial_density_map, double xl,
                               double yl, double xh, double yh,
                               double bin_size_x, double bin_size_y,
                               int num_movable_nodes, int num_filler_nodes) {
  CHECK_FLAT_CPU(pos);
  CHECK_EVEN(pos);
  CHECK_CONTIGUOUS(pos);

  int num_bins_x = int(ceil((xh - xl) / bin_size_x));
  int num_bins_y = int(ceil((yh - yl) / bin_size_y));
  at::Tensor density_map = initial_density_map.clone();

  DREAMPLACE_DISPATCH_FLOATING_TYPES(
      pos, "computeDensityMapLauncher", [&] {
        computeDensityMapLauncher<scalar_t>(
            DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + pos.numel() / 2,
            DREAMPLACE_TENSOR_DATA_PTR(node_size_x, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(node_size_y, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(bin_center_x, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(bin_center_y, scalar_t),
            num_movable_nodes,  // only compute that for movable cells
            num_bins_x, num_bins_y, xl, yl, xh, yh, bin_size_x, bin_size_y,
            at::get_num_threads(),
            DREAMPLACE_TENSOR_DATA_PTR(density_map, scalar_t));
      });
  if (num_filler_nodes) {
    DREAMPLACE_DISPATCH_FLOATING_TYPES(
        pos, "computeDensityMapLauncher", [&] {
          computeDensityMapLauncher<scalar_t>(
              DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + pos.numel() / 2 -
                  num_filler_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + pos.numel() -
                  num_filler_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(node_size_x, scalar_t) +
                  pos.numel() / 2 - num_filler_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(node_size_y, scalar_t) +
                  pos.numel() / 2 - num_filler_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(bin_center_x, scalar_t),
              DREAMPLACE_TENSOR_DATA_PTR(bin_center_y, scalar_t),
              num_filler_nodes,  // only compute that for movable cells
              num_bins_x, num_bins_y, xl, yl, xh, yh, bin_size_x, bin_size_y,
              at::get_num_threads(),
              DREAMPLACE_TENSOR_DATA_PTR(density_map, scalar_t));
        });
  }

  return density_map;
}

/// @brief Compute the density overflow for fixed cells.
/// This map can be used as the initial density map since it only needs to be
/// computed once.
/// @param bin_center_x_tensor bin center x locations
/// @param bin_center_y_tensor bin center y locations
/// @param xl left boundary
/// @param yl bottom boundary
/// @param xh right boundary
/// @param yh top boundary
/// @param bin_size_x bin width
/// @param bin_size_y bin height
/// @param num_movable_nodes number of movable cells
/// @param num_terminals number of fixed cells
/// @return a density map for fixed cells
at::Tensor fixed_density_map(at::Tensor pos, at::Tensor node_size_x,
                             at::Tensor node_size_y, at::Tensor bin_center_x,
                             at::Tensor bin_center_y, double xl, double yl,
                             double xh, double yh, double bin_size_x,
                             double bin_size_y, int num_movable_nodes,
                             int num_terminals) {
  int num_bins_x = int(ceil((xh - xl) / bin_size_x));
  int num_bins_y = int(ceil((yh - yl) / bin_size_y));
  int num_nodes = pos.numel() / 2;
  at::Tensor density_map = at::zeros({num_bins_x, num_bins_y}, pos.options());

  if (num_terminals) {
    // Call the cuda kernel launcher
    DREAMPLACE_DISPATCH_FLOATING_TYPES(
        pos, "computeDensityMapLauncher", [&] {
          computeDensityMapLauncher<scalar_t>(
              DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + num_movable_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + num_nodes +
                  num_movable_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(node_size_x, scalar_t) +
                  num_movable_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(node_size_y, scalar_t) +
                  num_movable_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(bin_center_x, scalar_t),
              DREAMPLACE_TENSOR_DATA_PTR(bin_center_y, scalar_t), num_terminals,
              num_bins_x, num_bins_y, xl, yl, xh, yh, bin_size_x, bin_size_y,
              at::get_num_threads(),
              DREAMPLACE_TENSOR_DATA_PTR(density_map, scalar_t));
        });
  }

  return density_map;
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &DREAMPLACE_NAMESPACE::density_map_forward,
        "DensityMap forward");
  m.def("fixed_density_map", &DREAMPLACE_NAMESPACE::fixed_density_map,
        "DensityMap Map for Fixed Cells");
}
