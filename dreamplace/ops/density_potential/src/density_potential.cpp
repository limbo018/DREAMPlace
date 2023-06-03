/**
 * @file   density_potential.cpp
 * @author Yibo Lin
 * @date   Jun 2018
 * @brief  Compute density potential on CPU according to NTUPlace3
 * (https://doi.org/10.1109/TCAD.2008.923063)
 */
#include "utility/src/torch.h"
#include "utility/src/utils.h"

DREAMPLACE_BEGIN_NAMESPACE

/// @brief compute density map, density cost, and gradient
/// @param x_tensor cell x locations
/// @param y_tensor cell y locations
/// @param node_size_x_tensor cell width array
/// @param node_size_y_tensor cell height array
/// @param ax_tensor ax tensor according to NTUPlace3 paper, for x direction
/// @param bx_tensor bx tensor according to NTUPlace3 paper, for x direction
/// @param cx_tensor cx tensor according to NTUPlace3 paper, for x direction
/// @param ay_tensor ay tensor according to NTUPlace3 paper, for y direction
/// @param by_tensor by tensor according to NTUPlace3 paper, for y direction
/// @param cy_tensor cy tensor according to NTUPlace3 paper, for y direction
/// @param bin_center_x_tensor bin center x locations
/// @param bin_center_y_tensor bin center y locations
/// @param num_impacted_bins_x number of impacted bins for any cell in x
/// direction
/// @param num_impacted_bins_y number of impacted bins for any cell in y
/// direction
/// @param num_nodes number of cells
/// @param num_bins_x number of bins in horizontal bins
/// @param num_bins_y number of bins in vertical bins
/// @param xl left boundary
/// @param yl bottom boundary
/// @param xh right boundary
/// @param yh top boundary
/// @param bin_size_x bin width
/// @param bin_size_y bin height
/// @param target_area target area computed from target density
/// @param density_map_tensor 2D density map in column-major to write
/// @param grad_tensor input gradient from backward propagation
/// @param grad_x_tensor density gradient of cell in x direction
/// @param grad_y_tensor density gradient of cell in y direction
template <typename T>
int computeDensityPotentialMapLauncher(
    const T* x_tensor, const T* y_tensor, const T* node_size_x_tensor,
    const T* node_size_y_tensor, const T* ax_tensor, const T* bx_tensor,
    const T* cx_tensor, const T* ay_tensor, const T* by_tensor,
    const T* cy_tensor, const T* bin_center_x_tensor,
    const T* bin_center_y_tensor, int num_impacted_bins_x,
    int num_impacted_bins_y, const int num_nodes, const int num_bins_x,
    const int num_bins_y, const int padding, const T xl, const T yl, const T xh,
    const T yh, const T bin_size_x, const T bin_size_y, const T target_area,
    T* density_map_tensor, const T* grad_tensor, int num_threads,
    T* grad_x_tensor, T* grad_y_tensor);

/// @brief compute density map, density cost, and gradient
/// @param pos cell locations. The array consists of all x locations and then y
/// locations.
/// @param node_size_x cell width array
/// @param node_size_y cell height array
/// @param ax ax tensor according to NTUPlace3 paper, for x direction
/// @param bx bx tensor according to NTUPlace3 paper, for x direction
/// @param cx cx tensor according to NTUPlace3 paper, for x direction
/// @param ay ay tensor according to NTUPlace3 paper, for y direction
/// @param by by tensor according to NTUPlace3 paper, for y direction
/// @param cy cy tensor according to NTUPlace3 paper, for y direction
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
/// @param num_impacted_bins_x number of impacted bins for any cell in x
/// direction
/// @param num_impacted_bins_y number of impacted bins for any cell in y
/// direction
std::vector<at::Tensor> density_potential_forward(
    at::Tensor pos, at::Tensor node_size_x, at::Tensor node_size_y,
    at::Tensor ax, at::Tensor bx, at::Tensor cx, at::Tensor ay, at::Tensor by,
    at::Tensor cy, at::Tensor bin_center_x, at::Tensor bin_center_y,
    at::Tensor initial_density_map,  // initial density map from fixed cells
    double target_density, double xl, double yl, double xh, double yh,
    double bin_size_x, double bin_size_y, int num_movable_nodes,
    int num_filler_nodes, int padding, int num_bins_x, int num_bins_y,
    int num_impacted_bins_x, int num_impacted_bins_y) {
  CHECK_FLAT_CPU(pos);
  CHECK_EVEN(pos);
  CHECK_CONTIGUOUS(pos);

  at::Tensor density_map = initial_density_map.clone();
  double target_area = target_density * bin_size_x * bin_size_y;

  // int num_nodes = pos.numel()/2;

  // Call the cuda kernel launcher
  DREAMPLACE_DISPATCH_FLOATING_TYPES(
      pos, "computeDensityPotentialMapLauncher", [&] {
        computeDensityPotentialMapLauncher<scalar_t>(
            DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + pos.numel() / 2,
            DREAMPLACE_TENSOR_DATA_PTR(node_size_x, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(node_size_y, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(ax, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(bx, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(cx, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(ay, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(by, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(cy, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(bin_center_x, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(bin_center_y, scalar_t),
            num_impacted_bins_x, num_impacted_bins_y, num_movable_nodes,
            num_bins_x, num_bins_y, padding, xl, yl, xh, yh, bin_size_x,
            bin_size_y, target_area,
            DREAMPLACE_TENSOR_DATA_PTR(density_map, scalar_t), nullptr,
            at::get_num_threads(), nullptr, nullptr);
      });
  if (num_filler_nodes) {
    DREAMPLACE_DISPATCH_FLOATING_TYPES(
        pos, "computeDensityPotentialMapLauncher", [&] {
          computeDensityPotentialMapLauncher<scalar_t>(
              DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + pos.numel() / 2 -
                  num_filler_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + pos.numel() -
                  num_filler_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(node_size_x, scalar_t) +
                  pos.numel() / 2 - num_filler_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(node_size_y, scalar_t) +
                  pos.numel() / 2 - num_filler_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(ax, scalar_t) + pos.numel() / 2 -
                  num_filler_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(bx, scalar_t) + pos.numel() / 2 -
                  num_filler_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(cx, scalar_t) + pos.numel() / 2 -
                  num_filler_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(ay, scalar_t) + pos.numel() / 2 -
                  num_filler_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(by, scalar_t) + pos.numel() / 2 -
                  num_filler_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(cy, scalar_t) + pos.numel() / 2 -
                  num_filler_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(bin_center_x, scalar_t),
              DREAMPLACE_TENSOR_DATA_PTR(bin_center_y, scalar_t),
              num_impacted_bins_x, num_impacted_bins_y, num_filler_nodes,
              num_bins_x, num_bins_y, padding, xl, yl, xh, yh, bin_size_x,
              bin_size_y, target_area,
              DREAMPLACE_TENSOR_DATA_PTR(density_map, scalar_t), nullptr,
              at::get_num_threads(), nullptr, nullptr);
        });
  }

  auto max_density = density_map.max();
  // (max(0, density-target_area))^2
  // auto delta = (density_map-target_area).clamp_min(0).pow(2);
  auto delta = (density_map - target_area).pow(2);
  auto density_cost = at::sum(delta);

  return {density_cost, density_map, max_density};
}

/// @brief Compute density potential gradient
/// @param grad_pos input gradient from backward propagation
/// @param num_bins_x number of bins in horizontal bins
/// @param num_bins_y number of bins in vertical bins
/// @param num_impacted_bins_x number of impacted bins for any cell in x
/// direction
/// @param num_impacted_bins_y number of impacted bins for any cell in y
/// direction
/// @param density_map current density map
/// @param pos cell locations. The array consists of all x locations and then y
/// locations.
/// @param node_size_x cell width array
/// @param node_size_y cell height array
/// @param ax ax tensor according to NTUPlace3 paper, for x direction
/// @param bx bx tensor according to NTUPlace3 paper, for x direction
/// @param cx cx tensor according to NTUPlace3 paper, for x direction
/// @param ay ay tensor according to NTUPlace3 paper, for y direction
/// @param by by tensor according to NTUPlace3 paper, for y direction
/// @param cy cy tensor according to NTUPlace3 paper, for y direction
/// @param bin_center_x bin center x locations
/// @param bin_center_y bin center y locations
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
at::Tensor density_potential_backward(
    at::Tensor grad_pos, int num_bins_x, int num_bins_y,
    int num_impacted_bins_x, int num_impacted_bins_y, at::Tensor density_map,
    at::Tensor pos, at::Tensor node_size_x, at::Tensor node_size_y,
    at::Tensor ax, at::Tensor bx, at::Tensor cx, at::Tensor ay, at::Tensor by,
    at::Tensor cy, at::Tensor bin_center_x, at::Tensor bin_center_y,
    double target_density, double xl, double yl, double xh, double yh,
    double bin_size_x, double bin_size_y, int num_movable_nodes,
    int num_filler_nodes, int padding) {
  CHECK_FLAT_CPU(pos);
  CHECK_EVEN(pos);
  CHECK_CONTIGUOUS(pos);

  AT_ASSERTM(!density_map.is_cuda() && density_map.ndimension() == 2 &&
                 density_map.size(0) == num_bins_x &&
                 density_map.size(1) == num_bins_y,
             "density_map must be a 2D tensor on CPU");
  double target_area = target_density * bin_size_x * bin_size_y;
  at::Tensor grad_out = at::zeros_like(pos);

  // int num_nodes = pos.numel()/2;

  // Call the cuda kernel launcher
  DREAMPLACE_DISPATCH_FLOATING_TYPES(
      pos, "computeDensityPotentialMapLauncher", [&] {
        computeDensityPotentialMapLauncher<scalar_t>(
            DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + pos.numel() / 2,
            DREAMPLACE_TENSOR_DATA_PTR(node_size_x, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(node_size_y, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(ax, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(bx, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(cx, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(ay, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(by, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(cy, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(bin_center_x, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(bin_center_y, scalar_t),
            num_impacted_bins_x, num_impacted_bins_y, num_movable_nodes,
            num_bins_x, num_bins_y, padding, xl, yl, xh, yh, bin_size_x,
            bin_size_y, target_area,
            DREAMPLACE_TENSOR_DATA_PTR(density_map, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(grad_pos, scalar_t),
            at::get_num_threads(),
            DREAMPLACE_TENSOR_DATA_PTR(grad_out, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(grad_out, scalar_t) + pos.numel() / 2);
      });
  if (num_filler_nodes) {
    DREAMPLACE_DISPATCH_FLOATING_TYPES(
        pos, "computeDensityPotentialMapLauncher", [&] {
          computeDensityPotentialMapLauncher<scalar_t>(
              DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + pos.numel() / 2 -
                  num_filler_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + pos.numel() -
                  num_filler_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(node_size_x, scalar_t) +
                  pos.numel() / 2 - num_filler_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(node_size_y, scalar_t) +
                  pos.numel() / 2 - num_filler_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(ax, scalar_t) + pos.numel() / 2 -
                  num_filler_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(bx, scalar_t) + pos.numel() / 2 -
                  num_filler_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(cx, scalar_t) + pos.numel() / 2 -
                  num_filler_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(ay, scalar_t) + pos.numel() / 2 -
                  num_filler_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(by, scalar_t) + pos.numel() / 2 -
                  num_filler_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(cy, scalar_t) + pos.numel() / 2 -
                  num_filler_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(bin_center_x, scalar_t),
              DREAMPLACE_TENSOR_DATA_PTR(bin_center_y, scalar_t),
              num_impacted_bins_x, num_impacted_bins_y, num_filler_nodes,
              num_bins_x, num_bins_y, padding, xl, yl, xh, yh, bin_size_x,
              bin_size_y, target_area,
              DREAMPLACE_TENSOR_DATA_PTR(density_map, scalar_t),
              DREAMPLACE_TENSOR_DATA_PTR(grad_pos, scalar_t),
              at::get_num_threads(),
              DREAMPLACE_TENSOR_DATA_PTR(grad_out, scalar_t) + pos.numel() / 2 -
                  num_filler_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(grad_out, scalar_t) + pos.numel() -
                  num_filler_nodes);
        });
  }

  return grad_out;
}

template <typename T>
int computeDensityOverflowMapLauncher(
    const T* x_tensor, const T* y_tensor, const T* node_size_x_tensor,
    const T* node_size_y_tensor, const T* bin_center_x_tensor,
    const T* bin_center_y_tensor, const int num_nodes, const int num_bins_x,
    const int num_bins_y, const T xl, const T yl, const T xh, const T yh,
    const T bin_size_x, const T bin_size_y, const T target_density,
    const int num_threads, T* density_map_tensor);

template <typename T>
int computeGaussianFilterLauncher(const int num_bins_x, const int num_bins_y,
                                  const T sigma, const int num_threads,
                                  T* gaussian_filter_tensor);

/// @brief compute density map for fixed cells
at::Tensor fixed_density_potential_map(
    at::Tensor pos, at::Tensor node_size_x, at::Tensor node_size_y,
    at::Tensor ax, at::Tensor bx, at::Tensor cx, at::Tensor ay, at::Tensor by,
    at::Tensor cy, at::Tensor bin_center_x, at::Tensor bin_center_y, double xl,
    double yl, double xh, double yh, double bin_size_x, double bin_size_y,
    int num_movable_nodes, int num_terminals, int num_bins_x, int num_bins_y,
    int num_impacted_bins_x, int num_impacted_bins_y, double sigma,
    double delta) {
  CHECK_FLAT_CPU(pos);
  CHECK_EVEN(pos);
  CHECK_CONTIGUOUS(pos);

  at::Tensor density_map = at::zeros({num_bins_x, num_bins_y}, pos.options());

  int num_nodes = pos.numel() / 2;

  // Call the cuda kernel launcher
  if (num_terminals && num_impacted_bins_x && num_impacted_bins_y) {
    DREAMPLACE_DISPATCH_FLOATING_TYPES(
        pos, "computeDensityOverflowMapLauncher", [&] {
          computeDensityOverflowMapLauncher<scalar_t>(
              DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + num_movable_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + num_nodes +
                  num_movable_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(node_size_x, scalar_t) +
                  num_movable_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(node_size_y, scalar_t) +
                  num_movable_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(bin_center_x, scalar_t),
              DREAMPLACE_TENSOR_DATA_PTR(bin_center_y, scalar_t), num_terminals,
              num_bins_x, num_bins_y, xl, yl, xh, yh, bin_size_x, bin_size_y, 0,
              at::get_num_threads(),
              DREAMPLACE_TENSOR_DATA_PTR(density_map, scalar_t));
        });
  }

  return density_map;
}

template <typename T>
int computeDensityPotentialMapLauncher(
    const T* x_tensor, const T* y_tensor, const T* node_size_x_tensor,
    const T* node_size_y_tensor, const T* ax_tensor, const T* bx_tensor,
    const T* cx_tensor, const T* ay_tensor, const T* by_tensor,
    const T* cy_tensor, const T* bin_center_x_tensor,
    const T* bin_center_y_tensor, int num_impacted_bins_x,
    int num_impacted_bins_y, const int num_nodes, const int num_bins_x,
    const int num_bins_y, const int padding, const T xl, const T yl, const T xh,
    const T yh, const T bin_size_x, const T bin_size_y, const T target_area,
    T* density_map_tensor, const T* grad_tensor, int num_threads,
    T* grad_x_tensor, T* grad_y_tensor) {
  // density_map_tensor should be initialized outside

  // density potential function
  auto computeDensityPotentialFunc = [](T x, T node_size, T bin_center,
                                        T bin_size, T a, T b, T c) {
    // from origin to center
    x += node_size / 2;
    // printf("x = %g, bin_center = %g\n", x, bin_center);
    T dist = fabs(x - bin_center);
    // printf("dist = %g\n", dist);
    T partition1 = node_size / 2 + bin_size;
    // printf("partition1 = %g\n", partition1);
    T partition2 = partition1 + bin_size;
    // printf("partition2 = %g\n", partition2);
    // printf("a = %g, b = %g, c = %g\n", a, b, c);
    if (dist < partition1) {
      return c * (1 - a * dist * dist);
    } else if (dist < partition2) {
      return c * (b * (dist - partition2) * (dist - partition2));
    } else {
      return T(0.0);
    }
  };
  // density potential gradient function
  auto computeDensityPotentialGradFunc = [](T x, T node_size, T bin_center,
                                            T bin_size, T a, T b, T c) {
    // from origin to center
    x += node_size / 2;
    T dist = fabs(x - bin_center);
    T partition1 = node_size / 2 + bin_size;
    T partition2 = partition1 + bin_size;
    if (dist < partition1) {
      return -2 * c * a * (x - bin_center);
    } else if (dist < partition2) {
      T sign = (x < bin_center) ? -1.0 : 1.0;
      return 2 * c * b * (dist - partition2) * sign;
    } else {
      return T(0.0);
    }
  };

#pragma omp parallel for num_threads(num_threads)
  for (int i = 0; i < num_nodes; ++i) {
    // x direction
    int bin_index_xl = int((x_tensor[i] - xl - 2 * bin_size_x) / bin_size_x);
    int bin_index_xh =
        int(ceil((x_tensor[i] - xl + node_size_x_tensor[i] + 2 * bin_size_x) /
                 bin_size_x)) +
        1;  // exclusive
    bin_index_xl = std::max(bin_index_xl, 0);
    bin_index_xh = std::min(bin_index_xh, num_bins_x);

    // y direction
    int bin_index_yl = int((y_tensor[i] - yl - 2 * bin_size_y) / bin_size_y);
    int bin_index_yh =
        int(ceil((y_tensor[i] - yl + node_size_y_tensor[i] + 2 * bin_size_y) /
                 bin_size_y)) +
        1;  // exclusive
    bin_index_yl = std::max(bin_index_yl, 0);
    bin_index_yh = std::min(bin_index_yh, num_bins_y);

    for (int k = bin_index_xl; k < bin_index_xh; ++k) {
      T px = computeDensityPotentialFunc(
          x_tensor[i], node_size_x_tensor[i], bin_center_x_tensor[k],
          bin_size_x, ax_tensor[i], bx_tensor[i], cx_tensor[i]);
      // printf("px[%d, %d] = %g\n", i, k, px);
      for (int h = bin_index_yl; h < bin_index_yh; ++h) {
        T py = computeDensityPotentialFunc(
            y_tensor[i], node_size_y_tensor[i], bin_center_y_tensor[h],
            bin_size_y, ay_tensor[i], by_tensor[i], cy_tensor[i]);
        // printf("py[%d, %d] = %g\n", i, h, py);
        // printf("px[%d, %d] = %g, py[%d, %d] = %g\n", k, h, px, k, h, py);

        // still area
        T& density = density_map_tensor[k * num_bins_y + h];
#pragma omp atomic
        density += px * py;
      }
    }
  }

  if (grad_tensor)  // compute density gradient
  {
#pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < num_nodes; ++i) {
      int bin_index_xl = int((x_tensor[i] - xl - 2 * bin_size_x) / bin_size_x);
      // int bin_index_xh =
      // int(ceil((x_tensor[i]-xl+node_size_x_tensor[i]+2*bin_size_x)/bin_size_x))+1;
      // // exclusive
      bin_index_xl = std::max(bin_index_xl, 0);
      // be careful about the bin_index_xl and bin_index_xh here
      // the assumption is that num_bins_x >= num_impacted_bins_x
      // each row of the px matrix should be filled with num_impacted_bins_x
      // columns
      bin_index_xl = std::min(bin_index_xl, num_bins_x - num_impacted_bins_x);
      // bin_index_xh = std::min(bin_index_xh, num_bins_x);
      int bin_index_xh = bin_index_xl + num_impacted_bins_x;

      int bin_index_yl = int((y_tensor[i] - yl - 2 * bin_size_y) / bin_size_y);
      // int bin_index_yh =
      // int(ceil((y_tensor[i]-yl+node_size_y_tensor[i]+2*bin_size_y)/bin_size_y))+1;
      // // exclusive
      bin_index_yl = std::max(bin_index_yl, 0);
      // be careful about the bin_index_yl and bin_index_yh here
      // the assumption is that num_bins_y >= num_impacted_bins_y
      // each row of the py matrix should be filled with num_impacted_bins_y
      // columns
      bin_index_yl = std::min(bin_index_yl, num_bins_y - num_impacted_bins_y);
      // bin_index_yh = std::min(bin_index_yh, num_bins_y);
      int bin_index_yh = bin_index_yl + num_impacted_bins_y;

      grad_x_tensor[i] = 0;
      grad_y_tensor[i] = 0;
      // update density potential map
      for (int k = bin_index_xl; k < bin_index_xh; ++k) {
        T px = computeDensityPotentialFunc(
            x_tensor[i], node_size_x_tensor[i], bin_center_x_tensor[k],
            bin_size_x, ax_tensor[i], bx_tensor[i], cx_tensor[i]);
        T gradx = computeDensityPotentialGradFunc(
            x_tensor[i], node_size_x_tensor[i], bin_center_x_tensor[k],
            bin_size_x, ax_tensor[i], bx_tensor[i], cx_tensor[i]);
        for (int h = bin_index_yl; h < bin_index_yh; ++h) {
          T py = computeDensityPotentialFunc(
              y_tensor[i], node_size_y_tensor[i], bin_center_y_tensor[h],
              bin_size_y, ay_tensor[i], by_tensor[i], cy_tensor[i]);
          T grady = computeDensityPotentialGradFunc(
              y_tensor[i], node_size_y_tensor[i], bin_center_y_tensor[h],
              bin_size_y, ay_tensor[i], by_tensor[i], cy_tensor[i]);

          T delta = density_map_tensor[k * num_bins_y + h] - target_area;
          // delta = std::max(delta, (T)0);

          grad_x_tensor[i] += 2 * delta * py * gradx;
          grad_y_tensor[i] += 2 * delta * px * grady;
        }
      }

      grad_x_tensor[i] *= *grad_tensor;
      grad_y_tensor[i] *= *grad_tensor;
    }
  } else  // compute density cost
  {
  // handle padding
#pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < num_bins_x; ++i) {
      for (int j = 0; j < num_bins_y; ++j) {
        if (!(i >= padding && i + padding < num_bins_x && j >= padding &&
              j + padding < num_bins_y)) {
          density_map_tensor[i * num_bins_y + j] = target_area;
        }
      }
    }
  }

  return 0;
}

template <typename T>
int computeDensityOverflowMapLauncher(
    const T* x_tensor, const T* y_tensor, const T* node_size_x_tensor,
    const T* node_size_y_tensor, const T* bin_center_x_tensor,
    const T* bin_center_y_tensor, const int num_nodes, const int num_bins_x,
    const int num_bins_y, const T xl, const T yl, const T xh, const T yh,
    const T bin_size_x, const T bin_size_y, const T target_area,
    const int num_threads, T* density_map_tensor) {
  // initialize
  int num_bins = num_bins_x * num_bins_y;
#pragma omp parallel for num_threads(num_threads)
  for (int i = 0; i < num_bins; ++i) {
    density_map_tensor[i] = 0;
  }

  // density overflow function
  auto computeDensityOverflowFunc = [](T x, T node_size, T bin_center,
                                       T bin_size) {
    return std::max(T(0.0), std::min(x + node_size, bin_center + bin_size / 2) -
                                std::max(x, bin_center - bin_size / 2));
  };
#pragma omp parallel for num_threads(num_threads)
  for (int i = 0; i < num_nodes; ++i) {
    // x direction
    int bin_index_xl = int((x_tensor[i] - xl) / bin_size_x);
    int bin_index_xh =
        int(ceil((x_tensor[i] - xl + node_size_x_tensor[i]) / bin_size_x)) +
        1;  // exclusive
    bin_index_xl = std::max(bin_index_xl, 0);
    bin_index_xh = std::min(bin_index_xh, num_bins_x);

    // y direction
    int bin_index_yl = int((y_tensor[i] - yl - 2 * bin_size_y) / bin_size_y);
    int bin_index_yh =
        int(ceil((y_tensor[i] - yl + node_size_y_tensor[i] + 2 * bin_size_y) /
                 bin_size_y)) +
        1;  // exclusive
    bin_index_yl = std::max(bin_index_yl, 0);
    bin_index_yh = std::min(bin_index_yh, num_bins_y);

    for (int k = bin_index_xl; k < bin_index_xh; ++k) {
      T px = computeDensityOverflowFunc(x_tensor[i], node_size_x_tensor[i],
                                        bin_center_x_tensor[k], bin_size_x);
      for (int h = bin_index_yl; h < bin_index_yh; ++h) {
        T py = computeDensityOverflowFunc(y_tensor[i], node_size_y_tensor[i],
                                          bin_center_y_tensor[h], bin_size_y);
        // printf("px[%d, %d] = %g, py[%d, %d] = %g\n", k, h, px, k, h, py);

        // still area
        T& density = density_map_tensor[k * num_bins_y + h];
#pragma omp atomic
        density += px * py;
      }
    }
  }

  return 0;
}

template <typename T>
int computeGaussianFilterLauncher(const int num_bins_x, const int num_bins_y,
                                  const T sigma, const int num_threads,
                                  T* gaussian_filter_tensor) {
  T sigma_square = sigma * sigma;
#pragma omp parallel for num_threads(num_threads)
  for (int i = 0; i < num_bins_x; ++i) {
    for (int j = 0; j < num_bins_y; ++j) {
      T x2_y2 = (i - num_bins_x / 2) * (i - num_bins_x) +
                (j - num_bins_y / 2) * (j - num_bins_y);
      gaussian_filter_tensor[i * num_bins_y + j] =
          1.0 / (2 * M_PI * sigma_square) * exp(-x2_y2 / (2 * sigma_square));
    }
  }

  return 0;
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &DREAMPLACE_NAMESPACE::density_potential_forward,
        "DensityPotential forward");
  m.def("backward", &DREAMPLACE_NAMESPACE::density_potential_backward,
        "DensityPotential backward");
  m.def("fixed_density_map", &DREAMPLACE_NAMESPACE::fixed_density_potential_map,
        "DensityPotential Map for Fixed Cells");
}
