/**
 * @file   hpwl_cuda.cpp
 * @author Yibo Lin
 * @date   Jun 2018
 * @brief  Compute density potential on CUDA according to NTUPlace3
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
/// @param density_cost_tensor overall density overflow
/// @param grad_tensor input gradient from backward propagation
/// @param grad_x_tensor density gradient of cell in x direction
/// @param grad_y_tensor density gradient of cell in y direction
template <typename T>
int computeDensityPotentialMapCudaLauncher(
    const T* x_tensor, const T* y_tensor, const T* node_size_x_tensor,
    const T* node_size_y_tensor, const T* ax_tensor, const T* bx_tensor,
    const T* cx_tensor, const T* ay_tensor, const T* by_tensor,
    const T* cy_tensor, const T* bin_center_x_tensor,
    const T* bin_center_y_tensor, const int num_impacted_bins_x,
    const int num_impacted_bins_y, const int mat_size_x, const int mat_size_y,
    const int num_nodes, const int num_bins_x, const int num_bins_y,
    const int padding, const T xl, const T yl, const T xh, const T yh,
    const T bin_size_x, const T bin_size_y, const T target_area,
    T* density_map_tensor, const T* grad_tensor, T* grad_x_tensor,
    T* grad_y_tensor);

typedef double T;

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
  CHECK_FLAT_CUDA(pos);
  CHECK_EVEN(pos);
  CHECK_CONTIGUOUS(pos);

  at::Tensor density_map = initial_density_map.clone();
  double target_area = target_density * bin_size_x * bin_size_y;

  int num_nodes = pos.numel() / 2;
  int mat_size_x =
      (num_movable_nodes *
       num_impacted_bins_x);  // only need to compute for movable nodes
  int mat_size_y = (num_movable_nodes * num_impacted_bins_y);

  // Call the cuda kernel launcher
  DREAMPLACE_DISPATCH_FLOATING_TYPES(
      pos, "computeDensityPotentialMapCudaLauncher", [&] {
        computeDensityPotentialMapCudaLauncher<scalar_t>(
            DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + num_nodes,
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
            num_impacted_bins_x, num_impacted_bins_y, mat_size_x, mat_size_y,
            num_movable_nodes,  // only need to compute for movable nodes
            num_bins_x, num_bins_y, padding, xl, yl, xh, yh, bin_size_x,
            bin_size_y, target_area,
            DREAMPLACE_TENSOR_DATA_PTR(density_map, scalar_t), nullptr, nullptr,
            nullptr);
      });
  if (num_filler_nodes) {
    DREAMPLACE_DISPATCH_FLOATING_TYPES(
        pos, "computeDensityPotentialMapCudaLauncher", [&] {
          computeDensityPotentialMapCudaLauncher<scalar_t>(
              DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + num_nodes -
                  num_filler_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + num_nodes * 2 -
                  num_filler_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(node_size_x, scalar_t) + num_nodes -
                  num_filler_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(node_size_y, scalar_t) + num_nodes -
                  num_filler_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(ax, scalar_t) + num_nodes -
                  num_filler_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(bx, scalar_t) + num_nodes -
                  num_filler_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(cx, scalar_t) + num_nodes -
                  num_filler_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(ay, scalar_t) + num_nodes -
                  num_filler_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(by, scalar_t) + num_nodes -
                  num_filler_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(cy, scalar_t) + num_nodes -
                  num_filler_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(bin_center_x, scalar_t),
              DREAMPLACE_TENSOR_DATA_PTR(bin_center_y, scalar_t),
              num_impacted_bins_x, num_impacted_bins_y, mat_size_x, mat_size_y,
              num_filler_nodes,  // only need to compute for movable nodes
              num_bins_x, num_bins_y, padding, xl, yl, xh, yh, bin_size_x,
              bin_size_y, target_area,
              DREAMPLACE_TENSOR_DATA_PTR(density_map, scalar_t), nullptr,
              nullptr, nullptr);
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
  CHECK_FLAT_CUDA(pos);
  CHECK_EVEN(pos);
  CHECK_CONTIGUOUS(pos);

  AT_ASSERTM(density_map.is_cuda() && density_map.ndimension() == 2 &&
                 density_map.size(0) == num_bins_x &&
                 density_map.size(1) == num_bins_y,
             "density_map must be a 2D tensor on GPU");
  double target_area = target_density * bin_size_x * bin_size_y;
  at::Tensor grad_out = at::zeros_like(pos);

  int num_nodes = pos.numel() / 2;
  int mat_size_x = (num_movable_nodes * num_impacted_bins_x);
  int mat_size_y = (num_movable_nodes * num_impacted_bins_y);

  // Call the cuda kernel launcher
  DREAMPLACE_DISPATCH_FLOATING_TYPES(
      pos, "computeDensityPotentialMapCudaLauncher", [&] {
        computeDensityPotentialMapCudaLauncher<scalar_t>(
            DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + num_nodes,
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
            num_impacted_bins_x, num_impacted_bins_y, mat_size_x, mat_size_y,
            num_movable_nodes,  // only need to compute for movable nodes
            num_bins_x, num_bins_y, padding, xl, yl, xh, yh, bin_size_x,
            bin_size_y, target_area,
            DREAMPLACE_TENSOR_DATA_PTR(density_map, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(grad_pos, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(grad_out, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(grad_out, scalar_t) + num_nodes);
      });
  if (num_filler_nodes) {
    DREAMPLACE_DISPATCH_FLOATING_TYPES(
        pos, "computeDensityPotentialMapCudaLauncher", [&] {
          computeDensityPotentialMapCudaLauncher<scalar_t>(
              DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + num_nodes -
                  num_filler_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + num_nodes * 2 -
                  num_filler_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(node_size_x, scalar_t) + num_nodes -
                  num_filler_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(node_size_y, scalar_t) + num_nodes -
                  num_filler_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(ax, scalar_t) + num_nodes -
                  num_filler_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(bx, scalar_t) + num_nodes -
                  num_filler_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(cx, scalar_t) + num_nodes -
                  num_filler_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(ay, scalar_t) + num_nodes -
                  num_filler_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(by, scalar_t) + num_nodes -
                  num_filler_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(cy, scalar_t) + num_nodes -
                  num_filler_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(bin_center_x, scalar_t),
              DREAMPLACE_TENSOR_DATA_PTR(bin_center_y, scalar_t),
              num_impacted_bins_x, num_impacted_bins_y, mat_size_x, mat_size_y,
              num_filler_nodes,  // only need to compute for movable nodes
              num_bins_x, num_bins_y, padding, xl, yl, xh, yh, bin_size_x,
              bin_size_y, target_area,
              DREAMPLACE_TENSOR_DATA_PTR(density_map, scalar_t),
              DREAMPLACE_TENSOR_DATA_PTR(grad_pos, scalar_t),
              DREAMPLACE_TENSOR_DATA_PTR(grad_out, scalar_t) + num_nodes -
                  num_filler_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(grad_out, scalar_t) + num_nodes * 2 -
                  num_filler_nodes);
        });
  }

  return grad_out;
}

template <typename T>
int computeDensityOverflowMapCudaLauncher(
    const T* x_tensor, const T* y_tensor, const T* node_size_x_tensor,
    const T* node_size_y_tensor, const T* bin_center_x_tensor,
    const T* bin_center_y_tensor, const int num_nodes, const int num_bins_x,
    const int num_bins_y, const int num_impacted_bins_x,
    const int num_impacted_bins_y, const T xl, const T yl, const T xh,
    const T yh, const T bin_size_x, const T bin_size_y, T* density_map_tensor);

template <typename T>
int computeGaussianFilterLauncher(const int num_bins_x, const int num_bins_y,
                                  const T sigma, T* gaussian_filter_tensor);

/// @brief compute density map for fixed cells
at::Tensor fixed_density_potential_map(
    at::Tensor pos, at::Tensor node_size_x, at::Tensor node_size_y,
    at::Tensor ax, at::Tensor bx, at::Tensor cx, at::Tensor ay, at::Tensor by,
    at::Tensor cy, at::Tensor bin_center_x, at::Tensor bin_center_y, double xl,
    double yl, double xh, double yh, double bin_size_x, double bin_size_y,
    int num_movable_nodes, int num_terminals, int num_bins_x, int num_bins_y,
    int num_impacted_bins_x, int num_impacted_bins_y, double sigma,
    double delta) {
  CHECK_FLAT_CUDA(pos);
  CHECK_EVEN(pos);
  CHECK_CONTIGUOUS(pos);

  at::Tensor density_map = at::zeros({num_bins_x, num_bins_y}, pos.options());

  int num_nodes = pos.numel() / 2;

  // Call the cuda kernel launcher
  if (num_terminals && num_impacted_bins_x && num_impacted_bins_y) {
    DREAMPLACE_DISPATCH_FLOATING_TYPES(
        pos, "computeDensityOverflowMapCudaLauncher", [&] {
          // int mat_size_x =
          // ((num_nodes-num_movable_nodes)*num_impacted_bins_x);  int
          // mat_size_y = ((num_nodes-num_movable_nodes)*num_impacted_bins_y);
          // computeDensityPotentialMapCudaLauncher<scalar_t>(
          //        DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t)+num_movable_nodes,
          //        DREAMPLACE_TENSOR_DATA_PTR(pos,
          //        scalar_t)+num_nodes+num_movable_nodes,
          //        DREAMPLACE_TENSOR_DATA_PTR(node_size_x,
          //        scalar_t)+num_movable_nodes,
          //        DREAMPLACE_TENSOR_DATA_PTR(node_size_y,
          //        scalar_t)+num_movable_nodes, DREAMPLACE_TENSOR_DATA_PTR(ax,
          //        scalar_t), DREAMPLACE_TENSOR_DATA_PTR(bx, scalar_t),
          //        DREAMPLACE_TENSOR_DATA_PTR(cx, scalar_t),
          //        DREAMPLACE_TENSOR_DATA_PTR(ay, scalar_t),
          //        DREAMPLACE_TENSOR_DATA_PTR(by, scalar_t),
          //        DREAMPLACE_TENSOR_DATA_PTR(cy, scalar_t),
          //        DREAMPLACE_TENSOR_DATA_PTR(bin_center_x, scalar_t),
          //        DREAMPLACE_TENSOR_DATA_PTR(bin_center_y, scalar_t),
          //        num_impacted_bins_x, num_impacted_bins_y,
          //        mat_size_x, mat_size_y,
          //        num_nodes-num_movable_nodes,
          //        num_bins_x, num_bins_y, 0,
          //        xl, yl, xh, yh,
          //        bin_size_x, bin_size_y,
          //        0,
          //        DREAMPLACE_TENSOR_DATA_PTR(density_map, scalar_t),
          //        nullptr,
          //        nullptr, nullptr
          //        );
          computeDensityOverflowMapCudaLauncher<scalar_t>(
              DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + num_movable_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + num_nodes +
                  num_movable_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(node_size_x, scalar_t) +
                  num_movable_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(node_size_y, scalar_t) +
                  num_movable_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(bin_center_x, scalar_t),
              DREAMPLACE_TENSOR_DATA_PTR(bin_center_y, scalar_t), num_terminals,
              num_bins_x, num_bins_y, num_impacted_bins_x, num_impacted_bins_y,
              xl, yl, xh, yh, bin_size_x, bin_size_y,
              DREAMPLACE_TENSOR_DATA_PTR(density_map, scalar_t));

#if 0
                density_map.div_(bin_size_x*bin_size_y);

                // smooth with Gaussian filter
                T truncate = 4.0; 
                int radius = std::round(truncate*sigma);
                int kernel_size = 2*radius+1; 
                at::Tensor gaussian_filter = at::zeros({kernel_size, kernel_size}, density_map.options()); 
                computeGaussianFilterLauncher<T>(
                        gaussian_filter.size(0), gaussian_filter.size(1), 
                        sigma, 
                        DREAMPLACE_TENSOR_DATA_PTR(gaussian_filter, T)
                        ); 
                gaussian_filter.div_(gaussian_filter.sum());
                //std::cout << "density_map = " << density_map << "\n";
                //std::cout << "gaussian_filter = " << gaussian_filter << "\n";
                density_map = at::conv2d(density_map.view({1, 1, num_bins_x, num_bins_y}), gaussian_filter.view({1, 1, gaussian_filter.size(0), gaussian_filter.size(1)}), {}, 1, {{radius, radius}}).view({num_bins_x, num_bins_y}); 
                //std::cout << "density_map = " << density_map << "\n";
                // normalize to [0, 1]
                //density_map.div_(density_map.max());

                // level smoothing 
                at::Tensor density_mean = density_map.mean();
                at::Tensor delta_map = density_map-density_mean;
                density_map = density_mean + delta_map.sign().mul_(delta_map.abs().pow_(delta));

                density_map.mul_(bin_size_x*bin_size_y);
#endif
        });
  }

  return density_map;
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &DREAMPLACE_NAMESPACE::density_potential_forward,
        "DensityPotential forward (CUDA)");
  m.def("backward", &DREAMPLACE_NAMESPACE::density_potential_backward,
        "DensityPotential backward (CUDA)");
  m.def("fixed_density_map", &DREAMPLACE_NAMESPACE::fixed_density_potential_map,
        "DensityPotential Map for Fixed Cells (CUDA)");
}
