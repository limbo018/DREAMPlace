/**
 * @file   adjust_node_area.cpp
 * @author Zixuan Jiang, Jiaqi Gu, Yibo Lin
 * @date   Dec 2019
 * @brief  Adjust cell area according to congestion map.
 */

#include "utility/src/torch.h"
#include "utility/src/utils.h"
// local dependency
#include "adjust_node_area/src/scaling_function.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
DEFINE_AVERAGE_SCALING_FUNCTION(T);

template <typename T>
DEFINE_MAX_SCALING_FUNCTION(T);

template <typename T>
int computeInstanceRoutabilityOptimizationMapLauncher(
    const T *pos_x, const T *pos_y, const T *node_size_x, const T *node_size_y,
    const T *routing_utilization_map, T xl, T yl, T bin_size_x, T bin_size_y,
    int num_bins_x, int num_bins_y, int num_movable_nodes, int num_threads,
    T *instance_route_area) {
  const T inv_bin_size_x = 1.0 / bin_size_x;
  const T inv_bin_size_y = 1.0 / bin_size_y;

  int chunk_size = DREAMPLACE_STD_NAMESPACE::max(
      int(num_movable_nodes / num_threads / 16), 1);
#pragma omp parallel for num_threads(num_threads) schedule(dynamic, chunk_size)
  for (int i = 0; i < num_movable_nodes; ++i) {
    const T x_min = pos_x[i];
    const T x_max = x_min + node_size_x[i];
    const T y_min = pos_y[i];
    const T y_max = y_min + node_size_y[i];

    // compute the bin box that this net will affect
    // We do NOT follow Wuxi's implementation. Instead, we clamp the bounding
    // box.
    int bin_index_xl = int((x_min - xl) * inv_bin_size_x);
    int bin_index_xh = int((x_max - xl) * inv_bin_size_x) + 1;
    bin_index_xl = DREAMPLACE_STD_NAMESPACE::max(bin_index_xl, 0);
    bin_index_xh = DREAMPLACE_STD_NAMESPACE::min(bin_index_xh, num_bins_x);

    int bin_index_yl = int((y_min - yl) * inv_bin_size_y);
    int bin_index_yh = int((y_max - yl) * inv_bin_size_y) + 1;
    bin_index_yl = DREAMPLACE_STD_NAMESPACE::max(bin_index_yl, 0);
    bin_index_yh = DREAMPLACE_STD_NAMESPACE::min(bin_index_yh, num_bins_y);

    instance_route_area[i] =
        SCALING_OP(routing_utilization_map, xl, yl, bin_size_x, bin_size_y,
                   num_bins_x, num_bins_y, bin_index_xl, bin_index_yl,
                   bin_index_xh, bin_index_yh, x_min, y_min, x_max, y_max);
  }

  return 0;
}

at::Tensor adjust_node_area_forward(at::Tensor pos, at::Tensor node_size_x,
                                    at::Tensor node_size_y,
                                    at::Tensor routing_utilization_map,
                                    double bin_size_x, double bin_size_y,
                                    double xl, double yl, double xh, double yh,
                                    int num_movable_nodes, int num_bins_x,
                                    int num_bins_y) {
  CHECK_FLAT_CPU(pos);
  CHECK_EVEN(pos);
  CHECK_CONTIGUOUS(pos);

  CHECK_FLAT_CPU(node_size_x);
  CHECK_CONTIGUOUS(node_size_x);

  CHECK_FLAT_CPU(node_size_y);
  CHECK_CONTIGUOUS(node_size_y);

  int num_nodes = pos.numel() / 2;
  at::Tensor instance_route_area =
      at::empty({num_movable_nodes}, pos.options());

  // compute routability and density optimziation instance area
  DREAMPLACE_DISPATCH_FLOATING_TYPES(
      pos, "computeInstanceRoutabilityOptimizationMapLauncher", [&] {
        computeInstanceRoutabilityOptimizationMapLauncher<scalar_t>(
            DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + num_nodes,
            DREAMPLACE_TENSOR_DATA_PTR(node_size_x, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(node_size_y, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(routing_utilization_map, scalar_t), xl,
            yl, bin_size_x, bin_size_y, num_bins_x, num_bins_y,
            num_movable_nodes, at::get_num_threads(),
            DREAMPLACE_TENSOR_DATA_PTR(instance_route_area, scalar_t));
      });

  return instance_route_area;
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &DREAMPLACE_NAMESPACE::adjust_node_area_forward,
        "Compute adjusted area for routability optimization");
}
