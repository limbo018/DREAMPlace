/**
 * @file   pin_utilization_map.cpp
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
/// @param atomic_add_op functional object for atomic add 
/// @param buf_map 2D density map in column-major to write
template <typename T, typename AtomicOp>
int pinDemandMapLauncher(const T *node_x, const T *node_y, const T *node_size_x,
                         const T *node_size_y,
                         const T *half_node_size_stretch_x,
                         const T *half_node_size_stretch_y,
                         const T *pin_weights, T xl, T yl, T xh, T yh,
                         int num_bins_x,
                         int num_bins_y, int num_nodes, 
                         int num_threads, AtomicOp atomic_add_op,
                         typename AtomicOp::type* buf_map) {
  T bin_size_x = (xh - xl) / num_bins_x; 
  T bin_size_y = (yh - yl) / num_bins_y; 
  const T inv_bin_size_x = 1.0 / bin_size_x;
  const T inv_bin_size_y = 1.0 / bin_size_y;

  int chunk_size =
      DREAMPLACE_STD_NAMESPACE::max(int(num_nodes / num_threads / 16), 1);
#pragma omp parallel for num_threads(num_threads) schedule(dynamic, chunk_size)
  for (int i = 0; i < num_nodes; ++i) {
    const T node_center_x = node_x[i] + node_size_x[i] / 2;
    const T node_center_y = node_y[i] + node_size_y[i] / 2;

    const T x_min = node_center_x - half_node_size_stretch_x[i];
    const T x_max = node_center_x + half_node_size_stretch_x[i];
    int bin_index_xl = int((x_min - xl) * inv_bin_size_x);
    int bin_index_xh = int((x_max - xl) * inv_bin_size_x) + 1;
    bin_index_xl = DREAMPLACE_STD_NAMESPACE::max(bin_index_xl, 0);
    bin_index_xh = DREAMPLACE_STD_NAMESPACE::min(bin_index_xh, num_bins_x);

    const T y_min = node_center_y - half_node_size_stretch_y[i];
    const T y_max = node_center_y + half_node_size_stretch_y[i];
    int bin_index_yl = int((y_min - yl) * inv_bin_size_y);
    int bin_index_yh = int((y_max - yl) * inv_bin_size_y) + 1;
    bin_index_yl = DREAMPLACE_STD_NAMESPACE::max(bin_index_yl, 0);
    bin_index_yh = DREAMPLACE_STD_NAMESPACE::min(bin_index_yh, num_bins_y);

    T density = pin_weights[i] /
                (half_node_size_stretch_x[i] * half_node_size_stretch_y[i] * 4);
    for (int x = bin_index_xl; x < bin_index_xh; ++x) {
      for (int y = bin_index_yl; y < bin_index_yh; ++y) {
        T bin_xl = xl + x * bin_size_x;
        T bin_yl = yl + y * bin_size_y;
        T bin_xh = bin_xl + bin_size_x;
        T bin_yh = bin_yl + bin_size_y;
        T overlap = DREAMPLACE_STD_NAMESPACE::max(
                        DREAMPLACE_STD_NAMESPACE::min(x_max, bin_xh) -
                            DREAMPLACE_STD_NAMESPACE::max(x_min, bin_xl),
                        (T)0) *
                    DREAMPLACE_STD_NAMESPACE::max(
                        DREAMPLACE_STD_NAMESPACE::min(y_max, bin_yh) -
                            DREAMPLACE_STD_NAMESPACE::max(y_min, bin_yl),
                        (T)0);
        int index = x * num_bins_y + y;
        atomic_add_op(&buf_map[index], overlap * density);
      }
    }
  }
  return 0;
}

at::Tensor pin_utilization_map_forward(
    at::Tensor pos, at::Tensor node_size_x, at::Tensor node_size_y,
    at::Tensor half_node_size_stretch_x, at::Tensor half_node_size_stretch_y,
    at::Tensor pin_weights, double xl, double yl, double xh, double yh,
    int num_physical_nodes,
    int num_bins_x, int num_bins_y, 
    int deterministic_flag) {
  CHECK_FLAT_CPU(pos);
  CHECK_EVEN(pos);
  CHECK_CONTIGUOUS(pos);

  CHECK_FLAT_CPU(node_size_x);
  CHECK_CONTIGUOUS(node_size_x);
  CHECK_FLAT_CPU(node_size_y);
  CHECK_CONTIGUOUS(node_size_y);

  CHECK_FLAT_CPU(half_node_size_stretch_x);
  CHECK_CONTIGUOUS(half_node_size_stretch_x);

  CHECK_FLAT_CPU(half_node_size_stretch_y);
  CHECK_CONTIGUOUS(half_node_size_stretch_y);

  CHECK_FLAT_CPU(pin_weights);
  CHECK_CONTIGUOUS(pin_weights);

  CHECK_FLAT_CPU(node_size_x);
  CHECK_CONTIGUOUS(node_size_x);

  CHECK_FLAT_CPU(node_size_y);
  CHECK_CONTIGUOUS(node_size_y);

  at::Tensor pin_utilization_map =
      at::zeros({num_bins_x, num_bins_y}, pos.options());
  auto num_nodes = pos.numel() / 2;

  DREAMPLACE_DISPATCH_FLOATING_TYPES(pos, "pinDemandMapLauncher", [&] {
      if (deterministic_flag) {
          double diearea = (xh - xl) * (yh - yl);
          int integer_bits = DREAMPLACE_STD_NAMESPACE::max((int)ceil(log2(diearea)) + 1, 32);
          int fraction_bits = DREAMPLACE_STD_NAMESPACE::max(64 - integer_bits, 0);
          long scale_factor = (1L << fraction_bits);
          int num_bins = num_bins_x * num_bins_y;

          std::vector<long> buf_map(num_bins, 0);
          AtomicAdd<long> atomic_add_op(scale_factor);

          pinDemandMapLauncher<scalar_t, decltype(atomic_add_op)>(
              DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t),
              DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + num_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(node_size_x, scalar_t),
              DREAMPLACE_TENSOR_DATA_PTR(node_size_y, scalar_t),
              DREAMPLACE_TENSOR_DATA_PTR(half_node_size_stretch_x, scalar_t),
              DREAMPLACE_TENSOR_DATA_PTR(half_node_size_stretch_y, scalar_t),
              DREAMPLACE_TENSOR_DATA_PTR(pin_weights, scalar_t), xl, yl, xh, yh,
              num_bins_x, num_bins_y, num_physical_nodes,
              at::get_num_threads(),
              atomic_add_op, buf_map.data());

          scaleAdd(DREAMPLACE_TENSOR_DATA_PTR(pin_utilization_map, scalar_t),
                   buf_map.data(), 1.0 / scale_factor, num_bins,
                   at::get_num_threads());
      } else {
          AtomicAdd<scalar_t> atomic_add_op;
          pinDemandMapLauncher<scalar_t, decltype(atomic_add_op)>(
              DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t),
              DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + num_nodes,
              DREAMPLACE_TENSOR_DATA_PTR(node_size_x, scalar_t),
              DREAMPLACE_TENSOR_DATA_PTR(node_size_y, scalar_t),
              DREAMPLACE_TENSOR_DATA_PTR(half_node_size_stretch_x, scalar_t),
              DREAMPLACE_TENSOR_DATA_PTR(half_node_size_stretch_y, scalar_t),
              DREAMPLACE_TENSOR_DATA_PTR(pin_weights, scalar_t), xl, yl, xh, yh,
              num_bins_x, num_bins_y, num_physical_nodes,
              at::get_num_threads(),
              atomic_add_op, 
              DREAMPLACE_TENSOR_DATA_PTR(pin_utilization_map, scalar_t));
      }
  });

  return pin_utilization_map;
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &DREAMPLACE_NAMESPACE::pin_utilization_map_forward,
        "compute pin utilization map");
}
