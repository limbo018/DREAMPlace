/**
 * @file   rudy.cpp
 * @author Zixuan Jiang, Jiaqi Gu, Yibo Lin
 * @date   Dec 2019
 * @brief  Compute the RUDY/RISA map for routing demand.
 *         A routing/pin utilization estimator based on the following two papers
 *         "Fast and Accurate Routing Demand Estimation for efficient
 * Routability-driven Placement", by Peter Spindler, DATE'07 "RISA: Accurate and
 * Efficient Placement Routability Modeling", by Chih-liang Eric Cheng,
 * ICCAD'94
 */

#include "rudy/src/parameters.h"
#include "utility/src/torch.h"
#include "utility/src/utils.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
inline DEFINE_NET_WIRING_DISTRIBUTION_MAP_WEIGHT;

// fill the demand map net by net
template <typename T, typename AtomicOp>
int rudyLauncher(const T *pin_pos_x, const T *pin_pos_y,
                 const int *netpin_start, const int *flat_netpin,
                 const T *net_weights, const T bin_size_x, const T bin_size_y,
                 T xl, T yl, T xh, T yh, int num_bins_x, int num_bins_y,
                 int num_nets, int num_threads, AtomicOp atomic_add_op,
                 typename AtomicOp::type* horizontal_buf_map, 
                 typename AtomicOp::type* vertical_buf_map) {
  const T inv_bin_size_x = 1.0 / bin_size_x;
  const T inv_bin_size_y = 1.0 / bin_size_y;

  int chunk_size =
      DREAMPLACE_STD_NAMESPACE::max(int(num_nets / num_threads / 16), 1);
#pragma omp parallel for num_threads(num_threads) schedule(dynamic, chunk_size)
  for (int i = 0; i < num_nets; ++i) {
    T x_max = -std::numeric_limits<T>::max();
    T x_min = std::numeric_limits<T>::max();
    T y_max = -std::numeric_limits<T>::max();
    T y_min = std::numeric_limits<T>::max();

    for (int j = netpin_start[i]; j < netpin_start[i + 1]; ++j) {
      int pin_id = flat_netpin[j];
      const T xx = pin_pos_x[pin_id];
      x_max = DREAMPLACE_STD_NAMESPACE::max(xx, x_max);
      x_min = DREAMPLACE_STD_NAMESPACE::min(xx, x_min);
      const T yy = pin_pos_y[pin_id];
      y_max = DREAMPLACE_STD_NAMESPACE::max(yy, y_max);
      y_min = DREAMPLACE_STD_NAMESPACE::min(yy, y_min);
    }

    // compute the bin box that this net will affect
    int bin_index_xl = int((x_min - xl) * inv_bin_size_x);
    int bin_index_xh = int((x_max - xl) * inv_bin_size_x) + 1;
    bin_index_xl = DREAMPLACE_STD_NAMESPACE::max(bin_index_xl, 0);
    bin_index_xh = DREAMPLACE_STD_NAMESPACE::min(bin_index_xh, num_bins_x);

    int bin_index_yl = int((y_min - yl) * inv_bin_size_y);
    int bin_index_yh = int((y_max - yl) * inv_bin_size_y) + 1;
    bin_index_yl = DREAMPLACE_STD_NAMESPACE::max(bin_index_yl, 0);
    bin_index_yh = DREAMPLACE_STD_NAMESPACE::min(bin_index_yh, num_bins_y);

    T wt = netWiringDistributionMapWeight<T>(netpin_start[i + 1] -
                                             netpin_start[i]);
    if (net_weights) {
      wt *= net_weights[i];
    }

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
        overlap *= wt;
        int index = x * num_bins_y + y;
        // Following Wuxi's implementation, a tolerance is added to avoid 0-size
        // bounding box
        atomic_add_op(&horizontal_buf_map[index], overlap / (y_max - y_min + std::numeric_limits<T>::epsilon()));
        atomic_add_op(&vertical_buf_map[index], overlap / (x_max - x_min + std::numeric_limits<T>::epsilon())); 
      }
    }
  }
  return 0;
}

void rudy_forward(at::Tensor pin_pos, at::Tensor netpin_start,
                  at::Tensor flat_netpin, at::Tensor net_weights,
                  double bin_size_x, double bin_size_y, double xl, double yl,
                  double xh, double yh, int num_bins_x, int num_bins_y,
                  int deterministic_flag,
                  at::Tensor horizontal_utilization_map,
                  at::Tensor vertical_utilization_map) {
  CHECK_FLAT_CPU(pin_pos);
  CHECK_EVEN(pin_pos);
  CHECK_CONTIGUOUS(pin_pos);

  CHECK_FLAT_CPU(netpin_start);
  CHECK_CONTIGUOUS(netpin_start);

  CHECK_FLAT_CPU(flat_netpin);
  CHECK_CONTIGUOUS(flat_netpin);

  CHECK_FLAT_CPU(net_weights);
  CHECK_CONTIGUOUS(net_weights);

  int num_nets = netpin_start.numel() - 1;
  int num_pins = pin_pos.numel() / 2;

  DREAMPLACE_DISPATCH_FLOATING_TYPES(pin_pos, "rudyLauncher", [&] {
      if (deterministic_flag) {
          double diearea = (xh - xl) * (yh - yl);
          int integer_bits = DREAMPLACE_STD_NAMESPACE::max((int)ceil(log2(diearea)) + 1, 32);
          int fraction_bits = DREAMPLACE_STD_NAMESPACE::max(64 - integer_bits, 0);
          long scale_factor = (1L << fraction_bits);
          int num_bins = num_bins_x * num_bins_y;

          std::vector<long> horizontal_buf_map(num_bins, 0);
          std::vector<long> vertical_buf_map(num_bins, 0);
          AtomicAdd<long> atomic_add_op(scale_factor);

          rudyLauncher<scalar_t, decltype(atomic_add_op)>(
              DREAMPLACE_TENSOR_DATA_PTR(pin_pos, scalar_t),
              DREAMPLACE_TENSOR_DATA_PTR(pin_pos, scalar_t) + num_pins,
              DREAMPLACE_TENSOR_DATA_PTR(netpin_start, int),
              DREAMPLACE_TENSOR_DATA_PTR(flat_netpin, int),
              (net_weights.numel())
                  ? DREAMPLACE_TENSOR_DATA_PTR(net_weights, scalar_t)
                  : nullptr,
              bin_size_x, bin_size_y, xl, yl, xh, yh,

              num_bins_x, num_bins_y, num_nets, at::get_num_threads(),
              atomic_add_op, horizontal_buf_map.data(), vertical_buf_map.data());

          scaleAdd(DREAMPLACE_TENSOR_DATA_PTR(horizontal_utilization_map, scalar_t),
                   horizontal_buf_map.data(), 1.0 / scale_factor, num_bins,
                   at::get_num_threads());
          scaleAdd(DREAMPLACE_TENSOR_DATA_PTR(vertical_utilization_map, scalar_t),
                   vertical_buf_map.data(), 1.0 / scale_factor, num_bins,
                   at::get_num_threads());
      } else {
          AtomicAdd<scalar_t> atomic_add_op;
          rudyLauncher<scalar_t, decltype(atomic_add_op)>(
              DREAMPLACE_TENSOR_DATA_PTR(pin_pos, scalar_t),
              DREAMPLACE_TENSOR_DATA_PTR(pin_pos, scalar_t) + num_pins,
              DREAMPLACE_TENSOR_DATA_PTR(netpin_start, int),
              DREAMPLACE_TENSOR_DATA_PTR(flat_netpin, int),
              (net_weights.numel())
                  ? DREAMPLACE_TENSOR_DATA_PTR(net_weights, scalar_t)
                  : nullptr,
              bin_size_x, bin_size_y, xl, yl, xh, yh,

              num_bins_x, num_bins_y, num_nets, at::get_num_threads(),
              atomic_add_op, 
              DREAMPLACE_TENSOR_DATA_PTR(horizontal_utilization_map, scalar_t),
              DREAMPLACE_TENSOR_DATA_PTR(vertical_utilization_map, scalar_t));
      }
  });
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &DREAMPLACE_NAMESPACE::rudy_forward, "compute RUDY map");
}
