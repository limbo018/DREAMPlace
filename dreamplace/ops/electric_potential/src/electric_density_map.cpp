/**
 * @file   density_map.cpp
 * @author Yibo Lin
 * @date   Aug 2018
 * @brief  Compute density map according to e-place
 * (http://cseweb.ucsd.edu/~jlu/papers/eplace-todaes14/paper.pdf)
 */
#include "utility/src/torch.h"
#include "utility/src/utils.h"
// local dependency
#include "electric_potential/src/density_function.h"

DREAMPLACE_BEGIN_NAMESPACE

/// define triangle_density_function
template <typename T>
DEFINE_TRIANGLE_DENSITY_FUNCTION(T);
/// define exact_density_function
template <typename T>
DEFINE_EXACT_DENSITY_FUNCTION(T);

/// @brief The triangular density model from e-place.
/// The impact of a cell to bins is extended to two neighboring bins
template <typename T, typename AtomicOp>
int computeTriangleDensityMapLauncher(
    const T* x_tensor, const T* y_tensor, const T* node_size_x_tensor,
    const T* node_size_y_tensor, const T* offset_x_tensor,
    const T* offset_y_tensor, const T* ratio_tensor,
    const T* bin_center_x_tensor, const T* bin_center_y_tensor,
    const int num_nodes, const int num_bins_x, const int num_bins_y, const T xl,
    const T yl, const T xh, const T yh, const T bin_size_x, const T bin_size_y,
    const int num_threads, AtomicOp atomic_add_op,
    typename AtomicOp::type* buf_map);

/// @brief The exact density model.
/// Compute the exact overlap area for density
template <typename T, typename AtomicOp>
int computeExactDensityMapLauncher(
    const T* x_tensor, const T* y_tensor, const T* node_size_x_tensor,
    const T* node_size_y_tensor, const T* bin_center_x_tensor,
    const T* bin_center_y_tensor, const int num_nodes, const int num_bins_x,
    const int num_bins_y, const T xl, const T yl, const T xh, const T yh,
    const T bin_size_x, const T bin_size_y, bool fixed_node_flag,
    const int num_threads, AtomicOp atomic_add_op,
    typename AtomicOp::type* buf_map);

#define CALL_TRIANGLE_LAUNCHER(begin, end, atomic_add_op, map_ptr)       \
  computeTriangleDensityMapLauncher<scalar_t, decltype(atomic_add_op)>(  \
      DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + begin,                 \
      DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + num_nodes + begin,     \
      DREAMPLACE_TENSOR_DATA_PTR(node_size_x_clamped, scalar_t) + begin, \
      DREAMPLACE_TENSOR_DATA_PTR(node_size_y_clamped, scalar_t) + begin, \
      DREAMPLACE_TENSOR_DATA_PTR(offset_x, scalar_t) + begin,            \
      DREAMPLACE_TENSOR_DATA_PTR(offset_y, scalar_t) + begin,            \
      DREAMPLACE_TENSOR_DATA_PTR(ratio, scalar_t) + begin,               \
      DREAMPLACE_TENSOR_DATA_PTR(bin_center_x, scalar_t),                \
      DREAMPLACE_TENSOR_DATA_PTR(bin_center_y, scalar_t), end - (begin), \
      num_bins_x, num_bins_y, xl, yl, xh, yh, bin_size_x, bin_size_y,    \
      at::get_num_threads(), atomic_add_op, map_ptr)

#define CALL_EXACT_LAUNCHER(begin, end, atomic_add_op, map_ptr)             \
  computeExactDensityMapLauncher<scalar_t, decltype(atomic_add_op)>(        \
      DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + begin,                    \
      DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + num_nodes + begin,        \
      DREAMPLACE_TENSOR_DATA_PTR(node_size_x, scalar_t) + begin,            \
      DREAMPLACE_TENSOR_DATA_PTR(node_size_y, scalar_t) + begin,            \
      DREAMPLACE_TENSOR_DATA_PTR(bin_center_x, scalar_t),                   \
      DREAMPLACE_TENSOR_DATA_PTR(bin_center_y, scalar_t), end - (begin),    \
      num_bins_x, num_bins_y, xl, yl, xh, yh, bin_size_x, bin_size_y, true, \
      at::get_num_threads(), atomic_add_op, map_ptr)

/// @brief compute density map for movable and filler cells
/// @param pos cell locations. The array consists of all x locations and then y
/// locations.
/// @param node_size_x cell width array
/// @param node_size_y cell height array
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
at::Tensor density_map(
    at::Tensor pos, at::Tensor node_size_x_clamped,
    at::Tensor node_size_y_clamped, at::Tensor offset_x, at::Tensor offset_y,
    at::Tensor ratio, at::Tensor bin_center_x, at::Tensor bin_center_y,
    at::Tensor initial_density_map, double target_density, double xl, double yl,
    double xh, double yh, double bin_size_x, double bin_size_y,
    int num_movable_nodes, int num_filler_nodes, int padding, int num_bins_x,
    int num_bins_y, int num_movable_impacted_bins_x,
    int num_movable_impacted_bins_y, int num_filler_impacted_bins_x,
    int num_filler_impacted_bins_y, int deterministic_flag) {
  CHECK_FLAT_CPU(pos);
  CHECK_EVEN(pos);
  CHECK_CONTIGUOUS(pos);

  at::Tensor density_map = initial_density_map.clone();
  int num_nodes = pos.numel() / 2;

  // total die area
  double diearea = (xh - xl) * (yh - yl);
  int integer_bits =
      DREAMPLACE_STD_NAMESPACE::max((int)ceil(log2(diearea)) + 1, 32);
  int fraction_bits = DREAMPLACE_STD_NAMESPACE::max(64 - integer_bits, 0);
  long scale_factor = (1L << fraction_bits);
  int num_bins = num_bins_x * num_bins_y;

  // Call the cuda kernel launcher
  DREAMPLACE_DISPATCH_FLOATING_TYPES(
      pos, "computeTriangleDensityMapLauncher", [&] {
        if (deterministic_flag) {
          std::vector<long> buf(num_bins, 0);
          AtomicAdd<long> atomic_add_op(scale_factor);
          CALL_TRIANGLE_LAUNCHER(0, num_movable_nodes, atomic_add_op,
                                 buf.data());
          if (num_filler_nodes) {
            CALL_TRIANGLE_LAUNCHER(num_nodes - num_filler_nodes, num_nodes,
                                   atomic_add_op, buf.data());
          }
          scaleAdd(DREAMPLACE_TENSOR_DATA_PTR(density_map, scalar_t),
                   buf.data(), 1.0 / scale_factor, num_bins,
                   at::get_num_threads());
        } else {
          auto buf = DREAMPLACE_TENSOR_DATA_PTR(density_map, scalar_t);
          AtomicAdd<scalar_t> atomic_add_op;
          CALL_TRIANGLE_LAUNCHER(0, num_movable_nodes, atomic_add_op, buf);
          if (num_filler_nodes) {
            CALL_TRIANGLE_LAUNCHER(num_nodes - num_filler_nodes, num_nodes,
                                   atomic_add_op, buf);
          }
        }
      });

  return density_map;
}

/// @brief Compute density map for fixed cells
at::Tensor fixed_density_map(at::Tensor pos, at::Tensor node_size_x,
                             at::Tensor node_size_y, at::Tensor bin_center_x,
                             at::Tensor bin_center_y, double xl, double yl,
                             double xh, double yh, double bin_size_x,
                             double bin_size_y, int num_movable_nodes,
                             int num_terminals, int num_bins_x, int num_bins_y,
                             int num_fixed_impacted_bins_x,
                             int num_fixed_impacted_bins_y,
                             int deterministic_flag) {
  CHECK_FLAT_CPU(pos);
  CHECK_EVEN(pos);
  CHECK_CONTIGUOUS(pos);

  at::Tensor density_map = at::zeros({num_bins_x, num_bins_y}, pos.options());

  int num_nodes = pos.numel() / 2;

  // total die area
  double diearea = (xh - xl) * (yh - yl);
  int integer_bits =
      DREAMPLACE_STD_NAMESPACE::max((int)ceil(log2(diearea)) + 1, 32);
  int fraction_bits = DREAMPLACE_STD_NAMESPACE::max(64 - integer_bits, 0);
  long scale_factor = (1L << fraction_bits);
  int num_bins = num_bins_x * num_bins_y;

  // Call the cuda kernel launcher
  if (num_terminals) {
    DREAMPLACE_DISPATCH_FLOATING_TYPES(
        pos, "computeExactDensityMapLauncher", [&] {
          if (deterministic_flag) {
            dreamplacePrint(kDEBUG,
                            "deterministic mode: integer %d bits, fraction %d "
                            "bits, scale factor %ld\n",
                            integer_bits, fraction_bits, scale_factor);
            std::vector<long> buf(num_bins, 0);
            AtomicAdd<long> atomic_add_op(scale_factor);
            CALL_EXACT_LAUNCHER(num_movable_nodes,
                                num_movable_nodes + num_terminals,
                                atomic_add_op, buf.data());
            scaleAdd(DREAMPLACE_TENSOR_DATA_PTR(density_map, scalar_t),
                     buf.data(), 1.0 / scale_factor, num_bins,
                     at::get_num_threads());
          } else {
            auto buf = DREAMPLACE_TENSOR_DATA_PTR(density_map, scalar_t);
            AtomicAdd<scalar_t> atomic_add_op;
            CALL_EXACT_LAUNCHER(num_movable_nodes,
                                num_movable_nodes + num_terminals,
                                atomic_add_op, buf);
          }
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
    int num_movable_nodes, int num_filler_nodes);

template <typename T, typename AtomicOp>
int computeTriangleDensityMapLauncher(
    const T* x_tensor, const T* y_tensor, const T* node_size_x_clamped_tensor,
    const T* node_size_y_clamped_tensor, const T* offset_x_tensor,
    const T* offset_y_tensor, const T* ratio_tensor,
    const T* bin_center_x_tensor, const T* bin_center_y_tensor,
    const int num_nodes, const int num_bins_x, const int num_bins_y, const T xl,
    const T yl, const T xh, const T yh, const T bin_size_x, const T bin_size_y,
    const int num_threads, AtomicOp atomic_add_op,
    typename AtomicOp::type* buf_map) {
  // density_map_tensor should be initialized outside

  T inv_bin_size_x = 1.0 / bin_size_x;
  T inv_bin_size_y = 1.0 / bin_size_y;
  // do not use dynamic scheduling for determinism
  // int chunk_size =
  // DREAMPLACE_STD_NAMESPACE::max(int(num_nodes/num_threads/16), 1);
#pragma omp parallel for num_threads( \
    num_threads)  // schedule(dynamic, chunk_size)
  for (int i = 0; i < num_nodes; ++i) {
    // use stretched node size
    T node_size_x = node_size_x_clamped_tensor[i];
    T node_size_y = node_size_y_clamped_tensor[i];
    T node_x = x_tensor[i] + offset_x_tensor[i];
    T node_y = y_tensor[i] + offset_y_tensor[i];
    T ratio = ratio_tensor[i];

    int bin_index_xl = int((node_x - xl) * inv_bin_size_x);
    int bin_index_xh =
        int(((node_x + node_size_x - xl) * inv_bin_size_x)) + 1;  // exclusive
    bin_index_xl = DREAMPLACE_STD_NAMESPACE::max(bin_index_xl, 0);
    bin_index_xh = DREAMPLACE_STD_NAMESPACE::min(bin_index_xh, num_bins_x);

    int bin_index_yl = int((node_y - yl) * inv_bin_size_y);
    int bin_index_yh =
        int(((node_y + node_size_y - yl) * inv_bin_size_y)) + 1;  // exclusive
    bin_index_yl = DREAMPLACE_STD_NAMESPACE::max(bin_index_yl, 0);
    bin_index_yh = DREAMPLACE_STD_NAMESPACE::min(bin_index_yh, num_bins_y);

    // update density potential map
    for (int k = bin_index_xl; k < bin_index_xh; ++k) {
      T px = triangle_density_function(node_x, node_size_x, xl, k, bin_size_x);
      T px_by_ratio = px * ratio;

      for (int h = bin_index_yl; h < bin_index_yh; ++h) {
        T py =
            triangle_density_function(node_y, node_size_y, yl, h, bin_size_y);
        T area = px_by_ratio * py;

        atomic_add_op(&buf_map[k * num_bins_y + h], area);
      }
    }
  }

  return 0;
}

template <typename T, typename AtomicOp>
int computeExactDensityMapLauncher(
    const T* x_tensor, const T* y_tensor, const T* node_size_x_tensor,
    const T* node_size_y_tensor, const T* bin_center_x_tensor,
    const T* bin_center_y_tensor, const int num_nodes, const int num_bins_x,
    const int num_bins_y, const T xl, const T yl, const T xh, const T yh,
    const T bin_size_x, const T bin_size_y, bool fixed_node_flag,
    const int num_threads, AtomicOp atomic_add_op,
    typename AtomicOp::type* buf_map) {
  // density_map_tensor should be initialized outside

  auto box2bin = [&](T bxl, T byl, T bxh, T byh) {
    // x direction
    int bin_index_xl = int((bxl - xl) / bin_size_x);
    int bin_index_xh = int(ceil((bxh - xl) / bin_size_x)) + 1;  // exclusive
    bin_index_xl = DREAMPLACE_STD_NAMESPACE::max(bin_index_xl, 0);
    bin_index_xh = DREAMPLACE_STD_NAMESPACE::min(bin_index_xh, num_bins_x);

    // y direction
    int bin_index_yl = int((byl - yl) / bin_size_y);
    int bin_index_yh = int(ceil((byh - yl) / bin_size_y)) + 1;  // exclusive
    bin_index_yl = DREAMPLACE_STD_NAMESPACE::max(bin_index_yl, 0);
    bin_index_yh = DREAMPLACE_STD_NAMESPACE::min(bin_index_yh, num_bins_y);

    for (int k = bin_index_xl; k < bin_index_xh; ++k) {
      T px = exact_density_function(bxl, bxh - bxl, bin_center_x_tensor[k],
                                    bin_size_x, xl, xh, fixed_node_flag);
      for (int h = bin_index_yl; h < bin_index_yh; ++h) {
        T py = exact_density_function(byl, byh - byl, bin_center_y_tensor[h],
                                      bin_size_y, yl, yh, fixed_node_flag);
        auto area = px * py;

        atomic_add_op(&buf_map[k * num_bins_y + h], area);
      }
    }
  };

#pragma omp parallel for num_threads(num_threads)
  for (int i = 0; i < num_nodes; ++i) {
    T bxl = x_tensor[i];
    T byl = y_tensor[i];
    T bxh = bxl + node_size_x_tensor[i];
    T byh = byl + node_size_y_tensor[i];
    box2bin(bxl, byl, bxh, byh);
  }

  return 0;
}

#undef CALL_TRIANGLE_LAUNCHER
#undef CALL_EXACT_LAUNCHER

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("density_map", &DREAMPLACE_NAMESPACE::density_map,
        "ElectricPotential Density Map");
  m.def("fixed_density_map", &DREAMPLACE_NAMESPACE::fixed_density_map,
        "ElectricPotential Density Map for Fixed Cells");
  m.def("electric_force", &DREAMPLACE_NAMESPACE::electric_force,
        "ElectricPotential Electric Force");
}
