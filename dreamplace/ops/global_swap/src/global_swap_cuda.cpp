/**
 * @file   global_swap_cuda.cpp
 * @author Yibo Lin
 * @date   Jan 2019
 */
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <vector>
#include "utility/src/torch.h"
#include "utility/src/utils.h"
// database dependency
#include "utility/src/detailed_place_db.h"
#include "utility/src/make_placedb.h"

DREAMPLACE_BEGIN_NAMESPACE

/// @brief global swap algorithm for detailed placement
template <typename T>
int globalSwapCUDALauncher(DetailedPlaceDB<T> db, int batch_size, int max_iters,
                           int num_threads);

at::Tensor global_swap_cuda_forward(
    at::Tensor init_pos, at::Tensor node_size_x, at::Tensor node_size_y,
    at::Tensor flat_region_boxes, at::Tensor flat_region_boxes_start,
    at::Tensor node2fence_region_map, at::Tensor flat_net2pin_map,
    at::Tensor flat_net2pin_start_map, at::Tensor pin2net_map,
    at::Tensor flat_node2pin_map, at::Tensor flat_node2pin_start_map,
    at::Tensor pin2node_map, at::Tensor pin_offset_x, at::Tensor pin_offset_y,
    at::Tensor net_mask, double xl, double yl, double xh, double yh,
    double site_width, double row_height, int num_bins_x, int num_bins_y,
    int num_movable_nodes, int num_terminal_NIs, int num_filler_nodes,
    int batch_size, int max_iters) {
  CHECK_FLAT_CUDA(init_pos);
  CHECK_EVEN(init_pos);
  CHECK_CONTIGUOUS(init_pos);

  auto pos = init_pos.clone();

  // Call the cuda kernel launcher
  DREAMPLACE_DISPATCH_INT_FLOAT_TYPES(pos, "globalSwapCUDALauncher", [&] {
    auto db = make_placedb<scalar_t>(
        init_pos, pos, node_size_x, node_size_y, flat_region_boxes,
        flat_region_boxes_start, node2fence_region_map, flat_net2pin_map,
        flat_net2pin_start_map, pin2net_map, flat_node2pin_map,
        flat_node2pin_start_map, pin2node_map, pin_offset_x, pin_offset_y,
        net_mask, xl, yl, xh, yh, site_width, row_height, num_bins_x,
        num_bins_y, num_movable_nodes, num_terminal_NIs, num_filler_nodes);
    globalSwapCUDALauncher(db, batch_size, max_iters, at::get_num_threads());
  });

  return pos;
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("global_swap", &DREAMPLACE_NAMESPACE::global_swap_cuda_forward,
        "Global swap (CUDA)");
}
