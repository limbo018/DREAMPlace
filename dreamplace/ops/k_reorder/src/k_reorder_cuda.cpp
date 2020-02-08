/**
 * @file   k_reorder_cuda.cpp
 * @author Yibo Lin
 * @date   Jan 2019
 */
#include <vector>
#include <algorithm>
#include <numeric>
#include <cstdlib>
#include <cmath>
#include <limits>
#include "utility/src/torch.h"
#include "utility/src/DetailedPlaceDB.h"
#include "utility/src/DetailedPlaceDBUtils.h"

DREAMPLACE_BEGIN_NAMESPACE

/// @brief k-reorder algorithm for detailed placement 
template <typename T>
int kreorderCUDALauncher(DetailedPlaceDB<T> db, int K, int max_iters, int num_threads);

#define CHECK_FLAT(x) AT_ASSERTM(x.is_cuda() && x.ndimension() == 1, #x "must be a flat tensor on CPU")
#define CHECK_EVEN(x) AT_ASSERTM((x.numel()&1) == 0, #x "must have even number of elements")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x "must be contiguous")

/// I remove the support to Char, since int8_t does not compile for CUDA 
/// char does not compile for ATen either 
#define DISPATCH_CUSTOM_TYPES(TYPE, NAME, ...)                           \
      [&] {                                                                       \
          switch (TYPE.scalarType()) {                                          \
                AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)        \
                AT_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__)         \
                AT_PRIVATE_CASE_TYPE(at::ScalarType::Int, int, __VA_ARGS__)        \
                default:                                                                \
                  AT_ERROR(#NAME, " not implemented for '", TYPE.toString(), "'");  \
                  }                                                                         \
            }()


at::Tensor k_reorder_cuda_forward(
        at::Tensor init_pos,
        at::Tensor node_size_x,
        at::Tensor node_size_y,
        at::Tensor flat_region_boxes, 
        at::Tensor flat_region_boxes_start, 
        at::Tensor node2fence_region_map, 
        at::Tensor flat_net2pin_map, 
        at::Tensor flat_net2pin_start_map, 
        at::Tensor pin2net_map, 
        at::Tensor flat_node2pin_map, 
        at::Tensor flat_node2pin_start_map, 
        at::Tensor pin2node_map, 
        at::Tensor pin_offset_x, 
        at::Tensor pin_offset_y, 
        at::Tensor net_mask, 
        double xl, 
        double yl, 
        double xh, 
        double yh, 
        double site_width, double row_height, 
        int num_bins_x, 
        int num_bins_y,
        int num_movable_nodes, 
        int num_terminal_NIs, 
        int num_filler_nodes, 
        int K, 
        int max_iters, 
        int num_threads
        )
{
    CHECK_FLAT(init_pos); 
    CHECK_EVEN(init_pos);
    CHECK_CONTIGUOUS(init_pos);

    auto pos = init_pos.clone();

    // Call the cuda kernel launcher
    DISPATCH_CUSTOM_TYPES(pos.type(), "kreorderCUDALauncher", [&] {
            auto db = make_placedb<scalar_t>(
                    init_pos,
                    pos, 
                    node_size_x, node_size_y,
                    flat_region_boxes, flat_region_boxes_start, node2fence_region_map, 
                    flat_net2pin_map, flat_net2pin_start_map, pin2net_map, 
                    flat_node2pin_map, flat_node2pin_start_map, pin2node_map, 
                    pin_offset_x, pin_offset_y, 
                    net_mask, 
                    xl, yl, xh, yh, 
                    site_width, row_height, 
                    num_bins_x, num_bins_y,
                    num_movable_nodes, num_terminal_NIs, num_filler_nodes
                    );
            kreorderCUDALauncher(db, K, max_iters, num_threads);
            });

    return pos; 
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("k_reorder", &DREAMPLACE_NAMESPACE::k_reorder_cuda_forward, "K-reorder (CUDA)");
}
