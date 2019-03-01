/**
 * @file   src/move_boundary.cpp
 * @author Yibo Lin
 * @date   Jun 2018
 */
#include <torch/torch.h>
#include <limits>
#include "function_cpu.h"

/// @brief legalize layout with greedy legalization. 
/// Only movable nodes will be moved. Fixed nodes and filler nodes are fixed. 
/// 
/// @param init_x initial x location of nodes, including movable nodes, fixed nodes, and filler nodes, [0, num_movable_nodes) are movable nodes, [num_movable_nodes, num_nodes-num_filler_nodes) are fixed nodes, [num_nodes-num_filler_nodes, num_nodes) are filler nodes
/// @param init_y initial y location of nodes, including movable nodes, fixed nodes, and filler nodes, same as init_x
/// @param node_size_x width of nodes, including movable nodes, fixed nodes, and filler nodes, [0, num_movable_nodes) are movable nodes, [num_movable_nodes, num_nodes-num_filler_nodes) are fixed nodes, [num_nodes-num_filler_nodes, num_nodes) are filler nodes
/// @param node_size_y height of nodes, including movable nodes, fixed nodes, and filler nodes, same as node_size_x
/// @param xl left edge of bounding box of layout area 
/// @param yl bottom edge of bounding box of layout area 
/// @param xh right edge of bounding box of layout area 
/// @param yh top edge of bounding box of layout area 
/// @param site_width width of a placement site 
/// @param row_height height of a placement row 
/// @param num_bins_x number of bins in horizontal direction 
/// @param num_bins_y number of bins in vertical direction 
/// @param num_nodes total number of nodes, including movable nodes, fixed nodes, and filler nodes; fixed nodes are in the range of [num_movable_nodes, num_nodes-num_filler_nodes)
/// @param num_movable_nodes number of movable nodes, movable nodes are in the range of [0, num_movable_nodes)
/// @param number of filler nodes, filler nodes are in the range of [num_nodes-num_filler_nodes, num_nodes)
template <typename T>
int greedyLegalizationLauncher(
        const T* init_x, const T* init_y, 
        const T* node_size_x, const T* node_size_y, 
        T* x, T* y, 
        const T xl, const T yl, const T xh, const T yh, 
        const T site_width, const T row_height, 
        int num_bins_x, int num_bins_y, 
        const int num_nodes, 
        const int num_movable_nodes, 
        const int num_filler_nodes 
        )
{
    greedyLegalizationCPU(
            init_x, init_y, 
            node_size_x, node_size_y, 
            x, y, 
            xl, yl, xh, yh, 
            site_width, row_height, 
            num_bins_x, num_bins_y, 
            num_nodes, 
            num_movable_nodes, 
            num_filler_nodes
            );
    return 0; 
}

#define CHECK_FLAT(x) AT_ASSERTM(!x.is_cuda() && x.ndimension() == 1, #x "must be a flat tensor on CPU")
#define CHECK_EVEN(x) AT_ASSERTM((x.numel()&1) == 0, #x "must have even number of elements")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x "must be contiguous")

at::Tensor greedy_legalization_forward(
        at::Tensor init_pos,
        at::Tensor node_size_x,
        at::Tensor node_size_y,
        double xl, 
        double yl, 
        double xh, 
        double yh, 
        double site_width, double row_height, 
        int num_bins_x, 
        int num_bins_y,
        int num_movable_nodes, 
        int num_filler_nodes
        )
{
    CHECK_FLAT(init_pos); 
    CHECK_EVEN(init_pos);
    CHECK_CONTIGUOUS(init_pos);

    auto pos = init_pos.clone();
    int num_nodes = init_pos.numel()/2;

    // Call the cuda kernel launcher
    AT_DISPATCH_FLOATING_TYPES(pos.type(), "greedyLegalizationLauncher", [&] {
            greedyLegalizationLauncher<scalar_t>(
                    init_pos.data<scalar_t>(), init_pos.data<scalar_t>()+num_nodes, 
                    node_size_x.data<scalar_t>(), node_size_y.data<scalar_t>(), 
                    pos.data<scalar_t>(), pos.data<scalar_t>()+num_nodes, 
                    xl, yl, xh, yh, 
                    site_width, row_height, 
                    num_bins_x, num_bins_y, 
                    num_nodes, 
                    num_movable_nodes, 
                    num_filler_nodes
                    );
            });

    return pos; 
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &greedy_legalization_forward, "Greedy legalization forward");
}
