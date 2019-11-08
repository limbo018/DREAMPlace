/**
 * @file   abacus_legalize.cpp
 * @author Yibo Lin
 * @date   Jun 2018
 */
#include "utility/src/torch.h"
#include "utility/src/LegalizationDB.h"
#include "utility/src/LegalizationDBUtils.h"
#include "abacus_legalize/src/abacus_legalize_cpu.h"
#include "greedy_legalize/src/legality_check_cpu.h"

DREAMPLACE_BEGIN_NAMESPACE

/// @brief legalize layout with abacus legalization. 
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
int abacusLegalizationLauncher(LegalizationDB<T> db);

#define CHECK_FLAT(x) AT_ASSERTM(!x.is_cuda() && x.ndimension() == 1, #x "must be a flat tensor on CPU")
#define CHECK_EVEN(x) AT_ASSERTM((x.numel()&1) == 0, #x "must have even number of elements")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x "must be contiguous")

/// @brief legalize layout with abacus legalization. 
/// Only movable nodes will be moved. Fixed nodes and filler nodes are fixed. 
/// 
/// @param init_pos initial locations of nodes, including movable nodes, fixed nodes, and filler nodes, [0, num_movable_nodes) are movable nodes, [num_movable_nodes, num_nodes-num_filler_nodes) are fixed nodes, [num_nodes-num_filler_nodes, num_nodes) are filler nodes
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
at::Tensor abacus_legalization_forward(
        at::Tensor init_pos,
        at::Tensor pos, 
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

    auto pos_copy = pos.clone();

    hr_clock_rep timer_start, timer_stop; 
    timer_start = get_globaltime(); 
    // Call the cuda kernel launcher
    DREAMPLACE_DISPATCH_FLOATING_TYPES(pos.type(), "abacusLegalizationLauncher", [&] {
            auto db = make_placedb<scalar_t>(
                    init_pos, 
                    pos_copy, 
                    node_size_x, 
                    node_size_y, 
                    xl, yl, xh, yh, 
                    site_width, row_height, 
                    num_bins_x, 
                    num_bins_y, 
                    num_movable_nodes, 
                    num_filler_nodes
                    );
            abacusLegalizationLauncher<scalar_t>(db);
            });
    timer_stop = get_globaltime(); 
    dreamplacePrint(kINFO, "Abacus legalization takes %g ms\n", (timer_stop-timer_start)*get_timer_period());

    return pos_copy; 
}

template <typename T>
int abacusLegalizationLauncher(LegalizationDB<T> db)
{
    abacusLegalizationCPU(
            db.init_x, db.init_y, 
            db.node_size_x, db.node_size_y, 
            db.x, db.y, 
            db.xl, db.yl, db.xh, db.yh, 
            db.site_width, db.row_height, 
            1, db.num_bins_y, 
            db.num_nodes, 
            db.num_movable_nodes, 
            0
            );

    legalityCheckKernelCPU(
            db.init_x, db.init_y, 
            db.node_size_x, db.node_size_y, 
            db.x, db.y, 
            db.site_width, db.row_height, 
            db.xl, db.yl, db.xh, db.yh, 
            db.num_nodes, 
            db.num_movable_nodes
            );

    return 0; 
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &DREAMPLACE_NAMESPACE::abacus_legalization_forward, "Abacus legalization forward");
}
