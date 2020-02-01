/**
 * @file   macro_legalize.cpp
 * @author Yibo Lin
 * @date   Jun 2018
 */
#include "utility/src/torch.h"
#include "utility/src/LegalizationDB.h"
#include "utility/src/LegalizationDBUtils.h"
#include "macro_legalize/src/hannan_legalize.h"
#include "macro_legalize/src/lp_legalize.h"

DREAMPLACE_BEGIN_NAMESPACE

/// @brief The macro legalization follows the way of floorplanning, 
/// because macros have quite different sizes. 
/// @return true if legal 
template <typename T>
bool macroLegalizationLauncher(LegalizationDB<T> db);

#define CHECK_FLAT(x) AT_ASSERTM(!x.is_cuda() && x.ndimension() == 1, #x "must be a flat tensor on CPU")
#define CHECK_EVEN(x) AT_ASSERTM((x.numel()&1) == 0, #x "must have even number of elements")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x "must be contiguous")

/// @brief legalize layout with greedy legalization. 
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
at::Tensor macro_legalization_forward(
        at::Tensor init_pos,
        at::Tensor pos, 
        at::Tensor node_size_x,
        at::Tensor node_size_y,
        at::Tensor flat_region_boxes, 
        at::Tensor flat_region_boxes_start, 
        at::Tensor node2fence_region_map, 
        double xl, 
        double yl, 
        double xh, 
        double yh, 
        double site_width, double row_height, 
        int num_bins_x, 
        int num_bins_y,
        int num_movable_nodes, 
        int num_terminal_NIs, 
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
    DREAMPLACE_DISPATCH_FLOATING_TYPES(pos.type(), "macroLegalizationLauncher", [&] {
            auto db = make_placedb<scalar_t>(
                    init_pos, 
                    pos_copy, 
                    node_size_x, 
                    node_size_y, 
                    flat_region_boxes, flat_region_boxes_start, node2fence_region_map, 
                    xl, yl, xh, yh, 
                    site_width, row_height, 
                    num_bins_x, 
                    num_bins_y, 
                    num_movable_nodes, 
                    num_terminal_NIs, 
                    num_filler_nodes
                    );
            macroLegalizationLauncher<scalar_t>(db);
            });
    timer_stop = get_globaltime(); 
    dreamplacePrint(kINFO, "Macro legalization takes %g ms\n", (timer_stop-timer_start)*get_timer_period());

    return pos_copy; 
}

template <typename T>
bool check_macro_legality(LegalizationDB<T> db, const std::vector<int>& macros, bool fast_check)
{
    // check legality between movable and fixed macros 
    // for debug only, so it is slow 
    auto checkOverlap2Nodes = [&](int i, int node_id1, T xl1, T yl1, T width1, T height1, int j, int node_id2, T xl2, T yl2, T width2, T height2) {
        T xh1 = xl1 + width1; 
        T yh1 = yl1 + height1;
        T xh2 = xl2 + width2; 
        T yh2 = yl2 + height2; 
        if (std::min(xh1, xh2) > std::max(xl1, xl2) && std::min(yh1, yh2) > std::max(yl1, yl2))
        {
            dreamplacePrint((fast_check)? kWARN : kERROR, "macro %d (%g, %g, %g, %g) var %d overlaps with macro %d (%g, %g, %g, %g) var %d, fixed: %d\n", 
                    node_id1, xl1, yl1, xh1, yh1, i, 
                    node_id2, xl2, yl2, xh2, yh2, j, 
                    (int)(node_id2 >= db.num_movable_nodes)
                    ); 
            return true; 
        }
        return false; 
    };

    bool legal = true; 
    for (unsigned int i = 0, ie = macros.size(); i < ie; ++i)
    {
        int node_id1 = macros[i];
        T xl1 = db.x[node_id1];
        T yl1 = db.y[node_id1];
        T width1 = db.node_size_x[node_id1];
        T height1 = db.node_size_y[node_id1];
        // constraints with other macros 
        for (unsigned int j = i+1; j < ie; ++j)
        {
            int node_id2 = macros[j];
            T xl2 = db.x[node_id2];
            T yl2 = db.y[node_id2];
            T width2 = db.node_size_x[node_id2];
            T height2 = db.node_size_y[node_id2];

            bool overlap = checkOverlap2Nodes(i, node_id1, xl1, yl1, width1, height1, j, node_id2, xl2, yl2, width2, height2);
            if (overlap)
            {
                legal = false; 
                if (fast_check)
                {
                    return legal; 
                }
            }
        }
        // constraints with fixed macros 
        // when considering fixed macros, there is no guarantee to find legal solution 
        // with current ad-hoc constraint graphs 
        for (int j = db.num_movable_nodes; j < db.num_nodes; ++j)
        {
            int node_id2 = j; 
            T xl2 = db.init_x[node_id2];
            T yl2 = db.init_y[node_id2];
            T width2 = db.node_size_x[node_id2];
            T height2 = db.node_size_y[node_id2];

            bool overlap = checkOverlap2Nodes(i, node_id1, xl1, yl1, width1, height1, j, node_id2, xl2, yl2, width2, height2);
            if (overlap)
            {
                legal = false; 
                if (fast_check)
                {
                    return legal; 
                }
            }
        }
    }
    if (legal)
    {
        dreamplacePrint(kDEBUG, "Macro legality check PASSED\n");
    }
    else 
    {
        dreamplacePrint(kERROR, "Macro legality check FAILED\n");
    }

    return legal; 
}

template <typename T>
T compute_displace(const LegalizationDB<T>& db, const std::vector<int>& macros)
{
    T displace = 0; 
    for (auto node_id : macros)
    {
        displace += std::abs(db.init_x[node_id]-db.x[node_id]) + std::abs(db.init_y[node_id]-db.y[node_id]);
    }
    return displace;
}

template <typename T>
bool macroLegalizationLauncher(LegalizationDB<T> db)
{
    // collect macros 
    std::vector<int> macros; 
    for (int i = 0; i < db.num_movable_nodes; ++i)
    {
        if (db.is_dummy_fixed(i))
        {
            macros.push_back(i);
#ifdef DEBUG
            dreamplacePrint(kDEBUG, "macro %d %gx%g\n", i, db.node_size_x[i], db.node_size_y[i]);
#endif
        }
    }
    dreamplacePrint(kINFO, "Macro legalization: regard %lu cells as dummy fixed (movable macros)\n", macros.size());

    // in case there is no movable macros 
    if (macros.empty())
    {
        return true;
    }

    // store the best legalization solution found
    std::vector<T> best_x (macros.size());
    std::vector<T> best_y (macros.size());
    T best_displace = std::numeric_limits<T>::max();

    // update current best solution  
    auto update_best = [&](bool legal, T displace){
        if (legal && displace < best_displace)
        {
            for (unsigned int i = 0, ie = macros.size(); i < ie; ++i)
            {
                int macro_id = macros[i];
                best_x[i] = db.x[macro_id];
                best_y[i] = db.y[macro_id];
            }
            best_displace = displace; 
        }
    };

    // first round with LP 
    lpLegalizeLauncher(db, macros);
    dreamplacePrint(kINFO, "Macro displacement %g\n", compute_displace(db, macros));
    bool legal = check_macro_legality(db, macros, true);

    // try Hannan grid legalization if still not legal 
    if (!legal)
    {
        legal = hannanLegalizeLauncher(db, macros);
        T displace = compute_displace(db, macros);
        dreamplacePrint(kINFO, "Macro displacement %g\n", displace);
        legal = check_macro_legality(db, macros, true);
        update_best(legal, displace);

        // refine with LP if legal 
        if (legal)
        {
            lpLegalizeLauncher(db, macros);
            displace = compute_displace(db, macros);
            dreamplacePrint(kINFO, "Macro displacement %g\n", displace);
            legal = check_macro_legality(db, macros, true);
            update_best(legal, displace);
        }
    }

    // apply best solution 
    if (best_displace < std::numeric_limits<T>::max())
    {
        dreamplacePrint(kINFO, "use best macro displacement %g\n", best_displace);
        for (unsigned int i = 0, ie = macros.size(); i < ie; ++i)
        {
            int macro_id = macros[i];
            db.x[macro_id] = best_x[i];
            db.y[macro_id] = best_y[i];
        }
    }

    dreamplacePrint(kINFO, "Align macros to site and rows\n");
    // align the lower left corner to row and site
    for (unsigned int i = 0, ie = macros.size(); i < ie; ++i)
    {
        int node_id = macros[i];
        db.x[node_id] = db.align2site(db.x[node_id], db.node_size_x[node_id]);
        db.y[node_id] = db.align2row(db.y[node_id], db.node_size_y[node_id]);
    }

    legal = check_macro_legality(db, macros, false);

    return legal; 
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &DREAMPLACE_NAMESPACE::macro_legalization_forward, "Macro legalization forward");
}
