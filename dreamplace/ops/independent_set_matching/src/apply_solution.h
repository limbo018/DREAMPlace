/**
 * @file   apply_solution.h
 * @author Yibo Lin
 * @date   Mar 2019
 */
#ifndef _DREAMPLACE_INDEPENDENT_SET_MATCHING_APPLY_SOLUTION_H
#define _DREAMPLACE_INDEPENDENT_SET_MATCHING_APPLY_SOLUTION_H

#include "utility/src/utils.h"
#include "independent_set_matching/src/adjust_pos.h"

//#define DEBUG

DREAMPLACE_BEGIN_NAMESPACE

template <typename DetailedPlaceDBType,
          typename IndependentSetMatchingStateType>
void apply_solution(DetailedPlaceDBType& db,
                    IndependentSetMatchingStateType& state,
                    int i  ///< entry in the batch
) {
#ifdef DEBUG
  dreamplaceAssert(i < (int)state.independent_sets.size());
  dreamplaceAssert(i < (int)state.solutions.size());
  dreamplaceAssert(i < (int)state.target_pos_x.size());
  dreamplaceAssert(i < (int)state.target_pos_y.size());
#endif
  auto const& independent_set = state.independent_sets.at(i);
  auto& solution = state.solutions.at(i);
  auto& target_pos_x = state.target_pos_x.at(i);
  auto& target_pos_y = state.target_pos_y.at(i);
  auto& target_spaces = state.target_spaces.at(i);
  target_pos_x.resize(independent_set.size());
  target_pos_y.resize(independent_set.size());
  target_spaces.resize(independent_set.size());

  // apply solution
  if (state.target_costs[i] < state.orig_costs[i]) {
    // record the locations
    for (unsigned int j = 0; j < independent_set.size(); ++j) {
      int target_node_id = independent_set.at(j);
      if (target_node_id < db.num_movable_nodes) {
#ifdef DEBUG
        dreamplaceAssert(j < target_pos_x.size());
        dreamplaceAssert(j < target_pos_y.size());
        dreamplaceAssertMsg(target_node_id < db.num_movable_nodes, "%d < %d",
                            target_node_id, db.num_movable_nodes);
#endif
        target_pos_x[j] = db.x[target_node_id];
        target_pos_y[j] = db.y[target_node_id];
        target_spaces[j] = state.spaces[target_node_id];
      }
    }
#ifdef DEBUG
      // typename DetailedPlaceDBType::type orig_hpwl = db.compute_total_hpwl();
      // for (unsigned int j = 0; j < independent_set.size(); ++j)
      //{
      //    int node_id = independent_set[j];
      //    typename DetailedPlaceDBType::type target_hpwl = 0;
      //    for (int node2pin_id = db.flat_node2pin_start_map[node_id];
      //    node2pin_id < db.flat_node2pin_start_map[node_id+1]; ++node2pin_id)
      //    {
      //        int node_pin_id = db.flat_node2pin_map[node2pin_id];
      //        int net_id = db.pin2net_map[node_pin_id];
      //        dreamplacePrint(kNONE, "node %d(%d) net %d(%d)\n", j, node_id,
      //        net_id, db.net_mask[net_id]); if (db.net_mask[net_id])
      //        {
      //            typename DetailedPlaceDBType::type bxl = db.xh;
      //            typename DetailedPlaceDBType::type byl = db.yh;
      //            typename DetailedPlaceDBType::type bxh = db.xl;
      //            typename DetailedPlaceDBType::type byh = db.yl;
      //            for (int net2pin_id = db.flat_net2pin_start_map[net_id];
      //            net2pin_id < db.flat_net2pin_start_map[net_id+1];
      //            ++net2pin_id)
      //            {
      //                int net_pin_id = db.flat_net2pin_map[net2pin_id];
      //                int other_node_id = db.pin2node_map[net_pin_id];
      //                typename DetailedPlaceDBType::type xxl =
      //                db.x[other_node_id]; typename DetailedPlaceDBType::type
      //                yyl = db.y[other_node_id]; bxl = std::min(bxl,
      //                xxl+db.pin_offset_x[net_pin_id]); bxh = std::max(bxh,
      //                xxl+db.pin_offset_x[net_pin_id]); byl = std::min(byl,
      //                yyl+db.pin_offset_y[net_pin_id]); byh = std::max(byh,
      //                yyl+db.pin_offset_y[net_pin_id]);
      //            }
      //            target_hpwl += (bxh-bxl) + (byh-byl);
      //        }
      //    }
      //    dreamplacePrint(kDEBUG, "node %d(%d) original hpwl %g\n", j,
      //    node_id, target_hpwl);
      //}
#endif
    // move cells
    int count = 0;
    for (unsigned int j = 0; j < independent_set.size(); ++j) {
#ifdef DEBUG
      dreamplaceAssert(j < solution.size());
#endif
      int sol_j = solution.at(j);
#ifdef DEBUG
      dreamplaceAssert(sol_j < (int)solution.size());
#endif
      int target_node_id = independent_set.at(j);
      if (target_node_id < db.num_movable_nodes) {
        //#ifdef DEBUG
        int target_pos_id = independent_set.at(sol_j);
        dreamplaceAssert(target_pos_id < db.num_movable_nodes);
        //#endif
        if (j != (unsigned int)sol_j) {
          count += 1;

#ifdef DEBUG
          // dreamplacePrint(kDEBUG, "move node %d(%d) pos %d(%d): (%g, %g) =>
          // (%g, %g), displace (%g, %g)\n",
          //        j, target_node_id, sol_j, target_pos_id,
          //        db.x[target_node_id], db.y[target_node_id],
          //        target_pos_x[sol_j], target_pos_y[sol_j],
          //        target_pos_x[sol_j]-db.x[target_node_id],
          //        target_pos_y[sol_j]-db.y[target_node_id]
          //      );
          dreamplaceAssert(target_node_id < db.num_movable_nodes);
#endif
          bool ret = adjust_pos(target_pos_x[sol_j], db.node_size_x[target_node_id], target_spaces[sol_j]);
          dreamplaceAssertMsg(ret,
                "set %d (%lu nodes), node %d, width %g, %g, %g, pos %d, %g, %g, space %g, %g, orig_cost %d, target_cost %d, cost %d\n",
                i, independent_set.size(), target_node_id, db.node_size_x[target_node_id], 
                db.x[target_node_id],
                db.x[target_node_id] + db.node_size_x[target_node_id],
                target_pos_id, target_pos_x[sol_j],
                target_pos_x[sol_j] + db.node_size_x[target_pos_id],
                target_spaces[sol_j].xl, target_spaces[sol_j].xh, 
                state.orig_costs[i], state.target_costs[i], 
                state.cost_matrices.at(i).at(j * independent_set.size() + sol_j));
          // update position
          db.x[target_node_id] = target_pos_x[sol_j];
          db.y[target_node_id] = target_pos_y[sol_j];
          state.spaces.at(target_node_id) = target_spaces[sol_j];
          // error-prone when it comes to weird scaling factors 
          // due to numerical precisions 
          //dreamplaceAssertMsg(db.x[target_node_id] >= target_spaces[sol_j].xl &&
          //    db.x[target_node_id] +
          //    db.node_size_x[target_node_id] <=
          //    target_spaces[sol_j].xh, 
          //    "gap %g, %g", 
          //    db.x[target_node_id] - target_spaces[sol_j].xl, 
          //    db.x[target_node_id] +
          //    db.node_size_x[target_node_id] -
          //    target_spaces[sol_j].xh
          //    );
        }
      }
    }
#ifdef DEBUG
      // typename DetailedPlaceDBType::type target_hpwl =
      // db.compute_total_hpwl();  for (unsigned int j = 0; j <
      // independent_set.size(); ++j)
      //{
      //    int node_id = independent_set[j];
      //    typename DetailedPlaceDBType::type target_hpwl = 0;
      //    for (int node2pin_id = db.flat_node2pin_start_map[node_id];
      //    node2pin_id < db.flat_node2pin_start_map[node_id+1]; ++node2pin_id)
      //    {
      //        int node_pin_id = db.flat_node2pin_map[node2pin_id];
      //        int net_id = db.pin2net_map[node_pin_id];
      //        if (db.net_mask[net_id])
      //        {
      //            typename DetailedPlaceDBType::type bxl = db.xh;
      //            typename DetailedPlaceDBType::type byl = db.yh;
      //            typename DetailedPlaceDBType::type bxh = db.xl;
      //            typename DetailedPlaceDBType::type byh = db.yl;
      //            for (int net2pin_id = db.flat_net2pin_start_map[net_id];
      //            net2pin_id < db.flat_net2pin_start_map[net_id+1];
      //            ++net2pin_id)
      //            {
      //                int net_pin_id = db.flat_net2pin_map[net2pin_id];
      //                int other_node_id = db.pin2node_map[net_pin_id];
      //                typename DetailedPlaceDBType::type xxl =
      //                db.x[other_node_id]; typename DetailedPlaceDBType::type
      //                yyl = db.y[other_node_id]; bxl = std::min(bxl,
      //                xxl+db.pin_offset_x[net_pin_id]); bxh = std::max(bxh,
      //                xxl+db.pin_offset_x[net_pin_id]); byl = std::min(byl,
      //                yyl+db.pin_offset_y[net_pin_id]); byh = std::max(byh,
      //                yyl+db.pin_offset_y[net_pin_id]);
      //            }
      //            target_hpwl += (bxh-bxl) + (byh-byl);
      //        }
      //    }
      //    dreamplacePrint(kDEBUG, "node %d(%d) target hpwl %g\n", j, node_id,
      //    target_hpwl);
      //}
      // dreamplacePrint(kDEBUG, "original cost %g, target cost %g, delta %g\n",
      //        state.orig_costs[i], state.target_costs[i],
      //        state.target_costs[i]-state.orig_costs[i]);
      // dreamplacePrint(kDEBUG, "%d/%d: original hpwl %g, target hpwl %g, delta
      // %g(%g%%), cumulated delta %g%%, moved %d/%lu, cumulated moved %g%%\n",
      //        state.cur_idx, db.num_movable_nodes,
      //        orig_hpwl, target_hpwl, target_hpwl-orig_hpwl,
      //        (target_hpwl-orig_hpwl)/orig_hpwl*100,
      //        (target_hpwl-initial_hpwl)/initial_hpwl*100,
      //        count, independent_set.size(),
      //        state.num_moved/(double)db.num_movable_nodes*100);
      // dreamplaceAssert(target_hpwl-orig_hpwl ==
      // state.target_costs[i]-state.orig_costs[i]);
      // dreamplaceAssert(state.target_costs[i] <= state.orig_costs[i]);
#endif
  }
}

DREAMPLACE_END_NAMESPACE

#endif
