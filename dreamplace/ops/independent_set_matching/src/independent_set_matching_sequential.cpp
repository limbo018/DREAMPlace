/**
 * @file   independent_set_matching_sequential.cpp
 * @author Yibo Lin
 * @date   Apr 2019
 */
#include <chrono>
#include <limits>

#include "utility/src/torch.h"
#include "utility/src/utils.h"
// database dependency
#include "utility/src/detailed_place_db.h"
#include "utility/src/make_placedb.h"
// local dependency
#include "independent_set_matching/src/auction_cpu.h"
#include "independent_set_matching/src/hungarian_cpu.h"
#include "independent_set_matching/src/min_cost_flow_cpu.h"

//#define DEBUG
//#define DEBUG_PROFILE

//#define LAP_SOLVER HungarianAlgorithmCPULauncher
//#define LAP_SOLVER MinCostFlowCPULauncher
#define LAP_SOLVER AuctionAlgorithmCPULauncher

// local dependency
#include "independent_set_matching/src/bin2node_3d_map.h"
#include "independent_set_matching/src/bin2node_map.h"
#include "independent_set_matching/src/construct_spaces.h"
#include "independent_set_matching/src/cost_matrix_construction.h"
#include "independent_set_matching/src/mark_dependent_nodes.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
struct IndependentSetMatchingState {
  std::vector<int> ordered_nodes;
  std::vector<std::vector<int> > independent_sets;
  std::vector<unsigned char> dependent_markers;
  std::vector<unsigned char> selected_markers;
  std::vector<int> num_selected_markers;
  std::vector<GridIndex<int> > search_grids;
  // std::vector<unsigned char> bin_marker;

  std::vector<std::vector<int> > bin2node_map;  ///< the first dimension is
                                                ///< size, all the cells are
                                                ///< categorized by width
  std::vector<BinMapIndex> node2bin_map;
  std::vector<Space<T> > spaces;

  std::vector<std::vector<int> >
      cost_matrices;  ///< the convergence rate is related to numerical scale
  std::vector<std::vector<int> > solutions;
  std::vector<int> orig_costs;    ///< original cost before matching
  std::vector<int> target_costs;  ///< target cost after matching
  std::vector<std::vector<T> >
      target_pos_x;  ///< temporary storage of cell locations
  std::vector<std::vector<T> > target_pos_y;
  std::vector<std::vector<BinMapIndex> > target_node2bin_map;
  std::vector<std::vector<Space<T> > > target_spaces;

  int batch_size;
  int set_size;
  int grid_size;
  int max_diamond_search_sequence;
  int num_moved;
  T large_number;
};

template <typename DetailedPlaceDBType,
          typename IndependentSetMatchingStateType>
bool collect_independent_sets_sequential(const DetailedPlaceDBType& db,
                                         IndependentSetMatchingStateType& state,
                                         int seed_node,
                                         int i  ///< entry in batch
) {
  auto& independent_set = state.independent_sets[i];
  independent_set.clear();

  typename DetailedPlaceDBType::type seed_height = db.node_size_y[seed_node];
  auto const& seed_bin = state.node2bin_map.at(seed_node);
  int num_bins_x = db.num_bins_x;
  int num_bins_y = db.num_bins_y;
  int seed_bin_x = seed_bin.bin_id / num_bins_y;
  int seed_bin_y = seed_bin.bin_id % num_bins_y;
  // int seed_bin_id = seed_bin_x*num_bins_y + seed_bin_y;
  auto const& bin2node_map = state.bin2node_map;
  // if (state.bin_marker[seed_bin_id])
  //{
  //    return false;
  //}
  // else
  //{
  //    state.bin_marker[seed_bin_id] = 1;
  //}
  for (int j = 0; j < state.max_diamond_search_sequence; ++j) {
    // get bin (bx, by)
    int bx = seed_bin_x + state.search_grids[j].ic;
    int by = seed_bin_y + state.search_grids[j].ir;
    if (bx < 0 || bx >= num_bins_x || by < 0 || by >= num_bins_y) {
      continue;
    }
    int bin_id = bx * num_bins_y + by;
#ifdef DEBUG
    dreamplaceAssert(bin_id < (int)bin2node_map.size());
#endif
    auto const& bin2nodes = bin2node_map.at(bin_id);

    for (auto node_id : bin2nodes) {
#ifdef DEBUG
      dreamplaceAssert(db.node_size_x[node_id] == db.node_size_x[seed_node]);
#endif
      if (db.node_size_y[node_id] == seed_height &&
          !state.dependent_markers[node_id]) {
        independent_set.push_back(node_id);
        mark_dependent_nodes(db, state, node_id, 1);
        state.selected_markers[node_id] = 1;
        state.num_selected_markers[node_id] += 1;
        if (independent_set.size() >= (unsigned int)state.set_size) {
          break;
        }
      }
    }
    if (independent_set.size() >= (unsigned int)state.set_size) {
      break;
    }
  }

  for (auto node_id : independent_set) {
    mark_dependent_nodes(db, state, node_id, 0);
  }

#ifdef DEBUG
  assert(std::count(state.dependent_markers.begin(),
                    state.dependent_markers.end(), 1) == 0);
#endif

  return true;
}

template <typename DetailedPlaceDBType,
          typename IndependentSetMatchingStateType>
void apply_solution_sequential(DetailedPlaceDBType& db,
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
  auto& target_node2bin_map = state.target_node2bin_map.at(i);
  auto& target_spaces = state.target_spaces.at(i);
  solution.resize(independent_set.size());
  target_pos_x.resize(independent_set.size());
  target_pos_y.resize(independent_set.size());
  target_node2bin_map.resize(independent_set.size());
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
        target_node2bin_map[j] = state.node2bin_map[target_node_id];
        target_spaces[j] = state.spaces[target_node_id];
      }
    }
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
#ifdef DEBUG
        int target_pos_id = independent_set.at(sol_j);
        dreamplaceAssert(target_pos_id < db.num_movable_nodes);
#endif
        if (db.x[target_node_id] != target_pos_x[sol_j] ||
            db.y[target_node_id] != target_pos_y[sol_j]) {
          count += 1;
        }
        // update position
        adjust_pos(target_pos_x[sol_j], db.node_size_x[target_node_id],
                   target_spaces[sol_j]);
        db.x[target_node_id] = target_pos_x[sol_j];
        db.y[target_node_id] = target_pos_y[sol_j];
        auto const& bm_idx = target_node2bin_map[sol_j];
        state.bin2node_map.at(bm_idx.bin_id).at(bm_idx.sub_id) = target_node_id;
        state.node2bin_map[target_node_id] = bm_idx;
        state.spaces.at(target_node_id) = target_spaces[sol_j];
      }
    }
    //#pragma omp atomic
    state.num_moved += count;
  }
}

template <typename T>
void independentSetMatchingCPULauncher(DetailedPlaceDB<T> db, int set_size,
                                       int max_iters) {
  // fix random seed
  std::srand(1000);
  const double threshold = 0.00001 / 100;
  IndependentSetMatchingState<T> state;
  state.batch_size = 1;
  state.set_size = set_size;
  state.num_moved = 0;
  state.large_number = (db.xh - db.xl + db.yh - db.yl) * 10;

  make_bin2node_map(db, db.x, db.y, db.node_size_x, db.node_size_y, state);
  construct_spaces(db, db.x, db.y, state.spaces, 1);
#ifdef DEBUG
  for (int node_id = 0; node_id < db.num_movable_nodes; ++node_id) {
    dreamplaceAssert(state.spaces[node_id].xl <= db.x[node_id]);
    dreamplaceAssert(state.spaces[node_id].xh >=
                     db.x[node_id] + db.node_size_x[node_id]);
  }
#endif
  state.grid_size = ceil_power2(std::max(db.num_bins_x, db.num_bins_y) / 8);
  state.max_diamond_search_sequence = state.grid_size * state.grid_size / 2;
  dreamplacePrint(kINFO, "diamond search grid size %d, sequence length %d\n",
                  state.grid_size, state.max_diamond_search_sequence);

  state.ordered_nodes.resize(db.num_movable_nodes);
  std::iota(state.ordered_nodes.begin(), state.ordered_nodes.end(), 0);
  state.independent_sets.resize(state.batch_size,
                                std::vector<int>(state.set_size));
  state.dependent_markers.assign(db.num_nodes, 0);
  state.selected_markers.assign(db.num_movable_nodes, 0);
  state.num_selected_markers.assign(db.num_movable_nodes, 0);
  state.search_grids =
      diamond_search_sequence(state.grid_size, state.grid_size);
  // state.bin_marker.assign(db.num_bins_x*db.num_bins_y, 0);

  state.cost_matrices.resize(state.batch_size);
  state.solutions.resize(state.batch_size);
  state.orig_costs.resize(state.batch_size);
  state.target_costs.resize(state.batch_size);
  state.target_pos_x.resize(state.batch_size);
  state.target_pos_y.resize(state.batch_size);
  state.target_node2bin_map.resize(state.batch_size);
  state.target_spaces.resize(state.batch_size);
  LAP_SOLVER<int> solver;
  bool major = false;  // row major

  // runtime profiling
  CPUTimer::hr_clock_rep iter_timer_start, iter_timer_stop;
  CPUTimer::hr_clock_rep timer_start, timer_stop;
  int random_shuffle_runs = 0, collect_independent_sets_runs = 0,
      cost_matrix_construction_runs = 0, hungarian_runs = 0,
      apply_solution_runs = 0;
  CPUTimer::hr_clock_rep random_shuffle_time = 0,
                         collect_independent_sets_time = 0,
                         cost_matrix_construction_time = 0, hungarian_time = 0,
                         apply_solution_time = 0;
  int num_independent_sets = 0;

  std::vector<T> hpwls(max_iters + 1);
  hpwls[0] = db.compute_total_hpwl();
  dreamplacePrint(kINFO, "initial hpwl %g\n", hpwls[0]);
  for (int iter = 0; iter < max_iters; ++iter) {
    iter_timer_start = CPUTimer::getGlobaltime();

    timer_start = CPUTimer::getGlobaltime();
    if (iter) {
      for (auto& bin2nodes : state.bin2node_map) {
        std::sort(bin2nodes.begin(), bin2nodes.end(),
                  [&](int node_id1, int node_id2) {
                    return state.num_selected_markers[node_id1] <
                           state.num_selected_markers[node_id2];
                  });
        // std::random_shuffle(bin2nodes.begin(), bin2nodes.end());
      }
    }
    std::sort(state.ordered_nodes.begin(), state.ordered_nodes.end(),
              [&](int node_id1, int node_id2) {
                return state.num_selected_markers[node_id1] <
                       state.num_selected_markers[node_id2];
              });
    // std::random_shuffle(state.ordered_nodes.begin(),
    // state.ordered_nodes.end());
    timer_stop = CPUTimer::getGlobaltime();
    random_shuffle_time += timer_stop - timer_start;
    random_shuffle_runs += 1;
    std::fill(state.selected_markers.begin(), state.selected_markers.end(), 0);
    // std::fill(state.bin_marker.begin(), state.bin_marker.end(), 0);

    for (int ii = 0; ii < db.num_movable_nodes; ii += state.batch_size) {
      timer_start = CPUTimer::getGlobaltime();
      num_independent_sets = 0;
      for (int in_batch_id = 0; in_batch_id < state.batch_size; ++in_batch_id) {
        if (ii + in_batch_id < db.num_movable_nodes) {
          int node_id = state.ordered_nodes[ii + in_batch_id];
          if (state.selected_markers[node_id]) {
            continue;
          }
          num_independent_sets += collect_independent_sets_sequential(
              db, state, node_id, in_batch_id);
        }
      }
      timer_stop = CPUTimer::getGlobaltime();
      collect_independent_sets_time += timer_stop - timer_start;
      collect_independent_sets_runs += 1;

      timer_start = CPUTimer::getGlobaltime();
      //#pragma omp parallel for schedule(dynamic, 1)
      for (int i = 0; i < num_independent_sets; ++i) {
        auto const& independent_set = state.independent_sets.at(i);
        auto& cost_matrix = state.cost_matrices.at(i);
        cost_matrix.resize(independent_set.size() * independent_set.size());

        cost_matrix_construction(db, state, major, i);
      }
      timer_stop = CPUTimer::getGlobaltime();
      cost_matrix_construction_time += timer_stop - timer_start;
      cost_matrix_construction_runs += 1;

      timer_start = CPUTimer::getGlobaltime();
      //#pragma omp parallel for schedule(dynamic, 1)
      for (int i = 0; i < num_independent_sets; ++i) {
        auto const& independent_set = state.independent_sets.at(i);
        auto& cost_matrix = state.cost_matrices.at(i);
        auto& solution = state.solutions.at(i);
        auto& orig_cost = state.orig_costs.at(i);
        auto& target_cost = state.target_costs.at(i);
        solution.resize(independent_set.size());

        // solve bipartite assignment problem
        // compute initial cost
        orig_cost = 0;
        for (unsigned int j = 0; j < independent_set.size(); ++j) {
          orig_cost += cost_matrix[j * independent_set.size() + j];
        }
        target_cost = solver.run(cost_matrix.data(), solution.data(),
                                 independent_set.size());
      }
      timer_stop = CPUTimer::getGlobaltime();
      hungarian_time += timer_stop - timer_start;
      hungarian_runs += 1;

      timer_start = CPUTimer::getGlobaltime();
      //#pragma omp parallel for schedule(dynamic, 1)
      for (int i = 0; i < num_independent_sets; ++i) {
        apply_solution_sequential(db, state, i);
      }
      timer_stop = CPUTimer::getGlobaltime();
      apply_solution_time += timer_stop - timer_start;
      apply_solution_runs += 1;

      if ((ii % ((int)ceil(db.num_movable_nodes / 10.0))) == 0) {
        dreamplacePrint(kINFO, "%d%%\n",
                        (int(ii * 100 / db.num_movable_nodes)));
      }
    }

    iter_timer_stop = CPUTimer::getGlobaltime();
    hpwls[iter + 1] = db.compute_total_hpwl();
    dreamplacePrint(
        kINFO,
        "iteration %d, target hpwl %g, delta %g(%g%%), solved %d "
        "sets, moved %g%% cells, runtime %g ms\n",
        iter, hpwls[iter + 1], hpwls[iter + 1] - hpwls[0],
        (hpwls[iter + 1] - hpwls[0]) / hpwls[0] * 100, num_independent_sets,
        state.num_moved / (double)db.num_movable_nodes * 100,
        CPUTimer::getTimerPeriod() * (iter_timer_stop - iter_timer_start));
    dreamplacePrint(
        kDEBUG, "random_shuffle takes %g ms, %d runs, average %g ms\n",
        CPUTimer::getTimerPeriod() * random_shuffle_time, random_shuffle_runs,
        CPUTimer::getTimerPeriod() * random_shuffle_time / random_shuffle_runs);
    dreamplacePrint(
        kDEBUG,
        "collect_independent_sets takes %g ms, %d runs, average %g ms\n",
        CPUTimer::getTimerPeriod() * collect_independent_sets_time,
        collect_independent_sets_runs,
        CPUTimer::getTimerPeriod() * collect_independent_sets_time /
            collect_independent_sets_runs);
    dreamplacePrint(
        kDEBUG,
        "cost_matrix_construction takes %g ms, %d runs, average %g ms\n",
        CPUTimer::getTimerPeriod() * cost_matrix_construction_time,
        cost_matrix_construction_runs,
        CPUTimer::getTimerPeriod() * cost_matrix_construction_time /
            cost_matrix_construction_runs);
    dreamplacePrint(
        kDEBUG, "%s takes %g ms, %d runs, average %g ms\n", solver.name(),
        CPUTimer::getTimerPeriod() * hungarian_time, hungarian_runs,
        CPUTimer::getTimerPeriod() * hungarian_time / hungarian_runs);
    dreamplacePrint(
        kDEBUG, "apply solution takes %g ms, %d runs, average %g ms\n",
        CPUTimer::getTimerPeriod() * apply_solution_time, apply_solution_runs,
        CPUTimer::getTimerPeriod() * apply_solution_time / apply_solution_runs);
    random_shuffle_time = 0;
    random_shuffle_runs = 0;
    collect_independent_sets_time = 0;
    collect_independent_sets_runs = 0;
    cost_matrix_construction_time = 0;
    cost_matrix_construction_runs = 0;
    hungarian_time = 0;
    hungarian_runs = 0;
    apply_solution_time = 0;
    apply_solution_runs = 0;

    if (iter && hpwls[iter] - hpwls[iter + 1] < threshold * hpwls[iter]) {
      break;
    }
  }

  // drawPlaceLauncher<T>(
  //        db.x, db.y,
  //        db.node_size_x, db.node_size_y,
  //        db.pin_offset_x, db.pin_offset_y,
  //        db.pin2node_map,
  //        db.num_nodes,
  //        db.num_movable_nodes,
  //        0,
  //        db.flat_net2pin_start_map[db.num_nets],
  //        db.xl, db.yl, db.xh, db.yh,
  //        db.site_width, db.row_height,
  //        db.bin_size_x, db.bin_size_y,
  //        "final.gds"
  //        );
}

at::Tensor independent_set_matching_forward(
    at::Tensor init_pos, at::Tensor node_size_x, at::Tensor node_size_y,
    at::Tensor flat_region_boxes, at::Tensor flat_region_boxes_start,
    at::Tensor node2fence_region_map, at::Tensor flat_net2pin_map,
    at::Tensor flat_net2pin_start_map, at::Tensor pin2net_map,
    at::Tensor flat_node2pin_map, at::Tensor flat_node2pin_start_map,
    at::Tensor pin2node_map, at::Tensor pin_offset_x, at::Tensor pin_offset_y,
    at::Tensor net_mask, double xl, double yl, double xh, double yh,
    double site_width, double row_height, int num_bins_x, int num_bins_y,
    int num_movable_nodes, int num_terminal_NIs, int num_filler_nodes,
    int batch_size, int set_size, int max_iters) {
  CHECK_FLAT_CPU(init_pos);
  CHECK_EVEN(init_pos);
  CHECK_CONTIGUOUS(init_pos);

  auto pos = init_pos.clone();

  CPUTimer::hr_clock_rep timer_start, timer_stop;

  timer_start = CPUTimer::getGlobaltime();
  // Call the cuda kernel launcher
  DREAMPLACE_DISPATCH_FLOATING_TYPES(
      pos, "independentSetMatchingCPULauncher", [&] {
        auto db = make_placedb<scalar_t>(
            init_pos, pos, node_size_x, node_size_y, flat_region_boxes,
            flat_region_boxes_start, node2fence_region_map, flat_net2pin_map,
            flat_net2pin_start_map, pin2net_map, flat_node2pin_map,
            flat_node2pin_start_map, pin2node_map, pin_offset_x, pin_offset_y,
            net_mask, xl, yl, xh, yh, site_width, row_height, num_bins_x,
            num_bins_y, num_movable_nodes, num_terminal_NIs, num_filler_nodes);
        independentSetMatchingCPULauncher<scalar_t>(db, set_size, max_iters);
      });
  timer_stop = CPUTimer::getGlobaltime();
  dreamplacePrint(kINFO, "Independent set matching sequential takes %g ms\n",
                  (timer_stop - timer_start) * CPUTimer::getTimerPeriod());

  return pos;
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("independent_set_matching",
        &DREAMPLACE_NAMESPACE::independent_set_matching_forward,
        "Independent set matching");
}
