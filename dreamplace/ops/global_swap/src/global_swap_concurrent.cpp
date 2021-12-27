/**
 * @file   global_swap_concurrent.cpp
 * @author Yibo Lin
 * @date   Apr 2019
 */
#include <algorithm>
#include <chrono>
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

template <typename T>
struct SwapCandidate {
  T cost;
  T node_xl[2][2];  ///< [0][] for node, [1][] for target node, [][0] for old,
                    ///< [][1] for new
  T node_yl[2][2];
  int node_id[2];  ///< [0] for node, [1] for target node
};

template <typename T>
struct SwapState {
  std::vector<int> ordered_nodes;

  std::vector<std::vector<int> > row2node_map;
  std::vector<RowMapIndex> node2row_map;

  std::vector<std::vector<int> > bin2node_map;
  std::vector<BinMapIndex> node2bin_map;

  std::vector<int> search_bins;
  int search_bin_strategy;  ///< how to compute search bins for eahc cell: 0 for
                            ///< cell bin, 1 for optimal region

  std::vector<std::vector<SwapCandidate<T> > > candidates;

  std::vector<T> net_hpwls;                 ///< HPWL for each net
  std::vector<unsigned char> node_markers;  ///< markers for cells

  int batch_size;
  int max_num_candidates;
  int max_num_candidates_all;
  int num_threads;
};

template <typename T>
void compute_search_bins(const DetailedPlaceDB<T>& db, SwapState<T>& state,
                         int begin, int end) {
#pragma omp parallel for num_threads(state.num_threads)
  for (int node_id = begin; node_id < end; node_id += 1) {
    // compute optimal region
    Box<T> opt_box = (state.search_bin_strategy)
                         ? db.compute_optimal_region(node_id)
                         : Box<T>(db.x[node_id], db.y[node_id],
                                  db.x[node_id] + db.node_size_x[node_id],
                                  db.y[node_id] + db.node_size_y[node_id]);
    // Box<T> opt_box = Box<T>(db.x[node_id],
    //        db.y[node_id],
    //        db.x[node_id]+db.node_size_x[node_id],
    //        db.y[node_id]+db.node_size_y[node_id]);
    int cx = db.pos2bin_x(opt_box.center_x());
    int cy = db.pos2bin_y(opt_box.center_y());
    state.search_bins[node_id] = cx * db.num_bins_y + cy;
  }
}

template <typename T>
void reset_state(DetailedPlaceDB<T>& db, SwapState<T>& state) {
  state.candidates.resize(state.batch_size);
#pragma omp parallel for num_threads(state.num_threads)
  for (int i = 0; i < state.batch_size; ++i) {
    auto& candidates = state.candidates[i];
    candidates.clear();
    candidates.reserve(state.max_num_candidates);
  }

#ifdef DEBUG
  for (int i = 0; i < db.num_movable_nodes; ++i) {
    dreamplaceAssert(state.node_markers[0] == 0);
  }
#endif
}

template <typename T>
Space<T> get_space(const DetailedPlaceDB<T>& db, const SwapState<T>& state,
                   int node_id) {
  auto const& row_id = state.node2row_map.at(node_id);
  auto const& row2nodes = state.row2node_map.at(row_id.row_id);
  Space<T> space;
  space.xl = db.xl;
  space.xh = db.xh;
  if (row_id.sub_id) {
    int left_node_id = row2nodes[row_id.sub_id - 1];
    space.xl =
        std::max(space.xl, db.x[left_node_id] + db.node_size_x[left_node_id]);
  }
  if (row_id.sub_id + 1 < (int)row2nodes.size()) {
    int right_node_id = row2nodes[row_id.sub_id + 1];
    space.xh = std::min(space.xh, db.x[right_node_id]);
  }
  // align space to sites
  return db.align2site(space);
}

template <typename T>
T compute_positions_hint(const DetailedPlaceDB<T>& db,
                         const SwapState<T>& state, SwapCandidate<T>& cand,
                         T node_xl, T node_yl, T node_width,
                         const Space<T>& space) {
  // case I: two cells are horizontally abutting
  cand.node_xl[0][0] = node_xl;
  cand.node_yl[0][0] = node_yl;
  cand.node_xl[1][0] = db.x[cand.node_id[1]];
  cand.node_yl[1][0] = db.y[cand.node_id[1]];
  T target_node_width = db.node_size_x[cand.node_id[1]];
  auto target_space = get_space(db, state, cand.node_id[1]);
  if (space.xh >= target_space.xl && target_space.xh >= space.xl &&
      cand.node_yl[0][0] == cand.node_yl[1][0])  // case I: abutting, not
                                                 // exactly abutting, there
                                                 // might be space between two
                                                 // cells, this is a generalized
                                                 // case
  {
    if (cand.node_xl[0][0] < cand.node_xl[1][0]) {
      cand.node_xl[0][1] = cand.node_xl[1][0] + target_node_width - node_width;
      cand.node_xl[1][1] = cand.node_xl[0][0];
    } else {
      cand.node_xl[0][1] = cand.node_xl[1][0];
      cand.node_xl[1][1] = cand.node_xl[0][0] + node_width - target_node_width;
    }
  } else  // case II: not abutting
  {
    if (space.xh < target_node_width + space.xl ||
        target_space.xh < node_width + target_space.xl) {
      // some large number
      return std::numeric_limits<T>::max();
    }
    cand.node_xl[0][1] =
        cand.node_xl[1][0] + (target_node_width - node_width) / 2;
    cand.node_xl[1][1] =
        cand.node_xl[0][0] + (node_width - target_node_width) / 2;
    cand.node_xl[0][1] = db.align2site(cand.node_xl[0][1]);
    cand.node_xl[0][1] = std::max(cand.node_xl[0][1], target_space.xl);
    cand.node_xl[0][1] =
        std::min(cand.node_xl[0][1], target_space.xh - node_width);
    cand.node_xl[1][1] = db.align2site(cand.node_xl[1][1]);
    cand.node_xl[1][1] = std::max(cand.node_xl[1][1], space.xl);
    cand.node_xl[1][1] =
        std::min(cand.node_xl[1][1], space.xh - target_node_width);
  }
  cand.node_yl[0][1] = cand.node_yl[1][0];
  cand.node_yl[1][1] = cand.node_yl[0][0];

  return 0;
}

template <typename T>
void collect_candidates(const DetailedPlaceDB<T>& db, SwapState<T>& state,
                        int idx_bgn, int idx_end) {
#pragma omp parallel for num_threads(state.num_threads)
  for (int i = idx_bgn; i < idx_end; ++i) {
    int node_id = state.ordered_nodes.at(i);
    T node_xl = db.x[node_id];
    T node_yl = db.y[node_id];
    T node_width = db.node_size_x[node_id];
    auto space = get_space(db, state, node_id);
    int seed_bin_id = state.search_bins[node_id];
    int bx = seed_bin_id / db.num_bins_y;
    int by = seed_bin_id % db.num_bins_y;
    auto& candidates = state.candidates.at(i - idx_bgn);

    auto collect = [&](int ix, int iy) {
      int bin_id = ix * db.num_bins_y + iy;
      auto const& bin2nodes = state.bin2node_map.at(bin_id);
      int num_nodes_in_bin =
          state.bin2node_map.at(bin_id).size() *
          (db.node_size_y[node_id] ==
           db.row_height);  // only consider single-row height cell
      int iters = std::min(state.max_num_candidates / 5, num_nodes_in_bin);

      for (int j = 0; j < iters; ++j) {
        SwapCandidate<T> cand;
        cand.node_id[0] = node_id;
        cand.node_id[1] = bin2nodes.at(j);
        if (db.node_size_y[cand.node_id[1]] == db.row_height) {
          cand.cost = compute_positions_hint(db, state, cand, node_xl, node_yl,
                                             node_width, space);
          if (cand.cost == 0) {
            candidates.push_back(cand);
          }
        }
      }
    };

    // consider left, right, bottom, top bins
    collect(bx, by);
    if (bx) {
      collect(bx - 1, by);
    }
    if (bx + 1 < db.num_bins_x) {
      collect(bx + 1, by);
    }
    if (by) {
      collect(bx, by - 1);
    }
    if (by + 1 < db.num_bins_y) {
      collect(bx, by + 1);
    }
  }
}

template <typename T>
T compute_pair_hpwl_general(const DetailedPlaceDB<T>& db,
                            const SwapState<T>& state, int node_id, T node_xl,
                            T node_yl, int target_node_id, T target_node_xl,
                            T target_node_yl, int skip_node_id) {
  T cost = 0;
  int node2pin_id = db.flat_node2pin_start_map[node_id];
  const int node2pin_id_end = db.flat_node2pin_start_map[node_id + 1];
  for (; node2pin_id < node2pin_id_end; ++node2pin_id) {
    int node_pin_id = db.flat_node2pin_map[node2pin_id];
    int net_id = db.pin2net_map[node_pin_id];
    if (db.net_mask[net_id]) {
      Box<T> box(db.xh, db.yh, db.xl, db.yl);
      int net2pin_id = db.flat_net2pin_start_map[net_id];
      const int net2pin_id_end = db.flat_net2pin_start_map[net_id + 1];
      for (; net2pin_id < net2pin_id_end; ++net2pin_id) {
        int net_pin_id = db.flat_net2pin_map[net2pin_id];
        int other_node_id = db.pin2node_map[net_pin_id];
        if (other_node_id != skip_node_id) {
          T xxl;
          T yyl;
          if (other_node_id == node_id) {
            xxl = node_xl;
            yyl = node_yl;
          } else if (other_node_id == target_node_id) {
            xxl = target_node_xl;
            yyl = target_node_yl;
          } else {
            xxl = db.x[other_node_id];
            yyl = db.y[other_node_id];
          }
          // xxl+px
          xxl += db.pin_offset_x[net_pin_id];
          // yyl+py
          yyl += db.pin_offset_y[net_pin_id];
          box.xl = std::min(box.xl, xxl);
          box.xh = std::max(box.xh, xxl);
          box.yl = std::min(box.yl, yyl);
          box.yh = std::max(box.yh, yyl);
        }
      }
      cost += (box.xh - box.xl + box.yh - box.yl);
    }
  }
  return cost;
}

template <typename T>
void compute_candidate_cost(const DetailedPlaceDB<T>& db, SwapState<T>& state) {
#pragma omp parallel for num_threads(state.num_threads)
  for (int i = 0; i < state.batch_size; i += 1) {
    auto& candidates = state.candidates.at(i);
    for (unsigned int j = 0; j < candidates.size(); ++j) {
      auto& cand = candidates[j];
      if (cand.node_id[0] < db.num_movable_nodes &&
          cand.node_id[1] < db.num_movable_nodes) {
        // consider FENCE region
        if (db.num_regions &&
            (!db.inside_fence(cand.node_id[0], cand.node_xl[0][1],
                              cand.node_yl[0][1]) ||
             !db.inside_fence(cand.node_id[1], cand.node_xl[1][1],
                              cand.node_yl[1][1]))) {
          cand.cost = std::numeric_limits<T>::max();
        } else {
          cand.cost = -compute_pair_hpwl_general(
              db, state, cand.node_id[0], cand.node_xl[0][0],
              cand.node_yl[0][0], cand.node_id[1], cand.node_xl[1][0],
              cand.node_yl[1][0], std::numeric_limits<int>::max());
          cand.cost -= compute_pair_hpwl_general(
              db, state, cand.node_id[1], cand.node_xl[1][0],
              cand.node_yl[1][0], cand.node_id[0], cand.node_xl[0][0],
              cand.node_yl[0][0], cand.node_id[0]);
          cand.cost += compute_pair_hpwl_general(
              db, state, cand.node_id[0], cand.node_xl[0][1],
              cand.node_yl[0][1], cand.node_id[1], cand.node_xl[1][1],
              cand.node_yl[1][1], std::numeric_limits<int>::max());
          cand.cost += compute_pair_hpwl_general(
              db, state, cand.node_id[1], cand.node_xl[1][1],
              cand.node_yl[1][1], cand.node_id[0], cand.node_xl[0][1],
              cand.node_yl[0][1], cand.node_id[0]);
        }
      }
    }
  }
}

template <typename T>
void reduce_min_2d(const SwapState<T>& state,
                   std::vector<std::vector<SwapCandidate<T> > >& candidates,
                   int batch_size) {
#pragma omp parallel for num_threads(state.num_threads)
  for (int i = 0; i < batch_size; ++i) {
    auto& row_candidates = candidates.at(i);
    for (unsigned int j = 1; j < row_candidates.size(); ++j) {
      if (row_candidates[j].cost < row_candidates[0].cost) {
        row_candidates[0] = row_candidates[j];
      }
    }
    // if (!row_candidates.empty())
    //{
    //    dreamplacePrint(kDEBUG, "best candidate cost %g\n",
    //    (float)row_candidates.at(0).cost);
    //}
  }
}

/// @brief mark a node and as first level connected nodes as dependent
/// only nodes with the same sizes are marked
template <typename DetailedPlaceDBType,
          typename IndependentSetMatchingStateType>
void mark_dependent_nodes(const DetailedPlaceDBType& db,
                          IndependentSetMatchingStateType& state, int node_id,
                          unsigned char value) {
  // in case all nets are masked
  int node2pin_start = db.flat_node2pin_start_map[node_id];
  int node2pin_end = db.flat_node2pin_start_map[node_id + 1];
  for (int node2pin_id = node2pin_start; node2pin_id < node2pin_end;
       ++node2pin_id) {
    int node_pin_id = db.flat_node2pin_map[node2pin_id];
    int net_id = db.pin2net_map[node_pin_id];
    if (db.net_mask[net_id]) {
      int net2pin_start = db.flat_net2pin_start_map[net_id];
      int net2pin_end = db.flat_net2pin_start_map[net_id + 1];
      for (int net2pin_id = net2pin_start; net2pin_id < net2pin_end;
           ++net2pin_id) {
        int net_pin_id = db.flat_net2pin_map[net2pin_id];
        int other_node_id = db.pin2node_map[net_pin_id];
        if (other_node_id <
            db.num_nodes)  // other_node_id may exceed db.num_nodes like IO pins
        {
          state.node_markers[other_node_id] = value;
        }
      }
    }
  }
}

template <typename T>
void apply_candidates(DetailedPlaceDB<T>& db, SwapState<T>& state,
                      int num_candidates) {
  for (int i = 0; i < num_candidates; ++i) {
    auto const& row_candidates = state.candidates.at(i);
    if (row_candidates.empty()) {
      continue;
    }
    auto const& best_cand = row_candidates.at(0);

    if (best_cand.cost < 0 && !(state.node_markers.at(best_cand.node_id[0]) ||
                                state.node_markers.at(best_cand.node_id[1]))) {
      T node_width = db.node_size_x[best_cand.node_id[0]];
      T target_node_width = db.node_size_x[best_cand.node_id[1]];
      Space<T> space = get_space(db, state, best_cand.node_id[0]);
      Space<T> target_space = get_space(db, state, best_cand.node_id[1]);

      // space may no longer be large enough or the previously computed
      // locations may not be correct any more
      if (best_cand.node_xl[0][1] >= target_space.xl &&
          best_cand.node_xl[0][1] + node_width <= target_space.xh &&
          best_cand.node_xl[1][1] >= space.xl &&
          best_cand.node_xl[1][1] + target_node_width <= space.xh) {
        // state.node_markers[best_cand.node_id[0]] = 1;
        // state.node_markers[best_cand.node_id[1]] = 1;
        mark_dependent_nodes(db, state, best_cand.node_id[0], 1);
        mark_dependent_nodes(db, state, best_cand.node_id[1], 1);

#ifdef DEBUG
        dreamplaceAssert(best_cand.node_id[0] < db.num_movable_nodes &&
                         best_cand.node_id[1] < db.num_movable_nodes);
#endif

        BinMapIndex& bin_id = state.node2bin_map.at(best_cand.node_id[0]);
        BinMapIndex& target_bin_id =
            state.node2bin_map.at(best_cand.node_id[1]);
        RowMapIndex& row_id = state.node2row_map.at(best_cand.node_id[0]);
        RowMapIndex& target_row_id =
            state.node2row_map.at(best_cand.node_id[1]);
        auto& row2nodes = state.row2node_map.at(row_id.row_id);
        auto& target_row2nodes = state.row2node_map.at(target_row_id.row_id);
#ifdef DEBUG
        assert(state.bin2node_map(bin_id.bin_id, bin_id.sub_id) ==
               best_cand.node_id[0]);
        assert(state.bin2node_map(target_bin_id.bin_id, target_bin_id.sub_id) ==
               best_cand.node_id[1]);
        assert(db.x[best_cand.node_id[0]] == best_cand.node_xl[0][0]);
        assert(db.y[best_cand.node_id[0]] == best_cand.node_yl[0][0]);
        assert(db.x[best_cand.node_id[1]] == best_cand.node_xl[1][0]);
        assert(db.y[best_cand.node_id[1]] == best_cand.node_yl[1][0]);
#endif
        db.x[best_cand.node_id[0]] = best_cand.node_xl[0][1];
        db.y[best_cand.node_id[0]] = best_cand.node_yl[0][1];
        db.x[best_cand.node_id[1]] = best_cand.node_xl[1][1];
        db.y[best_cand.node_id[1]] = best_cand.node_yl[1][1];
        int& bin2node_map_node_id =
            state.bin2node_map.at(bin_id.bin_id).at(bin_id.sub_id);
        int& bin2node_map_target_node_id =
            state.bin2node_map.at(target_bin_id.bin_id)
                .at(target_bin_id.sub_id);
        std::swap(bin2node_map_node_id, bin2node_map_target_node_id);
        std::swap(bin_id, target_bin_id);

        // update row2node_map and node2row_map
        std::swap(row2nodes[row_id.sub_id],
                  target_row2nodes[target_row_id.sub_id]);
        std::swap(row_id, target_row_id);
      }
    }
  }

  for (int i = 0; i < num_candidates; ++i) {
    auto const& row_candidates = state.candidates.at(i);
    if (row_candidates.empty()) {
      continue;
    }
    auto const& best_cand = row_candidates.at(0);
    mark_dependent_nodes(db, state, best_cand.node_id[0], 0);
    mark_dependent_nodes(db, state, best_cand.node_id[1], 0);
  }

#ifdef DEBUG
  dreamplaceAssert(
      std::count(state.node_markers.begin(), state.node_markers.end(), 1) == 0);
#endif
}

template <typename T>
void check_candidate_costs(const DetailedPlaceDB<T>& db,
                           const SwapState<T>& state) {
  for (int i = 0; i < state.batch_size; ++i) {
    for (auto const& cand : state.candidates.at(i)) {
      if (cand.cost < 0) {
        dreamplaceAssert(cand.node_id[0] < db.num_movable_nodes &&
                         cand.node_id[1] < db.num_movable_nodes);
      }
    }
  }
}
template <typename T>
void global_swap(DetailedPlaceDB<T>& db, SwapState<T>& state) {
  CPUTimer::hr_clock_rep timer_start, timer_stop;
  CPUTimer::hr_clock_rep collect_candidates_time = 0,
                         compute_candidate_cost_time = 0,
                         reduce_min_2d_time = 0, apply_candidates_time = 0;
  int collect_candidates_runs = 0, compute_candidate_cost_runs = 0,
      reduce_min_2d_runs = 0, apply_candidates_runs = 0;

  timer_start = CPUTimer::getGlobaltime();
  compute_search_bins(db, state, 0, db.num_movable_nodes);
  timer_stop = CPUTimer::getGlobaltime();
  dreamplacePrint(kDEBUG, "compute_search_bins takes %g ms\n",
                  (timer_stop - timer_start) * CPUTimer::getTimerPeriod());

  for (int i = 0; i < db.num_movable_nodes; i += state.batch_size) {
    // all results are stored in state.candidates
    int idx_bgn = i;
    int idx_end = std::min(i + state.batch_size, db.num_movable_nodes);
    // dreamplacePrint(kDEBUG, "batch %d - %d\n", idx_bgn, idx_end);

    timer_start = CPUTimer::getGlobaltime();
    reset_state(db, state);

    collect_candidates(db, state, idx_bgn, idx_end);
    timer_stop = CPUTimer::getGlobaltime();
    collect_candidates_time += timer_stop - timer_start;
    collect_candidates_runs += 1;

    timer_start = CPUTimer::getGlobaltime();
    compute_candidate_cost(db, state);
    timer_stop = CPUTimer::getGlobaltime();
    compute_candidate_cost_time += timer_stop - timer_start;
    compute_candidate_cost_runs += 1;

    // check_candidate_costs(db, state);
    timer_start = CPUTimer::getGlobaltime();
    // reduce min and apply
    reduce_min_2d(state, state.candidates, state.batch_size);
    timer_stop = CPUTimer::getGlobaltime();
    reduce_min_2d_time += timer_stop - timer_start;
    reduce_min_2d_runs += 1;

    // check_candidate_costs(db, state);
    timer_start = CPUTimer::getGlobaltime();
    // must use single thread
    apply_candidates(db, state, idx_end - idx_bgn);
    timer_stop = CPUTimer::getGlobaltime();
    apply_candidates_time += timer_stop - timer_start;
    apply_candidates_runs += 1;
  }

  dreamplacePrint(kDEBUG,
                  "collect_candidates takes %g ms for %d runs, average %g ms\n",
                  collect_candidates_time * CPUTimer::getTimerPeriod(),
                  collect_candidates_runs,
                  collect_candidates_time * CPUTimer::getTimerPeriod() /
                      collect_candidates_runs);
  dreamplacePrint(
      kDEBUG, "compute_candidate_cost takes %g ms for %d runs, average %g ms\n",
      compute_candidate_cost_time * CPUTimer::getTimerPeriod(),
      compute_candidate_cost_runs,
      compute_candidate_cost_time * CPUTimer::getTimerPeriod() /
          compute_candidate_cost_runs);
  dreamplacePrint(
      kDEBUG, "reduce_min_2d takes %g ms for %d runs, average %g ms\n",
      reduce_min_2d_time * CPUTimer::getTimerPeriod(), reduce_min_2d_runs,
      reduce_min_2d_time * CPUTimer::getTimerPeriod() / reduce_min_2d_runs);
  dreamplacePrint(
      kDEBUG, "apply_candidates takes %g ms for %d runs, average %g ms\n",
      apply_candidates_time * CPUTimer::getTimerPeriod(), apply_candidates_runs,
      apply_candidates_time * CPUTimer::getTimerPeriod() /
          apply_candidates_runs);
}

template <typename T>
T compute_total_hpwl(const DetailedPlaceDB<T>& db, const SwapState<T>& state,
                     const T* x, const T* y, T* net_hpwls) {
#pragma omp parallel for num_threads(state.num_threads)
  for (int i = 0; i < db.num_nets; ++i) {
    net_hpwls[i] = db.compute_net_hpwl(i);
  }
  T hpwl = 0;
  // I found OpenMP reduction cannot guarantee run-to-run determinism
  //#pragma omp parallel for num_threads(state.num_threads) default(shared)
  // reduction(+:hpwl)
  for (int i = 0; i < db.num_nets; ++i) {
    hpwl += net_hpwls[i];
  }

  return hpwl;
}

/// @brief global swap algorithm for detailed placement
template <typename T>
int globalSwapCPULauncher(DetailedPlaceDB<T> db, int batch_size, int max_iters,
                          int num_threads) {
  dreamplacePrint(kDEBUG, "%dx%d bins, bin size %g x %g\n", db.num_bins_x,
                  db.num_bins_y, db.bin_size_x, db.bin_size_y);

  SwapState<T> state;
  state.num_threads = std::max(num_threads, 1);

  const float stop_threshold = 0.1 / 100;
  state.batch_size = batch_size;
  int max_num_candidates_per_row =
      (2 << (int)log2(
           ceil(sqrt(db.num_nodes / (db.num_bins_x * db.num_bins_y)))));
  state.max_num_candidates =
      (1 << (int)ceil(log2(ceil(db.bin_size_y / db.row_height)))) *
      max_num_candidates_per_row * 5;
  state.max_num_candidates_all = state.batch_size * state.max_num_candidates;
  dreamplacePrint(
      kDEBUG,
      "batch_size = %d, max_num_candidates = %d, max_num_candidates_all = %d\n",
      state.batch_size, state.max_num_candidates, state.max_num_candidates_all);
  state.search_bin_strategy = 1;

  // distribute cells to rows
  state.row2node_map.resize(db.num_sites_y);
  state.node2row_map.resize(db.num_nodes);
  db.make_row2node_map(db.x, db.y, state.row2node_map, num_threads);
  for (int i = 0; i < db.num_sites_y; ++i) {
    for (unsigned int j = 0; j < state.row2node_map[i].size(); ++j) {
      auto& row_id = state.node2row_map.at(state.row2node_map[i][j]);
      row_id.row_id = i;
      row_id.sub_id = j;
    }
  }
  // distribute cells to bin
  state.bin2node_map.resize(db.num_bins_x * db.num_bins_y);
  state.node2bin_map.resize(db.num_movable_nodes);
  db.make_bin2node_map(db.x, db.y, db.node_size_x, db.node_size_y,
                       state.bin2node_map, state.node2bin_map);

  // fix random seed
  std::srand(1000);

  state.ordered_nodes.resize(db.num_movable_nodes);
  std::iota(state.ordered_nodes.begin(), state.ordered_nodes.end(), 0);

  state.candidates.resize(state.batch_size);
  state.search_bins.resize(db.num_movable_nodes);
  state.net_hpwls.resize(db.num_nets);
  state.node_markers.assign(db.num_nodes, 0);

  CPUTimer::hr_clock_rep kernel_time_start, kernel_time_stop;
  CPUTimer::hr_clock_rep iter_time_start, iter_time_stop;

  kernel_time_start = CPUTimer::getGlobaltime();
  std::vector<T> hpwls(max_iters + 1);
  hpwls[0] = compute_total_hpwl(db, state, db.x, db.y, state.net_hpwls.data());
  dreamplacePrint(kINFO, "initial hpwl = %.3f\n", hpwls[0]);
  for (int iter = 0; iter < max_iters; ++iter) {
    iter_time_start = CPUTimer::getGlobaltime();
    std::random_shuffle(state.ordered_nodes.begin(), state.ordered_nodes.end());
    global_swap(db, state);
    iter_time_stop = CPUTimer::getGlobaltime();
    dreamplacePrint(
        kINFO, " Iteration time(ms) \t %g\n",
        CPUTimer::getTimerPeriod() * (iter_time_stop - iter_time_start));

    hpwls[iter + 1] =
        compute_total_hpwl(db, state, db.x, db.y, state.net_hpwls.data());
    dreamplacePrint(kINFO, "iteration %d: hpwl %.3f => %.3f (imp. %g%%)\n",
                    iter, hpwls[0], hpwls[iter + 1],
                    (1.0 - hpwls[iter + 1] / (double)hpwls[0]) * 100);
    state.search_bin_strategy = !state.search_bin_strategy;

    if ((iter & 1) &&
        hpwls[iter] - hpwls[iter - 1] > -stop_threshold * hpwls[0]) {
      break;
    }
  }
  kernel_time_stop = CPUTimer::getGlobaltime();
  dreamplacePrint(
      kINFO, " Global swap time: %g ms\n",
      CPUTimer::getTimerPeriod() * (kernel_time_stop - kernel_time_start));

  return 0;
}

at::Tensor global_swap_forward(
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
  CHECK_FLAT_CPU(init_pos);
  CHECK_EVEN(init_pos);
  CHECK_CONTIGUOUS(init_pos);

  auto pos = init_pos.clone();

  // Call the cuda kernel launcher
  DREAMPLACE_DISPATCH_FLOATING_TYPES(pos, "globalSwapCPULauncher", [&] {
    auto db = make_placedb<scalar_t>(
        init_pos, pos, node_size_x, node_size_y, flat_region_boxes,
        flat_region_boxes_start, node2fence_region_map, flat_net2pin_map,
        flat_net2pin_start_map, pin2net_map, flat_node2pin_map,
        flat_node2pin_start_map, pin2node_map, pin_offset_x, pin_offset_y,
        net_mask, xl, yl, xh, yh, site_width, row_height, num_bins_x,
        num_bins_y, num_movable_nodes, num_terminal_NIs, num_filler_nodes);
    globalSwapCPULauncher(db, batch_size, max_iters, at::get_num_threads());
  });

  return pos;
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("global_swap", &DREAMPLACE_NAMESPACE::global_swap_forward,
        "Global swap concurrent");
}
