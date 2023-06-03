/**
 * @file   global_swap.cpp
 * @author Yibo Lin
 * @date   Jan 2019
 */
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <vector>
#include "global_swap/src/SwapCandidate.h"
#include "utility/src/torch.h"
#include "utility/src/utils.h"
// database dependency
#include "utility/src/detailed_place_db.h"
#include "utility/src/make_placedb.h"

DREAMPLACE_BEGIN_NAMESPACE

/// @brief global swap algorithm for detailed placement
template <typename T>
int globalSwapCPULauncher(DetailedPlaceDB<T> db, int max_iters) {
  dreamplacePrint(kDEBUG, "%dx%d bins, bin size %g x %g\n", db.num_bins_x,
                  db.num_bins_y, db.bin_size_x, db.bin_size_y);

  auto compute_pair_hpwl = [&](int node_id, T node_xl, T node_yl,
                               int target_node_id, T target_node_xl,
                               T target_node_yl) {
    T cost = 0;
    for (int node2pin_id = db.flat_node2pin_start_map[node_id];
         node2pin_id < db.flat_node2pin_start_map[node_id + 1]; ++node2pin_id) {
      int node_pin_id = db.flat_node2pin_map[node2pin_id];
      int net_id = db.pin2net_map[node_pin_id];
      if (db.net_mask[net_id]) {
        Box<T> box(db.xh, db.yh, db.xl, db.yl);
        for (int net2pin_id = db.flat_net2pin_start_map[net_id];
             net2pin_id < db.flat_net2pin_start_map[net_id + 1]; ++net2pin_id) {
          int net_pin_id = db.flat_net2pin_map[net2pin_id];
          int other_node_id = db.pin2node_map[net_pin_id];
          if (other_node_id == node_id) {
            box.xl = std::min(box.xl, node_xl + db.pin_offset_x[net_pin_id]);
            box.xh = std::max(box.xh, node_xl + db.pin_offset_x[net_pin_id]);
            box.yl = std::min(box.yl, node_yl + db.pin_offset_y[net_pin_id]);
            box.yh = std::max(box.yh, node_yl + db.pin_offset_y[net_pin_id]);
          } else if (other_node_id == target_node_id) {
            box.xl =
                std::min(box.xl, target_node_xl + db.pin_offset_x[net_pin_id]);
            box.xh =
                std::max(box.xh, target_node_xl + db.pin_offset_x[net_pin_id]);
            box.yl =
                std::min(box.yl, target_node_yl + db.pin_offset_y[net_pin_id]);
            box.yh =
                std::max(box.yh, target_node_yl + db.pin_offset_y[net_pin_id]);
          } else {
            box.xl = std::min(
                box.xl, db.x[other_node_id] + db.pin_offset_x[net_pin_id]);
            box.xh = std::max(
                box.xh, db.x[other_node_id] + db.pin_offset_x[net_pin_id]);
            box.yl = std::min(
                box.yl, db.y[other_node_id] + db.pin_offset_y[net_pin_id]);
            box.yh = std::max(
                box.yh, db.y[other_node_id] + db.pin_offset_y[net_pin_id]);
          }
        }
        T hpwl = box.xh - box.xl + box.yh - box.yl;
        cost += hpwl;
      }
    }
    for (int node2pin_id = db.flat_node2pin_start_map[target_node_id];
         node2pin_id < db.flat_node2pin_start_map[target_node_id + 1];
         ++node2pin_id) {
      int node_pin_id = db.flat_node2pin_map[node2pin_id];
      int net_id = db.pin2net_map[node_pin_id];
      if (db.net_mask[net_id]) {
        Box<T> box(db.xh, db.yh, db.xl, db.yl);
        // when encounter nets that have both node_id and target_node_id
        // skip them
        bool duplicate_net_flag = false;
        for (int net2pin_id = db.flat_net2pin_start_map[net_id];
             net2pin_id < db.flat_net2pin_start_map[net_id + 1]; ++net2pin_id) {
          int net_pin_id = db.flat_net2pin_map[net2pin_id];
          int other_node_id = db.pin2node_map[net_pin_id];
          if (other_node_id == node_id) {
            // skip them
            duplicate_net_flag = true;
            break;
            // box.xl = std::min(box.xl, node_xl+db.pin_offset_x[net_pin_id]);
            // box.xh = std::max(box.xh, node_xl+db.pin_offset_x[net_pin_id]);
            // box.yl = std::min(box.yl, node_yl+db.pin_offset_y[net_pin_id]);
            // box.yh = std::max(box.yh, node_yl+db.pin_offset_y[net_pin_id]);
          } else if (other_node_id == target_node_id) {
            box.xl =
                std::min(box.xl, target_node_xl + db.pin_offset_x[net_pin_id]);
            box.xh =
                std::max(box.xh, target_node_xl + db.pin_offset_x[net_pin_id]);
            box.yl =
                std::min(box.yl, target_node_yl + db.pin_offset_y[net_pin_id]);
            box.yh =
                std::max(box.yh, target_node_yl + db.pin_offset_y[net_pin_id]);
          } else {
            box.xl = std::min(
                box.xl, db.x[other_node_id] + db.pin_offset_x[net_pin_id]);
            box.xh = std::max(
                box.xh, db.x[other_node_id] + db.pin_offset_x[net_pin_id]);
            box.yl = std::min(
                box.yl, db.y[other_node_id] + db.pin_offset_y[net_pin_id]);
            box.yh = std::max(
                box.yh, db.y[other_node_id] + db.pin_offset_y[net_pin_id]);
          }
        }
        if (duplicate_net_flag) {
          continue;
        }
        T hpwl = box.xh - box.xl + box.yh - box.yl;
        cost += hpwl;
      }
    }
    return cost;
  };

  // divide layout into rows
  // distribute cells into them
  std::vector<std::vector<int> > row2node_map(db.num_sites_y);
  // map node index to its location in row2node_map
  // we can compute the rows, so only the index within a row of row2node_map is
  // stored
  std::vector<int> node2row2node_index_map(db.num_nodes);

  // distribute cells to rows
  db.make_row2node_map(db.x, db.y, row2node_map, 1);

  // set node2row2node_index_map
  for (int i = 0; i < db.num_sites_y; ++i) {
    for (unsigned int j = 0; j < row2node_map[i].size(); ++j) {
      node2row2node_index_map[row2node_map[i][j]] = j;
    }
  }

#ifdef DEBUG
  // debug
  // check row2node_map
  for (unsigned int i = 0; i < row2node_map.size(); ++i) {
    for (unsigned int j = 1; j < row2node_map[i].size(); ++j) {
      assert(y[row2node_map.at(i).at(j - 1)] == y[row2node_map.at(i).at(j)] &&
             x[row2node_map.at(i).at(j - 1)] +
                     node_size_x[row2node_map.at(i).at(j - 1)] <=
                 x[row2node_map.at(i).at(j)]);
      assert(node2row2node_index_map.at(row2node_map.at(i).at(j)) == j);
    }
  }
  dreamplacePrint(kDEBUG, "passed row2node_map check\n");
#endif

  auto compute_cost = [&](int node_id, T& node_xl, T& node_yl,
                          int target_node_id, T& target_node_xl,
                          T& target_node_yl) {
    // case I: two cells are horizontally abutting
    int row_id = db.pos2site_y(db.y[node_id]);
    int target_row_id = db.pos2site_y(db.y[target_node_id]);
    if (row_id == target_row_id &&
        (db.x[node_id] + db.node_size_x[node_id] == db.x[target_node_id] ||
         db.x[target_node_id] + db.node_size_x[target_node_id] ==
             db.x[node_id])) {
      if (db.x[node_id] < db.x[target_node_id]) {
        node_xl = db.x[target_node_id] + db.node_size_x[target_node_id] -
                  db.node_size_x[node_id];
        target_node_xl = db.x[node_id];
      } else {
        node_xl = db.x[target_node_id];
        target_node_xl = db.x[node_id] + db.node_size_x[node_id] -
                         db.node_size_x[target_node_id];
      }
    } else  // case II: not abutting
    {
      node_xl = db.x[target_node_id] + db.node_size_x[target_node_id] / 2 -
                db.node_size_x[node_id] / 2;
      target_node_xl = db.x[node_id] + db.node_size_x[node_id] / 2 -
                       db.node_size_x[target_node_id] / 2;
      node_xl = db.align2site(node_xl);
      target_node_xl = db.align2site(target_node_xl);
      int node2row2node_index = node2row2node_index_map[node_id];
      T space_xl =
          (node2row2node_index > 0)
              ? std::max(
                    db.xl,
                    db.x[row2node_map[row_id][node2row2node_index - 1]] +
                        db.node_size_x[row2node_map[row_id]
                                                   [node2row2node_index - 1]])
              : db.xl;
      T space_xh =
          (node2row2node_index + 1 < (int)row2node_map[row_id].size())
              ? std::min(db.xh,
                         db.x[row2node_map[row_id][node2row2node_index + 1]])
              : db.xh;
      if (space_xh - space_xl < db.node_size_x[target_node_id]) {
        return (db.xh - db.xl) + (db.yh - db.yl);  // some large number
      }
      int target_node2row2node_index = node2row2node_index_map[target_node_id];
      T target_space_xl =
          (target_node2row2node_index > 0)
              ? std::max(
                    db.xl,
                    db.x[row2node_map[target_row_id]
                                     [target_node2row2node_index - 1]] +
                        db.node_size_x[row2node_map[target_row_id]
                                                   [target_node2row2node_index -
                                                    1]])
              : db.xl;
      T target_space_xh =
          (target_node2row2node_index + 1 <
           (int)row2node_map[target_row_id].size())
              ? std::min(db.xh,
                         db.x[row2node_map[target_row_id]
                                          [target_node2row2node_index + 1]])
              : db.xh;
      if (target_space_xh - target_space_xl < db.node_size_x[node_id]) {
        return (db.xh - db.xl) + (db.yh - db.yl);  // some large number
      }
      node_xl = std::min(std::max(node_xl, target_space_xl),
                         target_space_xh - db.node_size_x[node_id]);
      target_node_xl = std::min(std::max(target_node_xl, space_xl),
                                space_xh - db.node_size_x[target_node_id]);
    }
    node_yl = db.y[target_node_id];
    target_node_yl = db.y[node_id];
    T cost = 0;
    // consider FENCE region
    if (db.num_regions &&
        (!db.inside_fence(node_id, node_xl, node_yl) ||
         !db.inside_fence(target_node_id, target_node_xl, target_node_yl))) {
      cost = std::numeric_limits<T>::max();
    } else {
      T orig_cost = compute_pair_hpwl(node_id, db.x[node_id], db.y[node_id],
                                      target_node_id, db.x[target_node_id],
                                      db.y[target_node_id]);
      T target_cost =
          compute_pair_hpwl(node_id, node_xl, node_yl, target_node_id,
                            target_node_xl, target_node_yl);
      cost = target_cost - orig_cost;
    }
    return cost;
  };

  // nodes
  std::vector<int> ordered_nodes(db.num_movable_nodes);
  std::iota(ordered_nodes.begin(), ordered_nodes.end(), 0);
  // fix random seed
  std::srand(1000);
  // generate bin search sequence
  int grid_size = 10;  // width and height of the diamond shape
  const float stop_threshold = 0.1 / 100;
  std::vector<GridIndex<int> > search_grids(
      diamond_search_sequence(grid_size, grid_size));
  // swap candidates
  std::vector<SwapCandidate<T> > candidates;
  // optimal region
  // std::vector<Box<T> > optimal_regions (db.num_movable_nodes);

  // profiling variables
  CPUTimer::hr_clock_rep timer_start, timer_stop;
  CPUTimer::hr_clock_rep compute_search_region_time = 0;
  int compute_search_region_runs = 0;
  CPUTimer::hr_clock_rep compute_cost_time = 0;
  int compute_cost_runs = 0;
  CPUTimer::hr_clock_rep collect_candidates_time = 0;
  int collect_candidates_runs = 0;
  CPUTimer::hr_clock_rep find_best_candidate_time = 0;
  int find_best_candidate_runs = 0;
  CPUTimer::hr_clock_rep apply_solution_time = 0;
  int apply_solution_runs = 0;
  CPUTimer::hr_clock_rep iter_time_start, iter_time_stop;

  // count number of movement
  int num_moved = 0;
  std::vector<T> hpwls(max_iters + 1);
  hpwls[0] = db.compute_total_hpwl();
  dreamplacePrint(kINFO, "initial hpwl = %.3f\n", hpwls[0]);

  for (int iter = 0; iter < max_iters; ++iter) {
    iter_time_start = CPUTimer::getGlobaltime();
    num_moved = 0;
    // for (unsigned int i = 0; i < ordered_nodes.size(); ++i)
    //{
    //    optimal_regions[i] = db.compute_optimal_region(i);
    //}
    // std::random_shuffle(ordered_nodes.begin(), ordered_nodes.end());
    for (unsigned int i = 0; i < ordered_nodes.size(); ++i) {
      int node_id = ordered_nodes[i];
      // do not consider multi-row height cells yet
      if (db.node_size_y[node_id] != db.row_height) {
        continue;
      }
      Box<T> node_box(db.x[node_id], db.y[node_id],
                      db.x[node_id] + db.node_size_x[node_id],
                      db.y[node_id] + db.node_size_y[node_id]);
      // dreamplacePrint(kDEBUG, "iter %d, node %d\n", iter, node_id);

      // compute optimal region
      timer_start = CPUTimer::getGlobaltime();
      // Box<T> opt_box = (iter&1)? node_box : optimal_regions[node_id];
      Box<T> opt_box = node_box;
      timer_stop = CPUTimer::getGlobaltime();
      compute_search_region_time += timer_stop - timer_start;
      compute_search_region_runs += 1;
      // Box<T> opt_box = node_box;
      // cell already in optimal region, skip it
      // if (opt_box.contains(node_box.xl, node_box.yl, node_box.xh,
      // node_box.yh))
      //{
      //    continue;
      //}
      // extend optimal region
      // opt_box.encompass(node_box.xl, node_box.yl, node_box.xh, node_box.yh);
      // set opt_box to node_box
      // opt_box = node_box;
      int opt_center_bin_x = db.pos2bin_x(opt_box.center_x());
      int opt_center_bin_y = db.pos2bin_y(opt_box.center_y());
      int node_bin_x = (node_box.center_x() < opt_box.center_x())
                           ? db.pos2bin_x(node_box.xl)
                           : db.pos2bin_x(node_box.xh);
      int node_bin_y = (node_box.center_y() < opt_box.center_y())
                           ? db.pos2bin_y(node_box.yl)
                           : db.pos2bin_y(node_box.yh);
      int distance = std::abs(node_bin_x - opt_center_bin_x) +
                     std::abs(node_bin_y - opt_center_bin_y);
      unsigned int max_diamond_search_sequence =
          std::min((std::size_t)(distance * distance * 2), search_grids.size());

      SwapCandidate<T> best_cand;
      best_cand.node_id = node_id;
      best_cand.cost = 0;
      for (unsigned int j = 0; j < max_diamond_search_sequence; ++j) {
        // get bin (bx, by)
        int bx = opt_center_bin_x + search_grids[j].ic;
        int by = opt_center_bin_y + search_grids[j].ir;
        if (bx < 0 || bx >= db.num_bins_x || by < 0 || by >= db.num_bins_y) {
          continue;
        }
        Box<T> bin(db.xl + bx * db.bin_size_x, db.yl + by * db.bin_size_y,
                   db.xl + (bx + 1) * db.bin_size_x,
                   db.yl + (by + 1) * db.bin_size_y);
        // dreamplacePrint(kDEBUG, "node %d search bin (%d, %d) distance to opt
        // box %g/%g\n", node_id, bx, by, bin.center_distance(opt_box),
        // node_box.center_distance(opt_box));

        Box<int> sitebox = db.box2sitebox(bin);

        // enumerate sites within the site box and check any space that is large
        // enough to host the node
        candidates.clear();
        timer_start = CPUTimer::getGlobaltime();
        for (int sy = sitebox.yl; sy < sitebox.yh; ++sy) {
          std::vector<int>& row2nodes = row2node_map.at(sy);
          int row2node_index_begin = 0;
          // search for the starting cell in the bin
          // by scanning the row
          // binary search
          int low = 0;
          int high = row2nodes.size();
          while (low < high) {
            int mid = (low + high) / 2;
            int mid_node_id = row2nodes[mid];
            T mid_node_xl = db.x[mid_node_id];
            T mid_node_xh = mid_node_xl + db.node_size_x[mid_node_id];
            if (mid_node_xh < bin.xl) {
              low = mid + 1;
            } else {
              high = mid - 1;
            }
          }
          row2node_index_begin = low;
          for (unsigned int k = row2node_index_begin; k < row2nodes.size();
               ++k) {
            int target_node_id = row2nodes[k];
            // space is large enough for target node
            if (target_node_id < db.num_movable_nodes &&
                target_node_id != node_id &&
                db.node_size_y[target_node_id] == db.row_height) {
              SwapCandidate<T> cand;
              cand.node_id = node_id;
              cand.target_node_id = target_node_id;
              candidates.push_back(cand);
            }
            // target space is outside the bin
            if (db.x[target_node_id] > bin.xh) {
              break;
            }
          }
        }
        timer_stop = CPUTimer::getGlobaltime();
        collect_candidates_time += timer_stop - timer_start;
        collect_candidates_runs += 1;
        // dreamplacePrint(kDEBUG, "found %lu candidates\n", candidates.size());
        timer_start = CPUTimer::getGlobaltime();
        //#pragma omp parallel for num_threads (4)
        for (unsigned int k = 0; k < candidates.size(); ++k) {
          auto& cand = candidates[k];
          cand.cost = compute_cost(cand.node_id, cand.node_xl, cand.node_yl,
                                   cand.target_node_id, cand.target_node_xl,
                                   cand.target_node_yl);
        }
        timer_stop = CPUTimer::getGlobaltime();
        compute_cost_time += timer_stop - timer_start;
        compute_cost_runs += 1;
        if (!candidates.empty()) {
          timer_start = CPUTimer::getGlobaltime();
          auto local_best_cand = *std::min_element(
              candidates.begin(), candidates.end(),
              [](const SwapCandidate<T>& cand1, const SwapCandidate<T>& cand2) {
                return cand1.cost < cand2.cost;
              });
          if (local_best_cand.cost < best_cand.cost) {
            best_cand = local_best_cand;
          }
          candidates.clear();
          timer_stop = CPUTimer::getGlobaltime();
          find_best_candidate_time += timer_stop - timer_start;
          find_best_candidate_runs += 1;
        }
        if (best_cand.cost < 0)  // already sorted bins
        {
          break;
        }
      }
      // apply solution
      if (best_cand.cost < 0) {
        timer_start = CPUTimer::getGlobaltime();
        // dreamplacePrint(kDEBUG, "(%g%%) swap node %d and node %d, (%g, %g) =>
        // (%g, %g), (%g, %g) => (%g, %g), best_cost %g\n",
        //        i/(T)db.num_movable_nodes*100, best_cand.node_id,
        //        best_cand.target_node_id, db.x[best_cand.node_id],
        //        db.y[best_cand.node_id], best_cand.node_xl, best_cand.node_yl,
        //        db.x[best_cand.target_node_id],
        //        db.y[best_cand.target_node_id], best_cand.target_node_xl,
        //        best_cand.target_node_yl, best_cand.cost);
        if (db.x[best_cand.node_id] != best_cand.node_xl ||
            db.y[best_cand.node_id] != best_cand.node_yl) {
          ++num_moved;
        }
        int row_id = db.pos2site_y(db.y[best_cand.node_id]);
        int target_node_row_id = db.pos2site_y(db.y[best_cand.target_node_id]);
        db.x[best_cand.node_id] = best_cand.node_xl;
        db.y[best_cand.node_id] = best_cand.node_yl;
        db.x[best_cand.target_node_id] = best_cand.target_node_xl;
        db.y[best_cand.target_node_id] = best_cand.target_node_yl;
        std::swap(
            row2node_map[row_id][node2row2node_index_map[best_cand.node_id]],
            row2node_map[target_node_row_id]
                        [node2row2node_index_map[best_cand.target_node_id]]);
        std::swap(node2row2node_index_map[best_cand.node_id],
                  node2row2node_index_map[best_cand.target_node_id]);
        // T target_hpwl = compute_total_hpwl();
        // dreamplacePrint(kDEBUG, "total hpwl %g, delta %g\n", target_hpwl,
        // target_hpwl-orig_hpwl);
        timer_stop = CPUTimer::getGlobaltime();
        apply_solution_time += timer_stop - timer_start;
        apply_solution_runs += 1;
      }
    }
    iter_time_stop = CPUTimer::getGlobaltime();
    dreamplacePrint(
        kINFO, "Iter %d time (ms) \t %g\n", iter,
        CPUTimer::getTimerPeriod() * (iter_time_stop - iter_time_start));

    hpwls[iter + 1] = db.compute_total_hpwl();
    dreamplacePrint(kINFO, "iteration %d: hpwl %.3f => %.3f (imp. %g%%)\n",
                    iter, hpwls[0], hpwls[iter + 1],
                    (1.0 - hpwls[iter + 1] / (double)hpwls[0]) * 100);

    if ((iter & 1) &&
        hpwls[iter] - hpwls[iter - 1] > -stop_threshold * hpwls[0]) {
      break;
    }
  }

  dreamplacePrint(kINFO, "kernel \t time (ms) \t runs\n");

  dreamplacePrint(kINFO, "compute_search_region \t %g \t %d \t %g\n",
                  CPUTimer::getTimerPeriod() * compute_search_region_time,
                  compute_search_region_runs,
                  CPUTimer::getTimerPeriod() * compute_search_region_time /
                      compute_search_region_runs);
  dreamplacePrint(kINFO, "collect_candidates \t %g \t %d \t %g\n",
                  CPUTimer::getTimerPeriod() * collect_candidates_time,
                  collect_candidates_runs,
                  CPUTimer::getTimerPeriod() * collect_candidates_time /
                      collect_candidates_runs);
  dreamplacePrint(
      kINFO, "compute_cost \t %g \t %d \t %g\n",
      CPUTimer::getTimerPeriod() * compute_cost_time, compute_cost_runs,
      CPUTimer::getTimerPeriod() * compute_cost_time / compute_cost_runs);
  dreamplacePrint(kINFO, "find_best_candidate \t %g \t %d \t %g\n",
                  CPUTimer::getTimerPeriod() * find_best_candidate_time,
                  find_best_candidate_runs,
                  CPUTimer::getTimerPeriod() * find_best_candidate_time /
                      find_best_candidate_runs);
  dreamplacePrint(
      kINFO, "apply_solution \t %g \t %d \t %g\n",
      CPUTimer::getTimerPeriod() * apply_solution_time, apply_solution_runs,
      CPUTimer::getTimerPeriod() * apply_solution_time / apply_solution_runs);

  // db.draw_place("final.gds");

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
    int max_iters) {
  CHECK_FLAT_CPU(init_pos);
  CHECK_EVEN(init_pos);
  CHECK_CONTIGUOUS(init_pos);

  auto pos = init_pos.clone();

  CPUTimer::hr_clock_rep total_time_start, total_time_stop;
  total_time_start = CPUTimer::getGlobaltime();
  // Call the cuda kernel launcher
  DREAMPLACE_DISPATCH_FLOATING_TYPES(pos, "globalSwapCPULauncher", [&] {
    auto db = make_placedb<scalar_t>(
        init_pos, pos, node_size_x, node_size_y, flat_region_boxes,
        flat_region_boxes_start, node2fence_region_map, flat_net2pin_map,
        flat_net2pin_start_map, pin2net_map, flat_node2pin_map,
        flat_node2pin_start_map, pin2node_map, pin_offset_x, pin_offset_y,
        net_mask, xl, yl, xh, yh, site_width, row_height, num_bins_x,
        num_bins_y, num_movable_nodes, num_terminal_NIs, num_filler_nodes);
    globalSwapCPULauncher(db, max_iters);
  });
  total_time_stop = CPUTimer::getGlobaltime();
  dreamplacePrint(
      kINFO, "Global swap sequential takes %g ms\n",
      CPUTimer::getTimerPeriod() * (total_time_stop - total_time_start));

  return pos;
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("global_swap", &DREAMPLACE_NAMESPACE::global_swap_forward,
        "Global swap");
}
