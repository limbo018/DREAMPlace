/**
 * @file   DetailedPlaceDB.h
 * @author Yibo Lin
 * @date   Jan 2019
 */

#ifndef _DREAMPLACE_UTILITY_DETAILEDPLACEDB_H
#define _DREAMPLACE_UTILITY_DETAILEDPLACEDB_H

#include "utility/src/utils.h"
// helper functions
#include "draw_place/src/draw_place.h"
#include "legality_check/src/legality_check.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
struct Space {
  T xl;
  T xh;
};

struct BinMapIndex {
  int bin_id;
  int sub_id;
};

struct RowMapIndex {
  int row_id;
  int sub_id;
};

/// @brief a wrapper class of required data for detailed placement
template <typename T>
struct DetailedPlaceDB {
  typedef T type;

  const T* init_x;
  const T* init_y;
  const T* node_size_x;
  const T* node_size_y;
  const T* flat_region_boxes;          ///< number of boxes x 4
  const int* flat_region_boxes_start;  ///< number of regions + 1
  const int* node2fence_region_map;    ///< length of number of movable cells
  T* x;
  T* y;
  const int* flat_net2pin_map;
  const int* flat_net2pin_start_map;
  const int* pin2net_map;
  const int* flat_node2pin_map;
  const int* flat_node2pin_start_map;
  const int* pin2node_map;
  const T* pin_offset_x;
  const T* pin_offset_y;
  const unsigned char* net_mask;
  T xl;
  T yl;
  T xh;
  T yh;
  T site_width;
  T row_height;
  T bin_size_x;
  T bin_size_y;
  int num_bins_x;
  int num_bins_y;
  int num_sites_x;
  int num_sites_y;
  int num_nodes;
  int num_movable_nodes;
  int num_nets;
  int num_pins;
  int num_regions;  ///< number of regions for flat_region_boxes and
                    ///< flat_region_boxes_start

  inline int pos2site_x(T xx) const {
    int sx = floorDiv(xx - xl, site_width);
    sx = std::max(sx, 0);
    sx = std::min(sx, num_sites_x - 1);
    return sx;
  }
  inline int pos2site_y(T yy) const {
    int sy = floorDiv(yy - yl, row_height);
    sy = std::max(sy, 0);
    sy = std::min(sy, num_sites_y - 1);
    return sy;
  }
  /// @brief site index as an upper bound
  inline int pos2site_ub_x(T xx) const {
    int sx = ceilDiv(xx - xl, site_width);
    sx = std::max(sx, 1);
    sx = std::min(sx, num_sites_x);
    return sx;
  }
  /// @brief site index as an upper bound
  inline int pos2site_ub_y(T yy) const {
    int sy = ceilDiv(yy - yl, row_height);
    sy = std::max(sy, 1);
    sy = std::min(sy, num_sites_y);
    return sy;
  }
  inline int pos2bin_x(T xx) const {
    int bx = floorDiv(xx - xl, bin_size_x);
    bx = std::max(bx, 0);
    bx = std::min(bx, num_bins_x - 1);
    return bx;
  }
  inline int pos2bin_y(T yy) const {
    int by = floorDiv(yy - yl, bin_size_y);
    by = std::max(by, 0);
    by = std::min(by, num_bins_y - 1);
    return by;
  }
  inline void shift_box_to_layout(Box<T>& box) const {
    box.xl = std::max(box.xl, xl);
    box.xl = std::min(box.xl, xh);
    box.xh = std::max(box.xh, xl);
    box.xh = std::min(box.xh, xh);
    box.yl = std::max(box.yl, yl);
    box.yl = std::min(box.yl, yh);
    box.yh = std::max(box.yh, yl);
    box.yh = std::min(box.yh, yh);
  }
  inline Box<int> box2sitebox(const Box<T>& box) const {
    // xh, yh are exclusive
    Box<int> sitebox(pos2site_x(box.xl), pos2site_y(box.yl),
                     pos2site_ub_x(box.xh), pos2site_ub_y(box.yh));

    return sitebox;
  }
  inline Box<int> box2binbox(const Box<T>& box) const {
    Box<int> binbox(pos2bin_x(box.xl), pos2bin_y(box.yl), pos2bin_x(box.xh),
                    pos2bin_y(box.yh));

    return binbox;
  }
  /// @brief align x coordinate to site
  inline T align2site(T xx) const {
    return floorDiv(xx - xl, site_width) * site_width + xl;
  }
  /// @brief align x coordinate to site for a space;
  /// make sure the space is shrinked.
  inline Space<T> align2site(Space<T> space) const {
    space.xl = ceilDiv(space.xl - xl, site_width) * site_width + xl;
    space.xh = floorDiv(space.xh - xl, site_width) * site_width + xl;
    return space;
  }
  /// @brief compute optimal region for a cell
  /// The method to compute optimal region ignores the pin offsets of the target
  /// cell. If we want to consider the pin offsets, there may not be feasible
  /// box for the optimal region. Thus, this is just an approximate optimal
  /// region. When using the optimal region, one needs to refer to the center of
  /// the cell to the region, or the region completely covers the entire cell.
  Box<T> compute_optimal_region(int node_id) const {
    Box<T> box(std::numeric_limits<T>::max(), std::numeric_limits<T>::max(),
               -std::numeric_limits<T>::max(), -std::numeric_limits<T>::max());
    for (int node2pin_id = flat_node2pin_start_map[node_id];
         node2pin_id < flat_node2pin_start_map[node_id + 1]; ++node2pin_id) {
      int node_pin_id = flat_node2pin_map[node2pin_id];
      int net_id = pin2net_map[node_pin_id];
      if (net_mask[net_id]) {
        for (int net2pin_id = flat_net2pin_start_map[net_id];
             net2pin_id < flat_net2pin_start_map[net_id + 1]; ++net2pin_id) {
          int net_pin_id = flat_net2pin_map[net2pin_id];
          int other_node_id = pin2node_map[net_pin_id];
          if (node_id != other_node_id) {
            box.xl =
                std::min(box.xl, x[other_node_id] + pin_offset_x[net_pin_id]);
            box.xh =
                std::max(box.xh, x[other_node_id] + pin_offset_x[net_pin_id]);
            box.yl =
                std::min(box.yl, y[other_node_id] + pin_offset_y[net_pin_id]);
            box.yh =
                std::max(box.yh, y[other_node_id] + pin_offset_y[net_pin_id]);
          }
        }
      }
    }
    shift_box_to_layout(box);

    return box;
  }
  /// @brief compute HPWL for a net
  T compute_net_hpwl(int net_id) const {
    Box<T> box(std::numeric_limits<T>::max(), std::numeric_limits<T>::max(),
               -std::numeric_limits<T>::max(), -std::numeric_limits<T>::max());
    for (int net2pin_id = flat_net2pin_start_map[net_id];
         net2pin_id < flat_net2pin_start_map[net_id + 1]; ++net2pin_id) {
      int net_pin_id = flat_net2pin_map[net2pin_id];
      int other_node_id = pin2node_map[net_pin_id];
      box.xl = std::min(box.xl, x[other_node_id] + pin_offset_x[net_pin_id]);
      box.xh = std::max(box.xh, x[other_node_id] + pin_offset_x[net_pin_id]);
      box.yl = std::min(box.yl, y[other_node_id] + pin_offset_y[net_pin_id]);
      box.yh = std::max(box.yh, y[other_node_id] + pin_offset_y[net_pin_id]);
    }
    if (box.xl == std::numeric_limits<T>::max() ||
        box.yl == std::numeric_limits<T>::max()) {
      return (T)0;
    }
    return (box.xh - box.xl) + (box.yh - box.yl);
  }
  /// @brief compute HPWL for all nets
  T compute_total_hpwl() const {
    // dreamplacePrint(kDEBUG, "start compute_total_hpwl\n");
    T total_hpwl = 0;
    for (int net_id = 0; net_id < num_nets; ++net_id) {
      // if (net_mask[net_id])
      { total_hpwl += compute_net_hpwl(net_id); }
    }
    // dreamplacePrint(kDEBUG, "end compute_total_hpwl\n");
    return total_hpwl;
  }
  /// @brief distribute cells to rows
  void make_row2node_map(const T* vx, const T* vy,
                         std::vector<std::vector<int> >& row2node_map,
                         int num_threads) const {
    // distribute cells to rows
    for (int i = 0; i < num_nodes; ++i) {
      // T node_xl = vx[i];
      T node_yl = vy[i];
      // T node_xh = node_xl+node_size_x[i];
      T node_yh = node_yl + node_size_y[i];

      int row_idxl = floorDiv(node_yl - yl, row_height);
      int row_idxh = ceilDiv(node_yh - yl, row_height);
      row_idxl = std::max(row_idxl, 0);
      row_idxh = std::min(row_idxh, num_sites_y);

      for (int row_id = row_idxl; row_id < row_idxh; ++row_id) {
        T row_yl = yl + row_id * row_height;
        T row_yh = row_yl + row_height;

        if (node_yl < row_yh && node_yh > row_yl)  // overlap with row
        {
          row2node_map[row_id].push_back(i);
        }
      }
    }

      // sort cells within rows
      // it is safer to sort by center
      // sometimes there might be cells with 0 sizes
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads) schedule(dynamic, 1)
#endif
    for (int i = 0; i < num_sites_y; ++i) {
      auto& row2nodes = row2node_map[i];
      // sort cells within rows according to left edges
      std::sort(row2nodes.begin(), row2nodes.end(),
                [&](int node_id1, int node_id2) {
                  T x1 = vx[node_id1];
                  T x2 = vx[node_id2];
                  return x1 < x2 || (x1 == x2 && node_id1 < node_id2);
                });
      // After sorting by left edge,
      // there is a special case for fixed cells where
      // one fixed cell is completely within another in a row.
      // This will cause failure to detect some overlaps.
      // We need to remove the "small" fixed cell that is inside another.
      if (!row2nodes.empty()) {
        std::vector<int> tmp_nodes;
        tmp_nodes.reserve(row2nodes.size());
        tmp_nodes.push_back(row2nodes.front());
        for (int j = 1, je = row2nodes.size(); j < je; ++j) {
          int node_id1 = row2nodes.at(j - 1);
          int node_id2 = row2nodes.at(j);
          // two fixed cells
          if (node_id1 >= num_movable_nodes && node_id2 >= num_movable_nodes) {
            T xl1 = vx[node_id1];
            T xl2 = vx[node_id2];
            T width1 = node_size_x[node_id1];
            T width2 = node_size_x[node_id2];
            T xh1 = xl1 + width1;
            T xh2 = xl2 + width2;
            // only collect node_id2 if its right edge is righter than node_id1
            if (xh1 < xh2) {
              tmp_nodes.push_back(node_id2);
            }
          } else {
            tmp_nodes.push_back(node_id2);
          }
        }
        row2nodes.swap(tmp_nodes);

        // sort according to center
        std::sort(row2nodes.begin(), row2nodes.end(),
                  [&](int node_id1, int node_id2) {
                    T x1 = vx[node_id1] + node_size_x[node_id1] / 2;
                    T x2 = vx[node_id2] + node_size_x[node_id2] / 2;
                    return x1 < x2 || (x1 == x2 && node_id1 < node_id2);
                  });
      }
    }
  }
  /// @brief distribute movable cells to bins
  void make_bin2node_map(const T* host_x, const T* host_y,
                         const T* host_node_size_x, const T* host_node_size_y,
                         std::vector<std::vector<int> >& bin2node_map,
                         std::vector<BinMapIndex>& node2bin_map) const {
    // construct bin2node_map
    for (int i = 0; i < num_movable_nodes; ++i) {
      int node_id = i;
      T node_x = host_x[node_id] + host_node_size_x[node_id] / 2;
      T node_y = host_y[node_id] + host_node_size_y[node_id] / 2;

      int bx = std::min(std::max((int)floorDiv(node_x - xl, bin_size_x), 0),
                        num_bins_x - 1);
      int by = std::min(std::max((int)floorDiv(node_y - yl, bin_size_y), 0),
                        num_bins_y - 1);
      int bin_id = bx * num_bins_y + by;
      // int sub_id = bin2node_map.at(bin_id).size();
      bin2node_map.at(bin_id).push_back(node_id);
    }
    // construct node2bin_map
    for (unsigned int bin_id = 0; bin_id < bin2node_map.size(); ++bin_id) {
      for (unsigned int sub_id = 0; sub_id < bin2node_map[bin_id].size();
           ++sub_id) {
        int node_id = bin2node_map[bin_id][sub_id];
        BinMapIndex& bm_idx = node2bin_map.at(node_id);
        bm_idx.bin_id = bin_id;
        bm_idx.sub_id = sub_id;
      }
    }
#ifdef DEBUG
    int max_num_nodes_per_bin = 0;
    for (unsigned int i = 0; i < bin2node_map.size(); ++i) {
      max_num_nodes_per_bin =
          std::max(max_num_nodes_per_bin, (int)bin2node_map[i].size());
    }
    printf("[D] max_num_nodes_per_bin = %d\n", max_num_nodes_per_bin);
#endif
  }
  /// @brief check whether placement is legal
  bool check_legality() const {
    return legalityCheckKernelCPU(
        x, y, node_size_x, node_size_y, flat_region_boxes,
        flat_region_boxes_start, node2fence_region_map, xl, yl, xh, yh,
        site_width, row_height, num_nodes, num_movable_nodes, num_regions);
  }
  /// @brief check whether a cell is within its fence region
  bool inside_fence(int node_id, T xx, T yy) const {
    T node_xl = xx;
    T node_yl = yy;
    T node_xh = node_xl + node_size_x[node_id];
    T node_yh = node_yl + node_size_y[node_id];

    bool legal_flag = true;
    int region_id = node2fence_region_map[node_id];
    if (region_id < num_regions) {
      int box_bgn = flat_region_boxes_start[region_id];
      int box_end = flat_region_boxes_start[region_id + 1];
      T node_area = (node_xh - node_xl) * (node_yh - node_yl);
      // I assume there is no overlap between boxes of a region
      // otherwise, preprocessing is required
      for (int box_id = box_bgn; box_id < box_end; ++box_id) {
        int box_offset = box_id * 4;
        T box_xl = flat_region_boxes[box_offset];
        T box_yl = flat_region_boxes[box_offset + 1];
        T box_xh = flat_region_boxes[box_offset + 2];
        T box_yh = flat_region_boxes[box_offset + 3];

        T dx = std::max(std::min(node_xh, box_xh) - std::max(node_xl, box_xl),
                        (T)0);
        T dy = std::max(std::min(node_yh, box_yh) - std::max(node_yl, box_yl),
                        (T)0);
        T overlap = dx * dy;
        if (overlap > 0) {
          node_area -= overlap;
        }
      }
      if (node_area > 0)  // not consumed by boxes within a region
      {
        legal_flag = false;
      }
    }
    return legal_flag;
  }
  /// @brief draw placement
  void draw_place(const char* filename) const {
    drawPlaceLauncher<T>(x, y, node_size_x, node_size_y, pin_offset_x,
                         pin_offset_y, pin2node_map, num_nodes,
                         num_movable_nodes, 0, flat_net2pin_start_map[num_nets],
                         xl, yl, xh, yh, site_width, row_height, bin_size_x,
                         bin_size_y, filename);
  }
};

DREAMPLACE_END_NAMESPACE

#endif
