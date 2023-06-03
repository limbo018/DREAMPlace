/**
 * @file   abacus_legalize_cpu.h
 * @author Yibo Lin
 * @date   Nov 2019
 */

#ifndef DREAMPLACEPLACE_ABACUS_LEGALIZE_CPU_H
#define DREAMPLACEPLACE_ABACUS_LEGALIZE_CPU_H

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <algorithm>
#include <iostream>
#include <vector>
#include "utility/src/utils.h"

DREAMPLACE_BEGIN_NAMESPACE

/// @brief helper function for distributing cells to rows 
/// sort cells within a row and clean overlapping fixed cells 
template <typename T>
void sortNodesInRow(const T* host_x, const T* host_y, 
    const T* host_node_size_x, const T* host_node_size_y, int num_movable_nodes, 
    std::vector<int>& nodes_in_row) {
  // sort cells within rows according to left edges 
  std::sort(nodes_in_row.begin(), nodes_in_row.end(), 
      [&] (int node_id1, int node_id2) {
      T x1 = host_x[node_id1];
      T x2 = host_x[node_id2];
      // put larger width front will help remove 
      // overlapping fixed cells, especially when 
      // x1 == x2, then we need the wider one comes first
      T w1 = host_node_size_x[node_id1]; 
      T w2 = host_node_size_x[node_id2]; 
      return x1 < x2 || (x1 == x2 && (w1 > w2 || (w1 == w2 && node_id1 < node_id2)));
      });
  // After sorting by left edge, 
  // there is a special case for fixed cells where 
  // one fixed cell is completely within another in a row. 
  // This will cause failure to detect some overlaps. 
  // We need to remove the "small" fixed cell that is inside another. 
  if (!nodes_in_row.empty()) {
    std::vector<int> tmp_nodes; 
    tmp_nodes.reserve(nodes_in_row.size());
    tmp_nodes.push_back(nodes_in_row.front()); 
    int j_1 = 0; 
    for (int j = 1, je = nodes_in_row.size(); j < je; ++j) {
      int node_id1 = nodes_in_row.at(j_1);
      int node_id2 = nodes_in_row.at(j);
      // two fixed cells 
      if (node_id1 >= num_movable_nodes && node_id2 >= num_movable_nodes) {
        T xl1 = host_x[node_id1]; 
        T xl2 = host_x[node_id2];
        T width1 = host_node_size_x[node_id1]; 
        T width2 = host_node_size_x[node_id2]; 
        T xh1 = xl1 + width1; 
        T xh2 = xl2 + width2; 
        // only collect node_id2 if its right edge is righter than node_id1 
        if (xh1 < xh2) {
          tmp_nodes.push_back(node_id2);
          j_1 = j; 
        }
      } else {
        tmp_nodes.push_back(node_id2);
        j_1 = j; 
      }
    }
    nodes_in_row.swap(tmp_nodes);

    // sort according to center 
    std::sort(nodes_in_row.begin(), nodes_in_row.end(), 
        [&] (int node_id1, int node_id2) {
        T x1 = host_x[node_id1] + host_node_size_x[node_id1]/2;
        T x2 = host_x[node_id2] + host_node_size_x[node_id2]/2;
        return x1 < x2 || (x1 == x2 && node_id1 < node_id2);
        });

    for (int j = 1, je = nodes_in_row.size(); j < je; ++j) {
      int node_id1 = nodes_in_row.at(j - 1);
      int node_id2 = nodes_in_row.at(j);
      T xl1 = host_x[node_id1]; 
      T xl2 = host_x[node_id2];
      T width1 = host_node_size_x[node_id1]; 
      T width2 = host_node_size_x[node_id2]; 
      T xh1 = xl1 + width1; 
      T xh2 = xl2 + width2; 
      T yl1 = host_y[node_id1]; 
      T yl2 = host_y[node_id2]; 
      T yh1 = yl1 + host_node_size_y[node_id1]; 
      T yh2 = yl2 + host_node_size_y[node_id2]; 
      dreamplaceAssertMsg(xl1 < xl2 && xh1 < xh2, 
          "node %d (%g, %g, %g, %g) overlaps with node %d (%g, %g, %g, %g)", 
          node_id1, xl1, yl1, xh1, yh1, node_id2, xl2, yl2, xh2, yh2);
    }
  }
}

/// A cluster recording abutting cells
/// behave liked a linked list but allocated on a continuous memory
template <typename T>
struct AbacusCluster {
  int prev_cluster_id;  ///< previous cluster, set to INT_MIN if the cluster is
                        ///< invalid
  int next_cluster_id;  ///< next cluster, set to INT_MIN if the cluster is
                        ///< invalid
  int bgn_row_node_id;  ///< id of first node in the row
  int end_row_node_id;  ///< id of last node in the row
  T e;                  ///< weight of displacement in the objective
  T q;                  ///< x = q/e
  T w;                  ///< width
  T x;                  ///< optimal location

  /// @return whether this is a valid cluster
  bool valid() const {
    return prev_cluster_id != INT_MIN && next_cluster_id != INT_MIN;
  }
};

template <typename T>
void distributeMovableAndFixedCells2BinsCPU(
    const T* x, const T* y, const T* node_size_x, const T* node_size_y,
    T bin_size_x, T bin_size_y, T xl, T yl, T xh, T yh, T site_width, 
    int num_bins_x, int num_bins_y, int num_nodes, int num_movable_nodes,
    std::vector<std::vector<int> >& bin_cells) {
  for (int i = 0; i < num_nodes; i += 1) {
    if (i < num_movable_nodes && roundDiv(node_size_y[i], bin_size_y) <= 1) {
      // single-row movable nodes only distribute to one bin
      int bin_id_x = (x[i] + node_size_x[i] / 2 - xl) / bin_size_x;
      int bin_id_y = (y[i] + node_size_y[i] / 2 - yl) / bin_size_y;

      bin_id_x = std::min(std::max(bin_id_x, 0), num_bins_x - 1);
      bin_id_y = std::min(std::max(bin_id_y, 0), num_bins_y - 1);

      int bin_id = bin_id_x * num_bins_y + bin_id_y;

      bin_cells[bin_id].push_back(i);
    } else {
      // fixed nodes may distribute to multiple bins
      int node_id = i;
      int bin_id_xl = std::max((x[node_id] - xl) / bin_size_x, (T)0);
      // tricky to control the tolerance in x direction, because bin_size_x is rather large 
      // not setting it properly will cause missing fixed nodes in the bin_cells 
      int bin_id_xh = std::min(
          (int)ceilDiv(x[node_id] + node_size_x[node_id] - xl, bin_size_x, NumericTolerance<T>::rtol * site_width / bin_size_x),
          num_bins_x);
      int bin_id_yl = std::max((y[node_id] - yl) / bin_size_y, (T)0);
      // the problem above usually does not appear in y direction, because bin_size_y is row height 
      int bin_id_yh = std::min(
          (int)ceilDiv(y[node_id] + node_size_y[node_id] - yl, bin_size_y),
          num_bins_y);

      for (int bin_id_x = bin_id_xl; bin_id_x < bin_id_xh; ++bin_id_x) {
        for (int bin_id_y = bin_id_yl; bin_id_y < bin_id_yh; ++bin_id_y) {
          int bin_id = bin_id_x * num_bins_y + bin_id_y;

          bin_cells[bin_id].push_back(node_id);
        }
      }
    }
  }
}

/// @param row_nodes node indices in this row
/// @param clusters pre-allocated clusters in this row with the same length as
/// that of row_nodes
/// @param num_row_nodes length of row_nodes
/// @return true if succeed, otherwise false
template <typename T>
bool abacusPlaceRowCPU(const T* init_x, const T* node_size_x,
                       const T* node_size_y, const T* node_weights, T* x,
                       const T site_width, const T row_height, const T xl, const T xh,
                       const int num_nodes, const int num_movable_nodes,
                       int* row_nodes, AbacusCluster<T>* clusters,
                       const int num_row_nodes) {
  // a very large number
  T M = std::pow(10, ceilDiv(std::log((xh - xl) * num_row_nodes), log(10)));
  // dreamplacePrint(kDEBUG, "M = %g\n", M);
  bool ret_flag = true;

  // merge two clusters
  // the second cluster will be invalid
  auto merge_cluster = [&](int dst_cluster_id, int src_cluster_id) {
    dreamplaceAssert(dst_cluster_id < num_row_nodes);
    AbacusCluster<T>& dst_cluster = clusters[dst_cluster_id];
    dreamplaceAssert(src_cluster_id < num_row_nodes);
    AbacusCluster<T>& src_cluster = clusters[src_cluster_id];

    dreamplaceAssert(dst_cluster.valid() && src_cluster.valid());
    for (int i = dst_cluster_id + 1; i < src_cluster_id; ++i) {
      dreamplaceAssert(!clusters[i].valid());
    }
    dst_cluster.end_row_node_id = src_cluster.end_row_node_id;
    dreamplaceAssert(dst_cluster.e < M && src_cluster.e < M);
    dst_cluster.e += src_cluster.e;
    dst_cluster.q += src_cluster.q - src_cluster.e * dst_cluster.w;
    dst_cluster.w += src_cluster.w;
    // update linked list
    if (src_cluster.next_cluster_id < num_row_nodes) {
      clusters[src_cluster.next_cluster_id].prev_cluster_id = dst_cluster_id;
    }
    dst_cluster.next_cluster_id = src_cluster.next_cluster_id;
    src_cluster.prev_cluster_id = std::numeric_limits<int>::min();
    src_cluster.next_cluster_id = std::numeric_limits<int>::min();
  };

  // collapse clusters between [0, cluster_id]
  // compute the locations and merge clusters
  auto collapse = [&](int cluster_id, T range_xl, T range_xh) {
    int cur_cluster_id = cluster_id;
    dreamplaceAssert(cur_cluster_id < num_row_nodes);
    int prev_cluster_id = clusters[cur_cluster_id].prev_cluster_id;
    AbacusCluster<T>* cluster = nullptr;
    AbacusCluster<T>* prev_cluster = nullptr;

    while (true) {
      dreamplaceAssert(cur_cluster_id < num_row_nodes);
      cluster = &clusters[cur_cluster_id];
      cluster->x = cluster->q / cluster->e;
      // make sure cluster >= range_xl, so fixed nodes will not be moved
      // in illegal case, cluster+w > range_xh may occur, but it is OK.
      // We can collect failed clusters later
      cluster->x = std::max(std::min(cluster->x, range_xh - cluster->w), range_xl);
      // there may be numeric precision issues
      dreamplaceAssertMsg(cluster->x + site_width * NumericTolerance<T>::rtol >= range_xl 
          && cluster->x + cluster->w <= range_xh + site_width * NumericTolerance<T>::rtol, 
                       "%.6f >= %.6f && %.6f + %.6f <= %.6f", 
                       cluster->x, range_xl, cluster->x, cluster->w,
                       range_xh + site_width * NumericTolerance<T>::rtol);

      prev_cluster_id = cluster->prev_cluster_id;
      if (prev_cluster_id >= 0) {
        prev_cluster = &clusters[prev_cluster_id];
        if (prev_cluster->x + prev_cluster->w > cluster->x) {
          merge_cluster(prev_cluster_id, cur_cluster_id);
          cur_cluster_id = prev_cluster_id;
        } else {
          break;
        }
      } else {
        break;
      }
    }
  };

  // initial cluster has only one cell
  for (int i = 0; i < num_row_nodes; ++i) {
    int node_id = row_nodes[i];
    AbacusCluster<T>& cluster = clusters[i];
    cluster.prev_cluster_id = i - 1;
    cluster.next_cluster_id = i + 1;
    cluster.bgn_row_node_id = i;
    cluster.end_row_node_id = i;
    cluster.e =
        (node_id < num_movable_nodes && node_size_y[node_id] <= row_height)
            ? 1.0 /*node_weights[node_id]*/
            : M;
    cluster.q = cluster.e * init_x[node_id];
    cluster.w = node_size_x[node_id];
    // this is required since we also include fixed nodes
    cluster.x =
        (node_id < num_movable_nodes && node_size_y[node_id] > row_height)
            ? x[node_id]
            : init_x[node_id];
  }

  // kernel algorithm for placeRow
  T range_xl = xl;
  T range_xh = xh;
  for (int j = 0; j < num_row_nodes; ++j) {
    const AbacusCluster<T>& next_cluster = clusters[j];
    if (next_cluster.e >= M)  // fixed node
    {
      range_xh = std::min(next_cluster.x, range_xh);
      break;
    } else {
      dreamplaceAssertMsg(std::abs(node_size_y[row_nodes[j]] - row_height) < 1e-6, 
          "node_size_y[row_nodes[%d] = %d] = %g, row_height = %g", 
          j, row_nodes[j], node_size_y[row_nodes[j]], row_height);
    }
  }
  for (int i = 0; i < num_row_nodes; ++i) {
    const AbacusCluster<T>& cluster = clusters[i];
    if (cluster.e < M) {
      dreamplaceAssertMsg(std::abs(node_size_y[row_nodes[i]] - row_height) < 1e-6, 
          "node_size_y[row_nodes[%d] = %d] = %g, row_height = %g", 
          i, row_nodes[i], node_size_y[row_nodes[i]], row_height);
      collapse(i, range_xl, range_xh);
    } else  // set range xl/xh according to fixed nodes
    {
      range_xl = cluster.x + cluster.w;
      range_xh = xh;
      for (int j = i + 1; j < num_row_nodes; ++j) {
        const AbacusCluster<T>& next_cluster = clusters[j];
        if (next_cluster.e >= M)  // fixed node
        {
          range_xh = std::min(next_cluster.x, range_xh);
          break;
        }
      }
    }
  }

  // apply solution
  for (int i = 0; i < num_row_nodes; ++i) {
    if (clusters[i].valid()) {
      const AbacusCluster<T>& cluster = clusters[i];
      T xc = cluster.x;
      for (int j = cluster.bgn_row_node_id; j <= cluster.end_row_node_id; ++j) {
        int node_id = row_nodes[j];
        if (node_id < num_movable_nodes && std::abs(node_size_y[node_id] - row_height) < 1e-6) {
          x[node_id] = xc;
        } else if (xc != x[node_id]) {
          if (node_id < num_movable_nodes)
            dreamplacePrint(kWARN,
                            "multi-row node %d tends to move from %.12f to "
                            "%.12f, ignored\n",
                            node_id, x[node_id], xc);
          else
            dreamplacePrint(
                kWARN,
                "fixed node %d tends to move from %.12f to %.12f, ignored\n",
                node_id, x[node_id], xc);
          ret_flag = false;
        }
        xc += node_size_x[node_id];
      }
    }
  }

  return ret_flag;
}

template <typename T>
void abacusLegalizeRowCPU(
    const T* init_x, const T* node_size_x, const T* node_size_y,
    const T* node_weights, T* x, T* y, const T xl, const T xh, const T yl, const T yh, 
    const T site_width, const T bin_size_x, const T bin_size_y, 
    const int num_bins_x, const int num_bins_y, const int num_nodes, const int num_movable_nodes,
    std::vector<std::vector<int> >& bin_cells,
    std::vector<std::vector<AbacusCluster<T> > >& bin_clusters) {
  for (unsigned int i = 0; i < bin_cells.size(); i += 1) {
    auto& row2nodes = bin_cells.at(i);

    // sort bin cells from left to right
    // a quick fix from cpp branch... may heavily affect results on ICCAD2015.
    sortNodesInRow(x, y, 
        node_size_x, node_size_y, 
        num_movable_nodes, row2nodes);

    auto& clusters = bin_clusters.at(i);
    int num_row_nodes = row2nodes.size();

    int bin_id_x = i / num_bins_y;
    // int bin_id_y = i-bin_id_x*num_bins_y;

    T bin_xl = xl + bin_size_x * bin_id_x;
    T bin_xh = std::min(bin_xl + bin_size_x, xh);

    abacusPlaceRowCPU(init_x, node_size_x, node_size_y, node_weights, x,
                      site_width, bin_size_y,  // must be equal to row_height
                      bin_xl, bin_xh, num_nodes, num_movable_nodes,
                      row2nodes.data(), clusters.data(), num_row_nodes);
  }
  T displace = 0;
  for (int i = 0; i < num_movable_nodes; ++i) {
    displace += fabs(x[i] - init_x[i]);
  }
  dreamplacePrint(kDEBUG, "average displace = %g\n",
                  displace / num_movable_nodes);
}

template <typename T>
void abacusLegalizationCPU(const T* init_x, const T* init_y,
                           const T* node_size_x, const T* node_size_y,
                           const T* node_weights, T* x, T* y, const T xl,
                           const T yl, const T xh, const T yh,
                           const T site_width, const T row_height,
                           int num_bins_x, int num_bins_y, const int num_nodes,
                           const int num_movable_nodes) {
  // adjust bin sizes
  T bin_size_x = (xh - xl) / num_bins_x;
  T bin_size_y = row_height;
  // num_bins_x = ceilDiv(xh - xl, bin_size_x);
  num_bins_y = ceilDiv(yh - yl, bin_size_y);

  // include both movable and fixed nodes
  std::vector<std::vector<int> > bin_cells(num_bins_x * num_bins_y);
  // distribute cells to bins
  distributeMovableAndFixedCells2BinsCPU(
      x, y, node_size_x, node_size_y, bin_size_x, bin_size_y, xl, yl, xh, yh,
      site_width, num_bins_x, num_bins_y, num_nodes, num_movable_nodes, bin_cells);

  std::vector<std::vector<AbacusCluster<T> > > bin_clusters(num_bins_x *
                                                            num_bins_y);
  for (unsigned int i = 0; i < bin_cells.size(); ++i) {
    bin_clusters[i].resize(bin_cells[i].size());
  }

  abacusLegalizeRowCPU(init_x, node_size_x, node_size_y, node_weights, x, y, 
      xl, xh, yl, yh, site_width, bin_size_x, bin_size_y, 
      num_bins_x, num_bins_y, num_nodes, num_movable_nodes, 
      bin_cells, bin_clusters);
  // need to align nodes to sites
  // this also considers cell width which is not integral times of site_width
  for (auto const& cells : bin_cells) {
    T xxl = xl;
    for (auto node_id : cells) {
      if (node_id < num_movable_nodes) {
        x[node_id] = std::max(std::min(x[node_id], xh - node_size_x[node_id]), xxl);
        x[node_id] = floorDiv(x[node_id] - xl, site_width) * site_width + xl;
        xxl = x[node_id] + node_size_x[node_id]; 
      } else if (node_id < num_nodes) {
        xxl = ceilDiv(x[node_id] + node_size_x[node_id] - xl, site_width) * site_width + xl;
      }
    }
  }
  // align2SiteCPU(
  //        node_size_x,
  //        x,
  //        xl, xh,
  //        site_width,
  //        num_movable_nodes
  //        );
}

DREAMPLACE_END_NAMESPACE

#endif
