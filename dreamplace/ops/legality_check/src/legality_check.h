/**
 * @file   legality_check.h
 * @author Yibo Lin
 * @date   Oct 2018
 */

#ifndef DREAMPLACE_LEGALITY_CHECK_H
#define DREAMPLACE_LEGALITY_CHECK_H

#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>
#include "utility/src/utils.h"

DREAMPLACE_BEGIN_NAMESPACE

/// compare nodes with x center
/// resolve ambiguity by index
template <typename T>
struct CompareByNodeXCenter {
  const T* x;
  const T* node_size_x;

  CompareByNodeXCenter(const T* xx, const T* size_x)
      : x(xx), node_size_x(size_x) {}

  bool operator()(int i, int j) const {
    T xc1 = x[i] + node_size_x[i] / 2;
    T xc2 = x[j] + node_size_x[j] / 2;
    return (xc1 < xc2) || (xc1 == xc2 && i < j);
  }
};

template <typename T>
bool boundaryCheck(const T* x, const T* y, const T* node_size_x,
                   const T* node_size_y, const T scale_factor, T xl, T yl, T xh, T yh,
                   const int num_movable_nodes) {
  // use scale factor to control the precision
  T precision = (scale_factor == 1.0) ? 1e-6 : scale_factor * 0.1;
  bool legal_flag = true;
  // check node within boundary
  for (int i = 0; i < num_movable_nodes; ++i) {
    T node_xl = x[i];
    T node_yl = y[i];
    T node_xh = node_xl + node_size_x[i];
    T node_yh = node_yl + node_size_y[i];
    if (node_xl + precision < xl || node_xh > xh + precision || node_yl + precision < yl || node_yh > yh + precision) {
      dreamplacePrint(kDEBUG, "node %d (%g, %g, %g, %g) out of boundary\n", i,
                      node_xl, node_yl, node_xh, node_yh);
      legal_flag = false;
    }
  }
  return legal_flag;
}

template <typename T>
bool siteAlignmentCheck(const T* x, const T* y, const T site_width,
                        const T row_height, const T scale_factor, const T xl,
                        const T yl, const int num_movable_nodes) {
  // use scale factor to control the precision
  // T precision = (scale_factor == 1.0) ? 1e-6 : scale_factor * 0.1;
  T precision = 0.005; 
  bool legal_flag = true;
  // check row and site alignment
  for (int i = 0; i < num_movable_nodes; ++i) {
    T node_xl = x[i];
    T node_yl = y[i];

    T row_id_f = (node_yl - yl) / row_height;
    int row_id = floorDiv(node_yl - yl, row_height);
    T row_yl = yl + row_height * row_id;
    T row_yh = row_yl + row_height;

    if (std::abs(row_id_f - row_id) > precision) {
      dreamplacePrint(
          kERROR,
          "node %d (%g, %g) failed to align to row %d (%g, %g), gap %g, precision %g\n", i,
          node_xl, node_yl, row_id, row_yl, row_yh, std::abs(node_yl - row_yl), precision);
      legal_flag = false;
    }

    T site_id_f = (node_xl - xl) / site_width;
    int site_id = floorDiv(node_xl - xl, site_width);
    if (std::abs(site_id_f - site_id) > precision) {
      dreamplacePrint(
          kERROR,
          "node %d (%g, %g) failed to align to row %d (%g, %g) and site id %d, gap %g, precision %g\n", i,
          node_xl, node_yl, row_id, row_yl, row_yh, site_id, std::abs(site_id_f - site_id), precision);
      legal_flag = false;
    }
  }

  return legal_flag;
}

template <typename T>
bool fenceRegionCheck(const T* node_size_x, const T* node_size_y,
                      const T* flat_region_boxes,
                      const int* flat_region_boxes_start,
                      const int* node2fence_region_map, const T* x, const T* y,
                      const int num_movable_nodes, const int num_regions) {
  bool legal_flag = true;
  // check fence regions
  for (int i = 0; i < num_movable_nodes; ++i) {
    T node_xl = x[i];
    T node_yl = y[i];
    T node_xh = node_xl + node_size_x[i];
    T node_yh = node_yl + node_size_y[i];

    int region_id = node2fence_region_map[i];
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
        dreamplacePrint(kERROR,
                        "node %d (%g, %g, %g, %g), out of fence region %d", i,
                        node_xl, node_yl, node_xh, node_yh, region_id);
        for (int box_id = box_bgn; box_id < box_end; ++box_id) {
          int box_offset = box_id * 4;
          T box_xl = flat_region_boxes[box_offset];
          T box_yl = flat_region_boxes[box_offset + 1];
          T box_xh = flat_region_boxes[box_offset + 2];
          T box_yh = flat_region_boxes[box_offset + 3];

          dreamplacePrint(kNONE, " (%g, %g, %g, %g)", box_xl, box_yl, box_xh,
                          box_yh);
        }
        dreamplacePrint(kNONE, "\n");
        legal_flag = false;
      }
    }
  }
  return legal_flag;
}

template <typename T>
bool overlapCheck(const T* node_size_x, const T* node_size_y, const T* x,
                  const T* y, T site_width, T row_height, T scale_factor, T xl, T yl, T xh,
                  T yh, const int num_nodes, const int num_movable_nodes) {
  bool legal_flag = true;
  int num_rows = ceilDiv(yh - yl, row_height);
  dreamplaceAssert(num_rows > 0);
  std::vector<std::vector<int> > row_nodes(num_rows);

  // general to node and fixed boxes
  auto getXL = [&](int id) { return x[id]; };
  auto getYL = [&](int id) { return y[id]; };
  auto getXH = [&](int id) { return x[id] + node_size_x[id]; };
  auto getYH = [&](int id) { return y[id] + node_size_y[id]; };

  // potential numerical issue (fix from cpp branch)
  auto getSiteXL = [&](T xx) { return int(floorDiv(xx - xl, site_width)); };
  auto getSiteYL = [&](T yy) { return int(floorDiv(yy - yl, row_height)); };
  auto getSiteXH = [&](T xx) { return int(ceilDiv(xx - xl, site_width)); };
  auto getSiteYH = [&](T yy) { return int(ceilDiv(yy - yl, row_height)); };

  // add a box to row
  auto addBox2Row = [&](int id, T bxl, T byl, T bxh, T byh) {
    int row_idxl = floorDiv(byl - yl, row_height);
    int row_idxh = ceilDiv(byh - yl, row_height);
    row_idxl = std::max(row_idxl, 0);
    row_idxh = std::min(row_idxh, num_rows);

    for (int row_id = row_idxl; row_id < row_idxh; ++row_id) {
      T row_yl = yl + row_id * row_height;
      T row_yh = row_yl + row_height;

      if (byl < row_yh && byh > row_yl)  // overlap with row
      {
        row_nodes[row_id].push_back(id);
      }
    }
  };
  // distribute movable cells to rows
  for (int i = 0; i < num_nodes; ++i) {
    T node_xl = x[i];
    T node_yl = y[i];
    T node_xh = node_xl + node_size_x[i];
    T node_yh = node_yl + node_size_y[i];

    addBox2Row(i, node_xl, node_yl, node_xh, node_yh);
  }

  // sort cells within rows
  for (int i = 0; i < num_rows; ++i) {
    auto& nodes_in_row = row_nodes.at(i);
    // using left edge
    std::sort(nodes_in_row.begin(), nodes_in_row.end(),
              [&](int node_id1, int node_id2) {
                T x1 = getXL(node_id1);
                T x2 = getXL(node_id2);
                return x1 < x2 || (x1 == x2 && (node_id1 < node_id2));
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
      for (int j = 1, je = nodes_in_row.size(); j < je; ++j) {
        int node_id1 = nodes_in_row.at(j - 1);
        int node_id2 = nodes_in_row.at(j);
        // two fixed cells
        if (node_id1 >= num_movable_nodes && node_id2 >= num_movable_nodes) {
          T xh1 = getXH(node_id1);
          T xh2 = getXH(node_id2);
          if (xh1 < xh2) {
            tmp_nodes.push_back(node_id2);
          }
        } else {
          tmp_nodes.push_back(node_id2);
        }
      }
      nodes_in_row.swap(tmp_nodes);
    }
  }

  // check overlap
  // use scale factor to control the precision
  // auto scaleBack2Integer = [&](T value) {
  //   return (scale_factor == 1.0)? value : std::round(value / scale_factor); 
  // };
  for (int i = 0; i < num_rows; ++i) {
    for (unsigned int j = 0; j < row_nodes.at(i).size(); ++j) {
      if (j > 0) {
        int node_id = row_nodes[i][j];
        int prev_node_id = row_nodes[i][j - 1];

        if (node_id < num_movable_nodes ||
            prev_node_id < num_movable_nodes)  // ignore two fixed nodes
        {
          T prev_xl = getXL(prev_node_id);
          T prev_yl = getYL(prev_node_id);
          T prev_xh = getXH(prev_node_id);
          T prev_yh = getYH(prev_node_id);
          T cur_xl = getXL(node_id);
          T cur_yl = getYL(node_id);
          T cur_xh = getXH(node_id);
          T cur_yh = getYH(node_id);
          int prev_site_xl = getSiteXL(prev_xl); 
          int prev_site_xh = getSiteXH(prev_xh); 
          int cur_site_xl = getSiteXL(cur_xl); 
          int cur_site_xh = getSiteXH(cur_xh); 
          // detect overlap
          // original criteria: scaleBack2Integer(prev_xh) > scaleBack2Integer(cur_xl)
          // the floating point comparison may introduce incorrect result
          if (prev_site_xh > cur_site_xl) {
            dreamplacePrint(
                kERROR,
                "row %d (%g, %g), overlap node %d (%g, %g, %g, %g) with "
                "node %d (%g, %g, %g, %g) site (%d, %d), gap %g\n",
                i, yl + i * row_height, yl + (i + 1) * row_height, prev_node_id,
                prev_xl, prev_yl, prev_xh, prev_yh, node_id, cur_xl, cur_yl,
                cur_xh, cur_yh, cur_site_xl, cur_site_xh,
                prev_xh - cur_xl);
            legal_flag = false;
          }
        }
      }
    }
  }

  return legal_flag;
}

template <typename T>
bool legalityCheckKernelCPU(const T* x, const T* y, const T* node_size_x,
                            const T* node_size_y, const T* flat_region_boxes,
                            const int* flat_region_boxes_start,
                            const int* node2fence_region_map, T xl, T yl, T xh,
                            T yh, T site_width, T row_height, T scale_factor,
                            const int num_nodes,  ///< movable and fixed cells
                            const int num_movable_nodes,
                            const int num_regions) {
  bool legal_flag = true;
  int num_rows = ceil((yh - yl) / row_height);
  dreamplaceAssert(num_rows > 0);
  fflush(stdout);
  std::vector<std::vector<int> > row_nodes(num_rows);

  // check node within boundary
  if (!boundaryCheck(x, y, node_size_x, node_size_y, scale_factor, xl, yl, xh, yh,
                     num_movable_nodes)) {
    legal_flag = false;
  }

  // check row and site alignment
  if (!siteAlignmentCheck(x, y, site_width, row_height, scale_factor, xl, yl,
                          num_movable_nodes)) {
    legal_flag = false;
  }

  if (!overlapCheck(node_size_x, node_size_y, x, y, site_width, row_height, scale_factor,
                    xl, yl, xh, yh, num_nodes, num_movable_nodes)) {
    legal_flag = false;
  }

  // check fence regions
  if (!fenceRegionCheck(node_size_x, node_size_y, flat_region_boxes,
                        flat_region_boxes_start, node2fence_region_map, x, y,
                        num_movable_nodes, num_regions)) {
    legal_flag = false;
  }

  return legal_flag;
}

template <typename T>
bool legalityCheckSiteMapKernelCPU(const T* init_x, const T* init_y,
                                   const T* node_size_x, const T* node_size_y,
                                   const T* x, const T* y, T xl, T yl, T xh,
                                   T yh, T site_width, T row_height,
                                   T scale_factor, const int num_nodes,
                                   const int num_movable_nodes) {
  int num_rows = ceilDiv(yh - yl, row_height);
  int num_sites = ceilDiv(xh - xl, site_width);
  std::vector<std::vector<unsigned char> > site_map(
      num_rows, std::vector<unsigned char>(num_sites, 0));

  // fixed macros
  for (int i = num_movable_nodes; i < num_nodes; ++i) {
    T node_xl = x[i];
    T node_yl = y[i];
    T node_xh = node_xl + node_size_x[i];
    T node_yh = node_yl + node_size_y[i];

    int idxl = floorDiv(node_xl - xl, site_width);
    int idxh = ceilDiv(node_xh - xl, site_width);
    int idyl = floorDiv(node_yl - yl, row_height);
    int idyh = ceilDiv(node_yh - yl, row_height);
    idxl = std::max(idxl, 0);
    idxh = std::min(idxh, num_sites);
    idyl = std::max(idyl, 0);
    idyh = std::min(idyh, num_rows);

    for (int iy = idyl; iy < idyh; ++iy) {
      for (int ix = idxl; ix < idxh; ++ix) {
        T site_xl = xl + ix * site_width;
        T site_xh = site_xl + site_width;
        T site_yl = yl + iy * row_height;
        T site_yh = site_yl + row_height;

        if (node_xl < site_xh && node_xh > site_xl && node_yl < site_yh &&
            node_yh > site_yl)  // overlap
        {
          site_map[iy][ix] = 255;
        }
      }
    }
  }

  bool legal_flag = true;
  // movable cells
  for (int i = 0; i < num_movable_nodes; ++i) {
    T node_xl = x[i];
    T node_yl = y[i];
    T node_xh = node_xl + node_size_x[i];
    T node_yh = node_yl + node_size_y[i];

    int idxl = floorDiv(node_xl - xl, site_width);
    int idxh = ceilDiv(node_xh - xl, site_width);
    int idyl = floorDiv(node_yl - yl, row_height);
    int idyh = ceilDiv(node_yh - yl, row_height);
    idxl = std::max(idxl, 0);
    idxh = std::min(idxh, num_sites);
    idyl = std::max(idyl, 0);
    idyh = std::min(idyh, num_rows);

    for (int iy = idyl; iy < idyh; ++iy) {
      for (int ix = idxl; ix < idxh; ++ix) {
        T site_xl = xl + ix * site_width;
        T site_xh = site_xl + site_width;
        T site_yl = yl + iy * row_height;
        T site_yh = site_yl + row_height;

        if (node_xl < site_xh && node_xh > site_xl && node_yl < site_yh &&
            node_yh > site_yl)  // overlap
        {
          if (site_map[iy][ix]) {
            dreamplacePrint(kERROR,
                            "detect overlap at site (%g, %g, %g, %g) for node "
                            "%d (%g, %g, %g, %g)\n",
                            site_xl, site_yl, site_xh, site_yh, i, node_xl,
                            node_yl, node_xh, node_yh);
            legal_flag = false;
          }
          site_map[iy][ix] += 1;
        }
      }
    }
  }

  return legal_flag;
}

DREAMPLACE_END_NAMESPACE

#endif
