/**
 * @file   bin_assignment.h
 * @author Yibo Lin
 * @date   Oct 2018
 */

#ifndef DREAMPLACE_BIN_ASSIGNMENT_H
#define DREAMPLACE_BIN_ASSIGNMENT_H

#include <cmath>
#include <vector>
#include "blank.h"
#include "utility/src/utils.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
void distributeCells2BinsCPU(const LegalizationDB<T>& db, const T* x,
                             const T* y, const T* node_size_x,
                             const T* node_size_y, T bin_size_x, T bin_size_y,
                             T xl, T yl, T xh, T yh, int num_bins_x,
                             int num_bins_y, int num_nodes,
                             int num_movable_nodes,
                             std::vector<std::vector<int> >& bin_cells) {
  // do not handle large macros
  // one cell cannot be distributed to one bin
  for (int i = 0; i < num_movable_nodes; i += 1) {
    if (!db.is_dummy_fixed(i)) {
      int bin_id_x = (x[i] + node_size_x[i] / 2 - xl) / bin_size_x;
      int bin_id_y = (y[i] + node_size_y[i] / 2 - yl) / bin_size_y;

      bin_id_x = std::min(std::max(bin_id_x, 0), num_bins_x - 1);
      bin_id_y = std::min(std::max(bin_id_y, 0), num_bins_y - 1);

      int bin_id = bin_id_x * num_bins_y + bin_id_y;

      bin_cells[bin_id].push_back(i);
    }
  }
}

template <typename T>
void distributeFixedCells2BinsCPU(const LegalizationDB<T>& db, const T* x,
                                  const T* y, const T* node_size_x,
                                  const T* node_size_y, T bin_size_x,
                                  T bin_size_y, T xl, T yl, T xh, T yh,
                                  int num_bins_x, int num_bins_y, int num_nodes,
                                  int num_movable_nodes,
                                  std::vector<std::vector<int> >& bin_cells) {
  // one cell can be assigned to multiple bins
  for (int i = 0; i < num_nodes; i += 1) {
    if (db.is_dummy_fixed(i) || i >= num_movable_nodes) {
      int node_id = i;
      int bin_id_xl = std::max((int)floorDiv(x[node_id] - xl, bin_size_x, (T)0), 0);
      int bin_id_xh = std::min(
          (int)ceilDiv((x[node_id] + node_size_x[node_id] - xl), bin_size_x, (T)0),
          num_bins_x);
      int bin_id_yl = std::max((int)floorDiv(y[node_id] - yl, bin_size_y, (T)0), 0);
      int bin_id_yh = std::min(
          (int)ceilDiv((y[node_id] + node_size_y[node_id] - yl), bin_size_y, (T)0),
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

template <typename T>
void distributeBlanks2BinsCPU(
    const T* x, const T* y, const T* node_size_x, const T* node_size_y,
    const std::vector<std::vector<int> >& bin_fixed_cells, T bin_size_x,
    T bin_size_y, T blank_bin_size_y, T xl, T yl, T xh, T yh, T site_width,
    T row_height, int num_bins_x, int num_bins_y, int blank_num_bins_y,
    std::vector<std::vector<Blank<T> > >& bin_blanks) {
  for (int i = 0; i < num_bins_x * num_bins_y; i += 1) {
    int bin_id_x = i / num_bins_y;
    int bin_id_y = i - bin_id_x * num_bins_y;
    int blank_num_bins_per_bin = roundDiv(bin_size_y, blank_bin_size_y);
    int blank_bin_id_yl = bin_id_y * blank_num_bins_per_bin;
    int blank_bin_id_yh =
        std::min(blank_bin_id_yl + blank_num_bins_per_bin, blank_num_bins_y);
    for (int blank_bin_id_y = blank_bin_id_yl; blank_bin_id_y < blank_bin_id_yh;
         ++blank_bin_id_y) {
      T bin_xl = xl + bin_id_x * bin_size_x;
      T bin_xh = std::min(bin_xl + bin_size_x, xh);
      T bin_yl = yl + blank_bin_id_y * blank_bin_size_y;
      T bin_yh = std::min(bin_yl + blank_bin_size_y, yh);
      int blank_bin_id = bin_id_x * blank_num_bins_y + blank_bin_id_y;

      for (T by = bin_yl; by < bin_yh; by += row_height) {
        Blank<T> blank;
        blank.xl = floorDiv((bin_xl - xl), site_width) * site_width +
                   xl;  // align blanks to sites
        blank.xh = floorDiv((bin_xh - xl), site_width) * site_width +
                   xl;  // align blanks to sites
        blank.yl = by;
        blank.yh = by + row_height;

        bin_blanks.at(blank_bin_id).push_back(blank);
      }

      const std::vector<int>& cells = bin_fixed_cells.at(i);
      std::vector<Blank<T> >& blanks = bin_blanks.at(blank_bin_id);

      for (unsigned int bi = 0; bi < blanks.size(); ++bi) {
        Blank<T>& blank = blanks.at(bi);
        for (unsigned int ci = 0; ci < cells.size(); ++ci) {
          int node_id = cells.at(ci);
          T node_xl = x[node_id];
          T node_yl = y[node_id];
          T node_xh = node_xl + node_size_x[node_id];
          T node_yh = node_yl + node_size_y[node_id];

          if (node_yh > blank.yl && node_yl < blank.yh && node_xh > blank.xl &&
              node_xl < blank.xh)  // overlap
          {
            if (node_xl <= blank.xl && node_xh >= blank.xh)  // erase
            {
              bin_blanks.at(blank_bin_id)
                  .erase(bin_blanks.at(blank_bin_id).begin() + bi);
              --bi;
              break;
            } else if (node_xl <= blank.xl)  // one blank
            {
              blank.xl = ceilDiv((node_xh - xl), site_width) * site_width +
                         xl;                 // align blanks to sites
            } else if (node_xh >= blank.xh)  // one blank
            {
              blank.xh = floorDiv((node_xl - xl), site_width) * site_width +
                         xl;  // align blanks to sites
            } else            // two blanks
            {
              Blank<T> new_blank = blank;
              blank.xh = floorDiv((node_xl - xl), site_width) * site_width +
                         xl;  // align blanks to sites
              new_blank.xl = floorDiv((node_xh - xl), site_width) * site_width +
                             xl;  // align blanks to sites
              bin_blanks.at(blank_bin_id)
                  .insert(bin_blanks.at(blank_bin_id).begin() + bi + 1,
                          new_blank);
              --bi;
              break;
            }
          }
        }
      }
    }
  }
}

template <typename T>
void computeBinCapacityCPU(
    const T* x, const T* y, const T* node_size_x, const T* node_size_y,
    const std::vector<std::vector<int> >& bin_fixed_cells, T bin_size_x,
    T bin_size_y, T xl, T yl, T xh, T yh, T site_width, T row_height,
    int num_bins_x, int num_bins_y, int* bin_capacities) {
  for (int i = 0; i < num_bins_x * num_bins_y; i += 1) {
    int bin_id_x = i / num_bins_y;
    int bin_id_y = i - bin_id_x * num_bins_y;
    T bin_xl = xl + bin_id_x * bin_size_x;
    T bin_xh = std::min(bin_xl + bin_size_x, xh);
    T bin_yl = yl + bin_id_y * bin_size_y;
    T bin_yh = std::min(bin_yl + bin_size_y, yh);

    T capacity = (bin_xh - bin_xl) * (bin_yh - bin_yl);

    const std::vector<int>& cells = bin_fixed_cells[i];

    for (unsigned int ci = 0; ci < cells.size(); ++ci) {
      int node_id = cells[ci];
      T node_xl = x[node_id];
      T node_yl = y[node_id];
      T node_xh = node_xl + node_size_x[node_id];
      T node_yh = node_yl + node_size_y[node_id];

      T overlap =
          std::max(std::min(bin_xh, node_xh) - std::max(bin_xl, node_xl),
                   (T)0) *
          std::max(std::min(bin_yh, node_yh) - std::max(bin_yl, node_yl), (T)0);
      capacity -= overlap;
    }
    bin_capacities[i] = ceilDiv(capacity, (site_width * row_height));
  }
}

DREAMPLACE_END_NAMESPACE

#endif
