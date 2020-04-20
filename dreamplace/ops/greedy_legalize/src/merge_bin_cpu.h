/**
 * @file   merge_bin_cpu.h
 * @author Yibo Lin
 * @date   Oct 2018
 */
#ifndef DREAMPLACE_MERGE_BIN_CPU_H
#define DREAMPLACE_MERGE_BIN_CPU_H

#include <cstdio>
#include <vector>
#include "utility/src/utils.h"

DREAMPLACE_BEGIN_NAMESPACE

/// assume num_bins_x*num_bins_y is smaller than bin_objs.size
/// assume num_bins_x*num_bins_y*max_objs_per_bin is no larger than the numbers
/// in original bin_objs
template <typename T>
void resizeBinObjectsCPU(std::vector<std::vector<T> >& bin_objs, int num_bins_x,
                         int num_bins_y) {
  bin_objs.resize(num_bins_x * num_bins_y);
}

template <typename T>
void countBinObjects(const std::vector<std::vector<T> >& bin_objs) {
  int count = 0;
  for (unsigned int i = 0; i < bin_objs.size(); ++i) {
    count += bin_objs.at(i).size();
  }
  dreamplacePrint(kDEBUG, "#bin_objs = %d\n", count);
}

template <typename T>
void mergeBinBlanksCPU(
    const std::vector<std::vector<Blank<T> > >& src_bin_blanks,
    int src_num_bins_x, int src_num_bins_y,  // dimensions for the src
    std::vector<std::vector<Blank<T> > >& dst_bin_blanks, int dst_num_bins_x,
    int dst_num_bins_y,  // dimensions for the dst
    int scale_ratio_x,   // roughly src_num_bins_x/dst_num_bins_x
    T min_blank_width    // minimum blank width to consider
) {
  for (int i = 0; i < dst_num_bins_x * dst_num_bins_y; i += 1) {
    // assume src_num_bins_y == dst_num_bins_y
    int dst_bin_id_x = i / dst_num_bins_y;
    int dst_bin_id_y = i - dst_bin_id_x * dst_num_bins_y;

    int src_bin_id_x_bgn = dst_bin_id_x * scale_ratio_x;
    // int src_bin_id_y_bgn = dst_bin_id_y<<1;
    int src_bin_id_x_end =
        std::min(src_bin_id_x_bgn + scale_ratio_x, src_num_bins_x);
    // int src_bin_id_y_end = std::min(src_bin_id_y_bgn+2, src_num_bins_y);

    std::vector<Blank<T> >& dst_bin_blank = dst_bin_blanks.at(i);

    // dreamplacePrint(kDEBUG, "dst_bin_blanks[%d] (%d, %d) found src_bin_blanks
    // (%d, %d) (%d)\n", i, dst_bin_id_x, dst_bin_id_y, src_bin_id_x_bgn,
    // src_bin_id_x_end, dst_bin_id_y);

    for (int ix = src_bin_id_x_bgn; ix < src_bin_id_x_end; ++ix) {
      // for (int iy = src_bin_id_y_bgn; iy < src_bin_id_y_end; ++iy)
      int iy = dst_bin_id_y;  // same as src_bin_id_y
      {
        int src_bin_id = ix * src_num_bins_y + iy;

        const std::vector<Blank<T> >& src_bin_blank =
            src_bin_blanks.at(src_bin_id);

        int offset = 0;
        if (!dst_bin_blank.empty() && !src_bin_blank.empty()) {
          const Blank<T>& first_blank = src_bin_blank.at(0);
          Blank<T>& last_blank = dst_bin_blank.at(dst_bin_blank.size() - 1);
          if (last_blank.yl == first_blank.yl &&
              last_blank.xh == first_blank.xl) {
            last_blank.xh = first_blank.xh;
            offset = 1;
          }
        }
        // CVector::push_back(dst_bin_blanks, i,
        //        src_bin_blank+offset,
        //        src_bin_blank+src_bin_blanks.sizes[src_bin_id]
        //        );
        for (unsigned int k = offset; k < src_bin_blank.size(); ++k) {
          const Blank<T>& blank = src_bin_blank.at(k);
          // prune small blanks
          if (blank.xh - blank.xl >= min_blank_width) {
            dst_bin_blanks.at(i).push_back(blank);
          }
        }
      }
    }
  }
}

void mergeBinCellsCPU(
    const std::vector<std::vector<int> >& src_bin_cells, int src_num_bins_x,
    int src_num_bins_y,  // dimensions for the src
    std::vector<std::vector<int> >& dst_bin_cells, int dst_num_bins_x,
    int dst_num_bins_y,  // dimensions for the dst
    int scale_ratio_x,
    int scale_ratio_y  // roughly src_num_bins_x/dst_num_bins_x, but may not be
                       // exactly the same due to even/odd numbers
);

DREAMPLACE_END_NAMESPACE

#endif
