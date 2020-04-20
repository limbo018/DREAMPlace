/**
 * @file   function_cpu.h
 * @author Yibo Lin
 * @date   Oct 2018
 */
#ifndef DREAMPLACE_LEGALIZE_FUNCTION_CPU_H
#define DREAMPLACE_LEGALIZE_FUNCTION_CPU_H

#include "utility/src/torch.h"
#include "utility/src/utils.h"
// database dependency
#include "utility/src/legalization_db.h"
#include "utility/src/make_placedb.h"
// local dependency
#include "greedy_legalize/src/bin_assignment_cpu.h"
#include "greedy_legalize/src/compare_cpu.h"
#include "greedy_legalize/src/merge_bin_cpu.h"
#include "greedy_legalize/src/status_summary_cpu.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
void legalizeBinCPU(
    const T* init_x, const T* init_y, const T* node_size_x,
    const T* node_size_y,
    std::vector<std::vector<Blank<T> > >& bin_blanks,  // blanks in each bin,
                                                       // sorted from low to
                                                       // high, left to right
    std::vector<std::vector<int> >& bin_cells,  // unplaced cells in each bin
    T* x, T* y, int num_bins_x, int num_bins_y, int blank_num_bins_y,
    T bin_size_x, T bin_size_y, T blank_bin_size_y, T site_width, T row_height,
    T xl, T yl, T xh, T yh,
    T alpha,       // a parameter to tune anchor initial locations and current
                   // locations
    T beta,        // a parameter to tune space reserving
    bool lr_flag,  // from left to right
    int* num_unplaced_cells);

template <typename T>
int greedyLegalizationCPU(const LegalizationDB<T>& db, const T* init_x,
                          const T* init_y, const T* node_size_x,
                          const T* node_size_y, T* x, T* y, const T xl,
                          const T yl, const T xh, const T yh,
                          const T site_width, const T row_height,
                          int num_bins_x, int num_bins_y, const int num_nodes,
                          const int num_movable_nodes);

DREAMPLACE_END_NAMESPACE

#endif
