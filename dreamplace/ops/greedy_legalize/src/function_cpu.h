/**
 * @file   function_cpu.h
 * @author Yibo Lin
 * @date   Oct 2018
 */
#ifndef GPUPLACE_LEGALIZE_FUNCTION_CPU_H
#define GPUPLACE_LEGALIZE_FUNCTION_CPU_H

#include "utility/src/Msg.h"
#include "bin_assignment_cpu.h"
#include "merge_bin_cpu.h"
#include "legality_check_cpu.h"
#include "status_summary_cpu.h"
#include "compare_cpu.h"
#include "abacus_legalize_cpu.h"
#include "align2site_cpu.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
void binAssignmentCPU(
        const int* ordered_nodes, 
        const T* init_x, const T* init_y, 
        const T* node_size_x, const T* node_size_y, 
        T* x, T* y, 
        const T xl, const T yl, const T xh, const T yh, 
        const T site_width, const T row_height, 
        int num_bins_x, int num_bins_y, 
        const int num_nodes, 
        const int num_movable_nodes, 
        const int num_filler_nodes
        );

template <typename T>
void binAssignmentCPULauncher(
        const T* init_x, const T* init_y, 
        const T* node_size_x, const T* node_size_y, 
        T* x, T* y, 
        const T xl, const T yl, const T xh, const T yh, 
        const T site_width, const T row_height, 
        int num_bins_x, int num_bins_y, 
        const int num_nodes, 
        const int num_movable_nodes, 
        const int num_filler_nodes
        );

template <typename T>
void legalizeBinCPU(
        const T* init_x, const T* init_y, 
        const T* node_size_x, const T* node_size_y, 
        std::vector<std::vector<Blank<T> > >& bin_blanks, // blanks in each bin, sorted from low to high, left to right 
        std::vector<std::vector<int> >& bin_cells, // unplaced cells in each bin 
        T* x, T* y, 
        int num_bins_x, int num_bins_y, int blank_num_bins_y, 
        T bin_size_x, T bin_size_y, T blank_bin_size_y, 
        T site_width, T row_height, 
        T xl, T yl, T xh, T yh,
        T alpha, // a parameter to tune anchor initial locations and current locations 
        T beta, // a parameter to tune space reserving 
        bool lr_flag, // from left to right 
        int* num_unplaced_cells 
        );

template <typename T>
int greedyLegalizationCPU(
        const T* init_x, const T* init_y, 
        const T* node_size_x, const T* node_size_y, 
        T* x, T* y, 
        const T xl, const T yl, const T xh, const T yh, 
        const T site_width, const T row_height, 
        int num_bins_x, int num_bins_y, 
        const int num_nodes, 
        const int num_movable_nodes, 
        const int num_filler_nodes
        );

template <typename T>
int greedyLegalizationCPULauncher(
        const T* init_x, const T* init_y, 
        const T* node_size_x, const T* node_size_y, 
        T* x, T* y, 
        const T xl, const T yl, const T xh, const T yh, 
        const T site_width, const T row_height, 
        int num_bins_x, int num_bins_y, 
        const int num_nodes, 
        const int num_movable_nodes, 
        const int num_filler_nodes
        );

DREAMPLACE_END_NAMESPACE

#endif
