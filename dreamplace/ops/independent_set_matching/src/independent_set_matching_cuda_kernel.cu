/**
 * @file   independent_set_matching_cuda_kernel.cu
 * @author Yibo Lin
 * @date   Jan 2019
 */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <random>
#include <assert.h>
#include <chrono>
#include <cmath>
#include <map>
#include <iostream>
#include <fstream>
#include <curand.h>
#include <curand_kernel.h>

#define DETERMINISTIC

//#define DEBUG 
//#define DYNAMIC
#define NUM_NODE_SIZES 64 ///< number of different cell sizes

#include "utility/src/utils.cuh"
// database dependency
#include "utility/src/detailed_place_db.cuh"
#include "independent_set_matching/src/construct_spaces.cuh"
#include "independent_set_matching/src/auction.cuh"
// This shared memory version is still buggy;
// it gets very slow at ISPD 2018 test4/5/6 benchmarks and cannot cause illegal memory access 
//#include "independent_set_matching/src/auction_shared_memory.cuh"
//#include "independent_set_matching/src/auction_cuda2cpu.cuh"
#include "independent_set_matching/src/maximal_independent_set.cuh"
//#include "independent_set_matching/src/maximal_independent_set_cuda2cpu.cuh"
#include "independent_set_matching/src/cpu_state.cuh"
#include "independent_set_matching/src/collect_independent_sets.cuh"
//#include "independent_set_matching/src/collect_independent_sets_cuda2cpu.cuh"
#include "independent_set_matching/src/cost_matrix_construction.cuh"
//#include "independent_set_matching/src/cost_matrix_construction_cuda2cpu.cuh"
#include "independent_set_matching/src/apply_solution.cuh"
//#include "independent_set_matching/src/apply_solution_cuda2cpu.h"
#include "independent_set_matching/src/shuffle.cuh"

DREAMPLACE_BEGIN_NAMESPACE

struct SizedBinIndex
{
    int size_id; 
    int bin_id; 
};

template <typename T>
struct IndependentSetMatchingState
{
    typedef T type; 
    typedef int cost_type;

    int* ordered_nodes = nullptr; 
    ////int* node_size_id = nullptr; 
    Space<T>* spaces = nullptr; ///< array of cell spaces, each cell only consider the space on its left side except for the left and right boundary
    int num_node_sizes; ///< number of cell sizes considered 
    int* independent_sets = nullptr; ///< independent sets, length of batch_size*set_size  
    int* independent_set_sizes = nullptr; ///< size of each independent set 
    ////int* ordered_independent_sets = nullptr; ///< temporary storage for reordering independent sets, forward mapping  
    ////int* reordered_independent_sets = nullptr; ///< temporary storage for reordering independent sets, reverse mapping 
    int* selected_maximal_independent_set = nullptr; ///< storing the selected maximum independent set  
    int* select_scratch = nullptr; ///< temporary storage for selection kernel 
    int num_selected; ///< maximum independent set size 
    int* device_num_selected; ///< maximum independent set size 
    ////int* device_num_selected_prefix_sum = nullptr; ///< prefix sum for different sizes of cells in the maximum independent set 
    //int* device_num_clusters_prefix_sum = nullptr; ///< prefix sum of the number of clusters for different cell sizes 
    ////int* node2center = nullptr; ///< map cell to cluster center for kmeans 
    //int* centers = nullptr; ///< batch_size, cells for centers 
    ////T* center_xs = nullptr; ///< NUM_NODE_SIZES*batch_size, cluster centers of different sizes 
    ////T* center_ys = nullptr; ///< NUM_NODE_SIZES*batch_size, cluster centers of different sizes 
    //int* cluster_sizes = nullptr; ///< NUM_NODE_SIZES*batch_size, cluster sizes of different cell sizes 

    double* net_hpwls; ///< HPWL for each net, use integer to get consistent values 

    int* selected_markers = nullptr; ///< must be int for cub to compute prefix sum
    unsigned char* dependent_markers = nullptr; 
    int* independent_set_empty_flag = nullptr; ///< a stopping flag for maximum independent set 
    ////int* device_num_independent_sets = nullptr; ///< actual number of independent sets 
    int num_independent_sets; ///< host copy 

    cost_type* cost_matrices = nullptr; ///< cost matrices batch_size*set_size*set_size 
    cost_type* cost_matrices_copy = nullptr; ///< temporary copy of cost matrices 
    int* solutions = nullptr; ///< batch_size*set_size
    char* auction_scratch = nullptr; ///< temporary memory for auction solver 
    char* stop_flags = nullptr; ///< record stopping status from auction solver 
    T* orig_x = nullptr; ///< original locations of cells for applying solutions 
    T* orig_y = nullptr; 
    cost_type* orig_costs = nullptr; ///< original costs 
    cost_type* solution_costs = nullptr; ///< solution costs 
    Space<T>* orig_spaces = nullptr; ///< original spaces of cells for apply solutions 

    int batch_size; ///< pre-allocated number of independent sets 
    int set_size; 
    int cost_matrix_size; ///< set_size*set_size 
    int num_bins; ///< num_bins_x*num_bins_y
    int* device_num_moved; ///< device copy 
    int num_moved; ///< host copy, number of moved cells 
    int large_number; ///< a large number 

    float auction_max_eps; ///< maximum epsilon for auction solver
    float auction_min_eps; ///< minimum epsilon for auction solver
    float auction_factor; ///< decay factor for auction epsilon
    int auction_max_iterations; ///< maximum iteration 
    T skip_threshold; ///< ignore connections if cells are far apart 
};

/// @brief A function for debug. Dump out binary data to a file. 
template <typename T>
void write(const T* device_data, size_t size, std::string filename) 
{
    std::ofstream out (filename.c_str(), std::ios::out | std::ios::binary); 
    dreamplaceAssert(out.good());
    dreamplacePrint(kDEBUG, "write to %s size %llu\n", filename.c_str(), size);

    std::vector<T> host_data (size); 
    checkCUDA(cudaMemcpy(host_data.data(), device_data, sizeof(T)*size, cudaMemcpyDeviceToHost));
    checkCUDA(cudaDeviceSynchronize());
    out.write((char*)&size, sizeof(int));
    out.write((char*)host_data.data(), sizeof(T)*host_data.size());

    out.close();
}

/// @brief Corresponding read the binary data. 
template <typename T>
void read(std::vector<T>& data, const char* filename)
{
    std::ifstream in (filename, std::ios::in | std::ios::binary);
    assert(in.good());

    int size = 0; 
    in.read((char*)&size, sizeof(size)); 
    data.resize(size); 

    in.read((char*)data.data(), sizeof(T)*size);

    in.close();
}

__global__ void cost_matrix_init(int* cost_matrix, int set_size)
{
    for (int i = blockIdx.x; i < set_size; i += gridDim.x)
    {
        for (int j = threadIdx.x; j < set_size; j += blockDim.x)
        {
            cost_matrix[i*set_size+j] = (i == j)? 0 : cuda::numeric_limits<int>::max();
        }
    }
}

template <typename T>
__global__ void print_global(T* a, int n)
{
	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x;
    if (tid == 0 && bid == 0)
    {
        printf("[%d]\n", n);
        for (int i = 0; i < n; ++i)
        {
            printf("%g ", (double)a[i]);
        }
        printf("\n");
    }
}

template <typename T>
__global__ void print_cost_matrix(const T* cost_matrix, int set_size, bool major)
{
	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x;
    if (tid == 0 && bid == 0)
    {
        printf("[%dx%d]\n", set_size, set_size);
        for (int r = 0; r < set_size; ++r)
        {
            for (int c = 0; c < set_size; ++c)
            {
                if (major) // column major 
                {
                    printf("%g ", (double)cost_matrix[c*set_size+r]);
                }
                else 
                {
                    printf("%g ", (double)cost_matrix[r*set_size+c]);
                }
            }
            printf("\n");
        }
        printf("\n");
    }
}

template <typename T>
__global__ void print_solution(const T* solution, int n)
{
	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x;
    if (tid == 0 && bid == 0)
    {
        printf("[%d]\n", n);
        for (int i = 0; i < n; ++i)
        {
            printf("%g ", (double)solution[i]);
        }
        printf("\n");
    }
}

template <typename T>
int independentSetMatchingCUDALauncher(DetailedPlaceDB<T> db, 
        int batch_size, int set_size, int max_iters, int num_threads)
{
    //size_t printf_size = 0; 
    //cudaDeviceGetLimit(&printf_size,cudaLimitPrintfFifoSize);
    //cudaDeviceSetLimit(cudaLimitPrintfFifoSize, printf_size*10);
    // fix random seed 
    std::srand(1000);
    //const double threshold = 0.0001; 
    CPUTimer::hr_clock_rep timer_start, timer_stop; 
    CPUTimer::hr_clock_rep kernel_timer_start, kernel_timer_stop; 
    CPUTimer::hr_clock_rep total_timer_start, total_timer_stop; 

    total_timer_start = CPUTimer::getGlobaltime(); 

    IndependentSetMatchingState<T> state; 

    // initialize host database 
    DetailedPlaceCPUDB<T> host_db; 
    init_cpu_db(db, host_db);

    state.batch_size = batch_size; 
    state.set_size = set_size; 
    state.cost_matrix_size = state.set_size*state.set_size;
    state.num_bins = db.num_bins_x*db.num_bins_y;
    state.num_moved = 0; 
    state.large_number = ((db.xh-db.xl) + (db.yh-db.yl))*set_size;
    state.skip_threshold = ((db.xh-db.xl) + (db.yh-db.yl))*0.01;
    state.auction_max_eps = 10.0; 
    state.auction_min_eps = 1.0;  
    state.auction_factor = 0.1;  
    state.auction_max_iterations = 9999; 

    ////timer_start = CPUTimer::getGlobaltime(); 
    ////std::map<int, int> size2num_node_map; ///< number of cells with different sizes 
    ////std::map<int, int> size2id_map; ///< map width of a cell to an index 
    ////std::vector<T> host_x (db.num_nodes);
    ////std::vector<T> host_y (db.num_nodes);
    checkCUDA(cudaMemcpy(host_db.x.data(), db.x, sizeof(T)*db.num_nodes, cudaMemcpyDeviceToHost));
    checkCUDA(cudaMemcpy(host_db.y.data(), db.y, sizeof(T)*db.num_nodes, cudaMemcpyDeviceToHost));
    /////std::vector<T> host_node_size_x (db.num_nodes);
    /////std::vector<T> host_node_size_y (db.num_nodes);
    /////checkCUDA(cudaMemcpy(host_node_size_x.data(), db.node_size_x, sizeof(T)*db.num_nodes, cudaMemcpyDeviceToHost));
    /////checkCUDA(cudaMemcpy(host_node_size_y.data(), db.node_size_y, sizeof(T)*db.num_nodes, cudaMemcpyDeviceToHost));
    ////std::vector<int> host_node_size_id (db.num_movable_nodes, std::numeric_limits<int>::max()); 
    ////std::vector<SizedBinIndex> host_thread2bin_map;
    ////std::vector<int> host_ordered_nodes (db.num_movable_nodes);
    ////std::iota(host_ordered_nodes.begin(), host_ordered_nodes.end(), 0);
    std::vector<Space<T> > host_spaces (db.num_movable_nodes);
    construct_spaces(db, host_db.x.data(), host_db.y.data(), host_db.node_size_x.data(), host_db.node_size_y.data(), host_spaces, num_threads);

    ////// initialize size information 
    ////{
    ////    for (int i = 0; i < db.num_movable_nodes; ++i)
    ////    {
    ////        if (host_node_size_y[i] == db.row_height)
    ////        {
    ////            int width = (int)ceil(host_node_size_x[i]/db.site_width); 
    ////            if (size2num_node_map.count(width))
    ////            {
    ////                size2num_node_map[width] += 1; 
    ////            }
    ////            else 
    ////            {
    ////                size2num_node_map[width] = 1; 
    ////            }
    ////        }
    ////    }
    ////    int size_id = 0; 
    ////    for (auto kv : size2num_node_map)
    ////    {
    ////        if (kv.second < state.set_size || size_id >= NUM_NODE_SIZES)
    ////        {
    ////            dreamplacePrint(kINFO, "ignore %d cells of width %d\n", kv.second, kv.first);
    ////            continue; 
    ////        }
    ////        size2id_map[kv.first] = size_id; 
    ////        dreamplacePrint(kINFO, "map %d cells of width %d to %d\n", kv.second, kv.first, size_id);
    ////        ++size_id;
    ////    }
    ////    state.num_node_sizes = size_id; 
    ////    dreamplacePrint(kINFO, "consider %d kinds of cell sizes\n", state.num_node_sizes);

    ////    for (int i = 0; i < db.num_movable_nodes; ++i)
    ////    {
    ////        if (host_node_size_y[i] == db.row_height)
    ////        {
    ////            int width = (int)ceil(host_node_size_x[i]/db.site_width); 
    ////            if (size2id_map.count(width))
    ////            {
    ////                int size_id = size2id_map.at(width);
    ////                host_node_size_id[i] = size_id;
    ////            }
    ////        }
    ////    }
    ////}
    ////timer_stop = CPUTimer::getGlobaltime(); 
    ////dreamplacePrint(kINFO, "initializing cell size categories takes %g ms\n", CPUTimer::getTimerPeriod()*(timer_stop-timer_start));

    // initialize cuda state 
    timer_start = CPUTimer::getGlobaltime();
    {
        ////allocateCopyCUDA(state.node_size_id, host_node_size_id.data(), db.num_movable_nodes);
        allocateCopyCUDA(state.spaces, host_spaces.data(), db.num_movable_nodes);

        allocateCUDA(state.ordered_nodes, db.num_movable_nodes, int);
        iota<<<ceilDiv(db.num_movable_nodes, 512), 512>>>(state.ordered_nodes, db.num_movable_nodes);
        allocateCUDA(state.independent_sets, state.batch_size*state.set_size, int);
        allocateCUDA(state.independent_set_sizes, state.batch_size, int);
        ////allocateCUDA(state.ordered_independent_sets, state.batch_size, int);
        ////allocateCUDA(state.reordered_independent_sets, state.batch_size, int);
        allocateCUDA(state.selected_maximal_independent_set, db.num_movable_nodes, int);
        allocateCUDA(state.select_scratch, db.num_movable_nodes, int); 
        allocateCUDA(state.device_num_selected, 1, int);
        ////allocateCUDA(state.device_num_selected_prefix_sum, NUM_NODE_SIZES+1, int);
        //allocateCUDA(state.device_num_clusters_prefix_sum, NUM_NODE_SIZES+1, int);
        ////allocateCUDA(state.node2center, db.num_movable_nodes, int);
        //allocateCUDA(state.centers, state.batch_size, int);
        ////allocateCUDA(state.center_xs, state.batch_size*NUM_NODE_SIZES, T);
        ////allocateCUDA(state.center_ys, state.batch_size*NUM_NODE_SIZES, T);
        //allocateCUDA(state.cluster_sizes, state.batch_size*NUM_NODE_SIZES, T);
        ////allocateCUDA(state.device_num_independent_sets, 1, int);
        allocateCUDA(state.orig_x, state.batch_size*state.set_size, T); 
        allocateCUDA(state.orig_y, state.batch_size*state.set_size, T); 
        allocateCUDA(state.orig_spaces, state.batch_size*state.set_size, Space<T>); 

        allocateCUDA(state.selected_markers, db.num_nodes, int);
        allocateCUDA(state.dependent_markers, db.num_nodes, unsigned char);
        allocateCUDA(state.independent_set_empty_flag, 1, int); 

        allocateCUDA(state.cost_matrices, state.batch_size*state.set_size*state.set_size, typename IndependentSetMatchingState<T>::cost_type);
        allocateCUDA(state.cost_matrices_copy, state.batch_size*state.set_size*state.set_size, typename IndependentSetMatchingState<T>::cost_type);
        allocateCUDA(state.solutions, state.batch_size*state.set_size, int);
        allocateCUDA(state.orig_costs, state.batch_size*state.set_size, typename IndependentSetMatchingState<T>::cost_type);
        allocateCUDA(state.solution_costs, state.batch_size*state.set_size, typename IndependentSetMatchingState<T>::cost_type);
        allocateCUDA(state.net_hpwls, db.num_nets, typename std::remove_pointer<decltype(state.net_hpwls)>::type);
        
        allocateCopyCUDA(state.device_num_moved, &state.num_moved, 1);

        init_auction<T>(state.batch_size, state.set_size, state.auction_scratch, state.stop_flags);
    }
    Shuffler<int, unsigned int> shuffler (1234ULL, state.ordered_nodes, db.num_movable_nodes);

    // initialize host state 
    IndependentSetMatchingCPUState<T> host_state; 
    init_cpu_state(db, state, host_state);

    // initialize kmeans state 
    KMeansState<T> kmeans_state; 
    init_kmeans(db, state, kmeans_state);

    timer_stop = CPUTimer::getGlobaltime(); 
    dreamplacePrint(kINFO, "initializing GPU memory takes %g ms\n", CPUTimer::getTimerPeriod()*(timer_stop-timer_start));

    kernel_timer_start = CPUTimer::getGlobaltime(); 

    // runtime profiling 
    CPUTimer::hr_clock_rep iter_timer_start, iter_timer_stop; 
    int random_shuffle_runs = 0, maximal_independent_set_runs = 0, collect_independent_sets_runs = 0, 
        cost_matrix_construction_runs = 0, independent_sets_solving_runs = 0, apply_solution_runs = 0; 
    CPUTimer::hr_clock_rep random_shuffle_time = 0, maximal_independent_set_time = 0, collect_independent_sets_time = 0, 
                 cost_matrix_construction_time = 0, independent_sets_solving_time = 0, apply_solution_time = 0; 

    std::vector<T> hpwls (max_iters+1); 
    hpwls[0] = compute_total_hpwl(db, db.x, db.y, state.net_hpwls);
    dreamplacePrint(kINFO, "initial hpwl %g\n", hpwls[0]);
    for (int iter = 0; iter < max_iters; ++iter)
    {
        iter_timer_start = CPUTimer::getGlobaltime();

        timer_start = CPUTimer::getGlobaltime();
        //std::random_shuffle(host_ordered_nodes.begin(), host_ordered_nodes.end()); 
        //checkCUDA(cudaMemcpy(state.ordered_nodes, host_ordered_nodes.data(), sizeof(int)*db.num_movable_nodes, cudaMemcpyHostToDevice));
        shuffler(); 
        checkCUDA(cudaDeviceSynchronize());
        timer_stop = CPUTimer::getGlobaltime();
        random_shuffle_time += timer_stop-timer_start; 
        random_shuffle_runs += 1; 

        timer_start = CPUTimer::getGlobaltime(); 
        maximal_independent_set(db, state);
        checkCUDA(cudaDeviceSynchronize()); 
        timer_stop = CPUTimer::getGlobaltime(); 
        maximal_independent_set_time += timer_stop-timer_start; 
        maximal_independent_set_runs += 1; 

        timer_start = CPUTimer::getGlobaltime();
        collect_independent_sets(db, state, kmeans_state, host_db, host_state);
        //collect_independent_sets_cuda2cpu(db, state);
        // better copy here, because state is passed as a copy. 
        // there will not be any effect if copied inside any function 
        ////checkCUDA(cudaMemcpy(&state.num_independent_sets, state.device_num_independent_sets, sizeof(int), cudaMemcpyDeviceToHost));
        checkCUDA(cudaDeviceSynchronize());
        timer_stop = CPUTimer::getGlobaltime(); 
        collect_independent_sets_time += timer_stop-timer_start; 
        collect_independent_sets_runs += 1; 

        timer_start = CPUTimer::getGlobaltime();
        cost_matrix_construction(db, state);
        checkCUDA(cudaDeviceSynchronize());
        timer_stop = CPUTimer::getGlobaltime(); 
        cost_matrix_construction_time += timer_stop-timer_start; 
        cost_matrix_construction_runs += 1; 

        // solve independent sets 
        //state.num_independent_sets = 4; 
        //print_cost_matrix<<<1, 1>>>(state.cost_matrices + state.cost_matrix_size*3, state.set_size, 0);
        timer_start = CPUTimer::getGlobaltime();
        linear_assignment_auction(
                state.cost_matrices, 
                state.solutions, 
                state.num_independent_sets, 
                state.set_size, 
                state.auction_scratch, 
                state.stop_flags, 
                state.auction_max_eps, 
                state.auction_min_eps, 
                state.auction_factor, 
                state.auction_max_iterations
                );
        checkCUDA(cudaDeviceSynchronize());
        timer_stop = CPUTimer::getGlobaltime();
        independent_sets_solving_time += timer_stop-timer_start; 
        independent_sets_solving_runs += 1; 
        //print_solution<<<1, 1>>>(state.solutions + state.set_size*3, state.set_size);

        // apply solutions 
        timer_start = CPUTimer::getGlobaltime();
        apply_solution(db, state);
        checkCUDA(cudaDeviceSynchronize());
        timer_stop = CPUTimer::getGlobaltime();
        apply_solution_time += timer_stop-timer_start; 
        apply_solution_runs += 1; 

        iter_timer_stop = CPUTimer::getGlobaltime(); 
        hpwls[iter+1] = compute_total_hpwl(db, db.x, db.y, state.net_hpwls); 
        if ((iter%(max(max_iters/10, 1))) == 0 || iter+1 == max_iters)
        {
            dreamplacePrint(kINFO, "iteration %d, target hpwl %g, delta %g(%g%%), %d independent sets, moved %g%% cells, runtime %g ms\n", 
                    iter, 
                    hpwls[iter+1], hpwls[iter+1]-hpwls[0], (hpwls[iter+1]-hpwls[0])/hpwls[0]*100, 
                    state.num_independent_sets, 
                    state.num_moved/(double)db.num_movable_nodes*100, 
                    CPUTimer::getTimerPeriod()*(iter_timer_stop-iter_timer_start)
                    );
        }
    }
    kernel_timer_stop = CPUTimer::getGlobaltime(); 
    dreamplacePrint(kDEBUG, "random_shuffle takes %g ms, %d runs, average %g ms\n", 
            CPUTimer::getTimerPeriod()*random_shuffle_time, random_shuffle_runs, CPUTimer::getTimerPeriod()*random_shuffle_time/random_shuffle_runs);
    dreamplacePrint(kDEBUG, "maximal_independent_set takes %g ms, %d runs, average %g ms\n", 
            CPUTimer::getTimerPeriod()*maximal_independent_set_time, maximal_independent_set_runs, CPUTimer::getTimerPeriod()*maximal_independent_set_time/maximal_independent_set_runs);
    dreamplacePrint(kDEBUG, "collect_independent_sets takes %g ms, %d runs, average %g ms\n", 
            CPUTimer::getTimerPeriod()*collect_independent_sets_time, collect_independent_sets_runs, CPUTimer::getTimerPeriod()*collect_independent_sets_time/collect_independent_sets_runs);
    dreamplacePrint(kDEBUG, "cost_matrix_construction takes %g ms, %d runs, average %g ms\n", 
            CPUTimer::getTimerPeriod()*cost_matrix_construction_time, cost_matrix_construction_runs, CPUTimer::getTimerPeriod()*cost_matrix_construction_time/cost_matrix_construction_runs);
    dreamplacePrint(kDEBUG, "independent_sets_solving takes %g ms, %d runs, average %g ms\n", 
            CPUTimer::getTimerPeriod()*independent_sets_solving_time, independent_sets_solving_runs, CPUTimer::getTimerPeriod()*independent_sets_solving_time/independent_sets_solving_runs);
    dreamplacePrint(kDEBUG, "apply_solution takes %g ms, %d runs, average %g ms\n", 
            CPUTimer::getTimerPeriod()*apply_solution_time, apply_solution_runs, CPUTimer::getTimerPeriod()*apply_solution_time/apply_solution_runs);

    // destroy state 
    timer_start = CPUTimer::getGlobaltime();
    {
        ////destroyCUDA(state.node_size_id);
        destroyCUDA(state.spaces);
        destroyCUDA(state.ordered_nodes);
        destroyCUDA(state.independent_sets);
        destroyCUDA(state.independent_set_sizes);
        ////destroyCUDA(state.ordered_independent_sets);
        ////destroyCUDA(state.reordered_independent_sets);
        destroyCUDA(state.selected_maximal_independent_set);
        destroyCUDA(state.select_scratch);
        destroyCUDA(state.device_num_selected);
        ////destroyCUDA(state.device_num_selected_prefix_sum);
        //destroyCUDA(state.device_num_clusters_prefix_sum);
        ////destroyCUDA(state.node2center);
        //destroyCUDA(state.centers);
        ////destroyCUDA(state.center_xs);
        ////destroyCUDA(state.center_ys);
        //destroyCUDA(state.cluster_sizes);

        destroyCUDA(state.net_hpwls);
        destroyCUDA(state.cost_matrices);
        destroyCUDA(state.cost_matrices_copy);
        destroyCUDA(state.solutions);
        destroyCUDA(state.orig_costs);
        destroyCUDA(state.solution_costs);
        destroyCUDA(state.orig_x);
        destroyCUDA(state.orig_y);
        destroyCUDA(state.orig_spaces);
        destroyCUDA(state.selected_markers);
        destroyCUDA(state.dependent_markers);
        destroyCUDA(state.independent_set_empty_flag);
        ////destroyCUDA(state.device_num_independent_sets);
        destroyCUDA(state.device_num_moved);
        destroy_auction(state.auction_scratch, state.stop_flags);
        // destroy kmeans state 
        destroy_kmeans(kmeans_state); 
    }
    timer_stop = CPUTimer::getGlobaltime();
    dreamplacePrint(kINFO, "destroying GPU memory takes %g ms\n", CPUTimer::getTimerPeriod()*(timer_stop-timer_start));

    total_timer_stop = CPUTimer::getGlobaltime(); 

    dreamplacePrint(kINFO, "Kernel time %g ms\n", CPUTimer::getTimerPeriod()*(kernel_timer_stop-kernel_timer_start));
    dreamplacePrint(kINFO, "Independent set matching time %g ms\n", CPUTimer::getTimerPeriod()*(total_timer_stop-total_timer_start));

    return 0; 
}

#define REGISTER_KERNEL_LAUNCHER(T) \
    template int independentSetMatchingCUDALauncher<T>(\
            DetailedPlaceDB<T> db, \
            int batch_size, \
            int set_size, \
            int max_iters, \
            int num_threads \
            ); 

REGISTER_KERNEL_LAUNCHER(float);
REGISTER_KERNEL_LAUNCHER(double);

DREAMPLACE_END_NAMESPACE
