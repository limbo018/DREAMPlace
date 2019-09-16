/**
 * @file   global_swap.cpp
 * @author Yibo Lin
 * @date   Jan 2019
 */
#include <vector>
#include <algorithm>
#include <numeric>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <limits>
#include <omp.h>
#include "utility/src/torch.h"
#include "utility/src/Msg.h"
#include "utility/src/diamond_search.h"
#include "utility/src/DetailedPlaceDB.h"
#include "utility/src/DetailedPlaceDBUtils.h"
#include "k_reorder/src/quick_perm.h"
#include "k_reorder/src/compute_independent_rows.h"
#include "k_reorder/src/compute_reorder_instances.h"

DREAMPLACE_BEGIN_NAMESPACE

#define MAX_NUM_THREADS 128 
//#define DEBUG 

typedef std::chrono::high_resolution_clock::rep hr_clock_rep;

inline hr_clock_rep get_globaltime(void) 
{
	using namespace std::chrono;
	return high_resolution_clock::now().time_since_epoch().count();
}

// Returns the period in miliseconds
inline double get_timer_period(void) 
{
	using namespace std::chrono;
	return 1000.0 * high_resolution_clock::period::num / high_resolution_clock::period::den;
}

template <typename T>
struct KReorderState 
{
    std::vector<std::vector<int> > row2node_map; 
    std::vector<std::vector<int> > permutations; 
    std::vector<unsigned char> net_markers; 
    std::vector<T> node_space_x; ///< cell size with spaces 
    std::vector<T> target_x[MAX_NUM_THREADS]; 
    std::vector<T> target_sizes[MAX_NUM_THREADS]; 
    std::vector<int> target_nodes[MAX_NUM_THREADS]; 

    std::vector<unsigned char> adjacency_matrix; ///< adjacency matrix for row graph 
    std::vector<std::vector<int> > row_graph; ///< adjacency list for row graph 
    std::vector<std::vector<int> > independent_rows; 
    std::vector<std::vector<KReorderInstance> > reorder_instances;

    int K; 
    int num_moved; 
    int num_threads; 
};

template <typename T>
void compute_position(const DetailedPlaceDB<T>& db, KReorderState<T>& state, int row_id, int idx_bgn, int idx_end, int permute_id)
{
    auto row2nodes = state.row2node_map.at(row_id).data() + idx_bgn;
    auto const& permutation = state.permutations.at(permute_id);
    int K = idx_end-idx_bgn; 

    int tid = omp_get_thread_num();
    auto& target_x = state.target_x[tid]; 
    auto& target_sizes = state.target_sizes[tid]; 

    target_x.resize(idx_end-idx_bgn);
    target_sizes.resize(idx_end-idx_bgn);

    // find left boundary  
    if (K)
    {
        int node_id = row2nodes[0]; 
        target_x.at(0) = db.x[node_id]; 
    }
    // record sizes, and pack to left 
    for (int i = 0; i < K; ++i)
    {
        int node_id = row2nodes[i];
        target_sizes[permutation.at(i)] = state.node_space_x[node_id];
    }
    for (int i = 1; i < K; ++i)
    {
        target_x[i] = target_x[i-1] + target_sizes[i-1];
    }

#ifdef DEBUG 
    for (int i = 0; i < K; ++i)
    {
        int node_id = row2nodes[i];
        if (node_id == 123651 || node_id == 123493 || node_id == 123652)
        {
            dreamplacePrint(kNONE, "[%d, %d]\n", idx_bgn, idx_end);
            for (int j = 0; j < K; ++j)
            {
                int node = row2nodes[j];
                dreamplacePrint(kNONE, "node[%d] = %g, size %g, got size %g\n", node, target_x[j], state.node_space_x[node], target_sizes[j]); 
            }
            break; 
        }
    }
#endif
}

template <typename T>
T compute_reorder_hpwl(const DetailedPlaceDB<T>& db, KReorderState<T>& state, int row_id, int idx_bgn, int idx_end, int permute_id)
{
    auto const& row2nodes = state.row2node_map.at(row_id);
    auto const& permutation = state.permutations.at(permute_id);

#ifdef DEBUG
    for (int i = idx_bgn; i < idx_end; ++i)
    {
        int node_id = row2nodes.at(i); 
        dreamplaceAssert(node_id < db.num_movable_nodes);
    }
#endif

    for (int i = 0; i < idx_end-idx_bgn; ++i)
    {
        if (permutation.at(i) >= idx_end-idx_bgn)
        {
            return std::numeric_limits<T>::max();
        }
    }

    compute_position(db, state, row_id, idx_bgn, idx_end, permute_id);

    int tid = omp_get_thread_num(); 
    auto& target_x = state.target_x[tid]; 
    T cost = 0; 
    for (int i = idx_bgn; i < idx_end; ++i)
    {
        int node_id = row2nodes.at(i);
        for (int node2pin_id = db.flat_node2pin_start_map[node_id]; node2pin_id < db.flat_node2pin_start_map[node_id+1]; ++node2pin_id)
        {
            int node_pin_id = db.flat_node2pin_map[node2pin_id];
            int net_id = db.pin2net_map[node_pin_id];
            if (db.net_mask[net_id] && !state.net_markers[net_id])
            {
                T bxl = db.xh; 
                T bxh = db.xl; 
                for (int net2pin_id = db.flat_net2pin_start_map[net_id]; net2pin_id < db.flat_net2pin_start_map[net_id+1]; ++net2pin_id)
                {
                    int net_pin_id = db.flat_net2pin_map[net2pin_id];
                    int other_node_id = db.pin2node_map[net_pin_id];
                    T other_node_xl; 
                    auto found = std::find(row2nodes.begin()+idx_bgn, row2nodes.begin()+idx_end, other_node_id);
                    if (found != row2nodes.begin()+idx_end)
                    {
                        int distance = std::distance(row2nodes.begin()+idx_bgn, found);
                        int permuted_offset = permutation.at(distance);
                        //int permute_node_id = row2nodes.at(idx_bgn + permuted_offset);
                        other_node_xl = target_x.at(permuted_offset);
                    }
                    else 
                    {
                        other_node_xl = db.x[other_node_id];
                    }
                    other_node_xl += db.pin_offset_x[net_pin_id];
                    bxl = std::min(bxl, other_node_xl);
                    bxh = std::max(bxh, other_node_xl);
                }
                cost += bxh-bxl; 
                state.net_markers[net_id] = 1; 
            }
        }
    }
    for (int i = idx_bgn; i < idx_end; ++i)
    {
        int node_id = row2nodes.at(i);
        for (int node2pin_id = db.flat_node2pin_start_map[node_id]; node2pin_id < db.flat_node2pin_start_map[node_id+1]; ++node2pin_id)
        {
            int node_pin_id = db.flat_node2pin_map[node2pin_id];
            int net_id = db.pin2net_map[node_pin_id];
            state.net_markers[net_id] = 0; 
        }
    }
#ifdef DEBUG
    if (MAX_NUM_THREADS == 1)
    {
      dreamplaceAssert(std::count(state.net_markers.begin(), state.net_markers.end(), 1) == 0);
    }
#endif 
    return cost; 
}

template <typename T>
void apply_reorder(DetailedPlaceDB<T>& db, KReorderState<T>& state, int row_id, int idx_bgn, int idx_end, const std::vector<int>& permutation, const std::vector<T>& target_x)
{
    auto row2nodes = state.row2node_map.at(row_id).data() + idx_bgn;
    int K = idx_end-idx_bgn; 

    int tid = omp_get_thread_num(); 
    auto& target_nodes = state.target_nodes[tid]; 
    target_nodes.resize(K);

    for (int i = 0; i < K; ++i)
    {
        int node_id = row2nodes[i];
        target_nodes.at(i) = node_id; 
    }

    for (int i = 0; i < K; ++i)
    {
        int node_id = row2nodes[i];
        T xx = target_x.at(permutation.at(i));
        if (db.x[node_id] != xx)
        {
            state.num_moved += 1; 
        }
        db.x[node_id] = xx; 
    }

    for (int i = 0; i < K; ++i)
    {
        row2nodes[permutation.at(i)] = target_nodes.at(i);
    }
}

/// @brief mark a node and as first level connected nodes as dependent 
/// only nodes with the same sizes are marked 
template <typename DetailedPlaceDBType, typename StateType>
void mark_dependent_nodes(const DetailedPlaceDBType& db, StateType& state, int node_id, unsigned char value)
{
    // in case all nets are masked 
    int node2pin_start = db.flat_node2pin_start_map[node_id];
    int node2pin_end = db.flat_node2pin_start_map[node_id+1];
    for (int node2pin_id = node2pin_start; node2pin_id < node2pin_end; ++node2pin_id)
    {
        int node_pin_id = db.flat_node2pin_map[node2pin_id];
        int net_id = db.pin2net_map[node_pin_id];
        if (db.net_mask[net_id])
        {
            int net2pin_start = db.flat_net2pin_start_map[net_id];
            int net2pin_end = db.flat_net2pin_start_map[net_id+1];
            for (int net2pin_id = net2pin_start; net2pin_id < net2pin_end; ++net2pin_id)
            {
                int net_pin_id = db.flat_net2pin_map[net2pin_id];
                int other_node_id = db.pin2node_map[net_pin_id];
                state.dependent_markers[other_node_id] = value; 
            }
        }
    }
}

/// @brief global swap algorithm for detailed placement 
template <typename T>
int kreorderCPULauncher(DetailedPlaceDB<T>& db, int K, int max_iters, int
                        num_threads)
{
    dreamplacePrint(kDEBUG, "%d-reorder\n", K);
    T stop_threshold = 0.1/100; 

    // profiling variables 
	hr_clock_rep timer_start[MAX_NUM_THREADS], timer_stop[MAX_NUM_THREADS];
	hr_clock_rep compute_reorder_hpwl_time[MAX_NUM_THREADS] = {0};
	int compute_reorder_hpwl_runs[MAX_NUM_THREADS] = {0};
	hr_clock_rep apply_reorder_time[MAX_NUM_THREADS] = {0};
	int apply_reorder_runs[MAX_NUM_THREADS] = {0};
	hr_clock_rep iter_time_start, iter_time_stop;

    KReorderState<T> state; 
    state.K = K; 
    state.num_threads = std::min(std::max(num_threads, 1), MAX_NUM_THREADS); 

    // divide layout into rows
    // distribute cells into them 
    state.row2node_map.resize(db.num_sites_y);
    // map node index to its location in row2node_map 
    // we can compute the rows, so only the index within a row of row2node_map is stored 
    //state.node2row_map.resize(db.num_nodes);

    // distribute cells to rows 
    db.make_row2node_map(db.x, db.y, state.row2node_map); 

    state.node_space_x.resize(db.num_nodes);
    for (int i = 0; i < db.num_sites_y; ++i)
    {
        for (unsigned int j = 0; j < state.row2node_map.at(i).size(); ++j)
        {
            int node_id = state.row2node_map[i][j];
            auto& space = state.node_space_x[node_id];
            T space_xl = db.x[node_id]; 
            T space_xh = db.xh; 
            if (j+1 < state.row2node_map[i].size())
            {
                int right_node_id = state.row2node_map[i][j+1];
                space_xh = std::min(space_xh, db.x[right_node_id]);
            }
            space = space_xh-space_xl;
#ifdef DEBUG
            if (node_id < db.num_movable_nodes)
            {
              if (!(space >= db.node_size_x[node_id]))
              {
                dreamplacePrint(kNONE, "space (%g, %g), node %d (%g, %g), layout (%g, %g)\n",
                       space_xl, space_xh, node_id, db.x[node_id],
                       db.x[node_id]+db.node_size_x[node_id], 
                       db.xl, db.xh);
                if (j+1 < state.row2node_map[i].size())
                {
                  int right_node_id = state.row2node_map[i][j+1];
                  dreamplacePrint(kNONE, "right node %d (%g, %g)\n", right_node_id, 
                         db.x[right_node_id], db.x[right_node_id] +
                         db.node_size_x[right_node_id]);
                }

              }
                dreamplaceAssert(space >= db.node_size_x[node_id]);
            }
#endif
        }
    }

    state.permutations = quick_perm(K); 
    state.net_markers.assign(db.num_nets, 0);

    timer_start[0] = get_globaltime(); 
    compute_row_conflict_graph(db, state); 
    compute_independent_rows(db, state); 
    timer_stop[0] = get_globaltime(); 
    dreamplacePrint(kDEBUG, "compute_independent_rows takes %g ms\n", get_timer_period()*(timer_stop[0]-timer_start[0]));

    // fix random seed 
    std::srand(1000);

    std::vector<T> best_target_x[MAX_NUM_THREADS]; 

    // count number of movement 
    state.num_moved = 0; 
    T hpwls [max_iters+1]; 
    hpwls[0] = db.compute_total_hpwl();
    dreamplacePrint(kINFO, "initial hpwl = %.3f\n", hpwls[0]);

    for (int iter = 0; iter < max_iters; ++iter)
    {
        iter_time_start = get_globaltime();

        for (unsigned int group_id = 0; group_id < state.independent_rows.size(); ++group_id)
        {
            auto const& independent_rows = state.independent_rows[group_id]; 
            unsigned int num_independent_rows = independent_rows.size();
#pragma omp parallel for num_threads (state.num_threads) schedule(dynamic, 1)
            for (unsigned int group_row_id = 0; group_row_id < num_independent_rows; ++group_row_id)
            {
                int tid = omp_get_thread_num(); 
                auto& target_x = state.target_x[tid]; 
                auto& best_target_x_tid = best_target_x[tid]; 

                int row_id = independent_rows.at(group_row_id); 
                auto const& row2nodes = state.row2node_map.at(row_id);
                for (int sub_id = 0; sub_id < (int)row2nodes.size(); sub_id += K/2)
                {
                    int idx_bgn = sub_id; 
                    int idx_end = std::min(sub_id+K, (int)row2nodes.size());
                    // stop at fixed cells and multi-row height cells 
                    for (int i = idx_bgn; i < idx_end; ++i)
                    {
                        int node_id = row2nodes.at(i);
                        if (node_id >= db.num_movable_nodes || db.node_size_y[node_id] > db.row_height)
                        {
                            idx_end = i; 
                            break; 
                        }
                    }
                    if (idx_end-idx_bgn < 2)
                    {
                        continue; 
                    }
                    T best_cost = std::numeric_limits<T>::max();
                    int best_pi = std::numeric_limits<int>::max(); 
                    timer_start[tid] = get_globaltime();
                    for (unsigned int pi = 0; pi < state.permutations.size(); ++pi)
                    {
                        T cost = compute_reorder_hpwl(db, state, row_id, idx_bgn, idx_end, pi);
                        if (cost < best_cost)
                        {
                            best_cost = cost; 
                            best_pi = pi; 
                            best_target_x_tid = target_x; 
                        }
                    }
                    timer_stop[tid] = get_globaltime();
                    compute_reorder_hpwl_time[tid] += timer_stop[tid]-timer_start[tid]; 
                    compute_reorder_hpwl_runs[tid] += state.permutations.size();

                    timer_start[tid] = get_globaltime();
                    apply_reorder(db, state, row_id, idx_bgn, idx_end, state.permutations.at(best_pi), best_target_x_tid);
                    timer_stop[tid] = get_globaltime();
                    apply_reorder_time[tid] += timer_stop[tid]-timer_start[tid]; 
                    apply_reorder_runs[tid] += 1; 
                }
            }
        }

        iter_time_stop = get_globaltime();
        dreamplacePrint(kINFO, "Iter %d time (ms) \t %g\n", iter, get_timer_period() * (iter_time_stop - iter_time_start));

        hpwls[iter+1] = db.compute_total_hpwl();
        dreamplacePrint(kINFO, "iteration %d: hpwl %.3f => %.3f (imp. %g%%)\n", iter, hpwls[0], hpwls[iter+1], (1.0-hpwls[iter+1]/(double)hpwls[0])*100);

        if ((iter&1) && hpwls[iter]-hpwls[iter-1] > -stop_threshold*hpwls[0])
        {
            break; 
        }
    }

	dreamplacePrint(kINFO, "kernel \t time (ms) \t runs\n");

    {
        auto time = std::accumulate(compute_reorder_hpwl_time, compute_reorder_hpwl_time+MAX_NUM_THREADS, (T)0);
        auto runs = std::accumulate(compute_reorder_hpwl_runs, compute_reorder_hpwl_runs+MAX_NUM_THREADS, 0);
        dreamplacePrint(kINFO, "compute_reorder_hpwl \t %g \t %d \t %g\n", 
                get_timer_period() * time, runs, 
                get_timer_period() * time / runs);
    }
    {
        auto time = std::accumulate(apply_reorder_time, apply_reorder_time+MAX_NUM_THREADS, (T)0); 
        auto runs = std::accumulate(apply_reorder_runs, apply_reorder_runs+MAX_NUM_THREADS, 0);
        dreamplacePrint(kINFO, "apply_reorder \t %g \t %d \t %g\n", 
                get_timer_period() * time, runs, 
                get_timer_period() * time / runs);
    }

    bool legal_flag = db.check_legality();
    dreamplacePrint(kDEBUG, "legal_flag = %d\n", (int)legal_flag);

    //db.draw_place("final.gds");

    return 0; 
}

#define CHECK_FLAT(x) AT_ASSERTM(!x.is_cuda() && x.ndimension() == 1, #x "must be a flat tensor on CPU")
#define CHECK_EVEN(x) AT_ASSERTM((x.numel()&1) == 0, #x "must have even number of elements")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x "must be contiguous")

at::Tensor k_reorder_forward(
        at::Tensor init_pos,
        at::Tensor node_size_x,
        at::Tensor node_size_y,
        at::Tensor flat_net2pin_map, 
        at::Tensor flat_net2pin_start_map, 
        at::Tensor pin2net_map, 
        at::Tensor flat_node2pin_map, 
        at::Tensor flat_node2pin_start_map, 
        at::Tensor pin2node_map, 
        at::Tensor pin_offset_x, 
        at::Tensor pin_offset_y, 
        at::Tensor net_mask, 
        double xl, 
        double yl, 
        double xh, 
        double yh, 
        double site_width, double row_height, 
        int num_bins_x, 
        int num_bins_y,
        int num_movable_nodes, 
        int num_filler_nodes, 
        int K, 
        int max_iters, 
        int num_threads
        )
{
    CHECK_FLAT(init_pos); 
    CHECK_EVEN(init_pos);
    CHECK_CONTIGUOUS(init_pos);

    auto pos = init_pos.clone();

	hr_clock_rep total_time_start, total_time_stop;
    total_time_start = get_globaltime(); 

    // Call the cuda kernel launcher
    AT_DISPATCH_FLOATING_TYPES(pos.type(), "kreorderCPULauncher", [&] {
            auto db = make_placedb<scalar_t>(
                    init_pos,
                    pos, 
                    node_size_x, node_size_y,
                    flat_net2pin_map, flat_net2pin_start_map, pin2net_map, 
                    flat_node2pin_map, flat_node2pin_start_map, pin2node_map, 
                    pin_offset_x, pin_offset_y, 
                    net_mask, 
                    xl, yl, xh, yh, 
                    site_width, row_height, 
                    num_bins_x, num_bins_y,
                    num_movable_nodes, num_filler_nodes
                    );
            kreorderCPULauncher(db, K, max_iters, num_threads);
            });
    total_time_stop = get_globaltime();
    dreamplacePrint(kINFO, "K-reorder time: %g ms\n", get_timer_period()*(total_time_stop-total_time_start));

    return pos; 
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("k_reorder", &DREAMPLACE_NAMESPACE::k_reorder_forward, "K-reorder");
}
