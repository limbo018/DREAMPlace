/**
 * @file   cpu_state.cuh
 * @author Yibo Lin
 * @date   Jul 2019
 */
#ifndef _DREAMPLACE_INDEPENDENT_SET_MATCHING_CPU_STATE_CUH
#define _DREAMPLACE_INDEPENDENT_SET_MATCHING_CPU_STATE_CUH

#include "utility/src/utils.cuh"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
struct DetailedPlaceCPUDB
{
    typedef T type; 
    int num_movable_nodes; 
    int num_bins_x; 
    int num_bins_y; 
    T bin_size_x; 
    T bin_size_y;
    T xl, yl, xh, yh; 
    std::vector<T> node_size_x; 
    std::vector<T> node_size_y; 
    std::vector<T> x; 
    std::vector<T> y; 
};

template <typename T>
struct IndependentSetMatchingCPUState
{
    typedef T type; 
    int batch_size; 
    int set_size; 
    int grid_size; 
    int max_diamond_search_sequence; 
    int num_independent_sets; 
    std::vector<std::vector<int> > independent_sets; 
    std::vector<int> flat_independent_sets; ///< flat version of storage 
    std::vector<int> independent_set_sizes; ///< size of each set 
    std::vector<int> selected_nodes; 
    std::vector<unsigned char> selected_markers; 
    std::vector<int> ordered_nodes; 
    std::vector<BinMapIndex> node2bin_map;  
    std::vector<std::vector<int> > bin2node_map; ///< the first dimension is size, all the cells are categorized by width 
    std::vector<GridIndex<int> > search_grids; 
};

inline int ceil_power2(int v)
{
    return (1<<(int)ceil(log2((float)v)));
}

template <typename DetailedPlaceDBType>
void init_cpu_db(const DetailedPlaceDBType& db, 
        DetailedPlaceCPUDB<typename DetailedPlaceDBType::type>& host_db
        )
{
    host_db.num_movable_nodes = db.num_movable_nodes; 
    host_db.num_bins_x = db.num_bins_x; 
    host_db.num_bins_y = db.num_bins_y; 
    host_db.bin_size_x = db.bin_size_x; 
    host_db.bin_size_y = db.bin_size_y; 
    host_db.xl = db.xl; 
    host_db.yl = db.yl; 
    host_db.xh = db.xh; 
    host_db.yh = db.yh; 
    host_db.node_size_x.resize(db.num_nodes);
    checkCUDA(cudaMemcpy(host_db.node_size_x.data(), db.node_size_x, sizeof(typename DetailedPlaceDBType::type)*db.num_nodes, cudaMemcpyDeviceToHost));
    host_db.node_size_y.resize(db.num_nodes);
    checkCUDA(cudaMemcpy(host_db.node_size_y.data(), db.node_size_y, sizeof(typename DetailedPlaceDBType::type)*db.num_nodes, cudaMemcpyDeviceToHost));
    host_db.x.resize(db.num_nodes);
    host_db.y.resize(db.num_nodes);
}

template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
void init_cpu_state(const DetailedPlaceDBType& db, const IndependentSetMatchingStateType& state,
        IndependentSetMatchingCPUState<typename DetailedPlaceDBType::type>& host_state
        )
{
    host_state.batch_size = state.batch_size; 
    host_state.set_size = state.set_size; 
    host_state.grid_size = ceil_power2(std::max(db.num_bins_x, db.num_bins_y)/8);
    host_state.max_diamond_search_sequence = host_state.grid_size*host_state.grid_size/2; 
    dreamplacePrint(kINFO, "diamond search grid size %d, sequence length %d\n", host_state.grid_size, host_state.max_diamond_search_sequence);
    host_state.selected_nodes.reserve(db.num_movable_nodes);
    host_state.selected_markers.assign(db.num_movable_nodes, 1);
    host_state.ordered_nodes.resize(db.num_movable_nodes);
    host_state.search_grids = diamond_search_sequence(host_state.grid_size, host_state.grid_size); 
    host_state.independent_sets.resize(state.batch_size, std::vector<int>(state.set_size));
    host_state.flat_independent_sets.resize(state.batch_size*state.set_size);
    host_state.independent_set_sizes.resize(state.batch_size);
    host_state.node2bin_map.resize(db.num_movable_nodes);
    host_state.bin2node_map.resize(db.num_bins_x*db.num_bins_y);
}

DREAMPLACE_END_NAMESPACE

#endif
