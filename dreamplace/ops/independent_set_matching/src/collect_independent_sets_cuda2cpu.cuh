#ifndef _DREAMPLACE_INDEPENDENT_SET_MATCHING_COLLECT_INDEPENDENT_SETS_CUDA2CPU_CUH
#define _DREAMPLACE_INDEPENDENT_SET_MATCHING_COLLECT_INDEPENDENT_SETS_CUDA2CPU_CUH

#include "independent_set_matching/src/collect_independent_sets.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
struct DetailedPlaceCPUDBCollectIndependentSets
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
struct IndependentSetMatchingCPUStateCollectIndependentSets
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

    std::vector<std::vector<int> > solutions; 
    std::vector<std::vector<T> > target_pos_x; ///< temporary storage of cell locations 
    std::vector<std::vector<T> > target_pos_y; 
};

template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
int collect_independent_sets_cuda2cpu(const DetailedPlaceDBType& db, IndependentSetMatchingStateType& state)
{
    typedef typename DetailedPlaceDBType::type T; 
    DetailedPlaceCPUDBCollectIndependentSets<T> host_db; 
    IndependentSetMatchingCPUStateCollectIndependentSets<T> host_state; 

    host_db.xl = db.xl; 
    host_db.yl = db.yl; 
    host_db.xh = db.xh; 
    host_db.yh = db.yh; 
    host_db.num_movable_nodes = db.num_movable_nodes; 
    host_db.num_bins_x = db.num_bins_x; 
    host_db.num_bins_y = db.num_bins_y; 
    host_db.bin_size_x = db.bin_size_x; 
    host_db.bin_size_y = db.bin_size_y; 
    host_db.node_size_x.resize(db.num_nodes);
    checkCUDA(cudaMemcpy(host_db.node_size_x.data(), db.node_size_x, sizeof(T)*db.num_nodes, cudaMemcpyDeviceToHost));
    host_db.node_size_y.resize(db.num_nodes);
    checkCUDA(cudaMemcpy(host_db.node_size_y.data(), db.node_size_y, sizeof(T)*db.num_nodes, cudaMemcpyDeviceToHost));
    host_db.x.resize(db.num_nodes);
    checkCUDA(cudaMemcpy(host_db.x.data(), db.x, sizeof(T)*db.num_nodes, cudaMemcpyDeviceToHost));
    host_db.y.resize(db.num_nodes);
    checkCUDA(cudaMemcpy(host_db.y.data(), db.y, sizeof(T)*db.num_nodes, cudaMemcpyDeviceToHost));

    host_state.batch_size = state.batch_size; 
    host_state.set_size = state.set_size; 
    host_state.grid_size = ceil_power2(std::max(db.num_bins_x, db.num_bins_y)/8);
    host_state.max_diamond_search_sequence = host_state.grid_size*host_state.grid_size/2; 
    dreamplacePrint(kINFO, "diamond search grid size %d, sequence length %d\n", host_state.grid_size, host_state.max_diamond_search_sequence);
    host_state.selected_nodes.reserve(db.num_movable_nodes);
    checkCUDA(cudaMemcpy(host_state.selected_nodes.data(), state.selected_maximal_independent_set, sizeof(int)*state.num_selected, cudaMemcpyDeviceToHost));
    host_state.selected_markers.resize(db.num_movable_nodes);
    checkCUDA(cudaMemcpy(host_state.selected_markers.data(), state.selected_markers, sizeof(unsigned char)*db.num_movable_nodes, cudaMemcpyDeviceToHost));
    host_state.ordered_nodes.resize(db.num_movable_nodes);
    checkCUDA(cudaMemcpy(host_state.ordered_nodes.data(), state.ordered_nodes, sizeof(int)*db.num_movable_nodes, cudaMemcpyDeviceToHost));
    host_state.search_grids = diamond_search_sequence(host_state.grid_size, host_state.grid_size); 
    host_state.independent_sets.resize(state.batch_size, std::vector<int>(state.set_size));
    host_state.flat_independent_sets.assign(state.batch_size*state.set_size, std::numeric_limits<int>::max());
    host_state.independent_set_sizes.resize(state.batch_size);
    host_state.node2bin_map.resize(db.num_movable_nodes);
    host_state.bin2node_map.resize(db.num_bins_x*db.num_bins_y);
    host_state.solutions.resize(state.batch_size);
    host_state.target_pos_x.resize(state.batch_size);
    host_state.target_pos_y.resize(state.batch_size);

    // compute on cpu 
    int num_independent_sets = collect_independent_sets(host_db, host_state);

    // update device 
    state.num_independent_sets = num_independent_sets; 
    for (int i = 0; i < num_independent_sets; ++i)
    {
        std::copy(host_state.independent_sets.at(i).begin(), host_state.independent_sets.at(i).end(), 
                host_state.flat_independent_sets.begin()+i*state.set_size 
                );
        host_state.independent_set_sizes.at(i) = host_state.independent_sets.at(i).size();
    }
    checkCUDA(cudaMemcpy(state.independent_sets, host_state.flat_independent_sets.data(), sizeof(int)*state.batch_size*state.set_size, cudaMemcpyHostToDevice));
    checkCUDA(cudaMemcpy(state.independent_set_sizes, host_state.independent_set_sizes.data(), sizeof(int)*state.batch_size, cudaMemcpyHostToDevice));

    return state.num_independent_sets;
}

DREAMPLACE_END_NAMESPACE

#endif
