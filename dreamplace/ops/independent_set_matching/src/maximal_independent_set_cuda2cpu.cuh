#ifndef _DREAMPLACE_INDEPENDENT_SET_MATCHING_MAXIMAL_INDEPENDENT_SET_CUDA2CPU_CUH
#define _DREAMPLACE_INDEPENDENT_SET_MATCHING_MAXIMAL_INDEPENDENT_SET_CUDA2CPU_CUH

#include "independent_set_matching/src/maximal_independent_set.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
struct DetailedPlaceCPUDBMaximumIndependentSet
{
    typedef T type; 
    std::vector<T> node_size_x; 
    std::vector<T> node_size_y; 
    std::vector<T> x; 
    std::vector<T> y; 

    std::vector<int> flat_net2pin_map; 
    std::vector<int> flat_net2pin_start_map; 
    std::vector<int> pin2net_map; 
    std::vector<int> flat_node2pin_map; 
    std::vector<int> flat_node2pin_start_map; 
    std::vector<int> pin2node_map; 
    std::vector<T> pin_offset_x; 
    std::vector<T> pin_offset_y; 
    std::vector<unsigned char> net_mask; 

    T xl, yl, xh, yh; 
    int num_nodes; 
    int num_movable_nodes; 
    int num_nets; 
    int num_pins; 
};

template <typename T>
struct IndependentSetMatchingCPUStateMaximumIndependentSet
{
    typedef T type; 

    std::vector<int> ordered_nodes; 
    std::vector<unsigned char> selected_markers; 
    std::vector<unsigned char> dependent_markers; 
    std::vector<int> selected_nodes; 

    int num_selected; 
    T skip_threshold; 
};

template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
void maximal_independent_set_cuda2cpu(DetailedPlaceDBType db, IndependentSetMatchingStateType state)
{
    typedef typename DetailedPlaceDBType::type T; 
    DetailedPlaceCPUDBMaximumIndependentSet<T> host_db; 
    IndependentSetMatchingCPUStateMaximumIndependentSet<T> host_state; 

    host_db.xl = db.xl; 
    host_db.yl = db.yl; 
    host_db.xh = db.xh; 
    host_db.yh = db.yh; 
    host_db.num_nodes = db.num_nodes; 
    host_db.num_movable_nodes = db.num_movable_nodes; 
    host_db.num_nets = db.num_nets; 
    host_db.num_pins = db.num_pins; 
    host_db.node_size_x.resize(db.num_nodes);
    checkCUDA(cudaMemcpy(host_db.node_size_x.data(), db.node_size_x, sizeof(T)*db.num_nodes, cudaMemcpyDeviceToHost));
    host_db.node_size_y.resize(db.num_nodes);
    checkCUDA(cudaMemcpy(host_db.node_size_y.data(), db.node_size_y, sizeof(T)*db.num_nodes, cudaMemcpyDeviceToHost));
    host_db.x.resize(db.num_nodes);
    checkCUDA(cudaMemcpy(host_db.x.data(), db.x, sizeof(T)*db.num_nodes, cudaMemcpyDeviceToHost));
    host_db.y.resize(db.num_nodes);
    checkCUDA(cudaMemcpy(host_db.y.data(), db.y, sizeof(T)*db.num_nodes, cudaMemcpyDeviceToHost));
    host_db.flat_net2pin_map.resize(db.num_pins);
    checkCUDA(cudaMemcpy(host_db.flat_net2pin_map.data(), db.flat_net2pin_map, sizeof(int)*db.num_pins, cudaMemcpyDeviceToHost));
    host_db.flat_net2pin_start_map.resize(db.num_nets+1); 
    checkCUDA(cudaMemcpy(host_db.flat_net2pin_start_map.data(), db.flat_net2pin_start_map, sizeof(int)*(db.num_nets+1), cudaMemcpyDeviceToHost));
    host_db.pin2net_map.resize(db.num_pins);
    checkCUDA(cudaMemcpy(host_db.pin2net_map.data(), db.pin2net_map, sizeof(int)*db.num_pins, cudaMemcpyDeviceToHost));
    host_db.flat_node2pin_map.resize(db.num_pins); 
    checkCUDA(cudaMemcpy(host_db.flat_node2pin_map.data(), db.flat_node2pin_map, sizeof(int)*db.num_pins, cudaMemcpyDeviceToHost));
    host_db.flat_node2pin_start_map.resize(db.num_nodes+1);
    checkCUDA(cudaMemcpy(host_db.flat_node2pin_start_map.data(), db.flat_node2pin_start_map, sizeof(int)*(db.num_nodes+1), cudaMemcpyDeviceToHost));
    host_db.pin2node_map.resize(db.num_pins);
    checkCUDA(cudaMemcpy(host_db.pin2node_map.data(), db.pin2node_map, sizeof(int)*db.num_pins, cudaMemcpyDeviceToHost));
    host_db.pin_offset_x.resize(db.num_pins);
    checkCUDA(cudaMemcpy(host_db.pin_offset_x.data(), db.pin_offset_x, sizeof(int)*db.num_pins, cudaMemcpyDeviceToHost));
    host_db.pin_offset_y.resize(db.num_pins);
    checkCUDA(cudaMemcpy(host_db.pin_offset_y.data(), db.pin_offset_y, sizeof(int)*db.num_pins, cudaMemcpyDeviceToHost));
    host_db.net_mask.resize(db.num_nets);
    checkCUDA(cudaMemcpy(host_db.net_mask.data(), db.net_mask, sizeof(unsigned char)*db.num_nets, cudaMemcpyDeviceToHost));

    host_state.skip_threshold = state.skip_threshold; 
    host_state.ordered_nodes.resize(db.num_movable_nodes); 
    checkCUDA(cudaMemcpy(host_state.ordered_nodes.data(), state.ordered_nodes, sizeof(int)*db.num_movable_nodes, cudaMemcpyDeviceToHost));
    //std::iota(host_state.ordered_nodes.begin(), host_state.ordered_nodes.end(), 0);
    //std::random_shuffle(host_state.ordered_nodes.begin(), host_state.ordered_nodes.end());
    host_state.selected_markers.assign(db.num_nodes, 0); 
    host_state.dependent_markers.assign(db.num_nodes, 0); 
    host_state.selected_nodes.reserve(db.num_movable_nodes); 

    maximal_independent_set_parallel(host_db, host_state);

    for (int i = 0; i < db.num_movable_nodes; ++i)
    {
        if (host_state.selected_markers.at(i))
        {
            host_state.selected_nodes.push_back(i);
        }
    }

    state.num_selected = host_state.num_selected; 
    checkCUDA(cudaMemcpy(state.selected_markers, host_state.selected_markers.data(), sizeof(unsigned char)*db.num_movable_nodes, cudaMemcpyHostToDevice));
    checkCUDA(cudaMemcpy(state.selected_maximal_independent_set, host_state.selected_nodes.data(), sizeof(int)*host_state.selected_nodes.size(), cudaMemcpyHostToDevice));
}

DREAMPLACE_END_NAMESPACE

#endif
