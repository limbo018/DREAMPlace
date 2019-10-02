
#ifndef _DREAMPLACE_INDEPENDENT_SET_MATCHING_COST_MATRIX_CONSTRUCTION_CUDA2CPU_CUH
#define _DREAMPLACE_INDEPENDENT_SET_MATCHING_COST_MATRIX_CONSTRUCTION_CUDA2CPU_CUH

#include "independent_set_matching/src/cost_matrix_construction.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
struct DetailedPlaceCPUDBCostMatrixConstruction
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
struct IndependentSetMatchingCPUStateCostMatrixConstruction
{
    typedef T type; 

    std::vector<Space<T> > spaces; ///< not used yet 
    std::vector<std::vector<int> > independent_sets; 

    std::vector<std::vector<int> > cost_matrices; ///< the convergence rate is related to numerical scale 

    int large_number; 
};

template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
void cost_matrix_construction_cuda2cpu(const DetailedPlaceDBType& db, IndependentSetMatchingStateType& state)
{
    typedef typename DetailedPlaceDBType::type T; 
    DetailedPlaceCPUDBCostMatrixConstruction<T> host_db; 
    IndependentSetMatchingCPUStateCostMatrixConstruction<T> host_state; 

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

    host_state.large_number = state.large_number; 
    host_state.spaces.resize(db.num_movable_nodes); 
    checkCUDA(cudaMemcpy(host_state.spaces.data(), state.spaces, sizeof(Space<T>)*db.num_movable_nodes, cudaMemcpyDeviceToHost));
    std::vector<int> flat_independent_sets (state.batch_size*state.set_size);
    checkCUDA(cudaMemcpy(flat_independent_sets.data(), state.independent_sets, sizeof(int)*state.batch_size*state.set_size, cudaMemcpyDeviceToHost));
    host_state.independent_sets.resize(state.batch_size);
    for (int i = 0; i < state.num_independent_sets; ++i)
    {
        host_state.independent_sets.at(i).insert(host_state.independent_sets.at(i).begin(), 
                flat_independent_sets.begin()+i*state.set_size, 
                flat_independent_sets.begin()+(i+1)*state.set_size
                );
    }
    //std::vector<int> flat_cost_matrices (state.batch_size*state.set_size*state.set_size);
    //checkCUDA(cudaMemcpy(flat_cost_matrices.data(), state.cost_matrices, sizeof(int)*state.batch_size*state.set_size*state.set_size, cudaMemcpyDeviceToHost));
    host_state.cost_matrices.resize(state.batch_size); 
    //for (int i = 0; i < state.num_independent_sets; ++i)
    //{
    //    host_state.cost_matrices.at(i).insert(host_state.cost_matrices.at(i).begin(), 
    //            flat_cost_matrices.begin()+i*state.set_size*state.set_size, 
    //            flat_cost_matrices.begin()+(i+1)*state.set_size*state.set_size
    //            );
    //}

    // compute on cpu 
    for (int i = 0; i < state.num_independent_sets; ++i)
    {
        auto& independent_set = host_state.independent_sets.at(i);
        for (unsigned int j = 0; j < independent_set.size(); ++j)
        {
            if (independent_set.at(j) >= db.num_movable_nodes)
            {
                dreamplaceAssert(*std::min_element(independent_set.begin()+j, independent_set.end()) >= db.num_movable_nodes);
                independent_set.resize(j); 
                break; 
            }
        }
    }
    bool major = false; // row major 
    for (int i = 0; i < state.num_independent_sets; ++i)
    {
        auto const& independent_set = host_state.independent_sets.at(i);
        auto& cost_matrix = host_state.cost_matrices.at(i);
        cost_matrix.resize(independent_set.size()*independent_set.size());

        cost_matrix_construction(host_db, host_state, major, i);

        // map to large matrix 
        std::vector<int> tmp_cost_matrix (state.set_size*state.set_size, state.large_number);
        for (int k = 0; k < state.set_size; ++k)
        {
            tmp_cost_matrix.at(k*state.set_size + k) = 0; 
        }
        for (unsigned int k = 0; k < independent_set.size(); ++k)
        {
            for (unsigned int h = 0; h < independent_set.size(); ++h)
            {
                tmp_cost_matrix.at(k*state.set_size + h) = cost_matrix.at(k*independent_set.size() + h);
            }
        }
        //if (i == 0)
        //{
        //    dreamplacePrint(kDEBUG, "set[%lu]: ", independent_set.size());
        //    for (auto node_id : independent_set)
        //    {
        //        dreamplacePrint(kNONE, "%d ", node_id);
        //    }
        //    dreamplacePrint(kNONE, "\n");
        //    dreamplacePrint(kDEBUG, "cost matrix\n");
        //    for (unsigned int j = 0; j < independent_set.size(); ++j)
        //    {
        //        for (unsigned int k = 0; k < independent_set.size(); ++k)
        //        {
        //            dreamplacePrint(kNONE, "%d ", host_state.cost_matrices.at(i).at(j*independent_set.size()+k));
        //        }
        //        dreamplacePrint(kNONE, "\n");
        //    }
        //    dreamplacePrint(kDEBUG, "cost matrix\n");
        //    for (int j = 0; j < state.set_size; ++j)
        //    {
        //        for (int k = 0; k < state.set_size; ++k)
        //        {
        //            dreamplacePrint(kNONE, "%g, ", (double)tmp_cost_matrix.at(j*state.set_size+k));
        //        }
        //        dreamplacePrint(kNONE, "\n");
        //    }
        //}
        // convert to maximization 
        int large_number = *std::max_element(tmp_cost_matrix.begin(), tmp_cost_matrix.end());
        for (auto& value : tmp_cost_matrix)
        {
            value = large_number-value; 
        }
        cost_matrix = tmp_cost_matrix; 
    }

    // copy back to device 
    std::vector<int> flat_cost_matrices(state.num_independent_sets*state.set_size*state.set_size, 0);
    for (int i = 0; i < state.num_independent_sets; ++i)
    {
        for (int k = 0; k < state.set_size; ++k)
        {
            for (int h = 0; h < state.set_size; ++h)
            {
                flat_cost_matrices.at(i*state.set_size*state.set_size + k*state.set_size + h) = host_state.cost_matrices.at(i).at(k*state.set_size + h);
            }
        }
    }
    checkCUDA(cudaMemcpy(state.cost_matrices, flat_cost_matrices.data(), sizeof(int)*state.num_independent_sets*state.set_size*state.set_size, cudaMemcpyHostToDevice));
}

DREAMPLACE_END_NAMESPACE

#endif
