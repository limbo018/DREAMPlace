/**
 * @file   apply_solution_cuda2cpu.h
 * @author Yibo Lin
 * @date   Jul 2019
 */

#ifndef _DREAMPLACE_INDEPENDENT_SET_MATCHING_APPLY_SOLUTION_CUDA2CPU_H
#define _DREAMPLACE_INDEPENDENT_SET_MATCHING_APPLY_SOLUTION_CUDA2CPU_H

#include "independent_set_matching/src/apply_solution.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
struct DetailedPlaceCPUDBApplySolution
{
    typedef T type; 
    int num_movable_nodes; 
    std::vector<T> node_size_x; 
    std::vector<T> node_size_y; 
    std::vector<T> x; 
    std::vector<T> y; 
};

template <typename T>
struct IndependentSetMatchingCPUStateApplySolution
{
    typedef T type; 

    std::vector<std::vector<int> > independent_sets; 

    std::vector<Space<T> > spaces; ///< not used yet 

    std::vector<std::vector<int> > cost_matrices; ///< the convergence rate is related to numerical scale 
    std::vector<std::vector<int> > solutions; 
    std::vector<int> orig_costs; ///< original cost before matching 
    std::vector<int> target_costs; ///< target cost after matching
    std::vector<std::vector<T> > target_pos_x; ///< temporary storage of cell locations 
    std::vector<std::vector<T> > target_pos_y; 
    std::vector<std::vector<Space<T> > > target_spaces; ///< not used yet 

    std::vector<char> stop_flags; 

    int num_moved; 
};

template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
void apply_solution_cuda2cpu(DetailedPlaceDBType& db, IndependentSetMatchingStateType& state)
{
    typedef typename DetailedPlaceDBType::type T; 
    DetailedPlaceCPUDBApplySolution<T> host_db; 
    IndependentSetMatchingCPUStateApplySolution<T> host_state; 

    host_db.num_movable_nodes = db.num_movable_nodes; 
    host_db.node_size_x.resize(db.num_movable_nodes);
    checkCUDA(cudaMemcpy(host_db.node_size_x.data(), db.node_size_x, sizeof(T)*db.num_movable_nodes, cudaMemcpyDeviceToHost));
    host_db.node_size_y.resize(db.num_movable_nodes);
    checkCUDA(cudaMemcpy(host_db.node_size_y.data(), db.node_size_y, sizeof(T)*db.num_movable_nodes, cudaMemcpyDeviceToHost));
    host_db.x.resize(db.num_movable_nodes);
    checkCUDA(cudaMemcpy(host_db.x.data(), db.x, sizeof(T)*db.num_movable_nodes, cudaMemcpyDeviceToHost));
    host_db.y.resize(db.num_movable_nodes);
    checkCUDA(cudaMemcpy(host_db.y.data(), db.y, sizeof(T)*db.num_movable_nodes, cudaMemcpyDeviceToHost));

    host_state.num_moved = state.num_moved; 
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
    host_state.spaces.resize(db.num_movable_nodes); 
    checkCUDA(cudaMemcpy(host_state.spaces.data(), state.spaces, sizeof(Space<T>)*db.num_movable_nodes, cudaMemcpyDeviceToHost));
    std::vector<int> flat_cost_matrices (state.batch_size*state.set_size*state.set_size);
    checkCUDA(cudaMemcpy(flat_cost_matrices.data(), state.cost_matrices, sizeof(int)*state.batch_size*state.set_size*state.set_size, cudaMemcpyDeviceToHost));
    host_state.cost_matrices.resize(state.batch_size); 
    for (int i = 0; i < state.num_independent_sets; ++i)
    {
        host_state.cost_matrices.at(i).insert(host_state.cost_matrices.at(i).begin(), 
                flat_cost_matrices.begin()+i*state.set_size*state.set_size, 
                flat_cost_matrices.begin()+(i+1)*state.set_size*state.set_size
                );
    }
    std::vector<int> flat_solutions (state.batch_size*state.set_size);
    checkCUDA(cudaMemcpy(flat_solutions.data(), state.solutions, sizeof(int)*state.batch_size*state.set_size, cudaMemcpyDeviceToHost));
    host_state.solutions.resize(state.batch_size); 
    for (int i = 0; i < state.num_independent_sets; ++i)
    {
        host_state.solutions.at(i).insert(host_state.solutions.at(i).begin(), 
                flat_solutions.begin()+i*state.set_size, 
                flat_solutions.begin()+(i+1)*state.set_size
                );
        //printf("sol[%d]: ", i);
        //for (int j = 0; j < state.set_size; ++j)
        //{
        //    printf("%d ", host_state.solutions.at(i).at(j)); 
        //}
        //printf("\n");
    }
    host_state.stop_flags.resize(state.batch_size);
    checkCUDA(cudaMemcpy(host_state.stop_flags.data(), state.stop_flags, sizeof(char)*state.batch_size, cudaMemcpyDeviceToHost));
    host_state.orig_costs.resize(state.batch_size); 
    host_state.target_costs.resize(state.batch_size); 
    host_state.target_pos_x.assign(state.batch_size, std::vector<T>(state.set_size));
    host_state.target_pos_y.assign(state.batch_size, std::vector<T>(state.set_size));
    host_state.target_spaces.assign(state.batch_size, std::vector<Space<T> >(state.set_size));

    // compute on cpu 
    for (int i = 0; i < state.num_independent_sets; ++i)
    {
        host_state.orig_costs[i] = 0; 
        host_state.target_costs[i] = 0; 
        for (int j = 0; j < state.set_size; ++j)
        {
            // regard maximization costs as minimization 
            host_state.orig_costs[i] -= host_state.cost_matrices.at(i).at(j*state.set_size+j); 
            if (host_state.stop_flags.at(i))
            {
                int sol_j = host_state.solutions.at(i).at(j);
                dreamplaceAssert(sol_j < state.set_size);
                host_state.target_costs[i] -= host_state.cost_matrices.at(i).at(j*state.set_size+sol_j); 
            }
        }
    }
    for (int i = 0; i < state.num_independent_sets; ++i)
    {
        apply_solution(host_db, host_state, i);
    }

    // update to cuda 
    checkCUDA(cudaMemcpy(db.x, host_db.x.data(), sizeof(T)*db.num_movable_nodes, cudaMemcpyHostToDevice));
    checkCUDA(cudaMemcpy(db.y, host_db.y.data(), sizeof(T)*db.num_movable_nodes, cudaMemcpyHostToDevice));
    checkCUDA(cudaMemcpy(state.spaces, host_state.spaces.data(), sizeof(Space<T>)*db.num_movable_nodes, cudaMemcpyHostToDevice));
    state.num_moved = host_state.num_moved; 
}

DREAMPLACE_END_NAMESPACE

#endif
