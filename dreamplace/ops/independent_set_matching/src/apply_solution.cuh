/**
 * @file   apply_solution.cuh
 * @author Yibo Lin
 * @date   Mar 2019
 */
#ifndef _DREAMPLACE_INDEPENDENT_SET_MATCHING_APPLY_SOLUTION_CUH
#define _DREAMPLACE_INDEPENDENT_SET_MATCHING_APPLY_SOLUTION_CUH

#include "utility/src/utils.cuh"
#include "utility/src/utils_cub.cuh"
#include "independent_set_matching/src/adjust_pos.h"

//#define DEBUG

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
__global__ void copy_orig_cost_kernel(const T* cost_matrices, const char* stop_flags, const int set_size, T* costs)
{
    int i = blockIdx.x; // set 

    if (stop_flags[i])
    {
        auto cost_matrix = cost_matrices + i*set_size*set_size; 
        auto cost = costs + i*set_size; 
        for (int j = threadIdx.x; j < set_size; j += blockDim.x)
        {
            cost[j] = cost_matrix[j*set_size + j];
        }
    }
}

template <typename T>
__global__ void copy_solution_cost_kernel(const T* cost_matrices, const char* stop_flags, const int* solutions, const int set_size, T* costs)
{
    int i = blockIdx.x; // set 

    if (stop_flags[i])
    {
        auto cost_matrix = cost_matrices + i*set_size*set_size; 
        auto cost = costs + i*set_size; 
        auto solution = solutions + i*set_size; 
        for (int j = threadIdx.x; j < set_size; j += blockDim.x)
        {
            int sol_k = solution[j];
            cost[j] = cost_matrix[j*set_size + sol_k];
        }
    }
}

template <typename T, int BlockDim>
__global__ void block_reduce_sum(T* costs, const char* stop_flags, int batch_size, int set_size)
{
    int bid = blockIdx.x; // set 
    int tid = threadIdx.x; 

    if (stop_flags[bid])
    {
        // Specialize BlockReduce for a 1D block of BlockDim threads on type int
        typedef cub::BlockReduce<T, BlockDim> BlockReduce;
        // Allocate shared memory for BlockReduce
        __shared__ typename BlockReduce::TempStorage temp_storage;
        // Obtain a segment of consecutive items that are blocked across threads
        int thread_data[1];

        thread_data[0] = costs[bid*set_size + tid];

        __syncthreads();

        // Compute the block-wide sum for thread0
        T aggregate = BlockReduce(temp_storage).Sum(thread_data);

        __syncthreads();

        if (tid == 0)
        {
            costs[bid*set_size] = aggregate;
        }
    }
}

template <typename T>
void compute_costs(const char* stop_flags, const int batch_size, const int set_size, T* costs)
{
    switch (set_size)
    {
        case 2:
            block_reduce_sum<T, 2><<<batch_size, 2>>>(costs, stop_flags, batch_size, set_size);
            break; 
        case 4:
            block_reduce_sum<T, 4><<<batch_size, 4>>>(costs, stop_flags, batch_size, set_size);
            break; 
        case 8:
            block_reduce_sum<T, 8><<<batch_size, 8>>>(costs, stop_flags, batch_size, set_size);
            break; 
        case 16:
            block_reduce_sum<T, 16><<<batch_size, 16>>>(costs, stop_flags, batch_size, set_size);
            break; 
        case 32:
            block_reduce_sum<T, 32><<<batch_size, 32>>>(costs, stop_flags, batch_size, set_size);
            break; 
        case 64:
            block_reduce_sum<T, 64><<<batch_size, 64>>>(costs, stop_flags, batch_size, set_size);
            break; 
        case 128:
            block_reduce_sum<T, 128><<<batch_size, 128>>>(costs, stop_flags, batch_size, set_size);
            break; 
        case 256:
            block_reduce_sum<T, 256><<<batch_size, 256>>>(costs, stop_flags, batch_size, set_size);
            break; 
        case 512:
            block_reduce_sum<T, 512><<<batch_size, 512>>>(costs, stop_flags, batch_size, set_size);
            break; 
        case 1024:
            block_reduce_sum<T, 1024><<<batch_size, 1024>>>(costs, stop_flags, batch_size, set_size);
            break; 
        default:
            dreamplaceAssertMsg(0, "unsupported set size %d", set_size);
    }
}

template <typename T>
__global__ void print_copy_costs_kernel(const T* costs, int batch_size, int set_size)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        for (int i = 0; i < batch_size; ++i)
        {
            printf("[%d] orig/solution_costs ", i);
            for (int j = 0; j < set_size; ++j)
            {
                printf("%g ", (float)costs[i*set_size+j]);
            }
            printf("\n");
        }
    }
}

template <typename T>
void compute_orig_cost(const T* cost_matrices, const char* stop_flags, const int batch_size, const int set_size, T* costs)
{
    copy_orig_cost_kernel<<<batch_size, set_size>>>(cost_matrices, stop_flags, set_size, costs);
    //print_copy_costs_kernel<<<1, 1>>>(costs, batch_size, set_size);
    compute_costs(stop_flags, batch_size, set_size, costs);
}

template <typename T>
void compute_solution_cost(const T* cost_matrices, const int* solutions, const char* stop_flags, const int batch_size, const int set_size, T* costs)
{
    copy_solution_cost_kernel<<<batch_size, set_size>>>(cost_matrices, stop_flags, solutions, set_size, costs);
    //print_copy_costs_kernel<<<1, 1>>>(costs, batch_size, set_size);
    compute_costs(stop_flags, batch_size, set_size, costs);
}

template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
__global__ void store_orig_pos_kernel(DetailedPlaceDBType db, IndependentSetMatchingStateType state)
{
    int i = blockIdx.x; // set 
    const int* __restrict__ independent_set = state.independent_sets + i*state.set_size; 
    auto orig_x = state.orig_x + i*state.set_size; 
    auto orig_y = state.orig_y + i*state.set_size; 
    auto orig_spaces = state.orig_spaces + i*state.set_size; 
    for (int j = threadIdx.x; j < state.set_size; j += blockDim.x)
    {
        int node_id = independent_set[j];
        if (node_id < db.num_movable_nodes)
        {
            assert(node_id >= 0);
            orig_x[j] = db.x[node_id]; 
            orig_y[j] = db.y[node_id]; 
            orig_spaces[j] = state.spaces[node_id]; 
#ifdef DEBUG
            if (!(orig_x[j] >= db.xl && orig_x[j] < db.xh && orig_y[j] >= db.yl && orig_y[j] < db.yh))
            {
                printf("[E] node %d (%g, %g) (%g, %g) out of bounds\n", node_id, db.x[node_id], db.y[node_id], orig_x[j], orig_y[j]);
            }
            assert(orig_x[j] >= db.xl && orig_x[j] < db.xh && orig_y[j] >= db.yl && orig_y[j] < db.yh);
#endif
        }
    }
}

template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
__global__ void move_nodes_kernel(DetailedPlaceDBType db, IndependentSetMatchingStateType state)
{
    int i = blockIdx.x; // set 
    int idx = i*state.set_size; 

    if (state.stop_flags[i])
    {
        // encourage movement 
        if (state.orig_costs[idx] <= state.solution_costs[idx])
        {
            const int* __restrict__ independent_set = state.independent_sets + i*state.set_size; 
            const int* __restrict__ solution = state.solutions + i*state.set_size; 
            const typename IndependentSetMatchingStateType::type* __restrict__ orig_x = state.orig_x + i*state.set_size; 
            const typename IndependentSetMatchingStateType::type* __restrict__ orig_y = state.orig_y + i*state.set_size; 
            const Space<typename IndependentSetMatchingStateType::type>* __restrict__ orig_spaces = state.orig_spaces + i*state.set_size; 
            for (int j = threadIdx.x; j < state.set_size; j += blockDim.x)
            {
                int node_id = independent_set[j];
                int sol_k = solution[j]; 
#ifdef DEBUG
                assert(node_id >= 0);
                assert(sol_k >= 0 && sol_k < state.set_size);
#endif
                if (node_id < db.num_movable_nodes)
                {
                    auto node_width = db.node_size_x[node_id];

#ifdef DEBUG
                    int pos_id = independent_set[sol_k]; 
                    assert(pos_id >= 0 && pos_id < db.num_movable_nodes);
#endif

                    auto& x = db.x[node_id]; 
                    auto& y = db.y[node_id];
                    auto& space = state.spaces[node_id];
                    if (j != sol_k)
                    {
                        atomicAdd(state.device_num_moved, 1);
                        auto const& orig_space = orig_spaces[sol_k]; 
                        x = orig_x[sol_k]; 
                        bool ret = adjust_pos(x, node_width, orig_space);
                        assert(ret);
                        y = orig_y[sol_k]; 
                        space = orig_space; 
#ifdef DEBUG
                        assert(db.node_size_x[node_id] <= orig_space.xh-orig_space.xl);
                        if (x < db.xl || x > db.xh || y < db.yl || y > db.yh)
                        {
                            printf("[E] applying cost matrix %d, j %d, node_id %d, space (%g, %g), sol_k %d, pos_id %d, space (%g, %g)\n", i, j, node_id, space.xl, space.xh, sol_k, independent_set[sol_k], orig_space.xh, orig_space.xl);
                        }
                        assert(!(x < db.xl || x > db.xh || y < db.yl || y > db.yh));
#endif
                    }
                }
            }
        }
    }
}

template <typename IndependentSetMatchingStateType>
__global__ void print_orig_and_solution_costs_kernel(IndependentSetMatchingStateType state)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        for (int i = 0; i < state.num_independent_sets; ++i)
        {
            int stop = state.stop_flags[i];
            printf("[%d] orig_costs %g, solution_costs %g, delta %g, stop_flag %d\n", 
                    i, (float)state.orig_costs[i*state.set_size], (float)state.solution_costs[i*state.set_size], (float)(state.solution_costs[i*state.set_size]-state.orig_costs[i*state.set_size]), 
                        stop
                        );
        }
    }
}

template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
__global__ void check_hpwl_kernel(DetailedPlaceDBType db, IndependentSetMatchingStateType state, const int* independent_set)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        for (int i = 0; i < state.set_size; ++i)
        {
            int node_id = independent_set[i]; 
            if (node_id < db.num_movable_nodes)
            {
                printf("node %d (%g, %g)", node_id, db.x[node_id], db.y[node_id]);
                typename DetailedPlaceDBType::type target_hpwl = 0; 
                for (int node2pin_id = db.flat_node2pin_start_map[node_id]; node2pin_id < db.flat_node2pin_start_map[node_id+1]; ++node2pin_id)
                {
                    int node_pin_id = db.flat_node2pin_map[node2pin_id];
                    int net_id = db.pin2net_map[node_pin_id];
                    if (db.net_mask[net_id])
                    {
                        {
                            Box<typename DetailedPlaceDBType::type> box;
                            box.xl = db.xh; 
                            box.yl = db.yh; 
                            box.xh = db.xl; 
                            box.yh = db.yl; 
                            for (int net2pin_id = db.flat_net2pin_start_map[net_id]; net2pin_id < db.flat_net2pin_start_map[net_id+1]; ++net2pin_id)
                            {
                                int net_pin_id = db.flat_net2pin_map[net2pin_id];
                                int other_node_id = db.pin2node_map[net_pin_id];
                                auto xxl = db.x[other_node_id]+db.pin_offset_x[net_pin_id];
                                auto yyl = db.y[other_node_id]+db.pin_offset_y[net_pin_id];
                                box.xl = min(box.xl, xxl);
                                box.xh = max(box.xh, xxl);
                                box.yl = min(box.yl, yyl);
                                box.yh = max(box.yh, yyl);
                            }
                            typename DetailedPlaceDBType::type hpwl = box.xh-box.xl+box.yh-box.yl;
                            target_hpwl += hpwl; 
                            printf(", net %d hpwl %g", net_id, (double)hpwl);
                        }
                        {
                            auto const& box = state.net_boxes[net_id];
                            typename DetailedPlaceDBType::type xxl = db.x[node_id]+db.pin_offset_x[node_pin_id];
                            typename DetailedPlaceDBType::type yyl = db.y[node_id]+db.pin_offset_y[node_pin_id];
                            typename DetailedPlaceDBType::type bxl = min(box.xl, xxl);
                            typename DetailedPlaceDBType::type bxh = max(box.xh, xxl);
                            typename DetailedPlaceDBType::type byl = min(box.yl, yyl);
                            typename DetailedPlaceDBType::type byh = max(box.yh, yyl);
                            typename DetailedPlaceDBType::type hpwl = (bxh-bxl) + (byh-byl); 
                            printf(" (%g)", hpwl);
                        }
                    }
                }
                printf(", total hpwl %g\n", (double)target_hpwl);
            }
        }
    }
}

template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
void apply_solution(DetailedPlaceDBType& db, IndependentSetMatchingStateType& state)
{
    compute_orig_cost(state.cost_matrices, state.stop_flags, state.num_independent_sets, state.set_size, state.orig_costs); 
    compute_solution_cost(state.cost_matrices, state.solutions, state.stop_flags, state.num_independent_sets, state.set_size, state.solution_costs);
#ifdef DEBUG
    //print_orig_and_solution_costs_kernel<<<1, 1>>>(state);
    //check_hpwl_kernel<<<1, 1>>>(db, state, state.independent_sets+state.set_size*3);
#endif

    store_orig_pos_kernel<<<state.num_independent_sets, state.set_size>>>(db, state); 
    move_nodes_kernel<<<state.num_independent_sets, state.set_size>>>(db, state);
    checkCUDA(cudaMemcpy(&state.num_moved, state.device_num_moved, sizeof(int), cudaMemcpyDeviceToHost));
#ifdef DEBUG
    //check_hpwl_kernel<<<1, 1>>>(db, state, state.independent_sets+state.set_size*3);
#endif
}

DREAMPLACE_END_NAMESPACE

#endif
