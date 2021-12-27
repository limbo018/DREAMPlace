/**
 * @file   maximal_independent_set.cuh
 * @author Yibo Lin
 * @date   Jul 2019
 */
#ifndef _DREAMPLACE_INDEPENDENT_SET_MATCHING_MAXIMAL_INDEPENDENT_SET_CUH
#define _DREAMPLACE_INDEPENDENT_SET_MATCHING_MAXIMAL_INDEPENDENT_SET_CUH

#include "independent_set_matching/src/select.cuh"

DREAMPLACE_BEGIN_NAMESPACE

#define SOFT_DEPENDENCY

/// @brief mark a node and as first level connected nodes as dependent 
/// if cell distance is larger than state.skip_threshold, we will skip it 
template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
__device__ void mark_dependent_nodes(const DetailedPlaceDBType& db, IndependentSetMatchingStateType& state, int node_id, unsigned char value)
{
#ifdef SOFT_DEPENDENCY
    typename DetailedPlaceDBType::type node_xl = db.x[node_id];
    typename DetailedPlaceDBType::type node_yl = db.y[node_id];
#endif
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
#ifdef SOFT_DEPENDENCY
                typename DetailedPlaceDBType::type other_node_xl = db.x[other_node_id];
                typename DetailedPlaceDBType::type other_node_yl = db.y[other_node_id];
                typename DetailedPlaceDBType::type distance = abs(node_xl-other_node_xl) + abs(node_yl-other_node_yl); 
                if (distance < state.skip_threshold)
                {
#endif
                    if (other_node_id < db.num_nodes) // other_node_id may exceed db.num_nodes like IO pins
                    {
                        state.dependent_markers[other_node_id] = value; 
                    }
#ifdef SOFT_DEPENDENCY
                }
#endif
            }
        }
    }
    state.dependent_markers[node_id] = value; 
}

/// @brief for each node, check its first level neighbors, if they are selected, mark itself as dependent 
template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
__device__ void mark_dependent_nodes_self(const DetailedPlaceDBType& db, IndependentSetMatchingStateType& state, int node_id)
{
    if (state.selected_markers[node_id])
    {
        state.dependent_markers[node_id] = 1; 
        return;
    }
#ifdef SOFT_DEPENDENCY
    typename DetailedPlaceDBType::type node_xl = db.x[node_id];
    typename DetailedPlaceDBType::type node_yl = db.y[node_id];
#endif
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
#ifdef SOFT_DEPENDENCY
                typename DetailedPlaceDBType::type other_node_xl = db.x[other_node_id];
                typename DetailedPlaceDBType::type other_node_yl = db.y[other_node_id];
                if (std::abs(node_xl-other_node_xl) + std::abs(node_yl-other_node_yl) < state.skip_threshold)
                {
#endif
                    if (other_node_id < db.num_movable_nodes && state.selected_markers[other_node_id])
                    {
                        state.dependent_markers[node_id] = 1; 
                        return;
                    }
#ifdef SOFT_DEPENDENCY
                }
#endif
            }
        }
    }
}

template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
__global__ void maximal_independent_set_kernel(DetailedPlaceDBType db, IndependentSetMatchingStateType state, int *empty)
{
    const int from = blockIdx.x * blockDim.x + threadIdx.x;
    const int incr = gridDim.x * blockDim.x;

    //do 
    //{
        //empty = true;
        for (int node_id = from; node_id < db.num_movable_nodes; node_id += incr) 
        {
            if (!state.dependent_markers[node_id])
            {
                if (*empty)
                {
                    atomicExch(empty, false); 
                }
                //empty = false; 
                bool min_node_flag = true; 
                {
#ifdef SOFT_DEPENDENCY
                    typename DetailedPlaceDBType::type node_xl = db.x[node_id];
                    typename DetailedPlaceDBType::type node_yl = db.y[node_id];
#endif
                    int node_rank = state.ordered_nodes[node_id];
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
#ifdef SOFT_DEPENDENCY
                                typename DetailedPlaceDBType::type other_node_xl = db.x[other_node_id];
                                typename DetailedPlaceDBType::type other_node_yl = db.y[other_node_id];
                                typename DetailedPlaceDBType::type distance = abs(node_xl-other_node_xl) + abs(node_yl-other_node_yl); 
#endif
                                //if (other_node_id < db.num_movable_nodes 
                                //        && state.dependent_markers[other_node_id] == 0
                                //        && state.ordered_nodes[other_node_id] < node_rank)
                                if (other_node_id < db.num_movable_nodes 
#ifdef SOFT_DEPENDENCY
                                        && (distance < state.skip_threshold)
#endif
                                        && (state.selected_markers[other_node_id] ||
                                            (state.dependent_markers[other_node_id] == 0
                                        && state.ordered_nodes[other_node_id] < node_rank)))
                                {
                                    min_node_flag = false; 
                                    break; 
                                }
                            }
                            if (!min_node_flag)
                            {
                                break; 
                            }
                        }
                    }
                }
                if (min_node_flag)
                {
                    state.selected_markers[node_id] = 1; 
                }
            }
        }
    //} while (!empty);
}

template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
__global__ void mark_dependent_nodes_kernel(DetailedPlaceDBType db, IndependentSetMatchingStateType state)
{
    for (int node_id = blockIdx.x*blockDim.x + threadIdx.x; node_id < db.num_movable_nodes; node_id += blockDim.x*gridDim.x)
    {
        if (!state.dependent_markers[node_id])
        {
            mark_dependent_nodes_self(db, state, node_id);
        }
    }
}

template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
__global__ void init_markers_kernel(DetailedPlaceDBType db, IndependentSetMatchingStateType state)
{
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < db.num_nodes; i += blockDim.x*gridDim.x)
    {
        state.selected_markers[i] = 0; 
        // make sure multi-row height cells are not selected
        state.dependent_markers[i] = (db.node_size_y[i] > db.row_height); 
    }
}

//template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
//__global__ void postprocess_markers_kernel(DetailedPlaceDBType db, IndependentSetMatchingStateType state)
//{
//    // remove cells with sizes not considered 
//    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < db.num_movable_nodes; i += blockDim.x*gridDim.x)
//    {
//        if (state.selected_markers[i])
//        {
//            int size_id = state.node_size_id[i]; 
//            if (size_id >= state.num_node_sizes)
//            {
//                state.selected_markers[i] = 0; 
//            }
//        }
//    }
//}

template <typename T>
__global__ void marker_sum(const T* selected_markers, int n)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        int sum = 0; 
        for (int i = 0; i < n; ++i)
        {
            sum += (int)selected_markers[i];
        }
        printf("sum = %d\n", sum);
    }
}

template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
__global__ void check_dependent_nodes(DetailedPlaceDBType db, IndependentSetMatchingStateType state)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        for (int i = 0; i < db.num_nodes; ++i)
        {
            state.dependent_markers[i] = 0; 
        }
        for (int node_id = 0; node_id < db.num_movable_nodes; ++node_id)
        {
            if (state.selected_markers[node_id])
            {
                if (state.dependent_markers[node_id])
                {
                    printf("node %d should not be selected\n", node_id);
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
                                printf("%d: node_rank %d, other_node_id %d, other_node_rank %d, dependent_markers %d, selected_markers %d\n", 
                                        node_id, state.ordered_nodes[node_id], other_node_id, state.ordered_nodes[other_node_id], 
                                        state.dependent_markers[other_node_id], state.selected_markers[other_node_id]);
                            }
                        }
                    }
                }
                assert(!state.dependent_markers[node_id]);
                mark_dependent_nodes(db, state, node_id, 1); 
            }
        }
        //for (int node_id = 0; node_id < db.num_movable_nodes; ++node_id)
        //{
        //    if (state.selected_markers[node_id])
        //    {
        //        printf("%d ", node_id);
        //    }
        //}
        //printf("\n");
    }
}

template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
__global__ void maximal_independent_set_dynamic(DetailedPlaceDBType db, IndependentSetMatchingStateType state)
{
    // if dependent_markers is 1, it means "cannot be selected"
    // if selected_markers is 1, it means "already selected"
    init_markers_kernel<<<ceilDiv(db.num_nodes, 256), 256>>>(db, state);

    int iteration = 0; 
    do {
        *state.independent_set_empty_flag = true; 
        maximal_independent_set_kernel<<<ceilDiv(db.num_movable_nodes, 256), 256>>>(db, state, state.independent_set_empty_flag);
        mark_dependent_nodes_kernel<<<ceilDiv(db.num_movable_nodes, 256), 256>>>(db, state);
        ++iteration; 
    } while (!*state.independent_set_empty_flag && iteration < 10); 
    //marker_sum<<<1, 1>>>(state.selected_markers, db.num_movable_nodes);
    //check_dependent_nodes<<<1, 1>>>(db, state);
    //postprocess_markers_kernel<<<ceilDiv(db.num_movable_nodes, 256), 256>>>(db, state);
}


template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
void maximal_independent_set(DetailedPlaceDBType const& db, IndependentSetMatchingStateType& state)
{
    // if dependent_markers is 1, it means "cannot be selected"
    // if selected_markers is 1, it means "already selected"
    init_markers_kernel<<<ceilDiv(db.num_nodes, 256), 256>>>(db, state);

    int host_empty; 

    int iteration = 0; 
    do {
        host_empty = true; 
        checkCUDA(cudaMemcpy(state.independent_set_empty_flag, &host_empty, sizeof(int), cudaMemcpyHostToDevice));
        maximal_independent_set_kernel<<<ceilDiv(db.num_movable_nodes, 256), 256>>>(db, state, state.independent_set_empty_flag);
        mark_dependent_nodes_kernel<<<ceilDiv(db.num_movable_nodes, 256), 256>>>(db, state);
        checkCUDA(cudaMemcpy(&host_empty, state.independent_set_empty_flag, sizeof(int), cudaMemcpyDeviceToHost));
        ++iteration; 
    } while (!host_empty && iteration < 10); 
    //marker_sum<<<1, 1>>>(state.selected_markers, db.num_movable_nodes);
    //check_dependent_nodes<<<1, 1>>>(db, state);
    //postprocess_markers_kernel<<<ceilDiv(db.num_movable_nodes, 256), 256>>>(db, state);

    select(state.selected_markers, state.selected_maximal_independent_set, db.num_movable_nodes, state.select_scratch, state.device_num_selected);
    checkCUDA(cudaMemcpy(&state.num_selected, state.device_num_selected, sizeof(int), cudaMemcpyDeviceToHost));
}

DREAMPLACE_END_NAMESPACE

#endif
