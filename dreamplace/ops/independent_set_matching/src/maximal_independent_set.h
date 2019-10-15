/**
 * @file   maximal_independent_set.h
 * @author Yibo Lin
 * @date   Jul 2019
 */
#ifndef _DREAMPLACE_INDEPENDENT_SET_MATCHING_MAXIMAL_INDEPENDENT_SET_H
#define _DREAMPLACE_INDEPENDENT_SET_MATCHING_MAXIMAL_INDEPENDENT_SET_H

#define SOFT_DEPENDENCY

#include "independent_set_matching/src/mark_dependent_nodes.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
void maximal_independent_set_sequential(const DetailedPlaceDBType& db, IndependentSetMatchingStateType& state)
{
    dreamplacePrint(kDEBUG, "%s\n", __func__);
    std::fill(state.selected_markers.begin(), state.selected_markers.end(), 0);
    std::fill(state.dependent_markers.begin(), state.dependent_markers.end(), 0);

    for (int i = 0; i < db.num_movable_nodes; ++i)
    {
        int node_id = state.ordered_nodes[i];
        if (!state.dependent_markers[node_id])
        {
            state.selected_markers[node_id] = 1; 
            mark_dependent_nodes(db, state, node_id, 1); 
        }
    }
}

template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
void maximal_independent_set_parallel(const DetailedPlaceDBType& db, IndependentSetMatchingStateType& state, int max_iters=10)
{
    dreamplacePrint(kDEBUG, "%s\n", __func__);
    // if dependent_markers is 1, it means "cannot be selected"
    // if selected_markers is 1, it means "already selected"
    std::fill(state.selected_markers.begin(), state.selected_markers.end(), 0);
    std::fill(state.dependent_markers.begin(), state.dependent_markers.end(), 0);

    bool empty = false; 
    int iteration = 0; 
    while (!empty && iteration < max_iters)
    {
        empty = true; 
#pragma omp parallel for num_threads(state.num_threads) 
        for (int node_id = 0; node_id < db.num_movable_nodes; ++node_id)
        {
            if (!state.dependent_markers[node_id])
            {
#pragma omp atomic
                empty &= false; 
                bool min_node_flag = true; 
#ifdef SOFT_DEPENDENCY
                typename DetailedPlaceDBType::type node_xl = db.x[node_id];
                typename DetailedPlaceDBType::type node_yl = db.y[node_id];
#endif
                int rank = state.ordered_nodes[node_id];
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
#endif
                            if (other_node_id < db.num_movable_nodes 
                                    && state.ordered_nodes[other_node_id] < rank && state.dependent_markers[other_node_id] == 0
#ifdef SOFT_DEPENDENCY
                                    && std::abs(node_xl-other_node_xl) + std::abs(node_yl-other_node_yl) < state.skip_threshold
#endif
                                    )
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
                if (min_node_flag)
                {
                    state.selected_markers[node_id] = 1; 
                }
            }
        }
#pragma omp parallel for num_threads(state.num_threads) 
        for (int node_id = 0; node_id < db.num_movable_nodes; ++node_id)
        {
            if (!state.dependent_markers[node_id])
            {
                // there wil be no data race if use *_self function 
                // the other version may result in parallel issue 
                mark_dependent_nodes_self(db, state, node_id);
            }
        }
        //dreamplacePrint(kDEBUG, "selected %lu nodes\n", std::count(state.selected_markers.begin(), state.selected_markers.end(), 1));
        ++iteration; 
    }
    dreamplacePrint(kDEBUG, "selected %lu nodes\n", std::count(state.selected_markers.begin(), state.selected_markers.end(), 1));
}

DREAMPLACE_END_NAMESPACE

#endif
