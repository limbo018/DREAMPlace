/**
 * @file   mark_dependent_nodes.h
 * @author Yibo Lin
 * @date   Apr 2019
 */
#ifndef _DREAMPLACE_INDEPENDENT_SET_MATCHING_MARK_DEPENDENT_NODES_H
#define _DREAMPLACE_INDEPENDENT_SET_MATCHING_MARK_DEPENDENT_NODES_H

DREAMPLACE_BEGIN_NAMESPACE

/// @brief mark a node and as first level connected nodes as dependent 
/// only nodes with the same sizes are marked 
template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
void mark_dependent_nodes(const DetailedPlaceDBType& db, IndependentSetMatchingStateType& state, int node_id, unsigned char value)
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

template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
void mark_dependent_nodes_soft(const DetailedPlaceDBType& db, IndependentSetMatchingStateType& state, int node_id, unsigned char value)
{
    typename DetailedPlaceDBType::type node_xl = db.x[node_id];
    typename DetailedPlaceDBType::type node_yl = db.y[node_id];
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
                typename DetailedPlaceDBType::type other_node_xl = db.x[other_node_id];
                typename DetailedPlaceDBType::type other_node_yl = db.y[other_node_id];
                if (std::abs(node_xl-other_node_xl) + std::abs(node_yl-other_node_yl) < state.skip_threshold)
                {
                    state.dependent_markers[other_node_id] = value; 
                }
            }
        }
    }
}

DREAMPLACE_END_NAMESPACE

#endif
