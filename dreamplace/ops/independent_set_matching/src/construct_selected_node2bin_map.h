/**
 * @file   construct_selected_node2bin_map.h
 * @author Yibo Lin
 * @date   Jul 2019
 */
#ifndef _DREAMPLACE_INDEPENDENT_SET_MATCHING_CONSTRUCT_SELECTED_NODE2BIN_MAP_H
#define _DREAMPLACE_INDEPENDENT_SET_MATCHING_CONSTRUCT_SELECTED_NODE2BIN_MAP_H

DREAMPLACE_BEGIN_NAMESPACE

template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
void construct_selected_node2bin_map(const DetailedPlaceDBType& db, IndependentSetMatchingStateType& state)
{
    for (auto& bin2nodes : state.bin2node_map)
    {
        bin2nodes.clear();
    }
    for (int node_id = 0; node_id < db.num_movable_nodes; ++node_id)
    {
        if (state.selected_markers[node_id])
        {
            typename DetailedPlaceDBType::type width = db.node_size_x[node_id];
            typename DetailedPlaceDBType::type height = db.node_size_y[node_id];
            auto& bm_idx = state.node2bin_map[node_id];
            int num_bins_x = db.num_bins_x;
            int num_bins_y = db.num_bins_y;
            typename DetailedPlaceDBType::type bin_size_x = db.bin_size_x;
            typename DetailedPlaceDBType::type bin_size_y = db.bin_size_y;

            typename DetailedPlaceDBType::type node_x = db.x[node_id] + width/2; 
            typename DetailedPlaceDBType::type node_y = db.y[node_id] + height/2;

            int bx = std::min(std::max((int)floorDiv((node_x-db.xl), bin_size_x), 0), num_bins_x-1);
            int by = std::min(std::max((int)floorDiv((node_y-db.yl), bin_size_y), 0), num_bins_y-1);
            bm_idx.bin_id = bx*num_bins_y+by; 

            auto& bin2nodes = state.bin2node_map.at(bm_idx.bin_id);
            bin2nodes.push_back(node_id);
        }
    }
}

DREAMPLACE_END_NAMESPACE

#endif
