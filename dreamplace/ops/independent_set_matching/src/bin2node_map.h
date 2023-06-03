/**
 * @file   bin2node_map.h
 * @author Yibo Lin
 * @date   Apr 2019
 */
#ifndef _DREAMPLACE_INDEPENDENT_SET_MATCHING_BIN2NODE_MAP_H
#define _DREAMPLACE_INDEPENDENT_SET_MATCHING_BIN2NODE_MAP_H

DREAMPLACE_BEGIN_NAMESPACE

template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
void make_bin2node_map(const DetailedPlaceDBType& db, const typename DetailedPlaceDBType::type* host_x, const typename DetailedPlaceDBType::type* host_y, 
        const typename DetailedPlaceDBType::type* host_node_size_x, const typename DetailedPlaceDBType::type* host_node_size_y, 
        IndependentSetMatchingStateType& state) 
{
    typedef typename DetailedPlaceDBType::type T; 
    // construct bin2node_map 
    state.bin2node_map.resize(db.num_bins_x*db.num_bins_y);
    for (int i = 0; i < db.num_movable_nodes; ++i)
    {
        int node_id = i; 
        T node_x = host_x[node_id] + host_node_size_x[node_id]/2; 
        T node_y = host_y[node_id] + host_node_size_y[node_id]/2;

        int bx = std::min(std::max((int)floorDiv((node_x-db.xl), db.bin_size_x), 0), db.num_bins_x-1);
        int by = std::min(std::max((int)floorDiv((node_y-db.yl), db.bin_size_y), 0), db.num_bins_y-1);
        int bin_id = bx*db.num_bins_y+by; 
        //int sub_id = bin2node_map.at(bin_id).size(); 
        state.bin2node_map.at(bin_id).push_back(node_id); 
    }
    // construct node2bin_map 
    state.node2bin_map.resize(db.num_movable_nodes);
    for (unsigned int bin_id = 0; bin_id < state.bin2node_map.size(); ++bin_id)
    {
        for (unsigned int sub_id = 0; sub_id < state.bin2node_map[bin_id].size(); ++sub_id)
        {
            int node_id = state.bin2node_map[bin_id][sub_id];
            BinMapIndex& bm_idx = state.node2bin_map.at(node_id); 
            bm_idx.bin_id = bin_id; 
            bm_idx.sub_id = sub_id; 
        }
    }
#ifdef DEBUG
    int max_num_nodes_per_bin = 0; 
    for (unsigned int i = 0; i < state.bin2node_map.size(); ++i)
    {
        max_num_nodes_per_bin = std::max(max_num_nodes_per_bin, (int)state.bin2node_map[i].size());
    }
    dreamplacePrint(kDEBUG, "max_num_nodes_per_bin = %d\n", max_num_nodes_per_bin);
#endif
}

DREAMPLACE_END_NAMESPACE

#endif
