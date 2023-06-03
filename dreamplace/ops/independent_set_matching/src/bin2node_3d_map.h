/**
 * @file   bin2node_3d_map.h
 * @author Yibo Lin
 * @date   Apr 2019
 */
#ifndef _DREAMPLACE_INDEPENDENT_SET_MATCHING_BIN2NODE_3D_MAP_H
#define _DREAMPLACE_INDEPENDENT_SET_MATCHING_BIN2NODE_3D_MAP_H

DREAMPLACE_BEGIN_NAMESPACE

struct BinMapIndex3D
{
    int size_id; 
    int bin_id; 
    int sub_id; 
};

template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
void make_bin2node_3d_map(const DetailedPlaceDBType& db, const typename DetailedPlaceDBType::type* x, const typename DetailedPlaceDBType::type* y, 
        const typename DetailedPlaceDBType::type* node_size_x, const typename DetailedPlaceDBType::type* node_size_y, 
        IndependentSetMatchingStateType& state) 
{
    for (int i = 0; i < db.num_movable_nodes; ++i)
    {
        if (node_size_y[i] == db.row_height)
        {
            int width = (int)ceilDiv(node_size_x[i], db.site_width); 
            if (state.size2num_node_map.count(width))
            {
                state.size2num_node_map[width] += 1; 
            }
            else 
            {
                state.size2num_node_map[width] = 1; 
            }
        }
    }
    int num_sizes = state.size2num_node_map.size();
#ifdef DEBUG
    dreamplacePrint(kDEBUG, "%lu width values\n", size2num_node_map.size());
    for (auto kv : state.size2num_node_map)
    {
        dreamplacePrint(kDEBUG, "width %d has %d cells\n", kv.first, kv.second);
    }
#endif

    state.bin2node_3d_map.resize(num_sizes);
    state.node2bin_3d_map.resize(db.num_movable_nodes);
    state.num_bins_xs.resize(num_sizes);
    state.num_bins_ys.resize(num_sizes);
    state.bin_size_xs.resize(num_sizes);
    state.bin_size_ys.resize(num_sizes);
    int size_id = 0; 
    for (auto kv : state.size2num_node_map)
    {
        state.num_bins_xs[size_id] = std::max((int)ceil(sqrt(kv.second/2)), 1);
        state.num_bins_ys[size_id] = state.num_bins_xs[size_id];
        state.bin_size_xs[size_id] = div((db.xh-db.xl), state.num_bins_xs[size_id]);
        state.bin_size_ys[size_id] = div((db.yh-db.yl), state.num_bins_ys[size_id]);
        state.bin2node_3d_map[size_id].resize(state.num_bins_xs[size_id]*state.num_bins_ys[size_id]);
        state.size2id_map[kv.first] = size_id; 
        dreamplacePrint(kDEBUG, "size id %d: prepare bins %dx%d for %d cells with width %g\n", size_id, state.num_bins_xs[size_id], state.num_bins_ys[size_id], kv.second, kv.first*db.site_width);
        ++size_id;
    }
}

inline int ceil_power2(int v)
{
    return (1<<(int)ceil(log2(v)));
}

DREAMPLACE_END_NAMESPACE

#endif
