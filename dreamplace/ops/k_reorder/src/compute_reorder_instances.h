/**
 * @file   compute_reorder_instances.h
 * @author Yibo Lin
 * @date   Apr 2019
 */
#ifndef _DREAMPLACE_K_REORDER_COMPUTE_REORDER_INSTANCES_H
#define _DREAMPLACE_K_REORDER_COMPUTE_REORDER_INSTANCES_H

DREAMPLACE_BEGIN_NAMESPACE

/// @brief An instance is a wrapper to an adjacent sequence of at most K cells. 
/// It records the row, group, and node begin/end indices in the row2node_map. 
struct KReorderInstance 
{
    int group_id; 
    int row_id; 
    int idx_bgn; 
    int idx_end; 
    //int permute_id; 
};

template <typename DetailedPlaceDBType>
void compute_reorder_instances(const DetailedPlaceDBType& db, 
        const std::vector<std::vector<int> >& state_row2node_map, 
        const std::vector<std::vector<int> >& state_independent_rows, 
        std::vector<std::vector<KReorderInstance> >& state_reorder_instances, 
        int K)
{
    state_reorder_instances.resize(state_independent_rows.size());

    for (unsigned int group_id = 0; group_id < state_independent_rows.size(); ++group_id)
    {
        auto const& independent_rows = state_independent_rows.at(group_id); 
        auto& reorder_instances = state_reorder_instances.at(group_id); 
        for (auto row_id : independent_rows)
        {
            auto const& row2nodes = state_row2node_map.at(row_id); 
            int num_nodes_in_row = row2nodes.size(); 
            for (int sub_id = 0; sub_id < num_nodes_in_row; sub_id += K)
            {
                int idx_bgn = sub_id; 
                int idx_end = std::min(sub_id+K, num_nodes_in_row);
                // stop at fixed cells and multi-row height cells 
                for (int i = idx_bgn; i < idx_end; ++i)
                {
                    int node_id = row2nodes.at(i);
                    if (node_id >= db.num_movable_nodes || db.node_size_y[node_id] > db.row_height)
                    {
                        idx_end = i; 
                        break; 
                    }
                }
                if (idx_end-idx_bgn >= 2)
                {
                    KReorderInstance inst; 
                    inst.group_id = group_id; 
                    inst.row_id = row_id; 
                    inst.idx_bgn = idx_bgn; 
                    inst.idx_end = idx_end; 
                    reorder_instances.push_back(inst);
                }
            }
        }
    }
#ifdef DEBUG
    for (unsigned int group_id = 0; group_id < state_independent_rows.size(); ++group_id)
    {
        dreamplacePrint(kDEBUG, "group[%u] has %lu instances\n", group_id, state_reorder_instances[group_id].size());
    }
#endif
}

DREAMPLACE_END_NAMESPACE

#endif
