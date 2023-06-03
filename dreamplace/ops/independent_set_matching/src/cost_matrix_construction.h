/**
 * @file   cost_matrix_construction.h
 * @author Yibo Lin
 * @date   Apr 2019
 */
#ifndef _DREAMPLACE_INDEPENDENT_SET_MATCHING_COST_MATRIX_CONSTRUCTION_H
#define _DREAMPLACE_INDEPENDENT_SET_MATCHING_COST_MATRIX_CONSTRUCTION_H

#include "independent_set_matching/src/adjust_pos.h"

DREAMPLACE_BEGIN_NAMESPACE

/// construct a NxN cost matrix 
/// row indices are for cells 
/// column indices are for locations 
template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
void cost_matrix_construction(const DetailedPlaceDBType& db, IndependentSetMatchingStateType& state, 
        bool major, ///< false: row major, true: column major 
        int i ///< entry in the batch 
        )
{
    typedef typename DetailedPlaceDBType::type T; 
    
    auto const& independent_set = state.independent_sets[i];
    auto& cost_matrix = state.cost_matrices[i]; 
    unsigned int independent_set_size = independent_set.size();
    std::vector<Box<T> > bboxes;
    // cells 
    for (unsigned int k = 0; k < independent_set_size; ++k)
    {
        int node_id = independent_set[k];
        T node_width = db.node_size_x[node_id];
        bboxes.resize(db.flat_node2pin_start_map[node_id+1]-db.flat_node2pin_start_map[node_id]);
        int idx = 0; 
        for (int node2pin_id = db.flat_node2pin_start_map[node_id]; node2pin_id < db.flat_node2pin_start_map[node_id+1]; ++node2pin_id, ++idx)
        {
            int node_pin_id = db.flat_node2pin_map[node2pin_id];
            int net_id = db.pin2net_map[node_pin_id];
            Box<T>& box = bboxes[idx];
            box.xl = db.xh;
            box.yl = db.yh;
            box.xh = db.xl;
            box.yh = db.yl;
            if (db.net_mask[net_id])
            {
                for (int net2pin_id = db.flat_net2pin_start_map[net_id]; net2pin_id < db.flat_net2pin_start_map[net_id+1]; ++net2pin_id)
                {
                    int net_pin_id = db.flat_net2pin_map[net2pin_id];
                    int other_node_id = db.pin2node_map[net_pin_id];
                    if (other_node_id != node_id)
                    {
                        T xxl = db.x[other_node_id];
                        T yyl = db.y[other_node_id];
                        box.xl = std::min(box.xl, xxl+db.pin_offset_x[net_pin_id]);
                        box.xh = std::max(box.xh, xxl+db.pin_offset_x[net_pin_id]);
                        box.yl = std::min(box.yl, yyl+db.pin_offset_y[net_pin_id]);
                        box.yh = std::max(box.yh, yyl+db.pin_offset_y[net_pin_id]);
                    }
                }
            }
        }
        for (unsigned int j = 0; j < independent_set_size; ++j)
        {
            int pos_id = independent_set[j]; 
            T target_x = db.x[pos_id]; 
            T target_y = db.y[pos_id]; 
            T target_hpwl = 0; 

            auto const& target_space = state.spaces[pos_id];
            if (adjust_pos(target_x, node_width, target_space))
            {
                //adjust_pos(target_x, db.node_size_x[pos_id], target_space);
                // consider FENCE region 
                if (db.num_regions && !db.inside_fence(node_id, target_x, target_y))
                {
                    target_hpwl = state.large_number; 
                }
                else 
                {
                    idx = 0; 
                    for (int node2pin_id = db.flat_node2pin_start_map[node_id]; node2pin_id < db.flat_node2pin_start_map[node_id+1]; ++node2pin_id, ++idx)
                    {
                        int node_pin_id = db.flat_node2pin_map[node2pin_id];
                        int net_id = db.pin2net_map[node_pin_id];
                        const Box<T>& box = bboxes[idx];
                        if (db.net_mask[net_id])
                        {
                            T xxl = target_x;
                            T yyl = target_y;
                            T bxl = std::min(box.xl, xxl+db.pin_offset_x[node_pin_id]);
                            T bxh = std::max(box.xh, xxl+db.pin_offset_x[node_pin_id]);
                            T byl = std::min(box.yl, yyl+db.pin_offset_y[node_pin_id]);
                            T byh = std::max(box.yh, yyl+db.pin_offset_y[node_pin_id]);
                            target_hpwl += (bxh-bxl) + (byh-byl); 
                        }
                    }
                }
            }
            else 
            {
                //dreamplaceAssertMsg(node_id != pos_id, "node %d, pos %d", node_id, pos_id);
                target_hpwl = state.large_number; 
            }

            if (!major) // row major 
            {
                cost_matrix[independent_set.size()*k + j] = target_hpwl; 
            }
            else // column major 
            {
                cost_matrix[independent_set.size()*j + k] = target_hpwl; 
            }
        }
    }
#ifdef DEBUG
    for (unsigned int k = 0; k < independent_set.size(); ++k)
    {
        for (unsigned int j = 0; j < independent_set.size(); ++j)
        {
            dreamplacePrint(kNONE, "%d ", cost_matrix[independent_set.size()*k + j]);
        }
        dreamplacePrint(kNONE, "\n");
    }
#endif
}

DREAMPLACE_END_NAMESPACE

#endif
