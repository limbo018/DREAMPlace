/**
 * @file   construct_spaces.h
 * @author Yibo Lin
 * @date   Mar 2019
 */
#ifndef _DREAMPLACE_INDEPENDENT_SET_MATCHING_CONSTRUCT_SPACES_CUH
#define _DREAMPLACE_INDEPENDENT_SET_MATCHING_CONSTRUCT_SPACES_CUH

#include "utility/src/utils.cuh"

DREAMPLACE_BEGIN_NAMESPACE

/// @brief generate array of spaces for each cell 
/// This is specifically designed for independent set, as we only consider the whitespace on the right side of a cell. 
/// It will make it much easier for apply_solution without keeping a structure of row2node_map. 
template <typename DetailedPlaceDBType>
void construct_spaces(const DetailedPlaceDBType& db, 
        const typename DetailedPlaceDBType::type* host_x, const typename DetailedPlaceDBType::type* host_y, 
        const typename DetailedPlaceDBType::type* host_node_size_x, const typename DetailedPlaceDBType::type* host_node_size_y, 
        std::vector<Space<typename DetailedPlaceDBType::type> >& host_spaces, 
        int num_threads
        )
{
    std::vector<std::vector<int> > row2node_map (db.num_sites_y);
    db.make_row2node_map(host_x, host_y, host_node_size_x, host_node_size_y, db.num_nodes, row2node_map, num_threads);

    // construct spaces 
    host_spaces.resize(db.num_movable_nodes);
    for (int i = 0; i < db.num_sites_y; ++i)
    {
        for (unsigned int j = 0; j < row2node_map[i].size(); ++j)
        {
            auto const& row2nodes = row2node_map[i];
            int node_id = row2nodes[j];
            auto& space = host_spaces[node_id];
            if (node_id < db.num_movable_nodes)
            {
                auto left_bound = db.xl; 
                if (j)
                {
                    left_bound = host_x[node_id];
                }
                space.xl = left_bound; 
                // make sure space aligns to site 
                space.xl = ceilDiv(space.xl - db.xl, db.site_width) * db.site_width + db.xl;

                auto right_bound = db.xh; 
                if (j+1 < row2nodes.size())
                {
                    int right_node_id = row2nodes[j+1];
                    right_bound = min(right_bound, host_x[right_node_id]);
                }
                space.xh = right_bound; 
                // make sure space aligns to site 
                space.xh = floorDiv(space.xh - db.xl, db.site_width) * db.site_width + db.xl; 

#ifdef DEBUG
                dreamplaceAssert(space.xl <= host_x[node_id]);
                dreamplaceAssert(space.xh >= host_x[node_id]+host_node_size_x[node_id]);
#endif
            }
        }
    }
}

DREAMPLACE_END_NAMESPACE

#endif
