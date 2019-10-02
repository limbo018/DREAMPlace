/**
 * @file   construct_spaces.h
 * @author Yibo Lin
 * @date   Mar 2019
 */
#ifndef _DREAMPLACE_INDEPENDENT_SET_MATCHING_CONSTRUCT_SPACES_H
#define _DREAMPLACE_INDEPENDENT_SET_MATCHING_CONSTRUCT_SPACES_H

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
inline T CPUDiv(T a, T b) 
{
    return a / b; 
}

template <typename T>
inline T CPUCeilDiv(T a, T b)
{
    return ceil(CPUDiv(a, b));
}

template <>
inline int CPUCeilDiv(int a, int b)
{
    return CPUDiv(a+b-1, b); 
}
template <>
inline unsigned int CPUCeilDiv(unsigned int a, unsigned int b)
{
    return CPUDiv(a+b-1, b); 
}

template <typename T>
inline bool adjust_pos(T& x, T width, const Space<T>& space)
{
    x = std::max(x, space.xl);
    x = std::min(x, space.xh-width);
    return width <= space.xh-space.xl; 
}

/// @brief generate array of spaces for each cell 
/// This is specifically designed for independent set, as we only consider the whitespace on the right side of a cell. 
/// It will make it much easier for apply_solution without keeping a structure of row2node_map. 
template <typename DetailedPlaceDBType>
void construct_spaces(const DetailedPlaceDBType& db, 
        const typename DetailedPlaceDBType::type* host_x, const typename DetailedPlaceDBType::type* host_y, 
        const typename DetailedPlaceDBType::type* host_node_size_x, const typename DetailedPlaceDBType::type* host_node_size_y, 
        std::vector<Space<typename DetailedPlaceDBType::type> >& host_spaces
        )
{
    std::vector<std::vector<int> > row2node_map (db.num_sites_y);
    // distribute cells to rows 
    for (int i = 0; i < db.num_nodes; ++i)
    {
        typename DetailedPlaceDBType::type node_yl = host_y[i];
        typename DetailedPlaceDBType::type node_yh = node_yl+host_node_size_y[i];

        int row_idxl = CPUDiv((node_yl-db.yl), db.row_height); 
        int row_idxh = CPUCeilDiv((node_yh-db.yl), db.row_height)+1;
        row_idxl = std::max(row_idxl, 0); 
        row_idxh = std::min(row_idxh, db.num_sites_y); 

        for (int row_id = row_idxl; row_id < row_idxh; ++row_id)
        {
            typename DetailedPlaceDBType::type row_yl = db.yl+row_id*db.row_height; 
            typename DetailedPlaceDBType::type row_yh = row_yl+db.row_height; 

            if (node_yl < row_yh && node_yh > row_yl) // overlap with row 
            {
                row2node_map[row_id].push_back(i); 
            }
        }
    }

    // sort cells within rows 
    // better to sort by center, because there might be 0-width cells 
    auto comp = [&] (int node_id1, int node_id2) {
        return host_x[node_id1]+host_node_size_x[node_id1]/2 <
            host_x[node_id2]+host_node_size_x[node_id2]/2;
    };
    for (int i = 0; i < db.num_sites_y; ++i)
    {
        std::sort(row2node_map[i].begin(), row2node_map[i].end(), comp);
    }

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

                auto right_bound = db.xh; 
                if (j+1 < row2nodes.size())
                {
                    int right_node_id = row2nodes[j+1];
                    right_bound = std::min(right_bound, host_x[right_node_id]);
                }
                space.xh = right_bound; 

#ifdef DEBUG
                dreamplaceAssert(space.xl <= db.x[node_id]);
                dreamplaceAssert(space.xh >= db.x[node_id]+db.node_size_x[node_id]);
#endif
            }
        }
    }
}

DREAMPLACE_END_NAMESPACE

#endif
