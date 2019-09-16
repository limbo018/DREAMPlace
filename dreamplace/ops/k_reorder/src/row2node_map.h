/**
 * @file   row2node_map.h
 * @author Yibo Lin
 * @date   Apr 2019
 */
#ifndef _DREAMPLACE_K_REORDER_ROW2NODE_MAP_H
#define _DREAMPLACE_K_REORDER_ROW2NODE_MAP_H

DREAMPLACE_BEGIN_NAMESPACE

/// @brief distribute cells to rows 
template <typename DetailedPlaceDBType>
void make_row2node_map(const DetailedPlaceDBType& db, const typename DetailedPlaceDBType::type* vx, const typename DetailedPlaceDBType::type* vy, std::vector<std::vector<int> >& row2node_map) 
{
    // distribute cells to rows 
    for (int i = 0; i < db.num_nodes; ++i)
    {
        //typename DetailedPlaceDBType::type node_xl = vx[i]; 
        typename DetailedPlaceDBType::type node_yl = vy[i];
        //typename DetailedPlaceDBType::type node_xh = node_xl+db.node_size_x[i];
        typename DetailedPlaceDBType::type node_yh = node_yl+db.node_size_y[i];

        int row_idxl = (node_yl-db.yl)/db.row_height; 
        int row_idxh = ceil((node_yh-db.yl)/db.row_height)+1;
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
    for (int i = 0; i < db.num_sites_y; ++i)
    {
        std::sort(row2node_map[i].begin(), row2node_map[i].end(), 
                [&] (int node_id1, int node_id2) {return
                    vx[node_id1]+db.node_size_x[node_id1]/2 <
                    vx[node_id2]+db.node_size_x[node_id2]/2;}
                );
    }
}

DREAMPLACE_END_NAMESPACE

#endif
