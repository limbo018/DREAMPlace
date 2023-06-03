/**
 * @file   compute_independent_rows.h
 * @author Yibo Lin
 * @date   Apr 2019
 */
#ifndef _DREAMPLACE_K_REORDER_COMPUTE_INDEPENDENT_ROWS_H
#define _DREAMPLACE_K_REORDER_COMPUTE_INDEPENDENT_ROWS_H

DREAMPLACE_BEGIN_NAMESPACE

template <typename DetailedPlaceDBType>
void compute_row_conflict_graph(const DetailedPlaceDBType& db, 
        const std::vector<std::vector<int> >& state_row2node_map, 
        std::vector<unsigned char>& state_adjacency_matrix, 
        std::vector<std::vector<int> >& state_row_graph, 
        int num_threads)
{
    // adjacency matrix 
    state_adjacency_matrix.assign(db.num_sites_y*db.num_sites_y, 0);
#pragma omp parallel for num_threads(num_threads) schedule(dynamic, 1)
    for (int net_id = 0; net_id < db.num_nets; ++net_id)
    {
        if (db.net_mask[net_id])
        {
            int net2pin_start = db.flat_net2pin_start_map[net_id]; 
            int net2pin_end = db.flat_net2pin_start_map[net_id+1];
            for (int net2pin_id1 = net2pin_start; net2pin_id1 < net2pin_end; ++net2pin_id1)
            {
                int net_pin_id1 = db.flat_net2pin_map[net2pin_id1];
                int node_id1 = db.pin2node_map[net_pin_id1];
                if (node_id1 < db.num_movable_nodes)
                {
                    int row_id1 = floorDiv(db.y[node_id1]-db.yl, db.row_height); 
                    row_id1 = std::min(std::max(row_id1, 0), db.num_sites_y-1);
                    for (int net2pin_id2 = net2pin_id1; net2pin_id2 < net2pin_end; ++net2pin_id2)
                    {
                        int net_pin_id2 = db.flat_net2pin_map[net2pin_id2];
                        int node_id2 = db.pin2node_map[net_pin_id2];
                        if (node_id2 < db.num_movable_nodes)
                        {
                            int row_id2 = floorDiv(db.y[node_id2]-db.yl, db.row_height); 
                            row_id2 = std::min(std::max(row_id2, 0), db.num_sites_y-1);
                            unsigned char& adjacency_matrix_element1 = state_adjacency_matrix.at(row_id1*db.num_sites_y + row_id2); 
                            unsigned char& adjacency_matrix_element2 = state_adjacency_matrix.at(row_id2*db.num_sites_y + row_id1); 
                            if (!adjacency_matrix_element1)
                            {
#pragma omp atomic 
                                adjacency_matrix_element1 |= 1; 
                            }
                            if (!adjacency_matrix_element2)
                            {
#pragma omp atomic 
                                adjacency_matrix_element2 |= 1; 
                            }
                        }
                    }
                }
            }
        }
    }
#ifdef DEBUG
    for (int row_id = 0; row_id < db.num_sites_y; ++row_id)
    {
        for (int other_row_id = 0; other_row_id < db.num_sites_y; ++other_row_id)
        {
            if (!(state_adjacency_matrix.at(row_id*db.num_sites_y+other_row_id) == state_adjacency_matrix.at(other_row_id*db.num_sites_y+row_id)))
            {
                dreamplacePrint(kDEBUG, "row %d, other_row %d, %d, %d\n", row_id, other_row_id, 
                        (int)state_adjacency_matrix.at(row_id*db.num_sites_y+other_row_id), 
                        (int)state_adjacency_matrix.at(other_row_id*db.num_sites_y+row_id)
                        );
            }
            dreamplaceAssert(state_adjacency_matrix.at(row_id*db.num_sites_y+other_row_id) == state_adjacency_matrix.at(other_row_id*db.num_sites_y+row_id));
            dreamplacePrint(kNONE, "%d", int(state_adjacency_matrix.at(row_id*db.num_sites_y+other_row_id))); 
        }
        dreamplacePrint(kNONE, "\n");
    }
#endif
    // adjacency list 
    state_row_graph.assign(db.num_sites_y, std::vector<int>()); 
#pragma omp parallel for num_threads(num_threads) 
    for (int row_id = 0; row_id < db.num_sites_y; ++row_id)
    {
        auto& adjacency_vec = state_row_graph[row_id]; 
        for (int other_row_id = 0; other_row_id < db.num_sites_y; ++other_row_id)
        {
            if (row_id != other_row_id && state_adjacency_matrix.at(row_id*db.num_sites_y+other_row_id))
            {
                adjacency_vec.push_back(other_row_id); 
            }
        }
    }

}

template <typename DetailedPlaceDBType, typename KReorderState>
void compute_row_conflict_graph(const DetailedPlaceDBType& db, KReorderState& state)
{
    compute_row_conflict_graph(db, state.row2node_map, state.adjacency_matrix, state.row_graph, state.num_threads); 
}


template <typename DetailedPlaceDBType>
void compute_independent_rows(const DetailedPlaceDBType& db, 
        const std::vector<std::vector<int> >& state_row_graph, 
        std::vector<std::vector<int> >& state_independent_rows
        )
{
    // generate independent sets of rows 
    std::vector<unsigned char> dependent_markers (db.num_sites_y, 0); 
    std::vector<unsigned char> selected_markers (db.num_sites_y, 0); 
    int num_selected = 0; 
    while (num_selected < db.num_sites_y)
    {
        std::vector<int> independent_rows; 
        for (int row_id = 0; row_id < db.num_sites_y; ++row_id)
        {
            if (!dependent_markers[row_id] && !selected_markers[row_id])
            {
                independent_rows.push_back(row_id); 
                dependent_markers[row_id] = 1; 
                selected_markers[row_id] = 1; 
                num_selected += 1; 

                for (auto other_row_id : state_row_graph[row_id])
                {
                    dependent_markers[other_row_id] = 1; 
                }
            }
        }
        // recover marker 
        for (auto i : independent_rows)
        {
            for (auto other_row_id : state_row_graph[i])
            {
#ifdef DEBUG
                dreamplaceAssert(std::count(independent_rows.begin(), independent_rows.end(), other_row_id) == 0); 
#endif
                dependent_markers[other_row_id] = 0; 
            }
        }
        state_independent_rows.push_back(independent_rows); 
    }
#ifdef DEBUG
    for (unsigned int i = 0; i < state_independent_rows.size(); ++i)
    {
        auto const& independent_rows = state_independent_rows.at(i); 
        dreamplacePrint(kDEBUG, "group[%d][%lu]: ", i, independent_rows.size());
        for (auto row_id : independent_rows)
        {
            dreamplacePrint(kNONE, "%d ", row_id); 
        }
        dreamplacePrint(kNONE, "\n");
        
        for (auto row_id : independent_rows)
        {
            for (auto other_row_id : independent_rows)
            {
                if (row_id != other_row_id)
                {
                    dreamplaceAssert(std::count(state_row_graph[row_id].begin(), state_row_graph[row_id].end(), other_row_id) == 0); 
                }
            }
        }
    }
    dreamplaceAssert(std::count(selected_markers.begin(), selected_markers.end(), 0) == 0);
#endif
}

template <typename DetailedPlaceDBType, typename KReorderState>
void compute_independent_rows(const DetailedPlaceDBType& db, KReorderState& state)
{
    compute_independent_rows(db, state.row_graph, state.independent_rows);
}


DREAMPLACE_END_NAMESPACE

#endif
