/**
 * @file   collect_independent_sets.h
 * @author Yibo Lin
 * @date   Mar 2019
 */
#ifndef _DREAMPLACE_INDEPENDENT_SET_MATCHING_COLLECT_INDEPENDENT_SETS_H
#define _DREAMPLACE_INDEPENDENT_SET_MATCHING_COLLECT_INDEPENDENT_SETS_H

#include <random>
#include "independent_set_matching/src/construct_selected_node2bin_map.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
int partitioning_diamond(const DetailedPlaceDBType& db, IndependentSetMatchingStateType& state)
{
	// assume cells have been distributed to bins
    state.independent_sets.resize(state.batch_size);
    state.solutions.resize(state.batch_size);
    state.target_pos_x.resize(state.batch_size);
    state.target_pos_y.resize(state.batch_size);
    int num_independent_sets = 0; 
    for (int i = 0; i < db.num_movable_nodes; ++i)
    {
        int seed_node = state.ordered_nodes.at(i);
        if (state.selected_markers.at(seed_node))
        {
            typename DetailedPlaceDBType::type seed_height = db.node_size_y[seed_node];
            auto const& seed_bin = state.node2bin_map.at(seed_node);
            int num_bins_x = db.num_bins_x;
            int num_bins_y = db.num_bins_y;
            int seed_bin_x = seed_bin.bin_id/num_bins_y; 
            int seed_bin_y = seed_bin.bin_id%num_bins_y;
            auto const& bin2node_map = state.bin2node_map;
            auto& independent_set = state.independent_sets.at(num_independent_sets);
            ++num_independent_sets; 
            independent_set.clear();
            for (int j = 0; j < state.max_diamond_search_sequence; ++j)
            {
                // get bin (bx, by)
                int bx = seed_bin_x+state.search_grids.at(j).ic; 
                int by = seed_bin_y+state.search_grids.at(j).ir; 
                if (bx < 0 || bx >= num_bins_x || by < 0 || by >= num_bins_y)
                {
                    continue;
                }
                int bin_id = bx*num_bins_y + by; 
#ifdef DEBUG
                dreamplaceAssert(bin_id < (int)bin2node_map.size());
#endif
                auto const& bin2nodes = bin2node_map.at(bin_id);

                for (auto node_id : bin2nodes)
                {
#ifdef DEBUG
                    dreamplaceAssert(db.node_size_x[node_id] == db.node_size_x[seed_node]);
#endif 
                    if (db.node_size_y[node_id] == seed_height && state.selected_markers.at(node_id))
                    {
                        independent_set.push_back(node_id);
                        state.selected_markers.at(node_id) = 0; 
                        if (independent_set.size() >= (unsigned int)state.set_size)
                        {
                            break; 
                        }
                    }
                }
                if (independent_set.size() >= (unsigned int)state.set_size)
                {
                    break; 
                }
            }
            // make sure batch_size is large enough 
            if (num_independent_sets >= state.batch_size)
            {
                break; 
            }
        }
    }

#ifdef DEBUG
    std::vector<typename DetailedPlaceDBType::type> centers_x (num_independent_sets, 0); 
    std::vector<typename DetailedPlaceDBType::type> centers_y (num_independent_sets, 0); 
    std::vector<typename DetailedPlaceDBType::type> partition_distances (num_independent_sets, 0); 
	std::vector<int> partition_sizes (num_independent_sets); 
    for (int j = 0; j < num_independent_sets; ++j)
    {
        for (auto node_id : state.independent_sets.at(j))
        {
            centers_x.at(j) += db.x[node_id]; 
            centers_y.at(j) += db.y[node_id]; 
        }
        centers_x.at(j) /= state.independent_sets.at(j).size();
        centers_y.at(j) /= state.independent_sets.at(j).size();

        for (auto node_id : state.independent_sets.at(j))
        {
            partition_distances.at(j) += std::abs(db.x[node_id]-centers_x.at(j))
                + std::abs(db.y[node_id]-centers_y.at(j));
        }
    }
    for (int i = 0; i < num_independent_sets; ++i)
    {
        dreamplacePrint(kDEBUG, "partition[%d][%lu]: ", i, state.independent_sets.at(i).size());
        for (auto node_id : state.independent_sets.at(i))
        {
            dreamplacePrint(kNONE, "%d ", node_id);
        }
        if (state.independent_sets.at(i).size())
        {
            dreamplacePrint(kNONE, "; (%g, %g), avg dist %g\n", 
                    centers_x.at(i), 
                    centers_y.at(i), 
                    partition_distances.at(i)/state.independent_sets.at(i).size());
        }
        else 
        {
            dreamplacePrint(kNONE, ";\n");
        }
    }
	for (int i = 0; i < num_independent_sets; ++i)
	{
		partition_sizes.at(i) = state.independent_sets.at(i).size();
	}
	std::sort(partition_sizes.begin(), partition_sizes.end()); 
	dreamplacePrint(kDEBUG, "partition sizes: ");
	for (auto s : partition_sizes)
	{
		dreamplacePrint(kNONE, "%d ", s);
	}
	dreamplacePrint(kNONE, "\n");
#endif

    return num_independent_sets; 
}

template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
int partitioning_kmeans(const DetailedPlaceDBType& db, IndependentSetMatchingStateType& state)
{
    std::vector<typename DetailedPlaceDBType::type> centers_x (state.batch_size);
    std::vector<typename DetailedPlaceDBType::type> centers_y (state.batch_size);
    std::vector<typename DetailedPlaceDBType::type> centers_x_copy (state.batch_size);
    std::vector<typename DetailedPlaceDBType::type> centers_y_copy (state.batch_size);
    std::vector<typename DetailedPlaceDBType::type> weights (state.batch_size, 1.0);
    std::vector<int> partition_sizes (state.batch_size); 
	std::vector<int> partition_sorted_indices (state.batch_size);
    int num_selected = std::count(state.selected_markers.begin(), state.selected_markers.end(), 1);
    std::vector<int> selected_nodes; 
    selected_nodes.reserve(num_selected);
    std::vector<int> node2centers_map (num_selected);

    for (int node_id = 0; node_id < db.num_movable_nodes; ++node_id)
    {
        if (state.selected_markers.at(node_id))
        {
            selected_nodes.push_back(node_id);
        }
    }

    // kmeans 
    // initialize centers 
    std::mt19937 gen1(1234); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<> dis1(0, num_selected);
	for (int i = 0; i < state.batch_size; ++i)
	{
		int node_id = selected_nodes.at(dis1(gen1));
		centers_x.at(i) = db.x[node_id]; 
		centers_y.at(i) = db.y[node_id];
	}

    // kmeans iterations 
    for (int iter = 0; iter < 2; ++iter)
    {
#ifdef DEBUG
        dreamplacePrint(kDEBUG, "# iter %d\n", iter);
#endif
        // update node2centers_map
        for (int i = 0; i < num_selected; ++i)
        {
            int node_id = selected_nodes.at(i);
            auto node_x = db.x[node_id]; 
            auto node_y = db.y[node_id];

            int closest_center = std::numeric_limits<int>::max();
            auto closest_center_distance = std::numeric_limits<typename DetailedPlaceDBType::type>::max();

            for (int j = 0; j < state.batch_size; ++j)
            {
                auto distance = (std::abs(node_x-centers_x.at(j)) + std::abs(node_y-centers_y.at(j)))*weights.at(j); 
                if (distance < closest_center_distance)
                {
                    closest_center = j; 
                    closest_center_distance = distance; 
                }
            }

            node2centers_map.at(i) = closest_center; 
        }
        // update center 
        std::fill(centers_x_copy.begin(), centers_x_copy.end(), 0); 
        std::fill(centers_y_copy.begin(), centers_y_copy.end(), 0); 
        std::fill(partition_sizes.begin(), partition_sizes.end(), 0);
        for (int i = 0; i < num_selected; ++i)
        {
            int node_id = selected_nodes.at(i);
            auto node_x = db.x[node_id]; 
            auto node_y = db.y[node_id];

            int center = node2centers_map.at(i);
            centers_x_copy.at(center) += node_x; 
            centers_y_copy.at(center) += node_y; 
            partition_sizes.at(center) += 1; 
        }
        for (int j = 0; j < state.batch_size; ++j)
        {
            if (partition_sizes.at(j))
            {
                centers_x.at(j) = centers_x_copy.at(j) / partition_sizes.at(j); 
                centers_y.at(j) = centers_y_copy.at(j) / partition_sizes.at(j); 
            }
        }
        // update weight 
        for (int j = 0; j < state.batch_size; ++j)
        {
            if (partition_sizes.at(j) > state.set_size)
            {
                auto ratio = partition_sizes.at(j) / (typename DetailedPlaceDBType::type)state.set_size;
                ratio = 1.0 + 0.5*log(ratio);
#ifdef DEBUG
                dreamplacePrint(kDEBUG, "partition[%d] weight ratio %g, %d nodes\n", j, ratio, partition_sizes.at(j));
#endif
                weights.at(j) *= ratio; 
            }
        }
#if 0
		// move the center of the smallest partition to the largest partition 
		// sort from large to small 
		std::iota(partition_sorted_indices.begin(), partition_sorted_indices.end(), 0);
		std::sort(partition_sorted_indices.begin(), partition_sorted_indices.end(), 
				[&partition_sizes](int a, int b) {return partition_sizes.at(a) > partition_sizes.at(b);}
				);
		int reverse_i = state.batch_size-1; 
		for (int i = 0; i < state.batch_size; ++i)
		{
			int forward = partition_sorted_indices.at(i);
			int backward = partition_sorted_indices.at(reverse_i);

			if (partition_sizes.at(forward) > 2*state.set_size && partition_sizes.at(backward) == 0)
			{
				centers_x.at(backward) = centers_x.at(forward); 
				centers_y.at(backward) = centers_y.at(forward); 
				dreamplacePrint(kDEBUG, "move center %d to %d, %d -> %d\n", backward, forward, partition_sizes.at(backward), partition_sizes.at(forward));
				--reverse_i;
			}
			else 
			{
				break; 
			}
		}
#endif
    }

    // add to independent sets 
    state.independent_sets.resize(state.batch_size); 
    state.solutions.resize(state.batch_size);
    state.target_pos_x.resize(state.batch_size);
    state.target_pos_y.resize(state.batch_size);
    for (int i = 0; i < num_selected; ++i)
    {
        int node_id = selected_nodes.at(i);
        int partition_id = node2centers_map.at(i); 
        state.independent_sets.at(partition_id).push_back(node_id); 
    }
#ifdef DEBUG
    std::vector<typename DetailedPlaceDBType::type> partition_distances (state.batch_size, 0); 
    for (int i = 0; i < num_selected; ++i)
    {
        int node_id = selected_nodes.at(i);
        int partition_id = node2centers_map.at(i);
        partition_distances.at(partition_id) += std::abs(db.x[node_id]-centers_x.at(partition_id))
            + std::abs(db.y[node_id]-centers_y.at(partition_id));
    }
    for (int i = 0; i < state.batch_size; ++i)
    {
        dreamplacePrint(kDEBUG, "partition[%d][%d]: ", i, partition_sizes.at(i));
        for (auto node_id : state.independent_sets.at(i))
        {
            dreamplacePrint(kNONE, "%d ", node_id);
        }
        if (partition_sizes.at(i))
        {
            dreamplacePrint(kNONE, "; (%g, %g), avg dist %g\n", 
                    centers_x.at(i), 
                    centers_y.at(i), 
                    partition_distances.at(i)/partition_sizes.at(i));
        }
        else 
        {
            dreamplacePrint(kNONE, ";\n");
        }
    }
	std::sort(partition_sizes.begin(), partition_sizes.end()); 
	dreamplacePrint(kDEBUG, "partition sizes: ");
	for (auto s : partition_sizes)
	{
		dreamplacePrint(kNONE, "%d ", s);
	}
	dreamplacePrint(kNONE, "\n");
#endif

    return state.batch_size;
}

template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
int partitioning_parallel(const DetailedPlaceDBType& db, IndependentSetMatchingStateType& state)
{
    std::vector<int> node2partition_map (db.num_movable_nodes, std::numeric_limits<int>::max());

    std::vector<typename DetailedPlaceDBType::type> partition_centers_sum_x (state.batch_size, 0);
    std::vector<typename DetailedPlaceDBType::type> partition_centers_sum_y (state.batch_size, 0);
    std::vector<int> partition_sizes (state.batch_size, 0);
    std::vector<typename DetailedPlaceDBType::type> partition_centers_sum_x_new (partition_centers_sum_x);
    std::vector<typename DetailedPlaceDBType::type> partition_centers_sum_y_new (partition_centers_sum_y);
    std::vector<int> partition_sizes_new (partition_sizes);
    std::vector<char> selected_markers (state.selected_markers.begin(), state.selected_markers.end()); 
    std::vector<char> selected_markers_new (selected_markers); 

    int num_partitions = 0; 

    int iter = 0; 
    bool empty = false; 
    while (!empty)
    {
        //dreamplacePrint(kDEBUG, "# iter %d, %lu nodes\n", iter, std::count(selected_markers.begin(), selected_markers.end(), 1));
//#pragma omp parallel for num_threads(state.num_threads)
//        for (int i = 0; i < state.batch_size; ++i)
//        {
//            partition_centers_sum_x_new.at(i) = partition_centers_sum_x.at(i);
//            partition_centers_sum_y_new.at(i) = partition_centers_sum_y.at(i);
//            partition_sizes_new.at(i) = partition_sizes.at(i);
//        }

        empty = true; 
#pragma omp parallel for num_threads(state.num_threads)
        for (int seed_node = 0; seed_node < db.num_movable_nodes; ++seed_node)
        {
            if (selected_markers.at(seed_node))
            {
                auto seed_height = db.node_size_y[seed_node];
                auto seed_x = db.x[seed_node];
                auto seed_y = db.y[seed_node];

                auto const& seed_bin = state.node2bin_map.at(seed_node);
                int num_bins_x = db.num_bins_x;
                int num_bins_y = db.num_bins_y;
                int seed_bin_x = seed_bin.bin_id/num_bins_y; 
                int seed_bin_y = seed_bin.bin_id%num_bins_y;
                auto const& bin2node_map = state.bin2node_map;

                bool min_random_id_flag = true; 
                int random_id = state.ordered_nodes.at(seed_node);
                int closest_partition = std::numeric_limits<int>::max(); 
                typename DetailedPlaceDBType::type closest_partition_distance = std::numeric_limits<typename DetailedPlaceDBType::type>::max(); 
                for (int j = 0; j < state.max_diamond_search_sequence; ++j)
                {
                    // get bin (bx, by)
                    int bx = seed_bin_x+state.search_grids.at(j).ic; 
                    int by = seed_bin_y+state.search_grids.at(j).ir; 
                    if (bx < 0 || bx >= num_bins_x || by < 0 || by >= num_bins_y)
                    {
                        continue;
                    }
                    int bin_id = bx*num_bins_y + by; 
#ifdef DEBUG
                    dreamplaceAssert(bin_id < (int)bin2node_map.size());
#endif
                    auto const& bin2nodes = bin2node_map.at(bin_id);

                    for (auto node_id : bin2nodes)
                    {
#ifdef DEBUG
                        dreamplaceAssert(db.node_size_x[node_id] == db.node_size_x[seed_node]);
#endif 
                        if (db.node_size_y[node_id] == seed_height)
                        {
                            if (selected_markers.at(node_id)) // no partition yet 
                            {
                                if (state.ordered_nodes.at(node_id) < random_id)
                                {
                                    min_random_id_flag = false; 
                                    break; 
                                }
                            }
                            else // already has a partition 
                            {
                                int partition_id = node2partition_map.at(node_id);
#ifdef DEBUG
                                dreamplaceAssert(partition_id < state.batch_size);
#endif
                                // not full yet 
                                auto s = partition_sizes.at(partition_id);
                                if (s < state.set_size)
                                {
#ifdef DEBUG
                                    dreamplaceAssert(s >= 0);
#endif
                                    auto pcx = partition_centers_sum_x.at(partition_id)/s;
                                    auto pcy = partition_centers_sum_y.at(partition_id)/s;
                                    auto distance = std::abs(seed_x-pcx) + std::abs(seed_y-pcy);
                                    if (distance < closest_partition_distance)
                                    {
                                        closest_partition_distance = distance;
                                        closest_partition = partition_id;
                                    }
                                }
                            }
                        }
                    }
                    if (!min_random_id_flag)
                    {
                        break; 
                    }
                }
                // current node has the smallest random id in its search region 
                if (min_random_id_flag)
                {
                    if (closest_partition == std::numeric_limits<int>::max()) // no closest partition 
                    {
                        // create new partition 
                        int partition_id = num_partitions; 
#ifdef DEBUG
                        dreamplacePrint(kDEBUG, "create node %d to partition %d\n", seed_node, partition_id);
#endif
                        node2partition_map.at(seed_node) = partition_id; 
                        partition_centers_sum_x_new.at(partition_id) = seed_x;
                        partition_centers_sum_y_new.at(partition_id) = seed_y;
                        partition_sizes_new.at(partition_id) = 1; 
#pragma omp atomic
                        num_partitions += 1; 
                    }
                    else // has closest partition 
                    {
#ifdef DEBUG
                        dreamplacePrint(kDEBUG, "add node %d to partition %d\n", seed_node, closest_partition);
                        dreamplaceAssert(closest_partition < num_partitions);
#endif
                        // add to closest partition 
                        node2partition_map.at(seed_node) = closest_partition; 
                        auto& pcx = partition_centers_sum_x_new.at(closest_partition);
                        auto& pcy = partition_centers_sum_y_new.at(closest_partition);
                        int& s = partition_sizes_new.at(closest_partition);
#pragma omp critical 
                        {
                            pcx += seed_x; 
                            pcy += seed_y; 
                            s += 1; 
                        }
                    }
                    selected_markers_new.at(seed_node) = 0; 
#pragma omp atomic
                    empty &= false; 
                }
            }
        }

#pragma omp parallel for num_threads(state.num_threads)
        for (int i = 0; i < state.batch_size; ++i)
        {
            partition_centers_sum_x.at(i) = partition_centers_sum_x_new.at(i);
            partition_centers_sum_y.at(i) = partition_centers_sum_y_new.at(i);
            partition_sizes.at(i) = partition_sizes_new.at(i);
        }
#pragma omp parallel for num_threads(state.num_threads)
        for (int i = 0; i < db.num_movable_nodes; ++i)
        {
            selected_markers.at(i) = selected_markers_new.at(i);
        }
        ++iter; 
    }

    // add to independent sets 
    state.independent_sets.resize(state.batch_size); 
    state.solutions.resize(state.batch_size);
    state.target_pos_x.resize(state.batch_size);
    state.target_pos_y.resize(state.batch_size);
    for (int seed_node = 0; seed_node < db.num_movable_nodes; ++seed_node)
    {
        if (state.selected_markers.at(seed_node))
        {
            int partition_id = node2partition_map.at(seed_node);
            if (partition_id < std::numeric_limits<int>::max())
            {
                state.independent_sets.at(partition_id).push_back(seed_node);
            }
        }
    }
#ifdef DEBUG
        std::vector<typename DetailedPlaceDBType::type> partition_distances (state.batch_size, 0); 
        for (int seed_node = 0; seed_node < db.num_movable_nodes; ++seed_node)
        {
            if (state.selected_markers.at(seed_node))
            {
                int partition_id = node2partition_map.at(seed_node);
                if (partition_id < std::numeric_limits<int>::max())
                {
                    partition_distances.at(partition_id) += std::abs(db.x[seed_node]-partition_centers_sum_x.at(partition_id)/partition_sizes.at(partition_id))
                        + std::abs(db.y[seed_node]-partition_centers_sum_y.at(partition_id)/partition_sizes.at(partition_id));
                }
            }
        }
        for (int i = 0; i < state.batch_size; ++i)
        {
            dreamplacePrint(kDEBUG, "partition[%d][%d]: ", i, partition_sizes.at(i));
            for (auto node_id : state.independent_sets.at(i))
            {
                dreamplacePrint(kDEBUG, "%d ", node_id);
            }
            if (partition_sizes.at(i))
            {
                dreamplacePrint(kDEBUG, "; (%g, %g), avg dist %g\n", 
                        partition_centers_sum_x.at(i)/partition_sizes.at(i), 
                        partition_centers_sum_y.at(i)/partition_sizes.at(i), 
                        partition_distances.at(i)/partition_sizes.at(i));
            }
            else 
            {
                dreamplacePrint(kDEBUG, ";\n");
            }
        }
#endif

#ifdef DEBUG
    dreamplaceAssert(num_partitions == partition_sizes.size()-std::count(partition_sizes.begin(), partition_sizes.end(), 0));
#endif

    return num_partitions; 
}

template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
int collect_independent_sets(const DetailedPlaceDBType& db, IndependentSetMatchingStateType& state)
{
    construct_selected_node2bin_map(db, state);
    for (auto& independent_set : state.independent_sets)
    {
        independent_set.clear();
    }

	int num_independent_sets = partitioning_diamond(db, state);
	//int num_independent_sets = partitioning_kmeans(db, state);
	//int num_independent_sets = partitioning_parallel(db, state);

    // sort sets according to large to small 
    std::sort(state.independent_sets.begin(), state.independent_sets.end(), 
            [&](const std::vector<int>& s1, const std::vector<int>& s2){
            return s1.size() > s2.size(); 
            });
    // clean small sets 
    for (int i = 0; i < (int)state.independent_sets.size(); ++i)
    {
        if (i >= state.batch_size || state.independent_sets.at(i).size() < 3U)
        {
            state.independent_sets.at(i).clear(); 
        }
        else 
        {
            num_independent_sets = i; 
        }
    }
	// shrink large sets 
	for (auto& independent_set : state.independent_sets)
	{
		if (independent_set.size() > (unsigned int)state.set_size)
		{
			independent_set.resize(state.set_size);
		}
	}
    //num_independent_sets = state.independent_sets.size();
    //for (int i = 0; i < num_independent_sets; )
    //{
    //    if (state.independent_sets[i].size() < 3U)
    //    {
    //        std::swap(state.independent_sets[i], state.independent_sets[num_independent_sets-1]); 
    //        state.independent_sets[num_independent_sets-1].clear(); 
    //        --num_independent_sets; 
    //    }
    //    else 
    //    {
    //        ++i; 
    //    }
    //}

    int avg_set_size = 0; 
    int max_set_size = 0; 
    for (int i = 0; i < num_independent_sets; ++i)
    {
        avg_set_size += state.independent_sets.at(i).size();
        max_set_size = std::max(max_set_size, (int)state.independent_sets.at(i).size());
    }
    dreamplacePrint(kDEBUG, "%d sets, average set size %d, max set size %d\n", 
            num_independent_sets, avg_set_size/num_independent_sets, max_set_size);

#ifdef DEBUG
    dreamplacePrint(kDEBUG, "#sizes = %lu, actual %d\n", size2id_map.size(), num_independent_sets);
    for (int i = 0; i < num_independent_sets; ++i)
    {
        auto const& independent_set = state.independent_sets.at(i);
        dreamplacePrint(kNONE, "%lu ", independent_set.size());
        typename DetailedPlaceDBType::type width = 0; 
        for (auto node_id : independent_set)
        {
            if (width == 0)
            {
                width = db.node_size_x[node_id];
            }
            else 
            {
                dreamplaceAssertMsg(width == db.node_size_x[node_id], "width inconsistent %g vs %g\n", width, db.node_size_x[node_id]);
            }
        }
    }
    dreamplacePrint(kNONE, "\n");
#endif

    return num_independent_sets; 
}

DREAMPLACE_END_NAMESPACE

#endif
