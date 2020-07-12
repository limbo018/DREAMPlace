/**
 * @file   partition_independent_sets_cuda2cpu.cuh
 * @author Yibo Lin
 * @date   Jul 2019
 */
#ifndef _DREAMPLACE_INDEPENDENT_SET_MATCHING_PARTITION_INDEPENDENT_SETS_CUDA2CPU_CUH
#define _DREAMPLACE_INDEPENDENT_SET_MATCHING_PARTITION_INDEPENDENT_SETS_CUDA2CPU_CUH

#include "independent_set_matching/src/construct_selected_node2bin_map.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
int partitioning_diamond(const DetailedPlaceDBType& db, IndependentSetMatchingStateType& state)
{
	// assume cells have been distributed to bins
    state.independent_sets.resize(state.batch_size);
    int num_independent_sets = 0; 
    //for (auto seed_node : state.selected_nodes)
    for (int i = 0; i < db.num_movable_nodes; ++i)
    {
        int seed_node = state.ordered_nodes[i];
        if (state.selected_markers[seed_node])
        {
            typename DetailedPlaceDBType::type seed_height = db.node_size_y[seed_node];
            auto const& seed_bin = state.node2bin_map.at(seed_node);
            int num_bins_x = db.num_bins_x;
            int num_bins_y = db.num_bins_y;
            int seed_bin_x = seed_bin.bin_id/num_bins_y; 
            int seed_bin_y = seed_bin.bin_id - num_bins_y*seed_bin_x;
            auto const& bin2node_map = state.bin2node_map;
            auto& independent_set = state.independent_sets.at(num_independent_sets);
            ++num_independent_sets; 
            independent_set.clear();
            for (int j = 0; j < state.max_diamond_search_sequence; ++j)
            {
                // get bin (bx, by)
                int bx = seed_bin_x+state.search_grids[j].ic; 
                int by = seed_bin_y+state.search_grids[j].ir; 
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
                    if (db.node_size_y[node_id] == seed_height && state.selected_markers[node_id])
                    {
                        independent_set.push_back(node_id);
                        state.selected_markers[node_id] = 0; 
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
    //for (auto seed_node : state.selected_nodes)
    //{
    //    state.selected_markers[seed_node] = 1; 
    //}

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

    //state.solutions.resize(num_independent_sets);
    //state.target_pos_x.resize(num_independent_sets);
    //state.target_pos_y.resize(num_independent_sets);

    return num_independent_sets; 
}

template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
int partitioning_kmeans(const DetailedPlaceDBType& db, IndependentSetMatchingStateType& state)
{
    int num_seeds = state.selected_nodes.size() / state.set_size; 
    std::vector<typename DetailedPlaceDBType::type> centers_x (num_seeds);
    std::vector<typename DetailedPlaceDBType::type> centers_y (num_seeds);
    std::vector<typename DetailedPlaceDBType::type> weights (num_seeds, 1.0);
    std::vector<int> partition_sizes (num_seeds); 
	std::vector<int> partition_sorted_indices (num_seeds);
    std::vector<int> node2centers_map (state.selected_nodes.size());

    // kmeans 
    // initialize centers 
    std::mt19937 gen1(1234); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<> dis1(0, state.selected_nodes.size());
	for (int i = 0; i < num_seeds; ++i)
	{
		int node_id = state.selected_nodes.at(dis1(gen1) % state.selected_nodes.size());
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
        for (unsigned int i = 0; i < state.selected_nodes.size(); ++i)
        {
            int node_id = state.selected_nodes.at(i);
            auto node_x = db.x[node_id]; 
            auto node_y = db.y[node_id];

            int closest_center = std::numeric_limits<int>::max();
            auto closest_center_distance = std::numeric_limits<typename DetailedPlaceDBType::type>::max();

            for (int j = 0; j < num_seeds; ++j)
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
        std::fill(partition_sizes.begin(), partition_sizes.end(), 0);
        for (unsigned int i = 0; i < state.selected_nodes.size(); ++i)
        {
            int center = node2centers_map.at(i);
            partition_sizes.at(center) += 1; 
        }
        for (int j = 0; j < num_seeds; ++j)
        {
            if (partition_sizes.at(j))
            {
                centers_x.at(j) = 0; 
                centers_y.at(j) = 0; 
            }
        }
        for (unsigned int i = 0; i < state.selected_nodes.size(); ++i)
        {
            int node_id = state.selected_nodes.at(i);
            auto node_x = db.x[node_id]; 
            auto node_y = db.y[node_id];

            int center = node2centers_map.at(i);
            centers_x.at(center) += node_x; 
            centers_y.at(center) += node_y; 
        }
        for (int j = 0; j < num_seeds; ++j)
        {
            if (partition_sizes.at(j))
            {
                centers_x.at(j) /= partition_sizes.at(j); 
                centers_y.at(j) /= partition_sizes.at(j); 
            }
        }
        // update weight 
        for (int j = 0; j < num_seeds; ++j)
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
    }

    // add to independent sets 
    state.independent_sets.resize(num_seeds); 
    for (auto& independent_set : state.independent_sets)
    {
        independent_set.clear();
    }
    for (unsigned int i = 0; i < state.selected_nodes.size(); ++i)
    {
        int node_id = state.selected_nodes.at(i);
        int partition_id = node2centers_map.at(i); 
        if (state.independent_sets.at(partition_id).size() < (unsigned int)state.set_size)
        {
            state.independent_sets.at(partition_id).push_back(node_id); 
        }
    }
#ifdef DEBUG
    std::vector<typename DetailedPlaceDBType::type> partition_distances (num_seeds, 0); 
    for (unsigned int i = 0; i < state.selected_nodes.size(); ++i)
    {
        int node_id = state.selected_nodes.at(i);
        int partition_id = node2centers_map.at(i);
        partition_distances.at(partition_id) += std::abs(db.x[node_id]-centers_x.at(partition_id))
            + std::abs(db.y[node_id]-centers_y.at(partition_id));
    }
    for (int i = 0; i < num_seeds; ++i)
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

    return num_seeds;
}

template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
void postprocess_independent_sets(const DetailedPlaceDBType& db, IndependentSetMatchingStateType& state, int& num_independent_sets)
{
    // sort sets according to large to small 
    std::sort(state.independent_sets.begin(), state.independent_sets.end(), 
            [&](const std::vector<int>& s1, const std::vector<int>& s2){
            return s1.size() > s2.size(); 
            });
    // clean small sets 
    for (unsigned int i = 0; i < state.independent_sets.size(); ++i)
    {
        if (i >= state.batch_size || state.independent_sets.at(i).size() < 3U)
        {
            num_independent_sets = i; 
            break; 
        }
    }
    for (unsigned int i = num_independent_sets; i < state.independent_sets.size(); ++i)
    {
        state.independent_sets.at(i).clear();
    }
	////// shrink large sets 
	////for (auto& independent_set : state.independent_sets)
	////{
	////	if (independent_set.size() > (unsigned int)state.set_size)
	////	{
	////		independent_set.resize(state.set_size);
	////	}
	////}
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

    // prepare flat 
    std::fill(state.independent_set_sizes.begin(), state.independent_set_sizes.end(), 0);
    std::fill(state.flat_independent_sets.begin(), state.flat_independent_sets.end(), std::numeric_limits<int>::max());
    for (int i = 0; i < state.num_independent_sets; ++i)
    {
        state.independent_set_sizes[i] = state.independent_sets.at(i).size();
        for (unsigned int j = 0; j < state.independent_sets.at(i).size(); ++j)
        {
            state.flat_independent_sets.at(i*state.set_size + j) = state.independent_sets.at(i).at(j);
        }
    }

    int avg_set_size = 0; 
    int max_set_size = 0; 
    for (int i = 0; i < num_independent_sets; ++i)
    {
        avg_set_size += state.independent_sets[i].size();
        max_set_size = std::max(max_set_size, (int)state.independent_sets[i].size());
    }
    dreamplacePrint(kDEBUG, "from %lu nodes, %d sets, average set size %d, max set size %d\n", state.selected_nodes.size(), num_independent_sets, avg_set_size/num_independent_sets, max_set_size);

#ifdef DEBUG
    dreamplacePrint(kDEBUG, "#sizes = %lu, actual %d\n", size2id_map.size(), num_independent_sets);
    for (int i = 0; i < num_independent_sets; ++i)
    {
        auto const& independent_set = state.independent_sets[i];
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

}

template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
void update_cpu_state(const DetailedPlaceDBType& db, const IndependentSetMatchingStateType& state,
        DetailedPlaceCPUDB<typename DetailedPlaceDBType::type>& host_db, 
        IndependentSetMatchingCPUState<typename DetailedPlaceDBType::type>& host_state
        )
{
    checkCUDA(cudaMemcpy(host_db.x.data(), db.x, sizeof(typename DetailedPlaceDBType::type)*db.num_movable_nodes, cudaMemcpyDeviceToHost));
    checkCUDA(cudaMemcpy(host_db.y.data(), db.y, sizeof(typename DetailedPlaceDBType::type)*db.num_movable_nodes, cudaMemcpyDeviceToHost));
    checkCUDA(cudaMemcpy(host_state.ordered_nodes.data(), state.ordered_nodes, sizeof(int)*db.num_movable_nodes, cudaMemcpyDeviceToHost));

    host_state.selected_nodes.resize(state.num_selected);
    checkCUDA(cudaMemcpy(host_state.selected_nodes.data(), state.selected_maximal_independent_set, sizeof(int)*state.num_selected, cudaMemcpyDeviceToHost));
    checkCUDA(cudaMemcpy(host_state.selected_markers.data(), state.selected_markers, sizeof(unsigned char)*db.num_movable_nodes, cudaMemcpyDeviceToHost));

    std::random_shuffle(host_state.selected_nodes.begin(), host_state.selected_nodes.end());
}

template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
void partition_independent_sets_cuda2cpu(const DetailedPlaceDBType& db, IndependentSetMatchingStateType& state, 
        DetailedPlaceCPUDB<typename DetailedPlaceDBType::type>& host_db, 
        IndependentSetMatchingCPUState<typename DetailedPlaceDBType::type>& host_state
        )
{
    CPUTimer::hr_clock_rep timer_start, timer_stop; 
    
    update_cpu_state(db, state, host_db, host_state);

    timer_start = CPUTimer::getGlobaltime();
    construct_selected_node2bin_map(host_db, host_state);
    timer_stop = CPUTimer::getGlobaltime();
    dreamplacePrint(kDEBUG, "construct_selected_node2bin_map takes %g ms\n", 
            CPUTimer::getTimerPeriod()*(timer_stop-timer_start)
            );

    timer_start = CPUTimer::getGlobaltime();
	//host_state.num_independent_sets = partitioning_diamond(host_db, host_state);
	host_state.num_independent_sets = partitioning_kmeans(host_db, host_state);
    timer_stop = CPUTimer::getGlobaltime();
    dreamplacePrint(kDEBUG, "partitioning_diamond takes %g ms\n", 
            CPUTimer::getTimerPeriod()*(timer_stop-timer_start)
            );
    postprocess_independent_sets(host_db, host_state, host_state.num_independent_sets);
    dreamplaceAssert(host_state.num_independent_sets <= state.batch_size);
    dreamplacePrint(kDEBUG, "state.num_independent_sets = %d\n", host_state.num_independent_sets);

    // copy to device 
    state.num_independent_sets = host_state.num_independent_sets; 
    checkCUDA(cudaMemcpy(state.independent_sets, host_state.flat_independent_sets.data(), sizeof(int)*state.batch_size*state.set_size, cudaMemcpyHostToDevice));
    checkCUDA(cudaMemcpy(state.independent_set_sizes, host_state.independent_set_sizes.data(), sizeof(int)*state.batch_size, cudaMemcpyHostToDevice));
    ////checkCUDA(cudaMemcpy(state.device_num_independent_sets, &state.num_independent_sets, sizeof(int), cudaMemcpyHostToDevice));
}

DREAMPLACE_END_NAMESPACE

#endif
