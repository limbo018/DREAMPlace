/**
 * @file   collect_independent_sets.cuh
 * @author Yibo Lin
 * @date   Mar 2019
 */
#ifndef _DREAMPLACE_INDEPENDENT_SET_MATCHING_COLLECT_INDEPENDENT_SETS_CUH
#define _DREAMPLACE_INDEPENDENT_SET_MATCHING_COLLECT_INDEPENDENT_SETS_CUH

#include "utility/src/utils.cuh"
#include "independent_set_matching/src/kmeans.cuh"
#include "independent_set_matching/src/partition_independent_sets_cuda2cpu.cuh"

DREAMPLACE_BEGIN_NAMESPACE

template <typename IndependentSetMatchingStateType>
__global__ void init_independent_sets_kernel(IndependentSetMatchingStateType state)
{
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < state.batch_size*state.set_size; i += blockDim.x*gridDim.x)
    {
        if (i == 0)
        {
            *state.device_num_independent_sets = 0;
        }
        state.independent_sets[i] = cuda::numeric_limits<int>::max();
    }
}

template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
__global__ void check_db_kernel(DetailedPlaceDBType db, IndependentSetMatchingStateType state)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        printf("check_db_kernel\n"); 
        for (int i = 0; i < db.num_movable_nodes; ++i)
        {
            if (!(db.x[i] >= db.xl && db.x[i] < db.xh && db.y[i] >= db.yl && db.y[i] < db.yh))
            {
                printf("[E] collect_independent_sets found node %d (%g, %g) out of bounds\n", i, db.x[i], db.y[i]);
            }
            assert(db.x[i] >= db.xl && db.x[i] < db.xh && db.y[i] >= db.yl && db.y[i] < db.yh);
        }
    }
}

template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
__global__ void print_independent_set_sizes_kernel(DetailedPlaceDBType db, IndependentSetMatchingStateType state)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        int avg = 0; 
        int max_size = 0; 
        printf("independent_set_sizes = ");
        for (int i = 0; i < state.batch_size; ++i)
        {
            int count = 0; 
            for (int j = 0; j < state.set_size; ++j)
            {
                int node_id = state.independent_sets[i*state.set_size+j];
                count += (node_id < db.num_movable_nodes); 
            }
            printf("%d ", count);
            avg += count; 
            max_size = max(max_size, count);
        }
        printf("\n");
        printf("avg %d, max %d\n", avg/(state.batch_size), max_size);
        printf("num_independent_sets %d\n", *state.device_num_independent_sets);
    }
}

template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
__global__ void print_independent_set_sizes_kernel2(DetailedPlaceDBType db, IndependentSetMatchingStateType state)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        int avg = 0; 
        int max_size = 0; 
        printf("independent_set_sizes = ");
        for (int i = 0; i < state.batch_size; ++i)
        {
            int set_id = i;
            printf("%d ", state.independent_set_sizes[set_id]);
            avg += state.independent_set_sizes[set_id]; 
            max_size = max(max_size, state.independent_set_sizes[set_id]);
        }
        printf("\n");
        printf("avg %d, max %d\n", avg/(state.batch_size), max_size);
    }
}

template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
__global__ void print_ordered_independent_sets_kernel(DetailedPlaceDBType db, IndependentSetMatchingStateType state)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        printf("ordered_independent_sets = "); 
        for (int i = 0; i < state.batch_size; ++i)
        {
            int set_id = state.ordered_independent_sets[i];
            printf("%d ", set_id); 
        }
        printf("\n");
    }
}

template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
__global__ void print_ordered_independent_set_sizes_kernel(DetailedPlaceDBType db, IndependentSetMatchingStateType state)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        int avg = 0; 
        int max_size = 0; 
        printf("ordered independent_set_sizes = ");
        for (int i = 0; i < state.batch_size; ++i)
        {
            int set_id = state.ordered_independent_sets[i];
            printf("%d(%d) ", state.ordered_independent_sets[set_id], state.independent_set_sizes[set_id]);
            avg += state.independent_set_sizes[set_id]; 
            max_size = max(max_size, state.independent_set_sizes[set_id]);
        }
        printf("\n");
        printf("avg %d, max %d\n", avg/(state.batch_size), max_size);
    }
}

template <typename IndependentSetMatchingStateType>
__global__ void permute_independent_sets_kernel(IndependentSetMatchingStateType state)
{
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < state.batch_size; i += blockDim.x*gridDim.x)
    {
        state.reordered_independent_sets[state.ordered_independent_sets[i]] = i; 
    }
}

template <typename IndependentSetMatchingStateType>
__global__ void print_reordered_independent_sets_kernel(IndependentSetMatchingStateType state)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        printf("reordered_independent_sets\n");
        for (int i = 0; i < state.batch_size; i += 1)
        {
            printf("%d(%d) ", i, state.reordered_independent_sets[i]);
        }
        printf("\n");
    }
}

template <typename IndependentSetMatchingStateType>
__global__ void compute_num_independent_sets_kernel(IndependentSetMatchingStateType state)
{
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < state.batch_size; i += blockDim.x*gridDim.x)
    {
        int independent_set_size = state.independent_set_sizes[i]; 
        if (independent_set_size > 2)
        {
            atomicMax(state.device_num_independent_sets, i+1);
        }
    }
}

__global__ void print_independent_set_kernel(const int* independent_set, int set_size)
{
	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x;
    if (tid == 0 && bid == 0)
    {
        printf("[%d]\n", set_size);
        for (int i = 0; i < set_size; ++i)
        {
            printf("%d ", independent_set[i]);
        }
        printf("\n");
    }
}

struct CompareNodeBySizeId 
{
    const int* node_size_id; 
    CompareNodeBySizeId(const int* n)
        : node_size_id(n)
    {
    }
    __device__ bool operator()(int node_id1, int node_id2) const 
    {
        return node_size_id[node_id1] < node_size_id[node_id2];
    }
};

template <typename T>
struct CompareNodeBySpace 
{
    const Space<T>* spaces; 
    CompareNodeBySpace(const Space<T>* s)
        : spaces(s)
    {
    }
    __device__ bool operator()(int node_id1, int node_id2) const 
    {
        return spaces[node_id1] < spaces[node_id2];
    }
};

template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
__global__ void init_num_selected_with_sizes_kernel(DetailedPlaceDBType db, IndependentSetMatchingStateType state)
{
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < state.num_node_sizes+1; i += blockDim.x*gridDim.x)
    {
        state.device_num_selected_prefix_sum[i] = state.num_selected; 
    }
}

template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
__global__ void compute_num_selected_with_sizes_kernel(DetailedPlaceDBType db, IndependentSetMatchingStateType state)
{
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < state.num_selected-1; i += blockDim.x*gridDim.x)
    {
        int node_id = state.selected_maximal_independent_set[i]; 
        int size_id = state.node_size_id[node_id]; 
        if (i == 0)
        {
            for (int j = 0; j <= size_id; ++j)
            {
                state.device_num_selected_prefix_sum[j] = i;
            }
        }

        int next_node_id = state.selected_maximal_independent_set[i+1];
        int next_size_id = state.node_size_id[next_node_id]; 
        if (size_id != next_size_id && next_size_id <= state.num_node_sizes)
        {
            for (int j = size_id+1; j <= next_size_id; ++j)
            {
                state.device_num_selected_prefix_sum[j] = i+1;
            }
        }
    }
}

template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
__global__ void print_selected(DetailedPlaceDBType db, IndependentSetMatchingStateType state)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        printf("selected[%d]\n", state.num_selected);
        for (int i = 0; i < state.num_selected; ++i)
        {
            printf("%d(%d) ", state.independent_sets[i], state.node_size_id[state.independent_sets[i]]);
        }
        printf("\n");
    }
}

template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
__global__ void print_num_selected_with_sizes_kernel(DetailedPlaceDBType db, IndependentSetMatchingStateType state)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        printf("device_num_selected_prefix_sum[%d]\n", state.num_node_sizes);
        for (int i = 0; i < state.num_node_sizes; ++i)
        {
            printf("%d ", state.device_num_selected_prefix_sum[i]);
        }
        printf("\n");
        for (int i = 0; i < state.num_node_sizes; ++i)
        {
            for (int j = state.device_num_selected_prefix_sum[i]; j < state.device_num_selected_prefix_sum[i+1]; ++j)
            {
                int node_id = state.selected_maximal_independent_set[j]; 
                int size_id = state.node_size_id[node_id]; 
                if (size_id != i)
                {
                    printf("i %d, j %d, node_id %d, size_id %d\n", i, j, node_id, size_id);
                }
                assert(size_id == i); 
            }
        }
        printf("\n");
    }
}


template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
__global__ void estimate_clusters2set_sizes_kernel(DetailedPlaceDBType db, IndependentSetMatchingStateType state)
{
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < state.num_selected; i += blockDim.x*gridDim.x)
    {
        int node_id = state.selected_maximal_independent_set[i]; 
        int set_id = state.node2center[node_id]; 
        if (set_id < state.batch_size)
        {
            atomicAdd(state.independent_set_sizes + set_id, 1); 
        }
    }
}

template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
__global__ void assign_clusters2ordered_sets_kernel(DetailedPlaceDBType db, IndependentSetMatchingStateType state)
{
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < state.num_selected; i += blockDim.x*gridDim.x)
    {
        int node_id = state.selected_maximal_independent_set[i]; 
        int set_id = state.node2center[node_id]; 
        if (set_id < state.batch_size)
        {
            // reorder with sort by independent_set_sizes 
            set_id = state.reordered_independent_sets[set_id]; 
            int index = atomicAdd(state.independent_set_sizes + set_id, 1); 
            if (index < state.set_size)
            {
                state.independent_sets[set_id*state.set_size + index] = node_id;
            }
        }
    }
}

template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
void collect_independent_sets(const DetailedPlaceDBType& db, IndependentSetMatchingStateType& state, 
        KMeansState<typename DetailedPlaceDBType::type>& kmeans_state, 
        DetailedPlaceCPUDB<typename DetailedPlaceDBType::type>& host_db, 
        IndependentSetMatchingCPUState<typename DetailedPlaceDBType::type>& host_state 
        )
{
    //partition_independent_sets_cuda2cpu(db, state, host_db, host_state);
    partition_kmeans(db, state, kmeans_state);
}

DREAMPLACE_END_NAMESPACE

#endif
