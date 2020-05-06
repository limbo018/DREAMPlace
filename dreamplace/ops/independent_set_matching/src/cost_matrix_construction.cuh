
#ifndef _DREAMPLACE_INDEPENDENT_SET_MATCHING_COST_MATRIX_CONSTRUCTION_CUH
#define _DREAMPLACE_INDEPENDENT_SET_MATCHING_COST_MATRIX_CONSTRUCTION_CUH

#include "utility/src/utils.cuh"
#include "global_swap/src/reduce_min.cuh"
#include "independent_set_matching/src/adjust_pos.h"

DREAMPLACE_BEGIN_NAMESPACE

#define MAX_NODE_DEGREE 32

template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
__global__ void print_net_boxes_kernel(DetailedPlaceDBType db, IndependentSetMatchingStateType state)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        for (int node_id = 0; node_id < db.num_movable_nodes; ++node_id) 
        {
            if (state.selected_markers[node_id])
            {
                int node2pin_id = db.flat_node2pin_start_map[node_id];
                const int node2pin_id_end = db.flat_node2pin_start_map[node_id+1];
                for (; node2pin_id < node2pin_id_end; ++node2pin_id)
                {
                    int node_pin_id = db.flat_node2pin_map[node2pin_id];
                    int net_id = db.pin2net_map[node_pin_id];
                    auto const& box = state.net_boxes[net_id];
                    printf("node %d: net %d (%g, %g, %g, %g)\n", node_id, net_id, box.xl, box.yl, box.xh, box.yh);
                }
            }
        }
    }
}

template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
__global__ void compute_cost_matrix_kernel(DetailedPlaceDBType db, IndependentSetMatchingStateType state)
{
    int i = blockIdx.y; // set 
    int j = blockIdx.x; // node in set 
    const int* __restrict__ independent_set = state.independent_sets + i*state.set_size; 
    auto cost_matrix = state.cost_matrices + i*state.cost_matrix_size + j*state.set_size; 
    __shared__ int node_id; 
    __shared__ typename DetailedPlaceDBType::type node_width; 
    __shared__ SharedBox<typename DetailedPlaceDBType::type> net_boxes[MAX_NODE_DEGREE]; 
    __shared__ int node2pin_id_bgn; 
    __shared__ int node2pin_id_end; 
    if (threadIdx.x == 0)
    {
        node_id = independent_set[j];
        node_width = cuda::numeric_limits<typename DetailedPlaceDBType::type>::max();
        if (node_id < db.num_movable_nodes)
        {
            node_width = db.node_size_x[node_id];

            node2pin_id_bgn = db.flat_node2pin_start_map[node_id];
            node2pin_id_end = db.flat_node2pin_start_map[node_id+1];
            node2pin_id_end = min(node2pin_id_bgn+MAX_NODE_DEGREE, node2pin_id_end);

            int idx = 0; 
            for (int node2pin_id = node2pin_id_bgn; node2pin_id < node2pin_id_end; ++node2pin_id, ++idx)
            {
                int node_pin_id = db.flat_node2pin_map[node2pin_id];
                int net_id = db.pin2net_map[node_pin_id];
                auto& box = net_boxes[idx];
                box.xl = db.xh;
                box.yl = db.yh;
                box.xh = db.xl;
                box.yh = db.yl;
                if (db.net_mask[net_id])
                {
                    int net2pin_id_bgn = db.flat_net2pin_start_map[net_id];
                    int net2pin_id_end = db.flat_net2pin_start_map[net_id+1];
                    for (int net2pin_id = net2pin_id_bgn; net2pin_id < net2pin_id_end; ++net2pin_id)
                    {
                        int net_pin_id = db.flat_net2pin_map[net2pin_id];
                        int other_node_id = db.pin2node_map[net_pin_id];
                        if (other_node_id != node_id)
                        {
                            typename DetailedPlaceDBType::type xxl = db.x[other_node_id]+db.pin_offset_x[net_pin_id];
                            typename DetailedPlaceDBType::type yyl = db.y[other_node_id]+db.pin_offset_y[net_pin_id];
                            box.xl = min(box.xl, xxl);
                            box.xh = max(box.xh, xxl);
                            box.yl = min(box.yl, yyl);
                            box.yh = max(box.yh, yyl);
                        }
                    }
                }
            }
        }
    }

    __syncthreads();

    for (int k = threadIdx.x; k < state.set_size; k += blockDim.x) // pos in set 
    {
        int pos_id = independent_set[k]; 
        auto& cost = cost_matrix[k]; // row major 
        if (node_id < db.num_movable_nodes && pos_id < db.num_movable_nodes)
        {
#ifdef DEBUG
            assert(db.node_size_x[node_id] == db.node_size_x[pos_id]);
#endif
            typename DetailedPlaceDBType::type target_x = db.x[pos_id]; 
            typename DetailedPlaceDBType::type target_y = db.y[pos_id]; 
            auto const& target_space = state.spaces[pos_id];
            int target_hpwl = 0; 
            if (adjust_pos(target_x, node_width, target_space))
            {
                // consider FENCE region 
                if (db.num_regions && !db.inside_fence(node_id, target_x, target_y))
                {
                    cost = BIG_NEGATIVE; // as a marker for post processing 
                }
                else 
                {
                    int idx = 0; 
                    for (int node2pin_id = node2pin_id_bgn; node2pin_id < node2pin_id_end; ++node2pin_id, ++idx)
                    {
#ifdef DEBUG
                        assert(node2pin_id >= 0 && node2pin_id < db.num_pins);
#endif
                        int node_pin_id = db.flat_node2pin_map[node2pin_id];
#ifdef DEBUG
                        assert(node_pin_id >= 0 && node_pin_id < db.num_pins);
#endif
                        int net_id = db.pin2net_map[node_pin_id];
#ifdef DEBUG
                        assert(net_id >= 0 && net_id < db.num_nets);
#endif
                        auto const& box = net_boxes[idx];
                        if (db.net_mask[net_id])
                        {
                            typename DetailedPlaceDBType::type xxl = target_x+db.pin_offset_x[node_pin_id];
                            typename DetailedPlaceDBType::type yyl = target_y+db.pin_offset_y[node_pin_id];
                            typename DetailedPlaceDBType::type bxl = min(box.xl, xxl);
                            typename DetailedPlaceDBType::type bxh = max(box.xh, xxl);
                            typename DetailedPlaceDBType::type byl = min(box.yl, yyl);
                            typename DetailedPlaceDBType::type byh = max(box.yh, yyl);
                            target_hpwl += (bxh-bxl) + (byh-byl); 
                        }
                    }
                    //target_hpwl = target_hpwl*db.row_height + (abs(target_x-node_x) + abs(target_y-node_y));
                    // row major 
#ifdef DEBUG
                    assert(state.set_size*j + k >= 0 && state.set_size*j + k < state.cost_matrix_size);
                    assert(state.large_number > target_hpwl);
#endif
                    cost = target_hpwl; 
                }
            }
            else 
            {
                cost = BIG_NEGATIVE; // as a marker for post processing 
            }
        }
        else 
        {
            //cost = state.large_number*(j != k); 
            cost = BIG_NEGATIVE; // as a marker for post processing
        }
    }
}

/// @brief change from minimization problem for maximization problem with non-negative edge weights 
template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
__global__ void postprocess_cost_matrix_kernel(DetailedPlaceDBType db, IndependentSetMatchingStateType state)
{
    int i = blockIdx.y; // set 
    int j = blockIdx.x; // node in set 
    const int* __restrict__ independent_set = state.independent_sets + i*state.set_size; 
    auto cost_matrix = state.cost_matrices + i*state.cost_matrix_size + j*state.set_size; 
    auto max_cost = state.cost_matrices_copy[i*state.cost_matrix_size];
    for (int k = threadIdx.x; k < state.set_size; k += blockDim.x) // pos in set 
    {
        int node_id = independent_set[j]; 
        int pos_id = independent_set[k]; 
        auto& cost = cost_matrix[k]; // row major 
        if (node_id < db.num_movable_nodes && pos_id < db.num_movable_nodes)
        {
            if (cost >= 0)
            {
                cost = max_cost - cost; 
            }
            // cost < 0 is already assigned to negative  
        }
        else if (j == k)
        {
            cost = max_cost; // dummy cells or positions 
        }
        // j != k is already assigned to negative 
    }
}

template <typename T>
__global__ void print_cost_matrix_kernel(const T* cost_matrix, int set_size)
{
	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x;
    if (tid == 0 && bid == 0)
    {
        printf("[%dx%d]\n", set_size, set_size);
        for (int r = 0; r < set_size; ++r)
        {
            for (int c = 0; c < set_size; ++c)
            {
                auto cost = cost_matrix[r*set_size+c];
                if (cost == BIG_NEGATIVE)
                {
                    printf("X ");
                }
                else 
                {
                    printf("%g ", (double)cost);
                }
            }
            printf("\n");
        }
        printf("\n");
    }
}

template <typename IndependentSetMatchingStateType>
__global__ void print_max_cost_kernel(IndependentSetMatchingStateType state)
{
	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x;
    if (tid == 0 && bid == 0)
    {
        printf("[%d]\n", state.num_independent_sets);
        for (int i = 0; i < state.num_independent_sets; ++i)
        {
            printf("%g ", (double)state.cost_matrices_copy[i*state.cost_matrix_size]); 
        }
        printf("\n");
    }
}

template <typename IndependentSetMatchingStateType>
__global__ void check_cost_matrices_kernel(IndependentSetMatchingStateType state)
{
	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x;
    if (tid == 0 && bid == 0)
    {
        for (int i = 0; i < state.num_independent_sets; ++i)
        {
            for (int j = 0; j < state.cost_matrix_size; ++j)
            {
                auto cost = state.cost_matrices[i*state.cost_matrix_size+j];
                assert(cost == cuda::numeric_limits<typename IndependentSetMatchingStateType::cost_type>::lowest()
                        || cost >= 0);
            }
        }
    }
}

template <typename T>
struct CompareCost
{
    __host__ __device__ bool operator()(T cost1, T cost2) const 
    {
        return cost1 > cost2; 
    }
};

template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
void cost_matrix_construction(const DetailedPlaceDBType& db, IndependentSetMatchingStateType& state)
{
    dim3 grid (state.set_size, state.num_independent_sets, 1);
    compute_cost_matrix_kernel<<<grid, state.set_size>>>(db, state);
#ifdef DEBUG
    //print_cost_matrix_kernel<<<1, 1>>>(state.cost_matrices + state.cost_matrix_size*3, state.set_size);
#endif

    checkCUDA(cudaMemcpy(state.cost_matrices_copy, state.cost_matrices, sizeof(typename IndependentSetMatchingStateType::cost_type)*state.num_independent_sets*state.cost_matrix_size, cudaMemcpyDeviceToDevice));
    typename IndependentSetMatchingStateType::cost_type ref = 0; 
    reduce_2d(state.cost_matrices_copy, state.num_independent_sets, state.cost_matrix_size, ref, CompareCost<typename IndependentSetMatchingStateType::cost_type>());

    postprocess_cost_matrix_kernel<<<grid, state.set_size>>>(db, state);
#ifdef DEBUG
    //print_max_cost_kernel<<<1, 1>>>(state);

    //print_cost_matrix_kernel<<<1, 1>>>(state.cost_matrices + state.cost_matrix_size*3, state.set_size);
    //check_cost_matrices_kernel<<<1, 1>>>(state);
#endif
}

DREAMPLACE_END_NAMESPACE

#endif
