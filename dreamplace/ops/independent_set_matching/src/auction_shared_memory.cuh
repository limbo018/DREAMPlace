/**
 * @file   auction.cuh
 * @author Jiaqi Gu, Yibo Lin
 * @date   Jan 2019
 */

#ifndef _DREAMPLACE_INDEPENDENT_SET_MATCHING_AUCTION_CUH
#define _DREAMPLACE_INDEPENDENT_SET_MATCHING_AUCTION_CUH

#include <cstdlib>
#include <iostream>
#include <string>

#include <stdio.h>
#include <stdlib.h>

#include "utility/src/utils.cuh"

DREAMPLACE_BEGIN_NAMESPACE

#define BIG_NEGATIVE    -9999999
#define MAX_MINIBATCH         64

template <typename T>
inline void init_auction(
        const int num_graphs, 
        const int num_nodes, 
        char*& scratch, 
        char*& stop_flags
        )
{
    checkCUDA(cudaMalloc(&scratch, 
                num_graphs*num_nodes*num_nodes*sizeof(T)
                ));
    allocateCUDA(stop_flags, num_graphs, char);
}

inline void destroy_auction(
        char* scratch, 
        char* stop_flags
        )
{
    destroyCUDA(scratch);
    destroyCUDA(stop_flags); 
}

template <typename T>
__global__ void compute_orig_cost_kernel(const T* cost_matrices, const int num_nodes, T* costs)
{
    int i = blockIdx.x; // set 
    auto cost_matrix = cost_matrices + i*num_nodes*num_nodes; 
    for (int j = threadIdx.x; j < num_nodes; j += blockDim.x)
    {
        atomicAdd(costs+i, cost_matrix[j*num_nodes+j]);
    }
}

template <typename T>
__global__ void compute_solution_cost_kernel(const T* cost_matrices, const int* solutions, const int num_nodes, T* costs)
{
    int i = blockIdx.x; // set 
    auto cost_matrix = cost_matrices + i*num_nodes*num_nodes; 
    auto solution = solutions + i*num_nodes; 
    for (int j = threadIdx.x; j < num_nodes; j += blockDim.x)
    {
        atomicAdd(costs+i, cost_matrix[j*num_nodes+solution[j]]);
    }
}

template <typename T>
__global__ void print_costs_kernel(T* a, int n)
{
	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x;
    if (tid == 0 && bid == 0)
    {
        printf("[%d]\n", n);
        for (int i = 0; i < n; ++i)
        {
            printf("%g ", (double)a[i]);
        }
        printf("\n");
    }
}

template <typename T>
__global__ void check_costs_kernel(const T* a, const T* b, int n, T epsilon)
{
	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x;
    if (tid == 0 && bid == 0)
    {
        for (int i = 0; i < n; ++i)
        {
            if (a[i] > b[i]+epsilon)
            {
                printf("cost error %g > %g\n", a[i], b[i]);
            }
            assert(a[i] <= b[i]+epsilon);
        }
    }
}

template <typename T>
__global__ void print_solution_kernel(const T* solutions, int num_graphs, int num_nodes)
{
	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x;
    if (tid == 0 && bid == 0)
    {
        for (int i = 0; i < num_graphs; ++i)
        {
            printf("[%d]\n", i);
            for (int j = 0; j < num_nodes; ++j)
            {
                printf("%d ", solutions[i*num_nodes+j]);
            }
            printf("\n");
        }
    }
}

__global__ void print_stop_flags_kernel(char* stop_flags, int n)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        printf("[%dx64]\n", n);
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < 64; ++j)
            {
                printf("%d", stop_flags[i]);
            }
            printf("\n");
        }
    }
}


template <typename T>
__global__ void __launch_bounds__(1024, 16)
linear_assignment_auction_kernel(const int num_nodes,
                                const T* __restrict__ cost_ptr,
                                int* solution_ptr, 
                                T*  bids_ptr,
                                char* stop_flag_ptr,
                                const float auction_max_eps,
                                const float auction_min_eps,
                                const float auction_factor,
                                const int max_iterations)
{
    const int batch_id = blockIdx.x;
    const int node_id = threadIdx.x;
    __shared__ float auction_eps;
    __shared__ int num_iteration;
    __shared__ int num_assigned;
    
    extern __shared__ unsigned char s_lap_data[];
    T*   prices      = (T*)s_lap_data;
    int* sbids       = (int*)(prices + num_nodes);
    int* person2item = sbids + num_nodes;
    int* item2person = person2item + num_nodes;

    if(node_id == 0){
        auction_eps = auction_max_eps;
        num_iteration = 0;
    }

    const T* __restrict__ data  = cost_ptr + batch_id * num_nodes * num_nodes;
    int* solution_global        = solution_ptr + batch_id * num_nodes; 
    T* bids                     = bids_ptr + batch_id * num_nodes * num_nodes;
    char* stop_flag             = stop_flag_ptr + batch_id;
    
    prices[node_id] = 0;

    __syncthreads();

    while(auction_eps >= auction_min_eps && num_iteration < max_iterations)
    {
        //clear num_assigned
        if(node_id == 0){
            num_assigned = 0;
        }

        //pre-init
        person2item[node_id] = -1;
        item2person[node_id] = -1;
        
        __syncthreads();
    
        //start iterative solving
        while(num_assigned < num_nodes && num_iteration < max_iterations)
        {
            //phase 1: init bid and bids
            sbids[node_id] = 0;
            
            for(int i = node_id; i < num_nodes*num_nodes; i += blockDim.x){
                bids[i] = 0;
            }
            __syncthreads();

            //phase 2: bidding
            if(person2item[node_id] == -1)
            {
                T top1_val = BIG_NEGATIVE; 
                T top2_val = BIG_NEGATIVE; 
                int top1_col; 
                T tmp_val;

                #pragma unroll 32
                for (int col = 0; col < num_nodes; col++)
                {
                    tmp_val = data[node_id * num_nodes + col]; 
                    if (tmp_val < 0)
                    {
                        continue;
                    }
                    tmp_val = tmp_val - prices[col];
                    if (tmp_val >= top1_val)
                    {
                        top2_val = top1_val;
                        top1_col = col;
                        top1_val = tmp_val;
                    }
                    else if (tmp_val > top2_val)
                    {
                        top2_val = tmp_val;
                    }
                }
                if (top2_val == BIG_NEGATIVE)
                {
                    top2_val = top1_val;
                }
                T bid = top1_val - top2_val + auction_eps;
                bids[num_nodes * top1_col + node_id] = bid;
                atomicMax(sbids + top1_col, 1);
            }

            __syncthreads();

            //phase 3 : assignment
            if(sbids[node_id] != 0) 
            {
                T high_bid  = 0;
                int high_bidder = -1;
    
                T tmp_bid = -1;
                #pragma unroll 64
                for(int i = 0; i < num_nodes; i++)
                {
                    tmp_bid = bids[node_id * num_nodes + i];
                    if(tmp_bid > high_bid)
                    {
                        high_bid    = tmp_bid;
                        high_bidder = i;
                    }
                }
    
                int current_person = item2person[node_id];
                if(current_person >= 0)
                {
                    person2item[current_person] = -1;
                } 
                else 
                {
                    atomicAdd(&num_assigned, 1);
                }
    
                prices[node_id]                += high_bid;
                person2item[high_bidder]          = node_id;
                item2person[node_id]              = high_bidder;
            }
            __syncthreads();
            
            //update iteration
            if(node_id == 0)
            {
                num_iteration++;
            }
            __syncthreads();
        }
        //scale auction_eps
        if(node_id == 0)
        {
            auction_eps *= auction_factor;
        }
        __syncthreads();
    }
    __syncthreads();
    // report whether finish solving
    if(node_id == 0)
    {
        *stop_flag = (num_assigned == num_nodes);
    }
    // write result out
    solution_global[node_id] = person2item[node_id];
}

template <typename T>
void linear_assignment_auction(
                const T* cost_matrics,
                int* solutions,
                const int num_graphs,
                const int num_nodes,
                char* scratch,
                char *stop_flags,
                float auction_max_eps,
                float auction_min_eps,
                float auction_factor,
                int max_iterations)
{
    //get pointers from scratch (size: num_nodes*num_nodes*sizeof(T))
    T* bids           = (T* )scratch;

    //launch solver
    linear_assignment_auction_kernel<T><<<num_graphs, num_nodes, 4*num_nodes*sizeof(T)>>>
                                    (
                                        num_nodes,
                                        cost_matrics,
                                        solutions,
                                        bids,
                                        stop_flags,
                                        auction_max_eps,
                                        auction_min_eps,
                                        auction_factor,
                                        max_iterations
                                    );
    cudaDeviceSynchronize();
}

DREAMPLACE_END_NAMESPACE

#endif
