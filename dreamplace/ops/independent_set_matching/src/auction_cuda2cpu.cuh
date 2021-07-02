/**
 * @file   auction_gpu2cpu.cuh
 * @author Yibo Lin
 * @date   Jul 2019
 */

#ifndef _DREAMPLACE_INDEPENDENT_SET_MATCHING_AUCTION_CUDA2CPU_CUH
#define _DREAMPLACE_INDEPENDENT_SET_MATCHING_AUCTION_CUDA2CPU_CUH

#include <cstdlib> 
#include <iostream>
#include <string>

#include <stdio.h>
#include <stdlib.h>

#include "utility/src/utils.cuh"
#include "independent_set_matching/src/auction_cpu.h"

DREAMPLACE_BEGIN_NAMESPACE

__global__ void set_stop_flags_kernel(char* stop_flags, int num_graphs, char value)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < num_graphs)
    {
        stop_flags[i] = value; 
    }
}

template <typename T>
__global__ void print_cost_matrix_hehe(const T* cost_matrix, int set_size, bool major)
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
                if (major) // column major 
                {
                    printf("%g ", (double)cost_matrix[c*set_size+r]);
                }
                else 
                {
                    printf("%g ", (double)cost_matrix[r*set_size+c]);
                }
            }
            printf("\n");
        }
        printf("\n");
    }
}
template <typename T>
void linear_assignment_auction_cuda2cpu(
                const T* cost_matrices,
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
    std::vector<T> cost_matrices_host (num_graphs*num_nodes*num_nodes);
    std::vector<int> solutions_host (num_graphs*num_nodes);

    checkCUDA(cudaMemcpy(cost_matrices_host.data(), cost_matrices, sizeof(T)*num_graphs*num_nodes*num_nodes, cudaMemcpyDeviceToHost));
    checkCUDA(cudaMemcpy(solutions_host.data(), solutions, sizeof(int)*num_graphs*num_nodes, cudaMemcpyDeviceToHost));

    //print_cost_matrix_hehe<<<1, 1>>>(cost_matrices, num_nodes, 0);
    //checkCUDA(cudaDeviceSynchronize());
//#pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < num_graphs; ++i)
    {
        const T* cost_matrix = cost_matrices_host.data()+i*num_nodes*num_nodes; 
        int* solution = solutions_host.data()+i*num_nodes; 
        //if (i == 0)
        //{
        //    for (int j = 0; j < num_nodes; ++j)
        //    {
        //        for (int k = 0; k < num_nodes; ++k)
        //        {
        //            dreamplacePrint(kNONE, "%g, ", (double)cost_matrix[j*num_nodes+k]);
        //        }
        //        dreamplacePrint(kNONE, "\n");
        //    }
        //}
        //T orig_cost = 0; 
        //for (int j = 0; j < num_nodes; ++j)
        //{
        //    orig_cost += cost_matrix[j*num_nodes+j];
        //}
        //T target_cost = 
            AuctionAlgorithmCPULauncher(cost_matrix, solution, num_nodes, std::numeric_limits<T>::max(), 0);
        //dreamplacePrint(kDEBUG, "graph %d, cost %g -> %g, delta %g\n", i, (double)orig_cost, (double)target_cost, (double)target_cost-orig_cost);
    }

    set_stop_flags_kernel<<<ceilDiv(num_graphs, 256), 256>>>(stop_flags, num_graphs, 1);
    checkCUDA(cudaMemcpy(solutions, solutions_host.data(), sizeof(int)*num_graphs*num_nodes, cudaMemcpyHostToDevice));
}

DREAMPLACE_END_NAMESPACE

#endif
