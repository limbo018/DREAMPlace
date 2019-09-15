/**
 * @file   abacus_place_row_cpu.h
 * @author Yibo Lin
 * @date   Oct 2018
 */

#ifndef GPUPLACE_LEGALIZE_PLACEROW_CPU_H
#define GPUPLACE_LEGALIZE_PLACEROW_CPU_H

#include <cassert>
#include "utility/src/Msg.h"
#include "abacus_cluster.h"

DREAMPLACE_BEGIN_NAMESPACE

/// @param row_nodes node indices in this row 
/// @param clusters pre-allocated clusters in this row with the same length as that of row_nodes 
/// @param num_row_nodes length of row_nodes 
/// @return true if succeed, otherwise false 
template <typename T>
bool abacusPlaceRowCPU(
        const T* init_x, 
        const T* node_size_x, const T* node_size_y, 
        T* x, 
        const T row_height, 
        const T xl, const T xh, 
        const int num_nodes, 
        const int num_movable_nodes, 
        const int num_filler_nodes, 
        int* row_nodes, AbacusCluster<T>* clusters, const int num_row_nodes
        )
{
    // a very large number 
    T M = pow(10, ceil(log((xh-xl)*num_row_nodes)/log(10))); 
    //dreamplacePrint(kDEBUG, "M = %g\n", M);
    bool ret_flag = true; 

    // merge two clusters 
    // the second cluster will be invalid 
    auto merge_cluster = [&](int dst_cluster_id, int src_cluster_id){
        dreamplaceAssert(dst_cluster_id < num_row_nodes); 
        AbacusCluster<T>& dst_cluster = clusters[dst_cluster_id]; 
        dreamplaceAssert(src_cluster_id < num_row_nodes); 
        AbacusCluster<T>& src_cluster = clusters[src_cluster_id]; 

        dreamplaceAssert(dst_cluster.valid() && src_cluster.valid()); 
        for (int i = dst_cluster_id+1; i < src_cluster_id; ++i)
        {
            dreamplaceAssert(!clusters[i].valid());
        }
        dst_cluster.end_row_node_id = src_cluster.end_row_node_id; 
        dreamplaceAssert(dst_cluster.e < M && src_cluster.e < M); 
        dst_cluster.e += src_cluster.e; 
        dst_cluster.q += src_cluster.q - src_cluster.e*dst_cluster.w; 
        dst_cluster.w += src_cluster.w; 
        // update linked list 
        if (src_cluster.next_cluster_id < num_row_nodes)
        {
            clusters[src_cluster.next_cluster_id].prev_cluster_id = dst_cluster_id;
        }
        dst_cluster.next_cluster_id = src_cluster.next_cluster_id; 
        src_cluster.prev_cluster_id = INT_MIN; 
        src_cluster.next_cluster_id = INT_MIN; 
    };

    // collapse clusters between [0, cluster_id]
    // compute the locations and merge clusters 
    auto collapse = [&](int cluster_id, T range_xl, T range_xh){
        int cur_cluster_id = cluster_id; 
        dreamplaceAssert(cur_cluster_id < num_row_nodes); 
        int prev_cluster_id = clusters[cur_cluster_id].prev_cluster_id; 
        AbacusCluster<T>* cluster = nullptr;
        AbacusCluster<T>* prev_cluster = nullptr;

        while (true)
        {
            dreamplaceAssert(cur_cluster_id < num_row_nodes); 
            cluster = &clusters[cur_cluster_id]; 
            cluster->x = cluster->q/cluster->e; 
            // make sure cluster >= range_xl, so fixed nodes will not be moved 
            // in illegal case, cluster+w > range_xh may occur, but it is OK. 
            // We can collect failed clusters later 
            cluster->x = std::max(std::min(cluster->x, range_xh-cluster->w), range_xl);
            dreamplaceAssert(cluster->x >= range_xl && cluster->x+cluster->w <= range_xh);

            prev_cluster_id = cluster->prev_cluster_id; 
            if (prev_cluster_id >= 0)
            {
                prev_cluster = &clusters[prev_cluster_id];
                if (prev_cluster->x+prev_cluster->w > cluster->x)
                {
                    merge_cluster(prev_cluster_id, cur_cluster_id); 
                    cur_cluster_id = prev_cluster_id; 
                }
                else 
                {
                    break; 
                }
            }
            else 
            {
                break; 
            }
        }
    };

    // sort row_nodes from left to right 
    std::sort(row_nodes, row_nodes+num_row_nodes, CompareByNodeCenterXCPU<T>(x, node_size_x));

    // initial cluster has only one cell 
    for (int i = 0; i < num_row_nodes; ++i)
    {
        int node_id = row_nodes[i]; 
        AbacusCluster<T>& cluster = clusters[i]; 
        cluster.prev_cluster_id = i-1; 
        cluster.next_cluster_id = i+1; 
        cluster.bgn_row_node_id = i; 
        cluster.end_row_node_id = i; 
        cluster.e = (node_id < num_movable_nodes && node_size_y[node_id] <= row_height)? 1.0 : M; 
        cluster.q = cluster.e*init_x[node_id];
        cluster.w = node_size_x[node_id]; 
        // this is required since we also include fixed nodes 
        cluster.x = (node_id < num_movable_nodes && node_size_y[node_id] > row_height)? x[node_id] : init_x[node_id];
    }

    // kernel algorithm for placeRow 
    T range_xl = xl; 
    T range_xh = xh; 
    for (int j = 0; j < num_row_nodes; ++j)
    {
        const AbacusCluster<T>& next_cluster = clusters[j]; 
        if (next_cluster.e >= M) // fixed node 
        {
            range_xh = std::min(next_cluster.x, range_xh); 
            break;
        }
        else 
        {
            dreamplaceAssert(node_size_y[row_nodes[j]] == row_height);
        }
    }
    for (int i = 0; i < num_row_nodes; ++i)
    {
        const AbacusCluster<T>& cluster = clusters[i]; 
        if (cluster.e < M)
        {
            dreamplaceAssert(node_size_y[row_nodes[i]] == row_height);
            collapse(i, range_xl, range_xh); 
        }
        else // set range xl/xh according to fixed nodes 
        {
            range_xl = cluster.x+cluster.w; 
            range_xh = xh; 
            for (int j = i+1; j < num_row_nodes; ++j)
            {
                const AbacusCluster<T>& next_cluster = clusters[j]; 
                if (next_cluster.e >= M) // fixed node 
                {
                    range_xh = std::min(next_cluster.x, range_xh); 
                    break;
                }
            }
        }
    }
    
    // apply solution
    for (int i = 0; i < num_row_nodes; ++i)
    {
        if (clusters[i].valid())
        {
            const AbacusCluster<T>& cluster = clusters[i]; 
            T xc = cluster.x; 
            for (int j = cluster.bgn_row_node_id; j <= cluster.end_row_node_id; ++j)
            {
                int node_id = row_nodes[j]; 
                if (node_id < num_movable_nodes && node_size_y[node_id] <= row_height)
                {
                    x[node_id] = xc; 
                }
                else if (xc != x[node_id])
                {
                    if (node_id < num_movable_nodes)
                        dreamplacePrint(kWARN, "multi-row node %d tends to move from %.12f to %.12f, ignored\n", node_id, x[node_id], xc);
                    else
                        dreamplacePrint(kWARN, "fixed node %d tends to move from %.12f to %.12f, ignored\n", node_id, x[node_id], xc);
                    ret_flag = false; 
                }
                xc += node_size_x[node_id]; 
            }
        }
    }

    return ret_flag; 
}

template <typename T>
void abacusLegalizeRowCPU(
        const T* init_x, 
        const T* node_size_x, const T* node_size_y, 
        T* x, 
        const T xl, const T xh, 
        const T bin_size_x, const T bin_size_y, 
        const int num_bins_x, const int num_bins_y, 
        const int num_nodes, 
        const int num_movable_nodes, 
        const int num_filler_nodes, 
        std::vector<std::vector<int> >& bin_cells, 
        std::vector<std::vector<AbacusCluster<T> > >& bin_clusters
        )
{
    for (unsigned int i = 0; i < bin_cells.size(); i += 1)
    {
        int* cells = &(bin_cells.at(i)[0]);
        AbacusCluster<T>* clusters = &(bin_clusters.at(i)[0]);
        int num_row_nodes = bin_cells.at(i).size();

        int bin_id_x = i/num_bins_y; 
        //int bin_id_y = i-bin_id_x*num_bins_y; 

        T bin_xl = xl+bin_size_x*bin_id_x;
        T bin_xh = std::min(bin_xl+bin_size_x, xh);

        abacusPlaceRowCPU(
                init_x, 
                node_size_x, node_size_y, 
                x, 
                bin_size_y, // must be equal to row_height
                bin_xl, bin_xh, 
                num_nodes, 
                num_movable_nodes, 
                num_filler_nodes, 
                cells, 
                clusters, 
                num_row_nodes
                );
    }
    T displace = 0; 
    for (int i = 0; i < num_movable_nodes; ++i)
    {
        displace += fabs(x[i]-init_x[i]); 
    }
    dreamplacePrint(kDEBUG, "average displace = %g\n", displace/num_movable_nodes);
}

DREAMPLACE_END_NAMESPACE

#endif
