/**
 * @file   abacus_cluster.h
 * @author Yibo Lin
 * @date   Oct 2018
 */
#ifndef GPUPLACE_LEGALIZE_CLUSTER_H
#define GPUPLACE_LEGALIZE_CLUSTER_H

#include <limits.h>
#include "utility/src/Msg.h"

DREAMPLACE_BEGIN_NAMESPACE

/// A cluster recording abutting cells 
/// behave liked a linked list but allocated on a continuous memory
template <typename T>
struct AbacusCluster
{
    int prev_cluster_id; ///< previous cluster, set to INT_MIN if the cluster is invalid  
    int next_cluster_id; ///< next cluster, set to INT_MIN if the cluster is invalid 
    int bgn_row_node_id; ///< id of first node in the row 
    int end_row_node_id; ///< id of last node in the row 
    T e; ///< weight of displacement in the objective
    T q; ///< x = q/e 
    T w; ///< width 
    T x; ///< optimal location 

    /// @return whether this is a valid cluster 
    bool valid() const 
    {
        return prev_cluster_id != INT_MIN && next_cluster_id != INT_MIN;
    }
};

DREAMPLACE_END_NAMESPACE

#endif
