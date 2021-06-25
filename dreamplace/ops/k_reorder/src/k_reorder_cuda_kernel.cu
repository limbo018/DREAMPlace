/**
 * @file   k_reorder_cuda_kernel.cu
 * @author Yibo Lin
 * @date   Jan 2019
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
//#include <thrust/device_vector.h>
//#include <thrust/functional.h>
//#include <thrust/host_vector.h>
//#include <thrust/reduce.h>
//#include <thrust/swap.h>
#include <time.h>
#include <chrono>
#include <cmath>
#include <random>

//#define DEBUG
//#define DYNAMIC
//#define TIMER
#define DETERMINISTIC

#include "utility/src/utils.cuh"
// database dependency
#include "utility/src/detailed_place_db.cuh"
// local dependency
#include "k_reorder/src/compute_independent_rows.h"
#include "k_reorder/src/compute_reorder_instances.h"
#include "k_reorder/src/quick_perm.h"
#include "k_reorder/src/row2node_map.h"

DREAMPLACE_BEGIN_NAMESPACE

// maximum number of cells for reordering
#define MAX_K 4
// maximum number of nets per cell to be considered
#define MAX_NUM_NETS_PER_NODE 20
// maximum number of nets incident to cells per instance
#define MAX_NUM_NETS_PER_INSTANCE (MAX_NUM_NETS_PER_NODE * MAX_K)

/// Concepts in the algorith:
/// A group contains independent rows.
/// An KReorderInstance contains an adjacent sequence of cells to be solved.
///

/// a net for a reorder instance
template <typename T>
struct InstanceNet {
  int net_id;
  int node_marker;  ///< mark cells in one instance using bit
  T bxl;
  T bxh;
  T pin_offset_x[MAX_K];
};

template <typename T>
struct KReorderState {
  PitchNestedVector<int> row2node_map;
  int* permutations;  ///< num_permutations x K
  int num_permutations;

  T* node_space_x;  ///< cell size with spaces, a cell only considers its right
                    ///< space

  PitchNestedVector<KReorderInstance> reorder_instances;  ///< array of array
                                                          ///< for independent
                                                          ///< instances; each
                                                          ///< instance is a
                                                          ///< sequence of at
                                                          ///< most K cells to
                                                          ///< be solved.
  T* costs;              ///< maximum reorder_instances.size2 * num_permutations
  int* best_permute_id;  ///< maximum reorder_instances.size2
  InstanceNet<T>*
      instance_nets;  ///< reorder_instances.size2 * MAX_NUM_NETS_PER_INSTANCE
  int* instance_nets_size;      ///< reorder_instances.size2, number of nets for
                                ///< each instance
  int* node2inst_map;           ///< map cell to instance
  int* net_markers;             ///< whether a net is in this group
  unsigned char* node_markers;  ///< cell offset in instance

  int* device_num_moved;
  int K;  ///< number of cells to reorder

  double* net_hpwls;  ///< used for compute HPWL
};

template <typename DetailedPlaceDBType, typename StateType>
inline __device__ void compute_position(
    const DetailedPlaceDBType& db, const StateType& state,
    const KReorderInstance& inst, int permute_id,
    typename DetailedPlaceDBType::type target_x[],
    typename DetailedPlaceDBType::type target_sizes[]) {
  auto row2nodes = state.row2node_map(inst.row_id) + inst.idx_bgn;
  auto permutation = state.permutations + permute_id * state.K;
  int K = inst.idx_end - inst.idx_bgn;
#ifdef DEBUG
  if (!(inst.idx_end - inst.idx_bgn <= state.row2node_map.size(inst.row_id))) {
    printf("idx_bgn %d, idx_end %d, row size %u\n", inst.idx_bgn, inst.idx_end,
           state.row2node_map.size(inst.row_id));
  }
  assert(inst.idx_end - inst.idx_bgn <= state.row2node_map.size(inst.row_id));
  assert(K <= MAX_K);
#endif

  // find left boundary
  if (K) {
    int node_id = row2nodes[0];
    target_x[0] = db.x[node_id];
  }
  // record sizes, and pack to left
  for (int i = 0; i < K; ++i) {
    int node_id = row2nodes[i];
    assert(node_id < db.num_movable_nodes);
    target_sizes[permutation[i]] = state.node_space_x[node_id];
  }
  for (int i = 1; i < K; ++i) {
    target_x[i] = target_x[i - 1] + target_sizes[i - 1];
  }
}

template <typename DetailedPlaceDBType, typename StateType>
inline __device__ typename DetailedPlaceDBType::type compute_instance_hpwl(
    const DetailedPlaceDBType& db, const StateType& state,
    const KReorderInstance& inst, int permute_id,
    const int* __restrict__ row2nodes, int K,
    typename DetailedPlaceDBType::type target_x[],
    typename DetailedPlaceDBType::type target_sizes[]) {
  typedef typename DetailedPlaceDBType::type T;
  auto check_node_exist = [&](int bgn, int end, int node_id) {
    for (int i = bgn; i < end; ++i) {
      if (row2nodes[i] == node_id) {
        return i;
      }
    }
    return cuda::numeric_limits<int>::max();
  };

  T row_yl = db.yl + inst.row_id * db.row_height;
  auto permutation = state.permutations + permute_id * state.K;
  T cost = 0;
  for (int i = 0; i < K; ++i) {
    int node_id = row2nodes[i];
    int node2pin_id = db.flat_node2pin_start_map[node_id];
    const int node2pin_id_end = db.flat_node2pin_start_map[node_id + 1];
    for (; node2pin_id < node2pin_id_end; ++node2pin_id) {
      int node_pin_id = db.flat_node2pin_map[node2pin_id];
      int net_id = db.pin2net_map[node_pin_id];
      T bxl = db.xh;
      T bxh = db.xl;
      int flag = db.net_mask[net_id];
      if (flag) {
        int net2pin_id = db.flat_net2pin_start_map[net_id];
        const int net2pin_id_end = db.flat_net2pin_start_map[net_id + 1] * flag;
        for (; net2pin_id < net2pin_id_end; ++net2pin_id) {
          int net_pin_id = db.flat_net2pin_map[net2pin_id];
          int other_node_id = db.pin2node_map[net_pin_id];
          // if other node found in previous
          // this net has already been computed
          int other_node_found = check_node_exist(0, K, other_node_id);
          if (other_node_found < i) {
            break;
          }
          T other_node_xl;
          if (other_node_found < K) {
            int permuted_offset = permutation[other_node_found];
            other_node_xl = target_x[permuted_offset];
          } else  // not found
          {
            other_node_xl = db.x[other_node_id];
            if (db.y[other_node_id] == row_yl)  // in the same row
            {
              if (other_node_xl < target_x[0])  // left of the segment
              {
                other_node_xl = db.xl;
              } else if (other_node_xl >
                         target_x[K - 1])  // right of the segment
              {
                other_node_xl = db.xh;
              }
            }
          }
          other_node_xl += db.pin_offset_x[net_pin_id];
          bxl = min(bxl, other_node_xl);
          bxh = max(bxh, other_node_xl);
        }
        cost += bxh - bxl;
      }
    }
  }
  return cost;
}

template <typename DetailedPlaceDBType, typename StateType>
__global__ void compute_instance_net_boxes(DetailedPlaceDBType db,
                                           StateType state, int group_id,
                                           int offset) {
  typedef typename DetailedPlaceDBType::type T;
  __shared__ int group_size;
  if (threadIdx.x == 0) {
    group_size = state.reorder_instances.size(group_id);
  }
  __syncthreads();

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < group_size;
       i += blockDim.x * gridDim.x) {
    int inst_id = i;
    // this is a copy
    auto inst = state.reorder_instances(group_id, inst_id);
    inst.idx_bgn += offset;
    inst.idx_end =
        min(inst.idx_end + offset, state.row2node_map.size(inst.row_id));
    auto row2nodes = state.row2node_map(inst.row_id) + inst.idx_bgn;
    int K = inst.idx_end - inst.idx_bgn;

    // after adding offset
    for (int idx = 0; idx < K; ++idx) {
      int node_id = row2nodes[idx];
      if (node_id >= db.num_movable_nodes ||
          db.node_size_y[node_id] > db.row_height) {
        inst.idx_end = inst.idx_bgn + idx;
        K = idx;
        break;
      }
    }

    if (K > 0) {
      T segment_xl = db.x[row2nodes[0]];
      T segment_xh = db.x[row2nodes[K - 1]];
      T row_yl = db.yl + inst.row_id * db.row_height;
      auto instance_nets =
          state.instance_nets + inst_id * MAX_NUM_NETS_PER_INSTANCE;
      auto instance_nets_size = state.instance_nets_size[inst_id];
      for (int idx = 0; idx < instance_nets_size; ++idx) {
        auto& instance_net = instance_nets[idx];
        instance_net.bxl = db.xh;
        instance_net.bxh = db.xl;

        int net2pin_id = db.flat_net2pin_start_map[instance_net.net_id];
        const int net2pin_id_end =
            db.flat_net2pin_start_map[instance_net.net_id + 1];
        for (; net2pin_id < net2pin_id_end; ++net2pin_id) {
          int net_pin_id = db.flat_net2pin_map[net2pin_id];
          int other_node_id = db.pin2node_map[net_pin_id];
          if (other_node_id < db.num_nodes)  // other_node_id may exceed
                                             // db.num_nodes like IO pins
          {
            int other_node_found =
                (state.node2inst_map[other_node_id] == inst_id);
            if (!other_node_found)  // not found
            {
              T other_node_xl = db.x[other_node_id];
              auto pin_offset_x = db.pin_offset_x[net_pin_id];
              if (abs(db.y[other_node_id] - row_yl) <
                  db.row_height)  // in the same row
              {
                if (other_node_xl < segment_xl)  // left of the segment
                {
                  other_node_xl = db.xl;
                } else if (other_node_xl > segment_xh)  // right of the segment
                {
                  other_node_xl = db.xh;
                }
              }
              other_node_xl += pin_offset_x;
              instance_net.bxl = min(instance_net.bxl, other_node_xl);
              instance_net.bxh = max(instance_net.bxh, other_node_xl);
            }
          }
        }
      }
    }
  }
}

template <typename DetailedPlaceDBType, typename StateType>
__global__ void compute_reorder_hpwl(DetailedPlaceDBType db, StateType state,
                                     int group_id, int offset) {
  typedef typename DetailedPlaceDBType::type T;
  __shared__ int group_size;
  __shared__ int group_size_with_permutation;
  if (threadIdx.x == 0) {
    group_size = state.reorder_instances.size(group_id);
    group_size_with_permutation = group_size * state.num_permutations;
  }
  __syncthreads();

  typename DetailedPlaceDBType::type target_x[MAX_K];
  typename DetailedPlaceDBType::type target_sizes[MAX_K];
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < group_size_with_permutation; i += blockDim.x * gridDim.x) {
    int inst_id = i / state.num_permutations;
    int permute_id = i - inst_id * state.num_permutations;
    // this is a copy
    auto inst = state.reorder_instances(group_id, inst_id);
    inst.idx_bgn += offset;
    inst.idx_end =
        min(inst.idx_end + offset, state.row2node_map.size(inst.row_id));
    auto row2nodes = state.row2node_map(inst.row_id) + inst.idx_bgn;
    auto permutation = state.permutations + permute_id * state.K;
    int K = inst.idx_end - inst.idx_bgn;

    // after adding offset
    for (int idx = 0; idx < K; ++idx) {
      int node_id = row2nodes[idx];
      if (node_id >= db.num_movable_nodes ||
          db.node_size_y[node_id] > db.row_height) {
        inst.idx_end = inst.idx_bgn + idx;
        K = idx;
        break;
      }
    }

    int valid_flag = (K > 0);
    for (int idx = 0; idx < K; ++idx) {
      if (permutation[idx] >= K) {
        valid_flag = 0;
        break;
      }
    }

    if (valid_flag) {
      compute_position(db, state, inst, permute_id, target_x, target_sizes);

      // state.costs[i] = compute_instance_hpwl(
      //        db,
      //        state,
      //        inst,
      //        permute_id,
      //        row2nodes,
      //        K,
      //        target_x,
      //        target_sizes
      //        );

      T cost = 0;
      // consider FENCE region
      if (db.num_regions) {
        for (int idx = 0; idx < K; ++idx) {
          int node_id = row2nodes[idx];
          int permuted_offset = permutation[idx];
          T node_xl = target_x[permuted_offset];
          T node_yl = db.y[node_id];
          if (!db.inside_fence(node_id, node_xl, node_yl)) {
            cost = cuda::numeric_limits<T>::max();
            break;
          }
        }
      }
      if (cost == 0) {
        auto instance_nets =
            state.instance_nets + inst_id * MAX_NUM_NETS_PER_INSTANCE;
        auto const& instance_nets_size = state.instance_nets_size[inst_id];
        for (int idx = 0; idx < instance_nets_size; ++idx) {
          auto& instance_net = instance_nets[idx];
          T bxl = instance_net.bxl;
          T bxh = instance_net.bxh;

          for (int j = 0; j < K; ++j) {
            int flag = (1 << j);
            if ((instance_net.node_marker & flag)) {
              int permuted_offset = permutation[j];
              T other_node_xl = target_x[permuted_offset];
              other_node_xl += instance_net.pin_offset_x[j];
              bxl = min(bxl, other_node_xl);
              bxh = max(bxh, other_node_xl);
            }
          }
          cost += bxh - bxl;
        }
      }
      state.costs[i] = cost;
    }
  }
}

template <typename T>
struct ItemWithIndex {
  T value;
  int index;
};

template <typename T>
struct ReduceMinOP {
  __host__ __device__ ItemWithIndex<T> operator()(
      const ItemWithIndex<T>& a, const ItemWithIndex<T>& b) const {
    return (a.value < b.value) ? a : b;
  }
};

template <typename T, int ThreadsPerBlock = 32>
__global__ void reduce_min_2d_cub(const T* __restrict__ costs,
                                  int* best_permute_id, int m, int n) {
  typedef cub::BlockReduce<ItemWithIndex<T>, ThreadsPerBlock> BlockReduce;

  __shared__ typename BlockReduce::TempStorage temp_storage;

  auto inst_costs = costs + blockIdx.x * n;
  auto inst_best_permute_id = best_permute_id + blockIdx.x;

  ItemWithIndex<T> thread_data;

  thread_data.value = cuda::numeric_limits<T>::max();
  thread_data.index = 0;
  for (int col = threadIdx.x; col < n; col += ThreadsPerBlock) {
    T cost = inst_costs[col];
    if (cost < thread_data.value) {
      thread_data.value = cost;
      thread_data.index = col;
    }
  }

  __syncthreads();

  // Compute the block-wide max for thread0
  ItemWithIndex<T> aggregate =
      BlockReduce(temp_storage).Reduce(thread_data, ReduceMinOP<T>(), n);

  __syncthreads();

  if (threadIdx.x == 0) {
    // printf("inst[%d] cost %g, permute_id %d\n", blockIdx.x, aggregate.value,
    // aggregate.index);
    *inst_best_permute_id = aggregate.index;
  }
}

template <typename DetailedPlaceDBType, typename StateType>
__global__ void apply_reorder(DetailedPlaceDBType db, StateType state,
                              int group_id, int offset) {
  __shared__ int group_size;
  if (threadIdx.x == 0) {
    group_size = state.reorder_instances.size(group_id);
  }
  __syncthreads();

  typename DetailedPlaceDBType::type target_x[MAX_K];
  typename DetailedPlaceDBType::type target_sizes[MAX_K];
  int target_nodes[MAX_K];

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < group_size;
       i += blockDim.x * gridDim.x) {
    int inst_id = i;
    int permute_id = state.best_permute_id[i];
#ifdef DEBUG
    // printf("inst[%d].permute_id = %d\n", inst_id, permute_id);
    assert(permute_id < state.num_permutations);
#endif
    // this is a copy for adding offset
    auto inst = state.reorder_instances(group_id, inst_id);
    inst.idx_bgn += offset;
    inst.idx_end =
        min(inst.idx_end + offset, state.row2node_map.size(inst.row_id));
    auto row2nodes = state.row2node_map(inst.row_id) + inst.idx_bgn;
    auto permutation = state.permutations + permute_id * state.K;
    int K = inst.idx_end - inst.idx_bgn;

    // after adding offset
    for (int idx = 0; idx < K; ++idx) {
      int node_id = row2nodes[idx];
      if (node_id >= db.num_movable_nodes ||
          db.node_size_y[node_id] > db.row_height) {
        inst.idx_end = inst.idx_bgn + idx;
        K = idx;
        break;
      }
    }

    if (K > 0) {
      compute_position(db, state, inst, permute_id, target_x, target_sizes);

      for (int i = 0; i < K; ++i) {
        int node_id = row2nodes[i];
        target_nodes[i] = node_id;
      }

      for (int i = 0; i < K; ++i) {
        int node_id = row2nodes[i];
        typename DetailedPlaceDBType::type xx = target_x[permutation[i]];
        if (db.x[node_id] != xx) {
          atomicAdd(state.device_num_moved, 1);
        }
        db.x[node_id] = xx;
      }

      for (int i = 0; i < K; ++i) {
        row2nodes[permutation[i]] = target_nodes[i];
      }
    }
  }
}

/// @brief Map each node to its instance.
/// For each instance in the group
///     For each node incident to the instance
///         update node2inst_map
///         update node_markers
/// Every time, we solve one group with all independent instances in the group.
/// For sliding window, offset can be different during iterations,
/// so node2inst_map and node_markers need to be recomputed.
template <typename T>
__global__ void compute_node2inst_map(DetailedPlaceDB<T> db,
                                      KReorderState<T> state, int group_id,
                                      int offset) {
  __shared__ int group_size;
  if (threadIdx.x == 0) {
    group_size = state.reorder_instances.size(group_id);
  }
  __syncthreads();

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < group_size;
       i += blockDim.x * gridDim.x) {
    int inst_id = i;
    // this is a copy
    auto inst = state.reorder_instances(group_id, inst_id);
    inst.idx_bgn += offset;
    inst.idx_end =
        min(inst.idx_end + offset, state.row2node_map.size(inst.row_id));
    auto row2nodes = state.row2node_map(inst.row_id) + inst.idx_bgn;
    int K = inst.idx_end - inst.idx_bgn;

    for (int j = 0; j < K; ++j) {
      int node_id = row2nodes[j];
      // do not update for fixed cells
      if (node_id < db.num_movable_nodes) {
        state.node2inst_map[node_id] = inst_id;
        state.node_markers[node_id] = j;
      }
    }
  }
}

/// @brief Mark target nets for all instances in this group.
template <typename T>
__global__ void compute_net_markers(DetailedPlaceDB<T> db,
                                    KReorderState<T> state) {
  for (int node_id = blockIdx.x * blockDim.x + threadIdx.x;
       node_id < db.num_movable_nodes; node_id += blockDim.x * gridDim.x) {
    if (state.node2inst_map[node_id] < cuda::numeric_limits<int>::max()) {
      int node2pin_id = db.flat_node2pin_start_map[node_id];
      const int node2pin_id_end = db.flat_node2pin_start_map[node_id + 1];
      for (; node2pin_id < node2pin_id_end; ++node2pin_id) {
        int node_pin_id = db.flat_node2pin_map[node2pin_id];
        int net_id = db.pin2net_map[node_pin_id];
        int flag = db.net_mask[net_id];
        atomicOr(state.net_markers + net_id, flag);
      }
    }
  }
}

template <typename T>
__global__ void print_net_markers(DetailedPlaceDB<T> db,
                                  KReorderState<T> state) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    for (int i = 0; i < db.num_nets; ++i) {
      printf("net_markers[%d] = %d\n", i, state.net_markers[i]);
    }
  }
}

#ifdef DETERMINISTIC
/// @brief Collect information of nets belong to each instance.
/// The net order is deterministic.
template <typename T>
__global__ void compute_instance_nets(DetailedPlaceDB<T> db,
                                      KReorderState<T> state, int group_id,
                                      int offset) {
  __shared__ int group_size;
  if (threadIdx.x == 0) {
    group_size = state.reorder_instances.size(group_id);
  }
  __syncthreads();

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < group_size;
       i += blockDim.x * gridDim.x) {
    int inst_id = i;
    // this is a copy
    auto inst = state.reorder_instances(group_id, inst_id);
    inst.idx_bgn += offset;
    inst.idx_end =
        min(inst.idx_end + offset, state.row2node_map.size(inst.row_id));
    const auto row2nodes = state.row2node_map(inst.row_id) + inst.idx_bgn;
    int K = inst.idx_end - inst.idx_bgn;
    auto& instance_nets_size = state.instance_nets_size[inst_id];
    auto instance_nets =
        state.instance_nets + inst_id * MAX_NUM_NETS_PER_INSTANCE;
    instance_nets_size = 0;

    // after adding offset
    for (int idx = 0; idx < K; ++idx) {
      int node_id = row2nodes[idx];
      if (node_id >= db.num_movable_nodes ||
          db.node_size_y[node_id] > db.row_height) {
        inst.idx_end = inst.idx_bgn + idx;
        K = idx;
        break;
      }
    }

    for (int j = 0; j < K; ++j) {
      int node_id = row2nodes[j];
      int node2pin_id = db.flat_node2pin_start_map[node_id];
      int node2pin_id_end = db.flat_node2pin_start_map[node_id + 1];
      for (; node2pin_id < node2pin_id_end; ++node2pin_id) {
        int node_pin_id = db.flat_node2pin_map[node2pin_id];
        int net_id = db.pin2net_map[node_pin_id];
        if (state.net_markers[net_id]) {
          if (instance_nets_size < MAX_NUM_NETS_PER_INSTANCE) {
            auto& instance_net = instance_nets[instance_nets_size];

            instance_net.net_id = net_id;
            instance_net.node_marker = (1 << state.node_markers[node_id]);
            instance_net.pin_offset_x[state.node_markers[node_id]] =
                db.pin_offset_x[node_pin_id];
            instance_nets_size += 1;
          }
        }
      }
    }
  }
}
#else
/// @brief Collect information of nets belong to each instance.
/// The net orders may not be deterministic.
template <typename T>
__global__ void compute_instance_nets(DetailedPlaceDB<T> db,
                                      KReorderState<T> state) {
  for (int net_id = blockIdx.x * blockDim.x + threadIdx.x; net_id < db.num_nets;
       net_id += blockDim.x * gridDim.x) {
    if (state.net_markers[net_id]) {
      int net2pin_id = db.flat_net2pin_start_map[net_id];
      const int net2pin_id_end = db.flat_net2pin_start_map[net_id + 1];
      for (; net2pin_id < net2pin_id_end; ++net2pin_id) {
        int net_pin_id = db.flat_net2pin_map[net2pin_id];
        int other_node_id = db.pin2node_map[net_pin_id];

        if (other_node_id <
            db.num_nodes)  // other_node_id may exceed db.num_nodes like IO pins
        {
          int inst_id = state.node2inst_map[other_node_id];
          if (inst_id < cuda::numeric_limits<int>::max()) {
            auto instance_nets_size = state.instance_nets_size + inst_id;

            int index = atomicAdd(instance_nets_size, 1);
            if (index < MAX_NUM_NETS_PER_INSTANCE) {
              auto& instance_net =
                  state.instance_nets[inst_id * MAX_NUM_NETS_PER_INSTANCE +
                                      index];

              instance_net.net_id = net_id;
              instance_net.node_marker =
                  (1 << state.node_markers[other_node_id]);
              instance_net.pin_offset_x[state.node_markers[other_node_id]] =
                  db.pin_offset_x[net_pin_id];
            }
            // I do not expect the else condition to happen
          }
        }
      }
    }
  }
}
#endif

/// @brief Remove duplicate nets in an instance.
template <typename T>
__global__ void unique_instance_nets(DetailedPlaceDB<T> db,
                                     KReorderState<T> state, int group_id) {
  __shared__ int group_size;
  if (threadIdx.x == 0) {
    group_size = state.reorder_instances.size(group_id);
  }
  __syncthreads();

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < group_size;
       i += blockDim.x * gridDim.x) {
    int inst_id = i;
    auto inst = state.reorder_instances(group_id, inst_id);
    auto instance_nets =
        state.instance_nets + inst_id * MAX_NUM_NETS_PER_INSTANCE;
    auto& instance_nets_size = state.instance_nets_size[inst_id];

    for (int j = 0; j < instance_nets_size; ++j) {
      for (int k = j + 1; k < instance_nets_size;) {
        if (instance_nets[j].net_id == instance_nets[k].net_id) {
          // copy marker and pin offset
          instance_nets[j].node_marker |= instance_nets[k].node_marker;
          for (int l = 0; l < state.K; ++l) {
            if ((instance_nets[k].node_marker & (1 << l))) {
              instance_nets[j].pin_offset_x[l] =
                  instance_nets[k].pin_offset_x[l];
            }
          }
          --instance_nets_size;
          host_device_swap(instance_nets[k], instance_nets[instance_nets_size]);
        } else {
          ++k;
        }
      }
    }
  }
}

template <typename StateType>
__global__ void print_costs(StateType state, int group_id, int offset) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    printf("group_id %d, offset %d, %s\n", group_id, offset, __func__);
    for (int i = 0; i < state.reorder_instances.size(group_id); ++i) {
      printf("inst[%d][%d] costs: ", i, state.num_permutations);
      for (int j = 0; j < state.num_permutations; ++j) {
        printf("%g ", state.costs[i * state.num_permutations + j]);
      }
      printf("\n");
    }
  }
}

template <typename StateType>
__global__ void print_best_permute_id(StateType state, int group_id,
                                      int offset) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    printf("group_id %d, offset %d, %s\n", group_id, offset, __func__);
    for (int i = 0; i < state.reorder_instances.size(group_id); ++i) {
      printf("[%d] = %d\n", i, state.best_permute_id[i]);
    }
  }
}

template <typename StateType>
__global__ void print_instance_nets(StateType state, int group_id, int offset) {
  assert(blockDim.x == 1 && gridDim.x == 1);
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    printf("group_id %d, offset %d, %s\n", group_id, offset, __func__);
    int size = state.reorder_instances.size(group_id);
    assert(size >= 0 && size < state.reorder_instances.size2);
    for (int i = 0; i < size; ++i) {
      int instance_nets_size = state.instance_nets_size[i];
      printf("inst[%d][%d] nets: ", i, instance_nets_size);
      assert(instance_nets_size >= 0 &&
             instance_nets_size < MAX_NUM_NETS_PER_INSTANCE);
      for (int j = 0; j < instance_nets_size; ++j) {
        int index = i * MAX_NUM_NETS_PER_INSTANCE + j;
        assert(index >= 0 && index < state.reorder_instances.size2 *
                                         MAX_NUM_NETS_PER_INSTANCE);
        printf("%d (%d) ", state.instance_nets[index].net_id,
               state.instance_nets[index].node_marker);
      }
      printf("\n");
    }
  }
}

template <typename StateType>
__global__ void print_instance_net_bboxes(StateType state, int group_id,
                                          int offset) {
  assert(blockDim.x == 1 && gridDim.x == 1);
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    printf("group_id %d, offset %d, %s\n", group_id, offset, __func__);
    int size = state.reorder_instances.size(group_id);
    assert(size >= 0 && size < state.reorder_instances.size2);
    for (int i = 0; i < size; ++i) {
      int instance_nets_size = state.instance_nets_size[i];
      printf("inst[%d][%d] nets: ", i, instance_nets_size);
      assert(instance_nets_size >= 0 &&
             instance_nets_size < MAX_NUM_NETS_PER_INSTANCE);
      for (int j = 0; j < instance_nets_size; ++j) {
        int index = i * MAX_NUM_NETS_PER_INSTANCE + j;
        assert(index >= 0 && index < state.reorder_instances.size2 *
                                         MAX_NUM_NETS_PER_INSTANCE);
        printf("%d/%d:%g/%g ", index, j, state.instance_nets[index].bxl,
               state.instance_nets[index].bxh);
      }
      printf("\n");
    }
  }
}

template <typename DetailedPlaceDBType, typename StateType>
__global__ void check_instance_nets(DetailedPlaceDBType db, StateType state,
                                    int group_id) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    for (int i = 0; i < state.reorder_instances.size(group_id); ++i) {
      auto const& inst = state.reorder_instances(group_id, i);
      auto row2nodes = state.row2node_map(inst.row_id) + inst.idx_bgn;
      int K = inst.idx_end - inst.idx_bgn;
      for (int j = 0; j < K; ++j) {
        int node_id = row2nodes[j];
        int node2pin_id = db.flat_node2pin_start_map[node_id];
        const int node2pin_id_end = db.flat_node2pin_start_map[node_id + 1];
        for (; node2pin_id < node2pin_id_end; ++node2pin_id) {
          int node_pin_id = db.flat_node2pin_map[node2pin_id];
          int net_id = db.pin2net_map[node_pin_id];

          if (db.net_mask[net_id]) {
            bool found = false;
            for (int k = 0; k < state.instance_nets_size[i]; ++k) {
              auto const& instance_net =
                  state.instance_nets[i * MAX_NUM_NETS_PER_INSTANCE + k];
              if (instance_net.net_id == net_id) {
                found = true;
                assert((instance_net.node_marker & (1 << j)));
                assert(instance_net.pin_offset_x[j] ==
                       db.pin_offset_x[node_pin_id]);
                break;
              }
            }
            assert(found);
          }
        }
      }
    }
  }
}

template <typename DetailedPlaceDBType>
__global__ void print_pos(DetailedPlaceDBType db, int group_id, int offset) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    printf("group_id %d, offset %d, pos[%d]\n", group_id, offset,
           db.num_movable_nodes);
    for (int i = 0; i < db.num_movable_nodes; ++i) {
      printf("[%d] = %g, %g\n", i, db.x[i], db.y[i]);
    }
  }
}

template <typename DetailedPlaceDBType, typename StateType>
__global__ void reset_state(DetailedPlaceDBType db, StateType state,
                            int group_id) {
  typedef typename DetailedPlaceDBType::type T;
  __shared__ int group_size;
  __shared__ int group_size_with_permutation;
  if (threadIdx.x == 0) {
    group_size = state.reorder_instances.size(group_id);
    group_size_with_permutation = group_size * state.num_permutations;
  }
  __syncthreads();

  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < group_size_with_permutation; i += blockDim.x * gridDim.x) {
    state.costs[i] = cuda::numeric_limits<T>::max();
  }
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < state.reorder_instances.size2; i += blockDim.x * gridDim.x) {
    state.instance_nets_size[i] = 0;
  }
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < db.num_nodes;
       i += blockDim.x * gridDim.x) {
    state.node_markers[i] = 0;
    state.node2inst_map[i] = cuda::numeric_limits<int>::max();
  }
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < db.num_nets;
       i += blockDim.x * gridDim.x) {
    state.net_markers[i] = 0;
  }
  int instance_nets_size =
      state.reorder_instances.size2 * MAX_NUM_NETS_PER_INSTANCE;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < instance_nets_size;
       i += blockDim.x * gridDim.x) {
    auto& instance_net = state.instance_nets[i];
    instance_net.net_id = cuda::numeric_limits<int>::max();
    instance_net.node_marker = 0;
    instance_net.bxl = db.xh;
    instance_net.bxh = db.xl;
    for (int j = 0; j < MAX_K; ++j) {
      instance_net.pin_offset_x[j] = 0;
    }
  }
}

#ifdef DYNAMIC
#define TIMER CUDATimer
#else
#define TIMER CPUTimer
#endif

template <typename T>
#ifdef DYNAMIC
__global__ void k_reorder(DetailedPlaceDB<T> db, KReorderState<T> state)
#else
void k_reorder(
    DetailedPlaceDB<T>& db, KReorderState<T>& state,
    const std::vector<std::vector<KReorderInstance>>& host_reorder_instances)
#endif
{
#ifdef TIMER
  TIMER::hr_clock_rep timer_start, timer_stop;
  TIMER::hr_clock_rep enumeration_time = 0, apply_reorder_time = 0;
  int enumeration_runs = 0, apply_reorder_runs = 0;
#endif
  for (int group_id = 0; group_id < state.reorder_instances.size1; ++group_id) {
#ifdef DYNAMIC
    int group_size = state.reorder_instances.size(group_id);
#else
    dreamplaceAssert(state.reorder_instances.size1 ==
                     host_reorder_instances.size());
    int group_size = host_reorder_instances[group_id].size();
#endif
    if (group_size) {
      for (int offset = 0; offset < state.K; offset += state.K / 2) {
#ifdef TIMER
        timer_start = TIMER::getGlobaltime();
#endif
        reset_state<<<64, 512>>>(db, state, group_id);
        compute_node2inst_map<<<ceilDiv(group_size, 256), 256>>>(
            db, state, group_id, offset);
        compute_net_markers<<<ceilDiv(db.num_movable_nodes, 256), 256>>>(db,
                                                                         state);
        // print_net_markers<<<1, 1>>>(db, state);
#ifdef DETERMINISTIC
        compute_instance_nets<<<ceilDiv(group_size, 256), 256>>>(
            db, state, group_id, offset);
#else
        compute_instance_nets<<<ceilDiv(db.num_nets, 256), 256>>>(db, state);
#endif
        // print_instance_nets<<<1, 1>>>(state, group_id);
        unique_instance_nets<<<ceilDiv(group_size, 256), 256>>>(db, state,
                                                                group_id);
        // print_instance_nets<<<1, 1>>>(state, group_id, offset);
        // check_instance_nets<<<1, 1>>>(db, state, group_id);
        compute_instance_net_boxes<<<ceilDiv(group_size, 256), 256>>>(
            db, state, group_id, offset);
        // print_instance_net_bboxes<<<1, 1>>>(state, group_id, offset);
        compute_reorder_hpwl<<<ceilDiv(group_size, 256), 256>>>(
            db, state, group_id, offset);
#ifdef TIMER
        checkCUDA(cudaDeviceSynchronize());
        timer_stop = TIMER::getGlobaltime();
        enumeration_time += timer_stop - timer_start;
        enumeration_runs += 1;
#endif

#ifdef TIMER
        timer_start = TIMER::getGlobaltime();
#endif
        // print_costs<<<1, 1>>>(state, group_id, offset);
        reduce_min_2d_cub<T, 32>
            <<<group_size, 32>>>(state.costs, state.best_permute_id, group_size,
                                 state.num_permutations);
        // print_best_permute_id<<<1, 1>>>(state, group_id, offset);
        apply_reorder<<<ceilDiv(group_size, 256), 256>>>(db, state, group_id,
                                                         offset);
#ifdef TIMER
        checkCUDA(cudaDeviceSynchronize());
        timer_stop = TIMER::getGlobaltime();
        apply_reorder_time += timer_stop - timer_start;
        apply_reorder_runs += 1;
#endif
        // print_pos<<<1, 1>>>(db, group_id, offset);
      }
    }
  }
#ifdef TIMER
  dreamplacePrint(
      kDEBUG, "enumeration takes %g ms for %d runs, average %g ms\n",
      TIMER::getTimerPeriod() * enumeration_time, enumeration_runs,
      TIMER::getTimerPeriod() * enumeration_time / enumeration_runs);
  dreamplacePrint(
      kDEBUG, "apply_reorder takes %g ms for %d runs, average %g ms\n",
      TIMER::getTimerPeriod() * apply_reorder_time, apply_reorder_runs,
      TIMER::getTimerPeriod() * apply_reorder_time / apply_reorder_runs);
#endif
}

template <typename T>
int kreorderCUDALauncher(DetailedPlaceDB<T> db, int K, int max_iters,
                         int num_threads) {
  dreamplacePrint(kDEBUG, "%d-reorder\n", K);
  CPUTimer::hr_clock_rep total_time_start, total_time_stop;
  CPUTimer::hr_clock_rep kernel_time_start, kernel_time_stop;
  CPUTimer::hr_clock_rep iter_time_start, iter_time_stop;
  total_time_start = CPUTimer::getGlobaltime();

  const float stop_threshold = 0.1 / 100;

  // fix random seed
  std::srand(1000);

  KReorderState<T> state;
  DetailedPlaceDB<T> cpu_db;
  state.K = K;

  // distribute cells to rows on host
  // copy cell locations from device to host
  std::vector<std::vector<int>> host_row2node_map(db.num_sites_y);
  std::vector<T> host_node_space_x(db.num_movable_nodes);
  std::vector<std::vector<int>> host_permutations = quick_perm(K);
  std::vector<unsigned char> host_adjacency_matrix;
  std::vector<std::vector<int>> host_row_graph;
  std::vector<std::vector<int>> host_independent_rows;
  std::vector<std::vector<KReorderInstance>> host_reorder_instances;
  dreamplacePrint(kDEBUG, "%lu permutations\n", host_permutations.size());

  // initialize cpu db from db
  {
    iter_time_start = CPUTimer::getGlobaltime();
    cpu_db.xl = db.xl;
    cpu_db.yl = db.yl;
    cpu_db.xh = db.xh;
    cpu_db.yh = db.yh;
    cpu_db.site_width = db.site_width;
    cpu_db.row_height = db.row_height;
    cpu_db.bin_size_x = db.bin_size_x;
    cpu_db.bin_size_y = db.bin_size_y;
    cpu_db.num_bins_x = db.num_bins_x;
    cpu_db.num_bins_y = db.num_bins_y;
    cpu_db.num_sites_x = db.num_sites_x;
    cpu_db.num_sites_y = db.num_sites_y;
    cpu_db.num_nodes = db.num_nodes;
    cpu_db.num_movable_nodes = db.num_movable_nodes;
    cpu_db.num_nets = db.num_nets;
    cpu_db.num_pins = db.num_pins;

    allocateCopyCPU(cpu_db.net_mask, db.net_mask, db.num_nets, unsigned char);
    allocateCopyCPU(cpu_db.flat_net2pin_start_map, db.flat_net2pin_start_map,
                    db.num_nets + 1, int);
    allocateCopyCPU(cpu_db.flat_net2pin_map, db.flat_net2pin_map, db.num_pins,
                    int);
    allocateCopyCPU(cpu_db.pin2node_map, db.pin2node_map, db.num_pins, int);
    allocateCopyCPU(cpu_db.x, db.x, db.num_nodes, T);
    allocateCopyCPU(cpu_db.y, db.y, db.num_nodes, T);
    allocateCopyCPU(cpu_db.node_size_x, db.node_size_x, db.num_nodes, T);
    allocateCopyCPU(cpu_db.node_size_y, db.node_size_y, db.num_nodes, T);

    make_row2node_map(cpu_db, cpu_db.x, cpu_db.y, host_row2node_map,
                      num_threads);
    host_node_space_x.resize(cpu_db.num_movable_nodes);
    for (int i = 0; i < cpu_db.num_sites_y; ++i) {
      for (unsigned int j = 0; j < host_row2node_map.at(i).size(); ++j) {
        int node_id = host_row2node_map[i][j];
        if (node_id < db.num_movable_nodes) {
          auto& space = host_node_space_x[node_id];
          T space_xl = cpu_db.x[node_id];
          T space_xh = cpu_db.xh;
          if (j + 1 < host_row2node_map[i].size()) {
            int right_node_id = host_row2node_map[i][j + 1];
            space_xh = min(space_xh, cpu_db.x[right_node_id]);
          }
          space = space_xh - space_xl;
          // align space to sites, as I assume space_xl aligns to sites
          // I also assume node width should be integral numbers of sites
          space = floorDiv(space, db.site_width) * db.site_width;
          T node_size_x = cpu_db.node_size_x[node_id];
          dreamplaceAssertMsg(space >= node_size_x,
                              "space %g, node_size_x[%d] %g, original space "
                              "(%g, %g), site_width %g",
                              space, node_id, node_size_x, space_xl, space_xh,
                              db.site_width);
        }
#ifdef DEBUG
        if (node_id < cpu_db.num_movable_nodes) {
          dreamplaceAssert(space >= cpu_db.node_size_x[node_id]);
        }
#endif
      }
    }
    iter_time_stop = CPUTimer::getGlobaltime();
    dreamplacePrint(
        kDEBUG, "initializing CPU DB takes %g ms\n",
        CPUTimer::getTimerPeriod() * (iter_time_stop - iter_time_start));

    iter_time_start = CPUTimer::getGlobaltime();
    compute_row_conflict_graph(cpu_db, host_row2node_map, host_adjacency_matrix,
                               host_row_graph, num_threads);
    compute_independent_rows(cpu_db, host_row_graph, host_independent_rows);
    iter_time_stop = CPUTimer::getGlobaltime();
    dreamplacePrint(
        kDEBUG, "computing independent rows takes %g ms\n",
        CPUTimer::getTimerPeriod() * (iter_time_stop - iter_time_start));

    iter_time_start = CPUTimer::getGlobaltime();
    compute_reorder_instances(cpu_db, host_row2node_map, host_independent_rows,
                              host_reorder_instances, state.K);
    iter_time_stop = CPUTimer::getGlobaltime();
    dreamplacePrint(
        kDEBUG, "computing reorder instances takes %g ms\n",
        CPUTimer::getTimerPeriod() * (iter_time_stop - iter_time_start));
  }
  // initialize cuda state
  iter_time_start = CPUTimer::getGlobaltime();
  {
    allocateCopyCUDA(state.node_space_x, host_node_space_x.data(),
                     db.num_movable_nodes);

    std::vector<int> host_permutations_flat(host_permutations.size() * K);
    for (unsigned int i = 0; i < host_permutations.size(); ++i) {
      std::copy(host_permutations[i].begin(), host_permutations[i].end(),
                host_permutations_flat.begin() + i * K);
    }
    state.num_permutations = host_permutations.size();
    allocateCopyCUDA(state.permutations, host_permutations_flat.data(),
                     state.num_permutations * state.K);

    state.row2node_map.initialize(host_row2node_map);
    state.reorder_instances.initialize(host_reorder_instances);

    allocateCUDA(state.costs,
                 state.reorder_instances.size2 * state.num_permutations, T);
    allocateCUDA(state.best_permute_id, state.reorder_instances.size2, int);
    allocateCUDA(state.instance_nets,
                 state.reorder_instances.size2 * MAX_NUM_NETS_PER_INSTANCE,
                 InstanceNet<T>);
    allocateCUDA(state.instance_nets_size, state.reorder_instances.size2, int);
    allocateCUDA(state.node2inst_map, db.num_nodes, int);
    allocateCUDA(state.net_markers, db.num_nets, int);
    allocateCUDA(state.node_markers, db.num_nodes, unsigned char);
    allocateCUDA(state.device_num_moved, 1, int);

    allocateCUDA(state.net_hpwls, db.num_nets, long);
  }
  iter_time_stop = CPUTimer::getGlobaltime();
  dreamplacePrint(
      kDEBUG, "initializing CUDA state takes %g ms\n",
      CPUTimer::getTimerPeriod() * (iter_time_stop - iter_time_start));

  kernel_time_start = CPUTimer::getGlobaltime();
  double hpwls[max_iters + 1];
  hpwls[0] = compute_total_hpwl(db, db.x, db.y, state.net_hpwls);
  dreamplacePrint(kINFO, "initial hpwl = %.3f\n", hpwls[0]);
  for (int iter = 0; iter < max_iters; ++iter) {
    iter_time_start = CPUTimer::getGlobaltime();
    k_reorder
#ifdef DYNAMIC
        <<<1, 1>>>(db, state);
#else
        (db, state, host_reorder_instances);
#endif
    checkCUDA(cudaDeviceSynchronize());
    iter_time_stop = CPUTimer::getGlobaltime();
    dreamplacePrint(
        kINFO, "Iteration time(ms) \t %g\n",
        CPUTimer::getTimerPeriod() * (iter_time_stop - iter_time_start));

    hpwls[iter + 1] = compute_total_hpwl(db, db.x, db.y, state.net_hpwls);
    dreamplacePrint(kINFO, "iteration %d: hpwl %.3f => %.3f (imp. %g%%)\n",
                    iter, hpwls[0], hpwls[iter + 1],
                    (1.0 - hpwls[iter + 1] / (double)hpwls[0]) * 100);

    if ((iter & 1) &&
        hpwls[iter] - hpwls[iter - 1] > -stop_threshold * hpwls[0]) {
      break;
    }
  }
  checkCUDA(cudaDeviceSynchronize());
  kernel_time_stop = CPUTimer::getGlobaltime();

  // destroy cuda state
  iter_time_start = CPUTimer::getGlobaltime();
  {
    destroyCUDA(state.node_space_x);
    destroyCUDA(state.permutations);
    state.row2node_map.destroy();
    state.reorder_instances.destroy();
    destroyCUDA(state.costs);
    destroyCUDA(state.best_permute_id);
    destroyCUDA(state.instance_nets);
    destroyCUDA(state.instance_nets_size);
    destroyCUDA(state.node2inst_map);
    destroyCUDA(state.net_markers);
    destroyCUDA(state.node_markers);
    destroyCUDA(state.device_num_moved);
    destroyCUDA(state.net_hpwls);
  }
  iter_time_stop = CPUTimer::getGlobaltime();
  dreamplacePrint(
      kDEBUG, "destroying CUDA state takes %g ms\n",
      CPUTimer::getTimerPeriod() * (iter_time_stop - iter_time_start));

  total_time_stop = CPUTimer::getGlobaltime();

  // destroy cpu db
  iter_time_start = CPUTimer::getGlobaltime();
  {
    destroyCPU(cpu_db.net_mask);
    destroyCPU(cpu_db.flat_net2pin_start_map);
    destroyCPU(cpu_db.flat_net2pin_map);
    destroyCPU(cpu_db.pin2node_map);
    destroyCPU(cpu_db.x);
    destroyCPU(cpu_db.y);
    destroyCPU(cpu_db.node_size_x);
    destroyCPU(cpu_db.node_size_y);
  }
  iter_time_stop = CPUTimer::getGlobaltime();
  dreamplacePrint(
      kDEBUG, "destroying CPU state takes %g ms\n",
      CPUTimer::getTimerPeriod() * (iter_time_stop - iter_time_start));

  dreamplacePrint(
      kINFO, "Kernel time: %g ms\n",
      CPUTimer::getTimerPeriod() * (kernel_time_stop - kernel_time_start));
  dreamplacePrint(
      kINFO, "K-reorder time: %g ms\n",
      CPUTimer::getTimerPeriod() * (total_time_stop - total_time_start));

  return 0;
}

#define REGISTER_KERNEL_LAUNCHER(T)                                      \
  template int kreorderCUDALauncher<T>(DetailedPlaceDB<T> db, int K,     \
                                       int max_iters, int num_threads); 

REGISTER_KERNEL_LAUNCHER(int);
REGISTER_KERNEL_LAUNCHER(float);
REGISTER_KERNEL_LAUNCHER(double);

DREAMPLACE_END_NAMESPACE
