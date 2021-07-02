/**
 * @file   global_swap_cuda_kernel.cu
 * @author Jiaqi Gu, Yibo Lin
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

#include "utility/src/utils.cuh"
#include "utility/src/utils_cub.cuh"
// database dependency
#include "utility/src/detailed_place_db.cuh"

#define MAX_NODE_DEGREE 20
#define MAX_NET_DEGREE 100

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
struct __align__(16) SwapCandidate {
  T cost;
  T node_xl[2][2];  ///< [0][] for node, [1][] for target node, [][0] for old,
                    ///< [][1] for new
  T node_yl[2][2];
  int node_id[2];  ///< [0] for node, [1] for target node
};

struct SearchBinInfo {
  int cx;
  int cy;
  // int size;
};

template <typename T>
struct __align__(16) NetPinPair {
  int net_id;
  T pin_offset_x;
  T pin_offset_y;
};

template <typename T>
struct __align__(16) NodePinPair {
  int node_id;
  T pin_offset_x;
  T pin_offset_y;
};

template <typename T>
struct SwapState {
  int* ordered_nodes = nullptr;

  Space<T>* spaces = nullptr;

  PitchNestedVector<int> row2node_map;
  RowMapIndex* node2row_map = nullptr;

  PitchNestedVector<int> bin2node_map;
  BinMapIndex* node2bin_map = nullptr;

  // PitchNestedVector<NetPinPair<T> > node2netpin_map;
  PitchNestedVector<int> node2net_map;
  PitchNestedVector<NodePinPair<T>> net2nodepin_map;

  int* search_bins = nullptr;
  int search_bin_strategy;  ///< how to compute search bins for eahc cell: 0 for
                            ///< cell bin, 1 for optimal region

  SwapCandidate<T>* candidates;

  double*
      net_hpwls;  ///< HPWL for each net, use integer to get consistent values
  unsigned char* node_markers;  ///< markers for cells

  int batch_size;
  int max_num_candidates_per_row;
  int max_num_candidates;
  int max_num_candidates_all;

  int pair_hpwl_computing_strategy;  ///< 0: for the original node2pin_map and
                                     ///< net2pin_map; 1: for node2net_map and
                                     ///< net2node_map, which requires
                                     ///< additional memory
};

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

template <typename T, int ThreadsPerBlock = 128>
__global__ void reduce_min_2d_cub(SwapCandidate<T>* candidates,
                                  int max_num_elements) {
  typedef cub::BlockReduce<ItemWithIndex<T>, ThreadsPerBlock> BlockReduce;

  __shared__ typename BlockReduce::TempStorage temp_storage;

  auto row_candidates = candidates + blockIdx.x * max_num_elements;

  ItemWithIndex<T> thread_data;

  thread_data.value = cuda::numeric_limits<T>::max();
  thread_data.index = 0;
  for (int col = threadIdx.x; col < max_num_elements; col += ThreadsPerBlock) {
    T cost = row_candidates[col].cost;
    if (cost < thread_data.value) {
      thread_data.value = cost;
      thread_data.index = col;
    }
  }

  __syncthreads();

  // Compute the block-wide max for thread0
  ItemWithIndex<T> aggregate =
      BlockReduce(temp_storage)
          .Reduce(thread_data, ReduceMinOP<T>(), max_num_elements);

  __syncthreads();

  if (threadIdx.x == 0) {
    row_candidates[0] = row_candidates[aggregate.index];
  }
}

template <typename T>
inline __device__ T compute_pair_hpwl_general(
    const int* __restrict__ flat_node2pin_start_map,
    const int* __restrict__ flat_node2pin_map,
    const int* __restrict__ pin2net_map, const T xh, const T yh, const T xl,
    const T yl, const unsigned char* __restrict__ net_mask,
    const int* __restrict__ flat_net2pin_start_map,
    const int* __restrict__ flat_net2pin_map,
    const int* __restrict__ pin2node_map, const T* __restrict__ x,
    const T* __restrict__ y, const T* __restrict__ pin_offset_x,
    const T* __restrict__ pin_offset_y, int node_id, T node_xl, T node_yl,
    int target_node_id, T target_node_xl, T target_node_yl, int skip_node_id) {
  T cost = 0;
  int node2pin_id = flat_node2pin_start_map[node_id];
  const int node2pin_id_end = flat_node2pin_start_map[node_id + 1];
  for (; node2pin_id < node2pin_id_end; ++node2pin_id) {
    int node_pin_id = flat_node2pin_map[node2pin_id];
    int net_id = pin2net_map[node_pin_id];
    Box<T> box(xh, yh, xl, yl);
    int flag = net_mask[net_id];
    int net2pin_id = flat_net2pin_start_map[net_id];
    const int net2pin_id_end = flat_net2pin_start_map[net_id + 1] * flag;
    for (; net2pin_id < net2pin_id_end; ++net2pin_id) {
      int net_pin_id = flat_net2pin_map[net2pin_id];
      int other_node_id = pin2node_map[net_pin_id];
      T xxl = x[other_node_id];
      T yyl = y[other_node_id];
      flag &= (other_node_id != skip_node_id);
      int cond1 = (other_node_id == node_id);
      int cond2 = (other_node_id == target_node_id);
      xxl =
          cond1 * node_xl + cond2 * target_node_xl + (!(cond1 || cond2)) * xxl;
      yyl =
          cond1 * node_yl + cond2 * target_node_yl + (!(cond1 || cond2)) * yyl;
      // xxl+px
      xxl += pin_offset_x[net_pin_id];
      // yyl+py
      yyl += pin_offset_y[net_pin_id];
      box.xl = min(box.xl, xxl);
      box.xh = max(box.xh, xxl);
      box.yl = min(box.yl, yyl);
      box.yh = max(box.yh, yyl);
    }
    cost += (box.xh - box.xl + box.yh - box.yl) * flag;
  }
  return cost;
}

template <typename T>
inline __device__ T compute_pair_hpwl_general_fast(
    PitchNestedVector<int>& node2net_map,
    PitchNestedVector<NodePinPair<T>>& net2nodepin_map, const T xh, const T yh,
    const T xl, const T yl, const unsigned char* __restrict__ net_mask,
    const T* __restrict__ x, const T* __restrict__ y, int node_id, T node_xl,
    T node_yl, int target_node_id, T target_node_xl, T target_node_yl,
    int skip_node_id) {
#if 0
    T cost = 0;
    auto node2nets = node2net_map(node_id);
    for(int i = 0; i<node2net_map.size(node_id); ++i)
    {
        int net_id = node2nets[i];
        int flag = net_mask[net_id];
        auto net2nodepins = net2nodepin_map(net_id);
        Box<T> box (
                        xh, 
                        yh, 
                        xl, 
                        yl
                    );
        int end = net2nodepin_map.size(net_id)*flag;
        for(int j = 0; j < end; ++j)
        {
            NodePinPair<T> & node_pin_pair = net2nodepins[j];
            int other_node_id = node_pin_pair.node_id;

            T xxl = x[other_node_id]; 
            T yyl = y[other_node_id];
            flag &= (other_node_id != skip_node_id);
            int cond1 = (other_node_id == node_id);
            int cond2 = (other_node_id == target_node_id);
            xxl = cond1*node_xl 
                + cond2*target_node_xl
                + (!(cond1||cond2))*xxl;
            yyl = cond1*node_yl 
                + cond2*target_node_yl
                + (!(cond1||cond2))*yyl;
            // xxl+px
            xxl += node_pin_pair.pin_offset_x; 
            // yyl+py
            yyl += node_pin_pair.pin_offset_y; 
            box.xl = min(box.xl, xxl);
            box.xh = max(box.xh, xxl);
            box.yl = min(box.yl, yyl);
            box.yh = max(box.yh, yyl);
        }
        cost += (box.xh-box.xl + box.yh-box.yl)*flag; 
    }
    return cost;
#endif

#if 1
  T cost = 0;
  auto node2nets = node2net_map(node_id);
  for (int i = 0; i < node2net_map.size(node_id); ++i) {
    int net_id = node2nets[i];
    int flag = net_mask[net_id];
    auto net2nodepins = net2nodepin_map(net_id);
    Box<T> box(xh, yh, xl, yl);

    int end = net2nodepin_map.size(net_id) * flag;
    for (int j = 0; j < end; ++j) {
      NodePinPair<T>& node_pin_pair = net2nodepins[j];
      int other_node_id = node_pin_pair.node_id;

      flag &= (other_node_id != skip_node_id);

      T xxl = x[other_node_id];
      T yyl = y[other_node_id];
      int cond1 = (other_node_id == node_id);
      int cond2 = (other_node_id == target_node_id);
      xxl =
          cond1 * node_xl + cond2 * target_node_xl + (!(cond1 || cond2)) * xxl;
      yyl =
          cond1 * node_yl + cond2 * target_node_yl + (!(cond1 || cond2)) * yyl;
      // xxl+px
      xxl += node_pin_pair.pin_offset_x;
      // yyl+py
      yyl += node_pin_pair.pin_offset_y;
      box.xl = min(box.xl, xxl);
      box.xh = max(box.xh, xxl);
      box.yl = min(box.yl, yyl);
      box.yh = max(box.yh, yyl);
    }
    cost += (box.xh - box.xl + box.yh - box.yl) * flag;
  }
  return cost;
#endif
}

template <typename T>
__device__ T compute_pair_hpwl(const DetailedPlaceDB<T>& db,
                               const SwapState<T>& state, int node_id,
                               T node_xl, T node_yl, int target_node_id,
                               T target_node_xl, T target_node_yl) {
  T cost = 0;
  for (int node2pin_id = db.flat_node2pin_start_map[node_id];
       node2pin_id < db.flat_node2pin_start_map[node_id + 1]; ++node2pin_id) {
    int node_pin_id = db.flat_node2pin_map[node2pin_id];
    int net_id = db.pin2net_map[node_pin_id];
    Box<T> box(db.xh, db.yh, db.xl, db.yl);
    if (db.net_mask[net_id]) {
      for (int net2pin_id = db.flat_net2pin_start_map[net_id];
           net2pin_id < db.flat_net2pin_start_map[net_id + 1]; ++net2pin_id) {
        int net_pin_id = db.flat_net2pin_map[net2pin_id];
        int other_node_id = db.pin2node_map[net_pin_id];
        int cond1 = (other_node_id == node_id);
        int cond2 = (other_node_id == target_node_id);
        T xxl = cond1 * node_xl + cond2 * target_node_xl +
                (!(cond1 || cond2)) * db.x[other_node_id];
        T yyl = cond1 * node_yl + cond2 * target_node_yl +
                (!(cond1 || cond2)) * db.y[other_node_id];
        T px = db.pin_offset_x[net_pin_id];
        T py = db.pin_offset_y[net_pin_id];
        box.xl = min(box.xl, xxl + px);
        box.xh = max(box.xh, xxl + px);
        box.yl = min(box.yl, yyl + py);
        box.yh = max(box.yh, yyl + py);
      }
      cost += box.xh - box.xl + box.yh - box.yl;
    }
  }
  for (int node2pin_id = db.flat_node2pin_start_map[target_node_id];
       node2pin_id < db.flat_node2pin_start_map[target_node_id + 1];
       ++node2pin_id) {
    int node_pin_id = db.flat_node2pin_map[node2pin_id];
    int net_id = db.pin2net_map[node_pin_id];
    Box<T> box(db.xh, db.yh, db.xl, db.yl);
    if (db.net_mask[net_id]) {
      // when encounter nets that have both node_id and target_node_id
      for (int net2pin_id = db.flat_net2pin_start_map[net_id];
           net2pin_id < db.flat_net2pin_start_map[net_id + 1]; ++net2pin_id) {
        int net_pin_id = db.flat_net2pin_map[net2pin_id];
        int other_node_id = db.pin2node_map[net_pin_id];
        int cond1 = (other_node_id == node_id);
        if (cond1) {
          // skip them
          box.xl = box.yl = box.xh = box.yh = 0;
          break;
        }
        int cond2 = (other_node_id == target_node_id);
        T xxl = cond1 * node_xl + cond2 * target_node_xl +
                (!(cond1 || cond2)) * db.x[other_node_id];
        T yyl = cond1 * node_yl + cond2 * target_node_yl +
                (!(cond1 || cond2)) * db.y[other_node_id];
        T px = db.pin_offset_x[net_pin_id];
        T py = db.pin_offset_y[net_pin_id];
        box.xl = min(box.xl, xxl + px);
        box.xh = max(box.xh, xxl + px);
        box.yl = min(box.yl, yyl + py);
        box.yh = max(box.yh, yyl + py);
      }
      cost += box.xh - box.xl + box.yh - box.yl;
    }
  }
  return cost;
}

template <typename T>
__device__ T compute_positions(const DetailedPlaceDB<T>& db,
                               const SwapState<T>& state,
                               SwapCandidate<T>& cand) {
  // case I: two cells are horizontally abutting
  int row_id = db.pos2site_y(db.y[cand.node_id[0]]);
  int target_row_id = db.pos2site_y(db.y[cand.node_id[1]]);
  cand.node_xl[0][0] = db.x[cand.node_id[0]];
  cand.node_yl[0][0] = db.y[cand.node_id[0]];
  cand.node_xl[1][0] = db.x[cand.node_id[1]];
  cand.node_yl[1][0] = db.y[cand.node_id[1]];
  // int cond = ((row_id == target_row_id)<<1);
  // cond += (cand.node_xl[0][0] + db.node_size_x[cand.node_id[0]] ==
  // cand.node_xl[1][0]);  cond += (cand.node_xl[1][0] +
  // db.node_size_x[cand.node_id[1]] == cand.node_xl[0][0]);
  if (row_id == target_row_id &&
      (cand.node_xl[0][0] + db.node_size_x[cand.node_id[0]] ==
           cand.node_xl[1][0] ||
       cand.node_xl[1][0] + db.node_size_x[cand.node_id[1]] ==
           cand.node_xl[0][0])) {
    if (cand.node_xl[0][0] < cand.node_xl[1][0]) {
      cand.node_xl[0][1] = cand.node_xl[1][0] +
                           db.node_size_x[cand.node_id[1]] -
                           db.node_size_x[cand.node_id[0]];
      cand.node_xl[1][1] = cand.node_xl[0][0];
    } else {
      cand.node_xl[0][1] = cand.node_xl[1][0];
      cand.node_xl[1][1] = cand.node_xl[0][0] +
                           db.node_size_x[cand.node_id[0]] -
                           db.node_size_x[cand.node_id[1]];
    }
  } else  // case II: not abutting
  {
    cand.node_xl[0][1] = cand.node_xl[1][0] +
                         db.node_size_x[cand.node_id[1]] / 2 -
                         db.node_size_x[cand.node_id[0]] / 2;
    cand.node_xl[1][1] = cand.node_xl[0][0] +
                         db.node_size_x[cand.node_id[0]] / 2 -
                         db.node_size_x[cand.node_id[1]] / 2;
    cand.node_xl[0][1] = db.align2site(cand.node_xl[0][1]);
    cand.node_xl[1][1] = db.align2site(cand.node_xl[1][1]);
    int node2row2node_index = state.node2row2node_index_map[cand.node_id[0]];
    T space_xl = db.xl;
    if (node2row2node_index) {
      int space_xl_node_id =
          state.row2node_map(row_id, node2row2node_index - 1);
      space_xl = max(space_xl,
                     db.x[space_xl_node_id] + db.node_size_x[space_xl_node_id]);
    }
    T space_xh = db.xh;
    if (node2row2node_index + 1 < (int)state.row2node_map.size(row_id)) {
      space_xh = min(space_xh,
                     db.x[state.row2node_map(row_id, node2row2node_index + 1)]);
    }
    if (space_xh < db.node_size_x[cand.node_id[1]] + space_xl) {
      // some large number
      return cuda::numeric_limits<T>::max();
    }
    int target_node2row2node_index =
        state.node2row2node_index_map[cand.node_id[1]];
    T target_space_xl =
        (target_node2row2node_index > 0)
            ? max(db.xl,
                  db.x[state.row2node_map(target_row_id,
                                          target_node2row2node_index - 1)] +
                      db.node_size_x[state.row2node_map(
                          target_row_id, target_node2row2node_index - 1)])
            : db.xl;
    T target_space_xh =
        (target_node2row2node_index + 1 <
         (int)state.row2node_map.size(target_row_id))
            ? min(db.xh, db.x[state.row2node_map(
                             target_row_id, target_node2row2node_index + 1)])
            : db.xh;
    if (target_space_xh < db.node_size_x[cand.node_id[0]] + target_space_xl) {
      // some large number
      return cuda::numeric_limits<T>::max();
    }
    cand.node_xl[0][1] = min(max(cand.node_xl[0][1], target_space_xl),
                             target_space_xh - db.node_size_x[cand.node_id[0]]);
    cand.node_xl[1][1] = min(max(cand.node_xl[1][1], space_xl),
                             space_xh - db.node_size_x[cand.node_id[1]]);
  }
  cand.node_yl[0][1] = cand.node_yl[1][0];
  cand.node_yl[1][1] = cand.node_yl[0][0];

  return 0;
}

template <typename T>
__device__ T compute_positions_hint(const DetailedPlaceDB<T>& db,
                                    const SwapState<T>& state,
                                    SwapCandidate<T>& cand, T node_xl,
                                    T node_yl, T node_width,
                                    const Space<T>& space) {
  // case I: two cells are horizontally abutting
  cand.node_xl[0][0] = node_xl;
  cand.node_yl[0][0] = node_yl;
  cand.node_xl[1][0] = db.x[cand.node_id[1]];
  cand.node_yl[1][0] = db.y[cand.node_id[1]];
  T target_node_width = db.node_size_x[cand.node_id[1]];
  auto target_space = db.align2site(state.spaces[cand.node_id[1]]);
  int cond = (space.xh >= target_space.xl);
  cond &= (target_space.xh >= space.xl);
  cond &= (cand.node_yl[0][0] == cand.node_yl[1][0]);
  if (cond)  // case I: abutting, not exactly abutting, there might be space
             // between two cells, this is a generalized case
  {
    cond = (space.xl < target_space.xl);
    cand.node_xl[0][1] =
        cand.node_xl[1][0] + (target_node_width - node_width) * cond;
    cand.node_xl[1][1] =
        cand.node_xl[0][0] - (target_node_width - node_width) * (!cond);
    // if (cand.node_xl[0][0] < cand.node_xl[1][0])
    //{
    //    cand.node_xl[0][1] = cand.node_xl[1][0]+target_node_width-node_width;
    //    cand.node_xl[1][1] = cand.node_xl[0][0];
    //}
    // else
    //{
    //    cand.node_xl[0][1] = cand.node_xl[1][0];
    //    cand.node_xl[1][1] = cand.node_xl[0][0]+node_width-target_node_width;
    //}
  } else  // case II: not abutting
  {
    cond = (space.xh < target_node_width + space.xl);
    cond |= (target_space.xh < node_width + target_space.xl);
    if (cond) {
      // some large number
      return cuda::numeric_limits<T>::max();
    }
    cand.node_xl[0][1] =
        cand.node_xl[1][0] + (target_node_width - node_width) / 2;
    cand.node_xl[1][1] =
        cand.node_xl[0][0] + (node_width - target_node_width) / 2;
    cand.node_xl[0][1] = db.align2site(cand.node_xl[0][1]);
    cand.node_xl[0][1] = max(cand.node_xl[0][1], target_space.xl);
    cand.node_xl[0][1] = min(cand.node_xl[0][1], target_space.xh - node_width);
    cand.node_xl[1][1] = db.align2site(cand.node_xl[1][1]);
    cand.node_xl[1][1] = max(cand.node_xl[1][1], space.xl);
    cand.node_xl[1][1] = min(cand.node_xl[1][1], space.xh - target_node_width);
  }
  cand.node_yl[0][1] = cand.node_yl[1][0];
  cand.node_yl[1][1] = cand.node_yl[0][0];

  return 0;
}

template <typename T>
struct CompareSwapCandidateCost {
  __host__ __device__ SwapCandidate<T> operator()(
      const SwapCandidate<T>& cand1, const SwapCandidate<T>& cand2) const {
    return (cand1.cost < cand2.cost) ? cand1 : cand2;
  }
};

template <typename T>
struct CompareSwapCandidateCostValue {
  __host__ __device__ bool operator()(const SwapCandidate<T>& cand1,
                                      const SwapCandidate<T>& cand2) const {
    return cand1.cost < cand2.cost;
  }
};

template <typename T>
__global__ void compute_search_bins(DetailedPlaceDB<T> db, SwapState<T> state,
                                    int begin, int end) {
  for (int node_id = begin + blockIdx.x * blockDim.x + threadIdx.x;
       node_id < end; node_id += blockDim.x * gridDim.x) {
    // compute optimal region
    Box<T> opt_box = (state.search_bin_strategy)
                         ? db.compute_optimal_region(node_id, db.x, db.y)
                         : Box<T>(db.x[node_id], db.y[node_id],
                                  db.x[node_id] + db.node_size_x[node_id],
                                  db.y[node_id] + db.node_size_y[node_id]);
    // cell already in optimal region, skip it
    // if (opt_box.contains(node_box.xl, node_box.yl, node_box.xh, node_box.yh))
    //{
    //    continue;
    //}
    // extend optimal region
    // opt_box.encompass(node_box.xl, node_box.yl, node_box.xh, node_box.yh);
    // SearchBinInfo& info = state.search_bins[node_id];
    int cx = db.pos2bin_x(opt_box.center_x());
    int cy = db.pos2bin_y(opt_box.center_y());
    state.search_bins[node_id] = cx * db.num_bins_y + cy;
    // int node_bin_x = (node_box.center_x() < opt_box.center_x())?
    // db.pos2bin_x(node_box.xl) : db.pos2bin_x(node_box.xh);  int node_bin_y =
    // (node_box.center_y() < opt_box.center_y())? db.pos2bin_y(node_box.yl) :
    // db.pos2bin_y(node_box.yh);  int distance = abs(node_bin_x-info.cx) +
    // abs(node_bin_y-info.cy);  info.size = min(distance*distance*2,
    // state.num_search_grids);

    // Box<T> search_box (
    //        max(opt_box.center_x()-distance*db.bin_size_x, db.xl),
    //        max(opt_box.center_y()-distance*db.bin_size_y, db.yl),
    //        min(opt_box.center_x()+distance*db.bin_size_x, db.xh),
    //        min(opt_box.center_y()+distance*db.bin_size_y, db.yh)
    //        );
    // Box<T>& bin = state.search_bins[node_id];
    // bin.xl = db.xl+cx*db.bin_size_x;
    // bin.yl = db.yl+cy*db.bin_size_y;
    // bin.xh = db.xl+(cx+1)*db.bin_size_x;
    // bin.yh = db.yl+(cy+1)*db.bin_size_y;
  }
}

template <typename T>
__global__ void reset_state(DetailedPlaceDB<T> db, SwapState<T> state) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < state.max_num_candidates_all; i += blockDim.x * gridDim.x) {
    SwapCandidate<T>& cand = state.candidates[i];
    cand.cost = 0;
    cand.node_id[0] = cuda::numeric_limits<int>::max();
    cand.node_id[1] = cuda::numeric_limits<int>::max();
    cand.node_xl[0][0] = 0;
    cand.node_xl[0][1] = 0;
    cand.node_yl[0][0] = 0;
    cand.node_yl[0][1] = 0;
    cand.node_xl[1][0] = 0;
    cand.node_xl[1][1] = 0;
    cand.node_yl[1][0] = 0;
    cand.node_yl[1][1] = 0;
  }
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < db.num_movable_nodes;
       i += blockDim.x * gridDim.x) {
    state.node_markers[i] = 0;
  }
}

template <typename T>
__global__ void check_state(DetailedPlaceDB<T> db, SwapState<T> state) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < db.num_movable_nodes;
       i += blockDim.x * gridDim.x) {
    const BinMapIndex& bm_idx = state.node2bin_map[i];
    if (state.bin2node_map(bm_idx.bin_id, bm_idx.sub_id) != i) {
      printf("[E] node %d @ (%g, %g), bin [%d, %d], found %d\n", i,
             (float)db.x[i], (float)db.y[i], bm_idx.bin_id, bm_idx.sub_id,
             state.bin2node_map(bm_idx.bin_id, bm_idx.sub_id));
      assert(0);
    }
  }
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < state.max_num_candidates_all; i += blockDim.x * gridDim.x) {
    SwapCandidate<T>& cand = state.candidates[i];
    if (cand.cost < 0 && (cand.node_id[0] >= db.num_movable_nodes ||
                          cand.node_id[1] >= db.num_movable_nodes)) {
      printf("[E] node %d, target_node %d, cost %g\n", cand.node_id[0],
             cand.node_id[1], (float)cand.cost);
      assert(0);
    }
    if (cand.cost < 0) {
      if (db.x[cand.node_id[0]] != cand.node_xl[0][0]) {
        printf("[E] node %d x %g node_xl %g\n", cand.node_id[0],
               (float)db.x[cand.node_id[0]], (float)cand.node_xl[0][0]);
      }
      if (db.y[cand.node_id[0]] != cand.node_yl[0][0]) {
        printf("[E] node %d y %g node_yl %g\n", cand.node_id[0],
               (float)db.y[cand.node_id[0]], (float)cand.node_yl[0][0]);
      }
      if (db.x[cand.node_id[1]] != cand.node_xl[1][0]) {
        printf("[E] node %d x %g target_node_xl %g\n", cand.node_id[1],
               (float)db.x[cand.node_id[1]], (float)cand.node_xl[1][0]);
      }
      if (db.y[cand.node_id[1]] != cand.node_yl[1][0]) {
        printf("[E] node %d y %g target_node_yl %g\n", cand.node_id[1],
               (float)db.y[cand.node_id[1]], (float)cand.node_yl[1][0]);
      }
      assert(db.x[cand.node_id[0]] == cand.node_xl[0][0]);
      assert(db.y[cand.node_id[0]] == cand.node_yl[0][0]);
      assert(db.x[cand.node_id[1]] == cand.node_xl[1][0]);
      assert(db.y[cand.node_id[1]] == cand.node_yl[1][0]);
    }
  }
}

template <typename T>
__global__ void __launch_bounds__(256, 4)
    collect_candidates(DetailedPlaceDB<T> db, SwapState<T> state, int idx_bgn,
                       int idx_end) {
  // assume following inequality
  // assert(gridDim.y == (idx_end-idx_bgn));
  // assert(gridDim.x == 5);
  // assert(blockDim.x == num_nodes_in_bin)
  __shared__ int node_id;
  __shared__ T node_xl, node_yl, node_width;
  __shared__ Space<T> space;
  __shared__ int max_num_candidates;
  __shared__ int bin_id;
  __shared__ const int* __restrict__ bin2nodes;
  __shared__ int num_nodes_in_bin;
  __shared__ float step_size;
  __shared__ int iters;
  __shared__ int block_offset;
  if (threadIdx.x == 0) {
    node_id = state.ordered_nodes[blockIdx.y + idx_bgn];
    node_xl = db.x[node_id];
    node_yl = db.y[node_id];
    node_width = db.node_size_x[node_id];
    space = db.align2site(state.spaces[node_id]);
    max_num_candidates = state.max_num_candidates / 5;

    block_offset =
        blockIdx.y * state.max_num_candidates + blockIdx.x * max_num_candidates;
    bin_id = state.search_bins[node_id];
    int bx = bin_id / db.num_bins_y;
    int by = bin_id - bx * db.num_bins_y;
    if (blockIdx.x == 1)  // left bin
    {
      if (bx > 0) {
        bin_id -= db.num_bins_y;
      } else {
        bin_id = -1;
      }
    } else if (blockIdx.x == 2)  // bottom bin
    {
      if (by > 0) {
        bin_id -= 1;
      } else {
        bin_id = -1;
      }
    } else if (blockIdx.x == 3)  // right bin
    {
      if (bx + 1 < db.num_bins_x) {
        bin_id += db.num_bins_y;
      } else {
        bin_id = -1;
      }
    } else if (blockIdx.x == 4)  // top bin
    {
      if (by + 1 < db.num_bins_y) {
        bin_id += 1;
      } else {
        bin_id = -1;
      }
    }
    // else is center bin

    if (bin_id >= 0) {
      bin2nodes = state.bin2node_map(bin_id);
      num_nodes_in_bin =
          state.bin2node_map.size(bin_id) *
          (db.node_size_y[node_id] ==
           db.row_height);  // only consider single-row height cell
      step_size =
          max(div((float)num_nodes_in_bin, (float)max_num_candidates),
              (float)1);
      iters = min(max_num_candidates, num_nodes_in_bin);
    }
  }
  __syncthreads();
  SwapCandidate<T> cand;
  cand.node_id[0] = node_id;
  if (bin_id >= 0) {
    for (int i = threadIdx.x; i < iters; i += blockDim.x) {
      cand.node_id[1] = bin2nodes[int(i * step_size)];
      int cond = (cand.node_id[0] != cand.node_id[1]);
      cond &= (db.node_size_y[cand.node_id[1]] == db.row_height);
      if (cond) {
        // target_cost - orig_cost
        // cand.cost = compute_positions(db, state, cand);
        cand.cost = compute_positions_hint(db, state, cand, node_xl, node_yl,
                                           node_width, space);
        cond = (cand.cost == 0);
        if (cond) {
          state.candidates[block_offset + i] = cand;
        }
      }
    }
  }
}

template <typename T>
__global__ void compute_candidate_position(DetailedPlaceDB<T> db,
                                           SwapState<T> state) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < state.max_num_candidates_all; i += blockDim.x * gridDim.x) {
    SwapCandidate<T>& cand = state.candidates[i];
    if (cand.node_id[0] < db.num_movable_nodes &&
        cand.node_id[1] < db.num_movable_nodes) {
      cand.cost = compute_positions(db, state, cand);
    }
  }
}

template <typename T>
__global__ void reset_candidate_costs(DetailedPlaceDB<T> db,
                                      SwapState<T> state) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < state.max_num_candidates_all; i += blockDim.x * gridDim.x) {
    state.candidates[i].cost = cuda::numeric_limits<T>::max();
  }
}

template <typename T>
__global__ void check_candidate_costs(DetailedPlaceDB<T> db,
                                      SwapState<T> state) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < state.max_num_candidates_all; i += blockDim.x * gridDim.x) {
    auto const& cand = state.candidates[i];
    if (cand.cost < 0) {
      assert(cand.node_id[0] < db.num_movable_nodes &&
             cand.node_id[1] < db.num_movable_nodes);
    }
  }
}

template <typename T>
__global__ void __launch_bounds__(64 * 4, 4)
    compute_candidate_cost(DetailedPlaceDB<T> db, SwapState<T> state) {
  extern __shared__ unsigned char cost_proxy[];
  __shared__ int num_candidates;
  T* cost = reinterpret_cast<T*>(cost_proxy);
  if (threadIdx.x == 0) {
    num_candidates = (state.max_num_candidates_all << 2);
  }
  __syncthreads();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_candidates;
       i += blockDim.x * gridDim.x) {
    SwapCandidate<T>& cand = state.candidates[i >> 2];
    int node_id_flag = ((threadIdx.x & 2) >> 1);
    int offset = (threadIdx.x & 1);
    int skip_node_id = cand.node_id[0] + INT_MIN * (!node_id_flag);
    if (cand.node_id[0] < db.num_movable_nodes &&
        cand.node_id[1] < db.num_movable_nodes) {
      int cost1 =
          (state.pair_hpwl_computing_strategy)
              ? compute_pair_hpwl_general_fast<T>(
                    state.node2net_map, state.net2nodepin_map, db.xh, db.yh,
                    db.xl, db.yl, db.net_mask, db.x, db.y,
                    cand.node_id[node_id_flag],
                    cand.node_xl[node_id_flag][offset],
                    cand.node_yl[node_id_flag][offset],
                    cand.node_id[!node_id_flag],
                    cand.node_xl[!node_id_flag][offset],
                    cand.node_yl[!node_id_flag][offset], skip_node_id)
              : compute_pair_hpwl_general<T>(
                    db.flat_node2pin_start_map, db.flat_node2pin_map,
                    db.pin2net_map, db.xh, db.yh, db.xl, db.yl, db.net_mask,
                    db.flat_net2pin_start_map, db.flat_net2pin_map,
                    db.pin2node_map, db.x, db.y, db.pin_offset_x,
                    db.pin_offset_y, cand.node_id[node_id_flag],
                    cand.node_xl[node_id_flag][offset],
                    cand.node_yl[node_id_flag][offset],
                    cand.node_id[!node_id_flag],
                    cand.node_xl[!node_id_flag][offset],
                    cand.node_yl[!node_id_flag][offset], skip_node_id);
      cost[threadIdx.x] = cost1;
    } else {
      cost[threadIdx.x] = 0;
    }
  }
  __syncthreads();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_candidates;
       i += blockDim.x * gridDim.x) {
    SwapCandidate<T>& cand = state.candidates[i >> 2];
    if ((threadIdx.x & 3) == 3)
    // if (threadIdx.x&1)
    {
      // consider FENCE region
      if (db.num_regions &&
          ((cand.node_id[0] < db.num_movable_nodes &&
            !db.inside_fence(cand.node_id[0], cand.node_xl[0][1],
                             cand.node_yl[0][1])) ||
           (cand.node_id[1] < db.num_movable_nodes &&
            !db.inside_fence(cand.node_id[1], cand.node_xl[1][1],
                             cand.node_yl[1][1])))) {
        cand.cost = cuda::numeric_limits<T>::max();
      } else {
        // target_cost - orig_cost
        // cand.cost += cost[threadIdx.x]-cost[threadIdx.x-1];
        cand.cost = cost[threadIdx.x] - cost[threadIdx.x - 1] +
                    cost[threadIdx.x - 2] - cost[threadIdx.x - 3];
      }
    }
  }
}

/// only allow 1 block
template <typename T>
__global__ void apply_candidates(DetailedPlaceDB<T> db, SwapState<T> state,
                                 int num_candidates) {
#ifdef DEBUG
  assert(gridDim.x == 1 && blockDim.x == 1);
#endif
  for (int i = 0; i < num_candidates; ++i) {
#ifdef DEBUG
    assert(i * state.max_num_candidates < state.max_num_candidates_all);
#endif
    const SwapCandidate<T>& best_cand =
        state.candidates[i * state.max_num_candidates];

#ifdef DEBUG
    if (best_cand.cost < 0) {
      if (!(best_cand.node_id[0] < db.num_movable_nodes &&
            best_cand.node_id[1] < db.num_movable_nodes)) {
        printf("node %d, %d, cost %g\n", best_cand.node_id[0],
               best_cand.node_id[1], best_cand.cost);
      }
      assert(best_cand.node_id[0] < db.num_movable_nodes &&
             best_cand.node_id[1] < db.num_movable_nodes);
    }
#endif
    if (best_cand.cost < 0 && !(state.node_markers[best_cand.node_id[0]] ||
                                state.node_markers[best_cand.node_id[1]])) {
      T node_width = db.node_size_x[best_cand.node_id[0]];
      T target_node_width = db.node_size_x[best_cand.node_id[1]];
      Space<T>& space = state.spaces[best_cand.node_id[0]];
      Space<T>& target_space = state.spaces[best_cand.node_id[1]];

      // space may no longer be large enough or the previously computed
      // locations may not be correct any more
      if (best_cand.node_xl[0][1] >= target_space.xl &&
          best_cand.node_xl[0][1] + node_width <= target_space.xh &&
          best_cand.node_xl[1][1] >= space.xl &&
          best_cand.node_xl[1][1] + target_node_width <= space.xh) {
        state.node_markers[best_cand.node_id[0]] = 1;
        state.node_markers[best_cand.node_id[1]] = 1;

#ifdef DEBUG
        assert(best_cand.node_id[0] < db.num_movable_nodes &&
               best_cand.node_id[1] < db.num_movable_nodes);
#endif

        BinMapIndex& bin_id = state.node2bin_map[best_cand.node_id[0]];
        BinMapIndex& target_bin_id = state.node2bin_map[best_cand.node_id[1]];
        RowMapIndex& row_id = state.node2row_map[best_cand.node_id[0]];
        RowMapIndex& target_row_id = state.node2row_map[best_cand.node_id[1]];
        // assert(row_id.row_id < db.num_sites_y);
        int* row2nodes = state.row2node_map(row_id.row_id);
        // assert(target_row_id.row_id < db.num_sites_y);
        int* target_row2nodes = state.row2node_map(target_row_id.row_id);
#ifdef DEBUG
        assert(row_id.sub_id > 0 &&
               row_id.sub_id + 1 < (int)state.row2node_map.size(row_id.row_id));
        assert(target_row_id.sub_id > 0 &&
               target_row_id.sub_id + 1 <
                   (int)state.row2node_map.size(target_row_id.row_id));
#endif
#ifdef DEBUG
        if (best_cand.node_id[0] == 44500 || best_cand.node_id[1] == 44500 ||
            best_cand.node_id[0] == 46123 || best_cand.node_id[1] == 46123) {
          printf(
              "[DEBUG  ] (%g%%) swap node %d (w %g) and node %d (w %g), (%g, "
              "%g) => (%g, %g), (%g, %g) => (%g, %g), space (%g, %g), (%g, "
              "%g), best_cost %g\n",
              i / (T)db.num_movable_nodes * 100, best_cand.node_id[0],
              (float)db.node_size_x[best_cand.node_id[0]], best_cand.node_id[1],
              (float)db.node_size_x[best_cand.node_id[1]],
              (float)db.x[best_cand.node_id[0]],
              (float)db.y[best_cand.node_id[0]], (float)best_cand.node_xl[0][1],
              (float)best_cand.node_yl[0][1], (float)db.x[best_cand.node_id[1]],
              (float)db.y[best_cand.node_id[1]], (float)best_cand.node_xl[1][1],
              (float)best_cand.node_yl[1][1],
              (float)state.spaces[best_cand.node_id[0]].xl,
              (float)state.spaces[best_cand.node_id[0]].xh,
              (float)state.spaces[best_cand.node_id[1]].xl,
              (float)state.spaces[best_cand.node_id[1]].xh,
              (float)best_cand.cost);
          int left_node_id = row2nodes[row_id.sub_id - 1];
          int right_node_id = row2nodes[row_id.sub_id + 1];
          int left_target_node_id = target_row2nodes[target_row_id.sub_id - 1];
          int right_target_node_id = target_row2nodes[target_row_id.sub_id + 1];
          printf(
              "[DEBUG  ] neighbor_node_id %d (%g, %g, %g, %g), %d (%g, %g, %g, "
              "%g)  %d (%g, %g, %g, %g), %d (%g, %g, %g, %g)\n",
              left_node_id, db.x[left_node_id], db.y[left_node_id],
              db.x[left_node_id] + db.node_size_x[left_node_id],
              db.y[left_node_id] + db.node_size_y[left_node_id], right_node_id,
              db.x[right_node_id], db.y[right_node_id],
              db.x[right_node_id] + db.node_size_x[right_node_id],
              db.y[right_node_id] + db.node_size_y[right_node_id],
              left_target_node_id, db.x[left_target_node_id],
              db.y[left_target_node_id],
              db.x[left_target_node_id] + db.node_size_x[left_target_node_id],
              db.y[left_target_node_id] + db.node_size_y[left_target_node_id],
              right_target_node_id, db.x[right_target_node_id],
              db.y[right_target_node_id],
              db.x[right_target_node_id] + db.node_size_x[right_target_node_id],
              db.y[right_target_node_id] +
                  db.node_size_y[right_target_node_id]);
        }
#endif
#ifdef DEBUG
        assert(state.bin2node_map(bin_id.bin_id, bin_id.sub_id) ==
               best_cand.node_id[0]);
        assert(state.bin2node_map(target_bin_id.bin_id, target_bin_id.sub_id) ==
               best_cand.node_id[1]);
        assert(db.x[best_cand.node_id[0]] == best_cand.node_xl[0][0]);
        assert(db.y[best_cand.node_id[0]] == best_cand.node_yl[0][0]);
        assert(db.x[best_cand.node_id[1]] == best_cand.node_xl[1][0]);
        assert(db.y[best_cand.node_id[1]] == best_cand.node_yl[1][0]);
#endif
        db.x[best_cand.node_id[0]] = best_cand.node_xl[0][1];
        db.y[best_cand.node_id[0]] = best_cand.node_yl[0][1];
        db.x[best_cand.node_id[1]] = best_cand.node_xl[1][1];
        db.y[best_cand.node_id[1]] = best_cand.node_yl[1][1];
        int& bin2node_map_node_id =
            state.bin2node_map(bin_id.bin_id, bin_id.sub_id);
        int& bin2node_map_target_node_id =
            state.bin2node_map(target_bin_id.bin_id, target_bin_id.sub_id);
        host_device_swap(bin2node_map_node_id, bin2node_map_target_node_id);
        host_device_swap(bin_id, target_bin_id);

        // update neighboring spaces
        {
          // assert(row_id.sub_id > 0);
          // assert(row_id.sub_id-1 < state.row2node_map.size(row_id.row_id));
          int neighbor_node_id = row2nodes[row_id.sub_id - 1];
          // left node of the node
          if (neighbor_node_id < db.num_movable_nodes) {
            Space<T>& neighbor_space = state.spaces[neighbor_node_id];
            neighbor_space.xh = min(neighbor_space.xh, best_cand.node_xl[1][1]);
          }
          // assert(row_id.sub_id+1 < state.row2node_map.size(row_id.row_id));
          // right node of the node
          neighbor_node_id = row2nodes[row_id.sub_id + 1];
          if (neighbor_node_id < db.num_movable_nodes) {
            Space<T>& neighbor_space = state.spaces[neighbor_node_id];
            neighbor_space.xl = max(
                neighbor_space.xl, best_cand.node_xl[1][1] + target_node_width);
          }
          // assert(target_row_id.sub_id > 0);
          // assert(target_row_id.sub_id-1 <
          // state.row2node_map.size(target_row_id.row_id));
          // left node of the target node
          neighbor_node_id = target_row2nodes[target_row_id.sub_id - 1];
          if (neighbor_node_id < db.num_movable_nodes) {
            Space<T>& neighbor_space = state.spaces[neighbor_node_id];
            neighbor_space.xh = min(neighbor_space.xh, best_cand.node_xl[0][1]);
          }
          // assert(target_row_id.sub_id+1 <
          // state.row2node_map.size(target_row_id.row_id));
          // right node of the target node
          neighbor_node_id = target_row2nodes[target_row_id.sub_id + 1];
          if (neighbor_node_id < db.num_movable_nodes) {
            Space<T>& neighbor_space = state.spaces[neighbor_node_id];
            neighbor_space.xl =
                max(neighbor_space.xl, best_cand.node_xl[0][1] + node_width);
          }
        }
        if ((best_cand.node_yl[0][0] == best_cand.node_yl[1][0]) &&
            (space.xh >= target_space.xl) &&
            (target_space.xh >= space.xl))  // case I: abutting, not exactly
                                            // abutting, there might be space
                                            // between two cells, this is a
                                            // generalized case
        {
          if (best_cand.node_xl[0][0] < best_cand.node_xl[1][0]) {
            space.xh = target_space.xh;
            target_space.xl = space.xl;
            space.xl = best_cand.node_xl[1][1] + target_node_width;
            target_space.xh = best_cand.node_xl[0][1];
          } else {
            target_space.xh = space.xh;
            space.xl = target_space.xl;
            target_space.xl = best_cand.node_xl[0][1] + node_width;
            space.xh = best_cand.node_xl[0][1];
          }
        } else  // case II: not abutting
        {
          // update spaces
          host_device_swap(space, target_space);
        }

        // update row2node_map and node2row_map
        host_device_swap(row2nodes[row_id.sub_id],
                     target_row2nodes[target_row_id.sub_id]);
        host_device_swap(row_id, target_row_id);
      }
    }
  }
}

/// generate array from 0 to n-1
__global__ void iota(int* ptr, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    ptr[i] = i;
  }
}

template <typename T>
#ifdef DYNAMIC
__global__ void global_swap(DetailedPlaceDB<T> db, SwapState<T> state)
#else
void global_swap(DetailedPlaceDB<T>& db, SwapState<T>& state)
#endif
{
  CPUTimer::hr_clock_rep timer_start, timer_stop;

  // const int num_streams = 32;
  // const int num_nodes_per_stream = ceilDiv(db.num_movable_nodes,
  // num_streams);  cudaStream_t streams[num_streams];
#ifdef DYNAMIC
  timer_start = CUDATimer::getGlobalTime();
#else
  timer_start = CPUTimer::getGlobaltime();
#endif
  compute_search_bins<<<ceilDiv(db.num_movable_nodes, 512), 512>>>(
      db, state, 0, db.num_movable_nodes);
  checkCUDA(cudaDeviceSynchronize());
#ifdef DYNAMIC
  timer_stop = CUDATimer::getGlobalTime();
  printf("[DEBUG  ] compute_search_bins takes %g ms\n",
         (timer_stop - timer_start) * CUDATimer::getTimerPeriod());
#else
  timer_stop = CPUTimer::getGlobaltime();
  dreamplacePrint(kDEBUG, "compute_search_bins takes %g ms\n",
                  (timer_stop - timer_start) * CPUTimer::getTimerPeriod());
#endif

#ifdef TIMER
  CPUTimer::hr_clock_rep collect_candidates_time = 0,
                         compute_candidate_cost_time = 0,
                         reduce_min_2d_time = 0, apply_candidates_time = 0;
  int collect_candidates_runs = 0, compute_candidate_cost_runs = 0,
      reduce_min_2d_runs = 0, apply_candidates_runs = 0;
#endif

  for (int i = 0; i < db.num_movable_nodes; i += state.batch_size) {
    // all results are stored in state.candidates
    int idx_bgn = i;
    int idx_end = min(i + state.batch_size, db.num_movable_nodes);
    // printf("[DEBUG  ] batch %d - %d\n", idx_bgn, idx_end);

#ifdef TIMER
    timer_start = CPUTimer::getGlobaltime();
#endif
    reset_state<<<ceilDiv(db.num_movable_nodes, 512), 512>>>(db, state);
    dim3 grid(5, (idx_end - idx_bgn), 1);
    collect_candidates<<<grid, 256>>>(db, state, idx_bgn, idx_end);
#ifdef TIMER
    checkCUDA(cudaDeviceSynchronize());
    timer_stop = CPUTimer::getGlobaltime();
    collect_candidates_time += timer_stop - timer_start;
    collect_candidates_runs += 1;
#endif

#ifdef TIMER
    timer_start = CPUTimer::getGlobaltime();
#endif
    reset_candidate_costs<<<ceilDiv(state.max_num_candidates_all, 256), 256>>>(
        db, state);

    // compute_candidate_position<<<(state.max_num_candidates_all/256),
    // 256>>>(db, state);
    compute_candidate_cost<<<ceilDiv(state.max_num_candidates_all, 64), 64 * 4,
                             64 * 4 * sizeof(T)>>>(db, state);
#ifdef TIMER
    checkCUDA(cudaDeviceSynchronize());
    timer_stop = CPUTimer::getGlobaltime();
    compute_candidate_cost_time += timer_stop - timer_start;
    compute_candidate_cost_runs += 1;
#endif
#ifdef DEBUG
    check_state<<<ceilDiv(db.num_movable_nodes, 512), 512>>>(db, state);
    check_candidate_costs<<<ceilDiv(state.max_num_candidates_all, 256), 256>>>(
        db, state);
#endif
    // reduce min and apply
#ifdef TIMER
    timer_start = CPUTimer::getGlobaltime();
#endif
    reduce_min_2d_cub<T, 256><<<idx_end - idx_bgn, 256>>>(
        state.candidates, state.max_num_candidates);
#ifdef TIMER
    checkCUDA(cudaDeviceSynchronize());
    timer_stop = CPUTimer::getGlobaltime();
    reduce_min_2d_time += timer_stop - timer_start;
    reduce_min_2d_runs += 1;
#endif

#ifdef TIMER
    timer_start = CPUTimer::getGlobaltime();
#endif
    // check_candidate_costs<<<ceilDiv(state.max_num_candidates_all, 256),
    // 256>>>(db, state);
    // must use single thread
    apply_candidates<<<1, 1>>>(db, state, idx_end - idx_bgn);
#ifdef TIMER
    checkCUDA(cudaDeviceSynchronize());
    timer_stop = CPUTimer::getGlobaltime();
    apply_candidates_time += timer_stop - timer_start;
    apply_candidates_runs += 1;
#endif
  }

#ifdef TIMER
  dreamplacePrint(kDEBUG,
                  "collect_candidates takes %g ms for %d runs, average %g ms\n",
                  collect_candidates_time * CPUTimer::getTimerPeriod(),
                  collect_candidates_runs,
                  collect_candidates_time * CPUTimer::getTimerPeriod() /
                      collect_candidates_runs);
  dreamplacePrint(
      kDEBUG, "compute_candidate_cost takes %g ms for %d runs, average %g ms\n",
      compute_candidate_cost_time * CPUTimer::getTimerPeriod(),
      compute_candidate_cost_runs,
      compute_candidate_cost_time * CPUTimer::getTimerPeriod() /
          compute_candidate_cost_runs);
  dreamplacePrint(
      kDEBUG, "reduce_min_2d takes %g ms for %d runs, average %g ms\n",
      reduce_min_2d_time * CPUTimer::getTimerPeriod(), reduce_min_2d_runs,
      reduce_min_2d_time * CPUTimer::getTimerPeriod() / reduce_min_2d_runs);
  dreamplacePrint(
      kDEBUG, "apply_candidates takes %g ms for %d runs, average %g ms\n",
      apply_candidates_time * CPUTimer::getTimerPeriod(), apply_candidates_runs,
      apply_candidates_time * CPUTimer::getTimerPeriod() /
          apply_candidates_runs);
#endif
}

template <typename T>
__global__ void initNode2NetMap_kernel(PitchNestedVector<int> node2net_map,
                                       DetailedPlaceDB<T> db,
                                       const int num_nodes) {
  const int node_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (node_id >= num_nodes) {
    return;
  }
  int num_elements = 0;
  int beg = db.flat_node2pin_start_map[node_id];
  int end = min(db.flat_node2pin_start_map[node_id + 1], beg + MAX_NODE_DEGREE);
  for (int node2pin_id = beg; node2pin_id < end;
       ++node2pin_id, ++num_elements) {
    if (num_elements < MAX_NODE_DEGREE)  // only consider MAX_NODE_DEGREE pins
    {
      int node_pin_id = db.flat_node2pin_map[node2pin_id];
      int net_id = db.pin2net_map[node_pin_id];
      node2net_map.flat_element_map[node_id * MAX_NODE_DEGREE + num_elements] =
          net_id;
    }
  }
  node2net_map.dim2_sizes[node_id] = num_elements;
}

template <typename T>
void initNode2NetMap(PitchNestedVector<int>& node2net_map,
                     DetailedPlaceDB<T>& db) {
  // allocate memory
  allocateCUDA(node2net_map.flat_element_map,
               db.num_movable_nodes * MAX_NODE_DEGREE, int);
  allocateCUDA(node2net_map.dim2_sizes, db.num_movable_nodes, unsigned int);
  node2net_map.size1 = db.num_movable_nodes;
  node2net_map.size2 = MAX_NODE_DEGREE;
  // init on GPU
  initNode2NetMap_kernel<T><<<ceilDiv(db.num_movable_nodes, 512), 512>>>(
      node2net_map, db, db.num_movable_nodes);
  checkCUDA(cudaDeviceSynchronize());
}

template <typename T>
__global__ void initNet2NodePinMap_kernel(
    PitchNestedVector<NodePinPair<T>> net2nodepin_map, DetailedPlaceDB<T> db,
    const int num_nets) {
  const int net_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (net_id >= num_nets) {
    return;
  }
  int num_elements = 0;
  int beg = db.flat_net2pin_start_map[net_id];
  int end = min(db.flat_net2pin_start_map[net_id + 1], beg + MAX_NET_DEGREE);
  for (int net2pin_id = beg; net2pin_id < end; ++net2pin_id, ++num_elements) {
    if (num_elements < MAX_NET_DEGREE)  // only consider MAX_NET_DEGREE pins
    {
      int net_pin_id = db.flat_net2pin_map[net2pin_id];
      T px = db.pin_offset_x[net_pin_id];
      T py = db.pin_offset_y[net_pin_id];
      int node_id = db.pin2node_map[net_pin_id];
      NodePinPair<T>& node_pin_pair =
          net2nodepin_map
              .flat_element_map[net_id * MAX_NET_DEGREE + num_elements];
      node_pin_pair.node_id = node_id;
      node_pin_pair.pin_offset_x = px;
      node_pin_pair.pin_offset_y = py;
    }
  }
  net2nodepin_map.dim2_sizes[net_id] = num_elements;
}

template <typename T>
void initNet2NodePinMap(PitchNestedVector<NodePinPair<T>>& net2nodepin_map,
                        DetailedPlaceDB<T>& db) {
  // allocate memory
  allocateCUDA(net2nodepin_map.flat_element_map, db.num_nets * MAX_NET_DEGREE,
               NodePinPair<T>);
  allocateCUDA(net2nodepin_map.dim2_sizes, db.num_nets, unsigned int);
  net2nodepin_map.size1 = db.num_nets;
  net2nodepin_map.size2 = MAX_NET_DEGREE;
  // init on GPU
  initNet2NodePinMap_kernel<T>
      <<<ceilDiv(db.num_nets, 512), 512>>>(net2nodepin_map, db, db.num_nets);
  checkCUDA(cudaDeviceSynchronize());
}

template <typename T>
__global__ void compute_num_nodes_in_bins(DetailedPlaceDB<T> db,
                                          int* node_count_map) {
  for (int node_id = blockIdx.x * blockDim.x + threadIdx.x;
       node_id < db.num_movable_nodes; node_id += blockDim.x * gridDim.x) {
    int bx = db.pos2bin_x(db.x[node_id]);
    int by = db.pos2bin_y(db.y[node_id]);
    int bin_id = bx * db.num_bins_y + by;
    atomicAdd(node_count_map + bin_id, 1);
  }
}

template <typename T>
int compute_max_num_nodes_per_bin(const DetailedPlaceDB<T>& db) {
  int num_bins = db.num_bins_x * db.num_bins_y;
  int* node_count_map = nullptr;
  allocateCUDA(node_count_map, num_bins, int);

  checkCUDA(cudaMemset(node_count_map, 0, sizeof(int) * num_bins));
  compute_num_nodes_in_bins<<<ceilDiv(db.num_movable_nodes, 256), 256>>>(
      db, node_count_map);

  //int max_num_nodes_per_bin =
  //    thrust::reduce(thrust::device, node_count_map, node_count_map + num_bins,
  //                   0, thrust::maximum<int>());

  int* d_out = NULL; 
  // Determine temporary device storage requirements
  void     *d_temp_storage = NULL;
  size_t   temp_storage_bytes = 0;
  cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, node_count_map, d_out, num_bins);
  // Allocate temporary storage
  checkCUDA(cudaMalloc(&d_temp_storage, temp_storage_bytes));
  checkCUDA(cudaMalloc(&d_out, sizeof(int))); 
  // Run max-reduction
  cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, node_count_map, d_out, num_bins);
    // copy d_out to hpwl  
  int max_num_nodes_per_bin = 0; 
  checkCUDA(cudaMemcpy(&max_num_nodes_per_bin, d_out, sizeof(int), cudaMemcpyDeviceToHost)); 
  destroyCUDA(d_temp_storage); 
  destroyCUDA(d_out); 

  destroyCUDA(node_count_map);

  return max_num_nodes_per_bin;
}

template <typename T>
int globalSwapCUDALauncher(DetailedPlaceDB<T> db, int batch_size, int max_iters,
                           int num_threads) {
  dreamplacePrint(
      kDEBUG, "bins %dx%d, bin sizes %gx%g, die size %g, %g, %g, %g\n",
      db.num_bins_x, db.num_bins_y, (float)db.bin_size_x, (float)db.bin_size_y,
      (float)db.xl, (float)db.yl, (float)db.xh, (float)db.yh);
  CPUTimer::hr_clock_rep total_time_start, total_time_stop;
  CPUTimer::hr_clock_rep kernel_time_start, kernel_time_stop;
  CPUTimer::hr_clock_rep iter_time_start, iter_time_stop;
  total_time_start = CPUTimer::getGlobaltime();

  SwapState<T> state;

  const float stop_threshold = 0.1 / 100;
  state.batch_size = batch_size;
  int max_num_nodes_per_bin = compute_max_num_nodes_per_bin(db);
  state.max_num_candidates = max_num_nodes_per_bin * 5;
  state.max_num_candidates_all = state.batch_size * state.max_num_candidates;
  dreamplacePrint(kDEBUG,
                  "batch_size = %d, max_num_nodes_per_bin = %d, "
                  "max_num_candidates = %d, max_num_candidates_all = %d\n",
                  state.batch_size, max_num_nodes_per_bin,
                  state.max_num_candidates, state.max_num_candidates_all);
  state.search_bin_strategy = 1;
  // use fast mode for small designs, because extra memory is required
  long estimate_memory_usage =
      db.num_nodes * MAX_NODE_DEGREE * sizeof(int)  // size of node2net_map
      + db.num_nets * MAX_NET_DEGREE *
            sizeof(NodePinPair<T>)  // size of net2nodepin_map
      ;
  if (estimate_memory_usage < 4e9)  // use 4GB as a switch threshold
  {
    dreamplacePrint(kDEBUG,
                    "estimate_memory_usage = %ld, use fast pair HPWL "
                    "computation strategy requires additional memory\n",
                    estimate_memory_usage);
    state.pair_hpwl_computing_strategy = 1;
  } else {
    dreamplacePrint(kDEBUG,
                    "estimate_memory_usage = %ld, use general pair HPWL\n",
                    estimate_memory_usage);
    state.pair_hpwl_computing_strategy = 0;
  }

  // fix random seed
  std::srand(1000);

  // allocate temporary memory to CPU
  // add dummy cells for xl and xh
  std::vector<T> host_x(db.num_nodes + 2);
  std::vector<T> host_y(db.num_nodes + 2);
  std::vector<T> host_node_size_x(db.num_nodes + 2);
  std::vector<T> host_node_size_y(db.num_nodes + 2);
  host_x[db.num_nodes] = db.xl - 1;
  host_y[db.num_nodes] = db.yl;
  host_node_size_x[db.num_nodes] = 1;
  host_node_size_y[db.num_nodes] = db.yh - db.yl;
  host_x[db.num_nodes + 1] = db.xh;
  host_y[db.num_nodes + 1] = db.yl;
  host_node_size_x[db.num_nodes + 1] = 1;
  host_node_size_y[db.num_nodes + 1] = db.yh - db.yl;
  checkCUDA(cudaMemcpy(host_x.data(), db.x, sizeof(T) * db.num_nodes,
                       cudaMemcpyDeviceToHost));
  checkCUDA(cudaMemcpy(host_y.data(), db.y, sizeof(T) * db.num_nodes,
                       cudaMemcpyDeviceToHost));
  checkCUDA(cudaMemcpy(host_node_size_x.data(), db.node_size_x,
                       sizeof(T) * db.num_nodes, cudaMemcpyDeviceToHost));
  checkCUDA(cudaMemcpy(host_node_size_y.data(), db.node_size_y,
                       sizeof(T) * db.num_nodes, cudaMemcpyDeviceToHost));

  // distribute cells to rows on host
  // copy cell locations from device to host
  std::vector<std::vector<int>> host_row2node_map(db.num_sites_y);
  std::vector<RowMapIndex> host_node2row_map(db.num_movable_nodes);
  std::vector<Space<T>> host_spaces(db.num_movable_nodes);
  db.make_row2node_map_with_spaces(host_x.data(), host_y.data(),
                                   host_node_size_x.data(),
                                   host_node_size_y.data(), host_row2node_map,
                                   host_node2row_map, host_spaces, num_threads);
  // distribute movable cells to bins on host
  // bin map is column-major
  std::vector<std::vector<int>> host_bin2node_map(db.num_bins_x *
                                                  db.num_bins_y);
  std::vector<BinMapIndex> host_node2bin_map(db.num_movable_nodes);
  db.make_bin2node_map(host_x.data(), host_y.data(), host_node_size_x.data(),
                       host_node_size_y.data(), host_bin2node_map,
                       host_node2bin_map);

  // initialize SwapState
  std::vector<int> host_ordered_nodes;
  host_ordered_nodes.reserve(db.num_movable_nodes);
  // std::iota(host_ordered_nodes.begin(), host_ordered_nodes.end(), 0);
  // reorder such that a batch of cells are distributed to different bins
  int sub_id_counter = 0;
  while ((int)host_ordered_nodes.size() < db.num_movable_nodes) {
    for (int i = 0; i < state.batch_size; ++i) {
      for (unsigned int j = i; j < host_bin2node_map.size();
           j += state.batch_size) {
        auto const& bin2nodes = host_bin2node_map[j];
        if (sub_id_counter < bin2nodes.size()) {
          host_ordered_nodes.push_back(bin2nodes[sub_id_counter]);
        }
      }
    }
    ++sub_id_counter;
  }
  allocateCopyCUDA(state.ordered_nodes, host_ordered_nodes.data(),
                   db.num_movable_nodes);
  // allocateCUDA(state.ordered_nodes, db.num_movable_nodes, int);
  // iota<<<ceilDiv(db.num_movable_nodes, 512), 512>>>(state.ordered_nodes,
  // db.num_movable_nodes);

  state.row2node_map.initialize(host_row2node_map);
  allocateCopyCUDA(state.node2row_map, host_node2row_map.data(),
                   host_node2row_map.size());
  allocateCopyCUDA(state.spaces, host_spaces.data(), host_spaces.size());

  state.bin2node_map.initialize(host_bin2node_map);
  ;
  allocateCopyCUDA(state.node2bin_map, host_node2bin_map.data(),
                   host_node2bin_map.size());

  allocateCUDA(state.candidates, state.max_num_candidates_all,
               SwapCandidate<T>);
  allocateCUDA(state.search_bins, db.num_movable_nodes, int);
  allocateCUDA(state.net_hpwls, db.num_nets,
               typename std::remove_pointer<decltype(state.net_hpwls)>::type);
  allocateCUDA(state.node_markers, db.num_movable_nodes, unsigned char);
  checkCUDA(cudaMemset(state.node_markers, 0,
                       sizeof(unsigned char) * db.num_movable_nodes));

  if (state.pair_hpwl_computing_strategy) {
    // initNode2NetPinMap(state.node2netpin_map, db);
    initNode2NetMap(state.node2net_map, db);
    initNet2NodePinMap(state.net2nodepin_map, db);
  }

  kernel_time_start = CPUTimer::getGlobaltime();
  double hpwls[max_iters + 1];
  hpwls[0] = compute_total_hpwl(db, db.x, db.y, state.net_hpwls);
  dreamplacePrint(kINFO, "initial hpwl = %.3f\n", hpwls[0]);
  for (int iter = 0; iter < max_iters; ++iter) {
    iter_time_start = CPUTimer::getGlobaltime();
    global_swap
#ifdef DYNAMIC
        <<<1, 1>>>
#endif
        (db, state);
    checkCUDA(cudaDeviceSynchronize());
    iter_time_stop = CPUTimer::getGlobaltime();
    dreamplacePrint(
        kINFO, "Iteration time(ms) \t %g\n",
        CPUTimer::getTimerPeriod() * (iter_time_stop - iter_time_start));

    hpwls[iter + 1] = compute_total_hpwl(db, db.x, db.y, state.net_hpwls);
    dreamplacePrint(kINFO, "iteration %d: hpwl %.3f => %.3f (imp. %g%%)\n",
                    iter, hpwls[0], hpwls[iter + 1],
                    (1.0 - hpwls[iter + 1] / (double)hpwls[0]) * 100);
    state.search_bin_strategy = !state.search_bin_strategy;

    if ((iter & 1) &&
        hpwls[iter] - hpwls[iter - 1] > -stop_threshold * hpwls[0]) {
      break;
    }
  }
  checkCUDA(cudaDeviceSynchronize());
  kernel_time_stop = CPUTimer::getGlobaltime();

  // destroy SwapState
  destroyCUDA(state.ordered_nodes);
  state.row2node_map.destroy();
  destroyCUDA(state.node2row_map);
  destroyCUDA(state.spaces);
  state.bin2node_map.destroy();
  destroyCUDA(state.node2bin_map);
  destroyCUDA(state.candidates);
  destroyCUDA(state.search_bins);
  destroyCUDA(state.net_hpwls);
  destroyCUDA(state.node_markers);

  if (state.pair_hpwl_computing_strategy) {
    state.node2net_map.destroy();
    state.net2nodepin_map.destroy();
  }

  checkCUDA(cudaDeviceSynchronize());
  total_time_stop = CPUTimer::getGlobaltime();

  dreamplacePrint(
      kINFO, "Kernel time: %g ms\n",
      CPUTimer::getTimerPeriod() * (kernel_time_stop - kernel_time_start));
  dreamplacePrint(
      kINFO, "Global swap time: %g ms\n",
      CPUTimer::getTimerPeriod() * (total_time_stop - total_time_start));

  return 0;
}

#define REGISTER_KERNEL_LAUNCHER(T)                                            \
  template int globalSwapCUDALauncher<T>(                                      \
      DetailedPlaceDB<T> db, int batch_size, int max_iters, int num_threads); 

REGISTER_KERNEL_LAUNCHER(int);
REGISTER_KERNEL_LAUNCHER(float);
REGISTER_KERNEL_LAUNCHER(double);

DREAMPLACE_END_NAMESPACE
