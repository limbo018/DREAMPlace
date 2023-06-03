/**
 * @file   electric_force_cuda_kernel.cu
 * @author Yibo Lin
 * @date   Aug 2018
 */
#include <float.h>
#include <math.h>
#include <stdio.h>
#include "cuda_runtime.h"
#include "utility/src/utils.cuh"
// local dependency
#include "electric_potential/src/density_function.h"

DREAMPLACE_BEGIN_NAMESPACE

/// define triangle_density_function
template <typename T>
inline __device__ DEFINE_TRIANGLE_DENSITY_FUNCTION(T);

template <typename T>
__global__ void __launch_bounds__(1024, 8) computeElectricForce(
    int num_bins_x, int num_bins_y, const T *field_map_x_tensor,
    const T *field_map_y_tensor, const T *x_tensor, const T *y_tensor,
    const T *node_size_x_clamped_tensor, const T *node_size_y_clamped_tensor,
    const T *offset_x_tensor, const T *offset_y_tensor, const T *ratio_tensor,
    const T *bin_center_x_tensor, const T *bin_center_y_tensor, T xl, T yl,
    T xh, T yh, const T half_bin_size_x, const T half_bin_size_y,
    const T bin_size_x, const T bin_size_y, const T inv_bin_size_x,
    const T inv_bin_size_y, int num_nodes, T *grad_x_tensor, T *grad_y_tensor,
    const int *sorted_node_map  ///< can be NULL if not sorted
) {
  int index = blockIdx.x * blockDim.z + threadIdx.z;
  if (index < num_nodes) {
    int i = (sorted_node_map) ? sorted_node_map[index] : index;

    // use stretched node size
    T node_size_x = node_size_x_clamped_tensor[i];
    T node_size_y = node_size_y_clamped_tensor[i];
    T node_x = x_tensor[i] + offset_x_tensor[i];
    T node_y = y_tensor[i] + offset_y_tensor[i];
    T ratio = ratio_tensor[i];

    // Yibo: looks very weird implementation, but this is how RePlAce implements
    // it Zixuan and Jiaqi: use the common practice of floor
    int bin_index_xl = int((node_x - xl) * inv_bin_size_x);
    int bin_index_xh =
        int(((node_x + node_size_x - xl) * inv_bin_size_x)) + 1;  // exclusive
    bin_index_xl = DREAMPLACE_STD_NAMESPACE::max(bin_index_xl, 0);
    bin_index_xh = DREAMPLACE_STD_NAMESPACE::min(bin_index_xh, num_bins_x);

    int bin_index_yl = int((node_y - yl) * inv_bin_size_y);
    int bin_index_yh =
        int(((node_y + node_size_y - yl) * inv_bin_size_y)) + 1;  // exclusive
    bin_index_yl = DREAMPLACE_STD_NAMESPACE::max(bin_index_yl, 0);
    bin_index_yh = DREAMPLACE_STD_NAMESPACE::min(bin_index_yh, num_bins_y);

    // blockDim.x * blockDim.y threads will be used to update one node
    // shared memory is used to privatize the atomic memory access to thread
    // block
    extern __shared__ unsigned char s_xy[];
    T *s_x = (T *)s_xy;
    T *s_y = s_x + blockDim.z;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      s_x[threadIdx.z] = s_y[threadIdx.z] = 0;
    }
    __syncthreads();

    T tmp_x, tmp_y;
    tmp_x = 0;
    tmp_y = 0;

    // update density potential map
    for (int k = bin_index_xl + threadIdx.y; k < bin_index_xh;
         k += blockDim.y) {
      T px = triangle_density_function(node_x, node_size_x, xl, k, bin_size_x);

      for (int h = bin_index_yl + threadIdx.x; h < bin_index_yh;
           h += blockDim.x) {
        T py =
            triangle_density_function(node_y, node_size_y, yl, h, bin_size_y);
        T area = px * py;

        int idx = k * num_bins_y + h;
        tmp_x += area * field_map_x_tensor[idx];
        tmp_y += area * field_map_y_tensor[idx];
      }
    }

    atomicAdd(&s_x[threadIdx.z], tmp_x * ratio);
    atomicAdd(&s_y[threadIdx.z], tmp_y * ratio);
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0) {
      grad_x_tensor[i] = s_x[threadIdx.z];
      grad_y_tensor[i] = s_y[threadIdx.z];
    }
  }
}

/// @brief An unrolled way to compute the force.
/// Currently it is not as efficient as computeElectricForce,
/// it has the potential to be better.
/// It is not used for now.
template <typename T>
__global__ void computeElectricForceUnroll(
    int num_bins_x, int num_bins_y, const T *field_map_x_tensor,
    const T *field_map_y_tensor, const T *x_tensor, const T *y_tensor,
    const T *node_size_x_clamped_tensor, const T *node_size_y_clamped_tensor,
    const T *offset_x_tensor, const T *offset_y_tensor, const T *ratio_tensor,
    const T *bin_center_x_tensor, const T *bin_center_y_tensor, T xl, T yl,
    T xh, T yh, const T half_bin_size_x, const T half_bin_size_y,
    const T bin_size_x, const T bin_size_y, const T inv_bin_size_x,
    const T inv_bin_size_y, int num_nodes, T *grad_x_tensor, T *grad_y_tensor,
    const int *sorted_node_map  ///< can be NULL if not sorted
) {
  int index = blockIdx.x * blockDim.y + threadIdx.y;
  if (index < num_nodes) {
    int i = (sorted_node_map) ? sorted_node_map[index] : index;

    // stretch node size to bin size
    T node_size_x = node_size_x_clamped_tensor[i];
    T node_size_y = node_size_y_clamped_tensor[i];
    T node_x = x_tensor[i] + offset_x_tensor[i];
    T node_y = y_tensor[i] + offset_y_tensor[i];
    T ratio = ratio_tensor[i];

    // Yibo: looks very weird implementation, but this is how RePlAce implements
    // it Zixuan and Jiaqi: use the common practice of floor
    int bin_index_xl = int((node_x - xl) * inv_bin_size_x);
    int bin_index_xh =
        int(((node_x + node_size_x - xl) * inv_bin_size_x));  // inclusive
    bin_index_xl = DREAMPLACE_STD_NAMESPACE::max(bin_index_xl, 0);
    bin_index_xh = DREAMPLACE_STD_NAMESPACE::min(bin_index_xh, num_bins_x - 1);

    int bin_index_yl = int((node_y - yl) * inv_bin_size_y);
    int bin_index_yh =
        int(((node_y + node_size_y - yl) * inv_bin_size_y));  // inclusive
    bin_index_yl = DREAMPLACE_STD_NAMESPACE::max(bin_index_yl, 0);
    bin_index_yh = DREAMPLACE_STD_NAMESPACE::min(bin_index_yh, num_bins_y - 1);

    int k, h;

    int cond =
        ((bin_index_xl == bin_index_xh) << 1) | (bin_index_yl == bin_index_yh);
    switch (cond) {
      case 0: {
        // blockDim.x threads will be used to update one node
        // shared memory is used to privatize the atomic memory access to thread
        // block
        extern __shared__ unsigned char shared_memory[];
        T *s_x = (T *)shared_memory;
        T *s_y = s_x + blockDim.y;
        if (threadIdx.x == 0) {
          s_x[threadIdx.y] = s_y[threadIdx.y] = 0;
        }
        __syncthreads();

        T tmp_x = 0;
        T tmp_y = 0;

        T px_c = bin_size_x;

        T py_l = yl + bin_index_yl * bin_size_y + bin_size_y - node_y;
        T py_c = bin_size_y;
        T py_h = node_y + node_size_y - (bin_index_yh * bin_size_y + yl);

        T area_xc_yl = px_c * py_l;
        T area_xc_yc = px_c * py_c;
        T area_xc_yh = px_c * py_h;

        k = bin_index_xl;
        if (threadIdx.x == 0) {
          T px_l = xl + bin_index_xl * bin_size_x + bin_size_x - node_x;
          T area_xl_yl = px_l * py_l;
          T area_xl_yc = px_l * py_c;
          T area_xl_yh = px_l * py_h;

          h = bin_index_yl;
          tmp_x = area_xl_yl * field_map_x_tensor[k * num_bins_y + h];
          tmp_y = area_xl_yl * field_map_y_tensor[k * num_bins_y + h];
          for (++h; h < bin_index_yh; ++h) {
            tmp_x += area_xl_yc * field_map_x_tensor[k * num_bins_y + h];
            tmp_y += area_xl_yc * field_map_y_tensor[k * num_bins_y + h];
          }
          tmp_x += area_xl_yh * field_map_x_tensor[k * num_bins_y + h];
          tmp_y += area_xl_yh * field_map_y_tensor[k * num_bins_y + h];
          k += blockDim.x;
        }

        for (k += threadIdx.x; k < bin_index_xh; k += blockDim.x) {
          h = bin_index_yl;
          tmp_x += area_xc_yl * field_map_x_tensor[k * num_bins_y + h];
          tmp_y += area_xc_yl * field_map_y_tensor[k * num_bins_y + h];
          for (++h; h < bin_index_yh; ++h) {
            tmp_x += area_xc_yc * field_map_x_tensor[k * num_bins_y + h];
            tmp_y += area_xc_yc * field_map_y_tensor[k * num_bins_y + h];
          }
          tmp_x += area_xc_yh * field_map_x_tensor[k * num_bins_y + h];
          tmp_y += area_xc_yh * field_map_y_tensor[k * num_bins_y + h];
        }

        if (k == bin_index_xh) {
          T px_h = node_x + node_size_x - (bin_index_xh * bin_size_x + xl);
          T area_xh_yl = px_h * py_l;
          T area_xh_yc = px_h * py_c;
          T area_xh_yh = px_h * py_h;

          h = bin_index_yl;
          tmp_x += area_xh_yl * field_map_x_tensor[k * num_bins_y + h];
          tmp_y += area_xh_yl * field_map_y_tensor[k * num_bins_y + h];
          for (++h; h < bin_index_yh; ++h) {
            tmp_x += area_xh_yc * field_map_x_tensor[k * num_bins_y + h];
            tmp_y += area_xh_yc * field_map_y_tensor[k * num_bins_y + h];
          }
          tmp_x += area_xh_yh * field_map_x_tensor[k * num_bins_y + h];
          tmp_y += area_xh_yh * field_map_y_tensor[k * num_bins_y + h];
        }

        atomicAdd(&s_x[threadIdx.y], tmp_x * ratio);
        atomicAdd(&s_y[threadIdx.y], tmp_y * ratio);
        __syncthreads();

        if (threadIdx.x == 0) {
          grad_x_tensor[i] = s_x[threadIdx.y];
          grad_y_tensor[i] = s_y[threadIdx.y];
        }

        return;
      }
      case 1: {
        extern __shared__ unsigned char shared_memory[];
        T *s_x = (T *)shared_memory;
        T *s_y = s_x + blockDim.y;
        if (threadIdx.x == 0) {
          s_x[threadIdx.y] = s_y[threadIdx.y] = 0;
        }
        __syncthreads();

        T tmp_x = 0;
        T tmp_y = 0;

        T py = yl + bin_index_yl * bin_size_y + bin_size_y - node_y;
        h = bin_index_yl;

        k = bin_index_xl;
        if (threadIdx.x == 0) {
          T px_l = xl + bin_index_xl * bin_size_x + bin_size_x - node_x;
          T area_xl = px_l * py;
          tmp_x = area_xl * field_map_x_tensor[k * num_bins_y + h];
          tmp_y = area_xl * field_map_y_tensor[k * num_bins_y + h];
          k += blockDim.x;
        }

        T px_c = bin_size_x;
        T area_xc = px_c * py;
        for (k += threadIdx.x; k < bin_index_xh; k += blockDim.x) {
          tmp_x += area_xc * field_map_x_tensor[k * num_bins_y + h];
          tmp_y += area_xc * field_map_y_tensor[k * num_bins_y + h];
        }

        if (k == bin_index_xh) {
          T px_h = node_x + node_size_x - (bin_index_xh * bin_size_x + xl);
          T area_xh = px_h * py;
          tmp_x += area_xh * field_map_x_tensor[k * num_bins_y + h];
          tmp_y += area_xh * field_map_y_tensor[k * num_bins_y + h];
        }

        atomicAdd(&s_x[threadIdx.y], tmp_x * ratio);
        atomicAdd(&s_y[threadIdx.y], tmp_y * ratio);
        __syncthreads();

        if (threadIdx.x == 0) {
          grad_x_tensor[i] = s_x[threadIdx.y];
          grad_y_tensor[i] = s_y[threadIdx.y];
        }

        return;
      }
      case 2: {
        extern __shared__ unsigned char shared_memory[];
        T *s_x = (T *)shared_memory;
        T *s_y = s_x + blockDim.y;
        if (threadIdx.x == 0) {
          s_x[threadIdx.y] = s_y[threadIdx.y] = 0;
        }
        __syncthreads();

        T tmp_x = 0;
        T tmp_y = 0;

        T px = xl + bin_index_xl * bin_size_x + bin_size_x - node_x;
        k = bin_index_xl;

        h = bin_index_yl;
        if (threadIdx.x == 0) {
          T py_l = yl + bin_index_yl * bin_size_y + bin_size_y - node_y;
          T area_yl = px * py_l;
          tmp_x = area_yl * field_map_x_tensor[k * num_bins_y + h];
          tmp_y = area_yl * field_map_y_tensor[k * num_bins_y + h];
          h += blockDim.x;
        }

        T py_c = bin_size_y;
        T area_yc = px * py_c;
        for (h += threadIdx.x; h < bin_index_yh; h += blockDim.x) {
          tmp_x += area_yc * field_map_x_tensor[k * num_bins_y + h];
          tmp_y += area_yc * field_map_y_tensor[k * num_bins_y + h];
        }

        if (h == bin_index_yh) {
          T py_h = node_y + node_size_y - (bin_index_yh * bin_size_y + yl);
          T area_yh = px * py_h;
          tmp_x += area_yh * field_map_x_tensor[k * num_bins_y + h];
          tmp_y += area_yh * field_map_y_tensor[k * num_bins_y + h];
        }

        atomicAdd(&s_x[threadIdx.y], tmp_x * ratio);
        atomicAdd(&s_y[threadIdx.y], tmp_y * ratio);
        __syncthreads();

        if (threadIdx.x == 0) {
          grad_x_tensor[i] = s_x[threadIdx.y];
          grad_y_tensor[i] = s_y[threadIdx.y];
        }

        return;
      }
      case 3: {
        if (threadIdx.x == 0) {
          T px = xl + bin_index_xl * bin_size_x + bin_size_x - node_x;
          T py = yl + bin_index_yl * bin_size_y + bin_size_y - node_y;
          T area_by_ratio = px * py * ratio;

          k = bin_index_xl;
          h = bin_index_yl;

          grad_x_tensor[i] =
              area_by_ratio * field_map_x_tensor[k * num_bins_y + h];
          grad_y_tensor[i] =
              area_by_ratio * field_map_y_tensor[k * num_bins_y + h];
        }

        return;
      }
      default:
        assert(0);
    }
  }
}

template <typename T>
__global__ void computeElectricForceSimpleLikeCPU(
    int num_bins_x, int num_bins_y, int num_impacted_bins_x,
    int num_impacted_bins_y, const T *field_map_x_tensor,
    const T *field_map_y_tensor, const T *x_tensor, const T *y_tensor,
    const T *node_size_x_clamped_tensor, const T *node_size_y_clamped_tensor,
    const T *offset_x_tensor, const T *offset_y_tensor, const T *ratio_tensor,
    const T *bin_center_x_tensor, const T *bin_center_y_tensor, T xl, T yl,
    T xh, T yh, T bin_size_x, T bin_size_y, int num_nodes, T *grad_x_tensor,
    T *grad_y_tensor) {
  // density_map_tensor should be initialized outside

  T inv_bin_size_x = 1.0 / bin_size_x;
  T inv_bin_size_y = 1.0 / bin_size_y;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_nodes) {
    // use stretched node size
    T node_size_x = node_size_x_clamped_tensor[i];
    T node_size_y = node_size_y_clamped_tensor[i];
    T node_x = x_tensor[i] + offset_x_tensor[i];
    T node_y = y_tensor[i] + offset_y_tensor[i];
    T ratio = ratio_tensor[i];

    // Yibo: looks very weird implementation, but this is how RePlAce implements
    // it the common practice should be floor Zixuan and Jiaqi: use the common
    // practice of floor
    int bin_index_xl = int((node_x - xl) * inv_bin_size_x);
    int bin_index_xh =
        int(((node_x + node_size_x - xl) * inv_bin_size_x)) + 1;  // exclusive
    bin_index_xl = DREAMPLACE_STD_NAMESPACE::max(bin_index_xl, 0);
    bin_index_xh = DREAMPLACE_STD_NAMESPACE::min(bin_index_xh, num_bins_x);
    // int bin_index_xh = bin_index_xl+num_impacted_bins_x;

    // Yibo: looks very weird implementation, but this is how RePlAce implements
    // it the common practice should be floor Zixuan and Jiaqi: use the common
    // practice of floor
    int bin_index_yl = int((node_y - yl) * inv_bin_size_y);
    int bin_index_yh =
        int(((node_y + node_size_y - yl) * inv_bin_size_y)) + 1;  // exclusive
    bin_index_yl = DREAMPLACE_STD_NAMESPACE::max(bin_index_yl, 0);
    bin_index_yh = DREAMPLACE_STD_NAMESPACE::min(bin_index_yh, num_bins_y);
    // int bin_index_yh = bin_index_yl+num_impacted_bins_y;

    T &gx = grad_x_tensor[i];
    T &gy = grad_y_tensor[i];
    gx = 0;
    gy = 0;
    // update density potential map
    for (int k = bin_index_xl; k < bin_index_xh; ++k) {
      T px = triangle_density_function(node_x, node_size_x, xl, k, bin_size_x);
      for (int h = bin_index_yl; h < bin_index_yh; ++h) {
        T py =
            triangle_density_function(node_y, node_size_y, yl, h, bin_size_y);
        T area = px * py;

        int idx = k * num_bins_y + h;
        gx += area * field_map_x_tensor[idx];
        gy += area * field_map_y_tensor[idx];
      }
    }
    gx *= ratio;
    gy *= ratio;
  }
}

template <typename T>
int computeElectricForceCudaLauncher(
    int num_bins_x, int num_bins_y, int num_impacted_bins_x,
    int num_impacted_bins_y, const T *field_map_x_tensor,
    const T *field_map_y_tensor, const T *x_tensor, const T *y_tensor,
    const T *node_size_x_clamped_tensor, const T *node_size_y_clamped_tensor,
    const T *offset_x_tensor, const T *offset_y_tensor, const T *ratio_tensor,
    const T *bin_center_x_tensor, const T *bin_center_y_tensor, T xl, T yl,
    T xh, T yh, T bin_size_x, T bin_size_y, int num_nodes, bool deterministic_flag, 
    T *grad_x_tensor, T *grad_y_tensor, const int *sorted_node_map) {
  int thread_count = 64;
  int block_count_nodes = ceilDiv(num_nodes, thread_count);

  if (deterministic_flag) {
    computeElectricForceSimpleLikeCPU<<<block_count_nodes, thread_count>>>(
        num_bins_x, num_bins_y,
        num_impacted_bins_x, num_impacted_bins_y,
        field_map_x_tensor, field_map_y_tensor,
        x_tensor, y_tensor,
        node_size_x_clamped_tensor, node_size_y_clamped_tensor,
        offset_x_tensor, offset_y_tensor,
        ratio_tensor,
        bin_center_x_tensor, bin_center_y_tensor,
        xl, yl, xh, yh,
        bin_size_x, bin_size_y,
        num_nodes,
        grad_x_tensor, grad_y_tensor);
  } else {
    dim3 blockSize(2, 2, thread_count);
    size_t shared_mem_size = sizeof(T) * thread_count * 2;
    computeElectricForce<<<block_count_nodes, blockSize, shared_mem_size>>>(
        num_bins_x, num_bins_y, field_map_x_tensor, field_map_y_tensor, x_tensor,
        y_tensor, node_size_x_clamped_tensor, node_size_y_clamped_tensor,
        offset_x_tensor, offset_y_tensor, ratio_tensor, bin_center_x_tensor,
        bin_center_y_tensor, xl, yl, xh, yh, bin_size_x / 2, bin_size_y / 2,
        bin_size_x, bin_size_y, 1 / bin_size_x, 1 / bin_size_y, num_nodes,
        grad_x_tensor, grad_y_tensor, sorted_node_map);
  }

  return 0;
}

#define REGISTER_KERNEL_LAUNCHER(T)                                            \
  template int computeElectricForceCudaLauncher<T>(                            \
      int num_bins_x, int num_bins_y, int num_impacted_bins_x,                 \
      int num_impacted_bins_y, const T *field_map_x_tensor,                    \
      const T *field_map_y_tensor, const T *x_tensor, const T *y_tensor,       \
      const T *node_size_x_clamped_tensor,                                     \
      const T *node_size_y_clamped_tensor, const T *offset_x_tensor,           \
      const T *offset_y_tensor, const T *ratio_tensor,                         \
      const T *bin_center_x_tensor, const T *bin_center_y_tensor, T xl, T yl,  \
      T xh, T yh, T bin_size_x, T bin_size_y, int num_nodes, bool deterministic_flag, \
      T *grad_x_tensor, T *grad_y_tensor, const int *sorted_node_map); 

REGISTER_KERNEL_LAUNCHER(float);
REGISTER_KERNEL_LAUNCHER(double);

DREAMPLACE_END_NAMESPACE
