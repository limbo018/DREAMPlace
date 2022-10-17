/**
 * @file   rudy_cuda_kernel.cu
 * @author Zixuan Jiang, Jiaqi Gu, Yibo Lin
 * @date   Dec 2019
 * @brief  Compute the RUDY/RISA map for routing demand. 
 *         A routing/pin utilization estimator based on the following two papers
 *         "Fast and Accurate Routing Demand Estimation for efficient Routability-driven Placement", by Peter Spindler, DATE'07
 *         "RISA: Accurate and Efficient Placement Routability Modeling", by Chih-liang Eric Cheng, ICCAD'94
 */

#include "utility/src/utils.cuh"
#include "pinrudy/src/parameters.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
inline __device__ DEFINE_NET_WIRING_DISTRIBUTION_MAP_WEIGHT;

template <typename T, typename AtomicOp>
__global__ void pinRudy(const T *pin_pos_x,
                              const T *pin_pos_y,
                              const int *netpin_start,
                              const int *flat_netpin,
                              const T *net_weights,
                              T bin_size_x, T bin_size_y,
                              T xl, T yl, T xh, T yh,

                              int num_bins_x, int num_bins_y,
                              int num_nets, AtomicOp atomic_add_op,
                              typename AtomicOp::type *horizontal_utilization_map,
                              typename AtomicOp::type *vertical_utilization_map)
{
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < num_nets)
    {
        const int start = netpin_start[i];
        const int end = netpin_start[i + 1];

        T x_max = -cuda::numeric_limits<T>::max();
        T x_min = cuda::numeric_limits<T>::max();
        T y_max = -cuda::numeric_limits<T>::max();
        T y_min = cuda::numeric_limits<T>::max();

        for (int j = start; j < end; ++j)
        {
            int pin_id = flat_netpin[j];
            const T xx = pin_pos_x[pin_id];
            x_max = DREAMPLACE_STD_NAMESPACE::max(xx, x_max);
            x_min = DREAMPLACE_STD_NAMESPACE::min(xx, x_min);
            const T yy = pin_pos_y[pin_id];
            y_max = DREAMPLACE_STD_NAMESPACE::max(yy, y_max);
            y_min = DREAMPLACE_STD_NAMESPACE::min(yy, y_min);
        }

        // compute the bin box that this net will affect
        int bin_index_xl = int((x_min - xl) / bin_size_x);
        int bin_index_xh = int((x_max - xl) / bin_size_x) + 1;
        bin_index_xl = DREAMPLACE_STD_NAMESPACE::max(bin_index_xl, 0);
        bin_index_xh = DREAMPLACE_STD_NAMESPACE::min(bin_index_xh, num_bins_x);

        int bin_index_yl = int((y_min - yl) / bin_size_y);
        int bin_index_yh = int((y_max - yl) / bin_size_y) + 1;
        bin_index_yl = DREAMPLACE_STD_NAMESPACE::max(bin_index_yl, 0);
        bin_index_yh = DREAMPLACE_STD_NAMESPACE::min(bin_index_yh, num_bins_y);

        T wt = netWiringDistributionMapWeight<T>(end - start);
        if (net_weights)
        {
            wt *= net_weights[i];
        }

        for (int j = start; j < end; ++j) {
            int pin_id = flat_netpin[j];
            const T xx = pin_pos_x[pin_id];
            const T yy = pin_pos_y[pin_id];

            int bin_index_x = int((xx - xl) / bin_size_x);
            int bin_index_y = int((yy - yl) / bin_size_y);

            
            int index = bin_index_x * num_bins_y + bin_index_y;

            atomic_add_op(&horizontal_utilization_map[index], wt / (bin_index_xh - bin_index_xl + cuda::numeric_limits<T>::epsilon()));
            atomic_add_op(&vertical_utilization_map[index], wt / (bin_index_yh - bin_index_yl + cuda::numeric_limits<T>::epsilon()));
        }
    }
}

// fill the demand map net by net
template <typename T>
int pinRudyCudaLauncher(const T *pin_pos_x,
                              const T *pin_pos_y,
                              const int *netpin_start,
                              const int *flat_netpin,
                              const T *net_weights,
                              T bin_size_x, T bin_size_y,
                              T xl, T yl, T xh, T yh,

                              int num_bins_x, int num_bins_y,
                              int num_nets, bool deterministic_flag,
                              T *horizontal_utilization_map,
                              T *vertical_utilization_map)
{
  if (deterministic_flag)  // deterministic implementation using unsigned long
                           // as fixed point number
  {
    // total die area
    double diearea = (xh - xl) * (yh - yl);
    int integer_bits = max((int)ceil(log2(diearea)) + 1, 32);
    int fraction_bits = max(64 - integer_bits, 0);
    unsigned long long int scale_factor = (1UL << fraction_bits);
    int num_bins = num_bins_x * num_bins_y;
    unsigned long long int *buf_map = NULL;
    allocateCUDA(buf_map, num_bins*2, unsigned long long int);
    unsigned long long int *horizontal_buf_map = buf_map;
    unsigned long long int *vertical_buf_map = buf_map + num_bins;

    AtomicAddCUDA<unsigned long long int> atomic_add_op(scale_factor);

    int thread_count = 512;
    int block_count = ceilDiv(num_bins, thread_count);
    copyScaleArray<<<block_count, thread_count>>>(
        horizontal_buf_map, horizontal_utilization_map, scale_factor, num_bins);
    copyScaleArray<<<block_count, thread_count>>>(
        vertical_buf_map, vertical_utilization_map, scale_factor, num_bins);

    block_count = ceilDiv(num_nets, thread_count);
    pinRudy<<<block_count, thread_count>>>(
            pin_pos_x,
            pin_pos_y,
            netpin_start,
            flat_netpin,
            net_weights,
            bin_size_x, bin_size_y,
            xl, yl, xh, yh,
            num_bins_x, num_bins_y,
            num_nets,
            atomic_add_op, 
            horizontal_buf_map,
            vertical_buf_map
            );

    block_count = ceilDiv(num_bins, thread_count);
    copyScaleArray<<<block_count, thread_count>>>(
        horizontal_utilization_map, horizontal_buf_map, T(1.0 / scale_factor), num_bins);
    copyScaleArray<<<block_count, thread_count>>>(
        vertical_utilization_map, vertical_buf_map, T(1.0 / scale_factor), num_bins);

    destroyCUDA(buf_map);
  } else {
    AtomicAddCUDA<T> atomic_add_op;
    int thread_count = 512;
    int block_count = ceilDiv(num_nets, thread_count);
    pinRudy<<<block_count, thread_count>>>(
            pin_pos_x,
            pin_pos_y,
            netpin_start,
            flat_netpin,
            net_weights,
            bin_size_x, bin_size_y,
            xl, yl, xh, yh,
            num_bins_x, num_bins_y,
            num_nets,
            atomic_add_op, 
            horizontal_utilization_map,
            vertical_utilization_map
            );
  }
  return 0;
}

#define REGISTER_KERNEL_LAUNCHER(T)                                           \
    template int pinRudyCudaLauncher<T>(const T *pin_pos_x,             \
                                              const T *pin_pos_y,             \
                                              const int *netpin_start,        \
                                              const int *flat_netpin,         \
                                              const T *net_weights,           \
                                              T bin_size_x, T bin_size_y,     \
                                              T xl, T yl, T xh, T yh,         \
                                                                              \
                                              int num_bins_x, int num_bins_y, \
                                              int num_nets,                   \
                                              bool deterministic_flag,        \
                                              T *horizontal_utilization_map,   \
                                              T *vertical_utilization_map);  \

REGISTER_KERNEL_LAUNCHER(float);
REGISTER_KERNEL_LAUNCHER(double);

DREAMPLACE_END_NAMESPACE
