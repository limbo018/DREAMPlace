#include <cfloat>
#include <stdio.h>
#include "assert.h"
#include "cuda_runtime.h"
#include "utility/src/print.h"
#include "utility/src/Msg.h"
#include "weighted_average_wirelength/src/functional_cuda.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
int computeWeightedAverageWirelengthCudaLauncher(
    const T *x, const T *y,
    const int *pin2net_map,
    const int *flat_netpin,
    const int *netpin_start,
    const unsigned char *net_mask,
    int num_nets,
    int num_pins,
    const T *inv_gamma,
    T *exp_xy, T *exp_nxy,
    T *exp_xy_sum, T *exp_nxy_sum,
    T *xyexp_xy_sum, T *xyexp_nxy_sum,
    T *partial_wl,
    const T *grad_tensor,
    T *grad_x_tensor, T *grad_y_tensor)
{
    if (grad_tensor)
    {
        int thread_count = 64;
        int block_count_pins = (num_pins - 1 + thread_count) / thread_count;
        dim3 block_size(thread_count, 2, 1);
        // computeWeightedAverageWirelengthGradInterleave<<<block_count_pins, block_size>>>(
        computeWeightedAverageWirelengthGrad<<<block_count_pins, thread_count>>>(
            x, y,
            exp_xy, exp_nxy,
            exp_xy_sum, exp_nxy_sum,
            xyexp_xy_sum, xyexp_nxy_sum,
            pin2net_map,
            net_mask,
            num_nets,
            num_pins,
            inv_gamma,
            grad_tensor,
            grad_x_tensor, grad_y_tensor);
    }
    else
    {
        int thread_count = 64;
        int block_count_nets = (num_nets + thread_count - 1) / thread_count;
        computeWeightedAverageWirelength<<<block_count_nets, thread_count>>>(
            x, y,
            flat_netpin,
            netpin_start,
            net_mask,
            num_nets,
            num_pins,
            inv_gamma,
            exp_xy, exp_nxy,
            exp_xy_sum, exp_nxy_sum,
            xyexp_xy_sum, xyexp_nxy_sum,
            partial_wl);
    }

    return 0;
}

#define REGISTER_KERNEL_LAUNCHER(T)                          \
    int instantiateComputeWeightedAverageWirelengthLauncher( \
        const T *x, const T *y,                              \
        const int *pin2net_map,                              \
        const int *flat_netpin,                              \
        const int *netpin_start,                             \
        const unsigned char *net_mask,                       \
        int num_nets,                                        \
        int num_pins,                                        \
        const T *inv_gamma,                                  \
        T *exp_xy, T *exp_nxy,                               \
        T *exp_xy_sum, T *exp_nxy_sum,                       \
        T *xyexp_xy_sum, T *xyexp_nxy_sum,                   \
        T *partial_wl,                                       \
        const T *grad_tensor,                                \
        T *grad_x_tensor, T *grad_y_tensor)                  \
    {                                                        \
        return computeWeightedAverageWirelengthCudaLauncher( \
            x, y,                                            \
            pin2net_map,                                     \
            flat_netpin,                                     \
            netpin_start,                                    \
            net_mask,                                        \
            num_nets,                                        \
            num_pins,                                        \
            inv_gamma,                                       \
            exp_xy, exp_nxy,                                 \
            exp_xy_sum, exp_nxy_sum,                         \
            xyexp_xy_sum, xyexp_nxy_sum,                     \
            partial_wl,                                      \
            grad_tensor,                                     \
            grad_x_tensor, grad_y_tensor);                   \
    }
REGISTER_KERNEL_LAUNCHER(float);
REGISTER_KERNEL_LAUNCHER(double);

DREAMPLACE_END_NAMESPACE
