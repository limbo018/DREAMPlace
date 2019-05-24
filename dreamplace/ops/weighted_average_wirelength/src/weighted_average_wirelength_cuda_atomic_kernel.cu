#include <cfloat>
#include <stdio.h>
#include "assert.h"
#include "cuda_runtime.h"
#include "utility/src/print.h"
#include "utility/src/Msg.h"
#include "weighted_average_wirelength/src/functional_cuda.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T, typename V>
int computeWeightedAverageWirelengthCudaAtomicLauncher(
    const T *x, const T *y,
    const int *pin2net_map,
    const unsigned char *net_mask,
    int num_nets,
    int num_pins,
    const T *inv_gamma,
    T *exp_xy, T *exp_nxy,
    T *exp_xy_sum, T *exp_nxy_sum,
    T *xyexp_xy_sum, T *xyexp_nxy_sum,
    V *xy_max, V *xy_min,
    T *partial_wl, // wirelength of each net
    const T *grad_tensor,
    T *grad_x_tensor, T *grad_y_tensor // the gradient is partial total wirelength to partial pin position
)
{
    int thread_count = 256;
    int block_count_pins = (num_pins - 1 + thread_count) / thread_count;

    if (grad_tensor)
    {
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
        // compute max and min in one kernel
        computeMaxMin<<<block_count_pins, thread_count>>>(
            x, y,
            pin2net_map,
            net_mask,
            num_pins,
            num_nets,
            xy_max,
            xy_min);
        // compute plus-minus exp, sum of plus-minus exp, sum of x*exp in one CUDA kernels
        // corresponding to the plus and minus a b c kernels in the DREAMPlace paper
        computeABCKernels<<<block_count_pins, thread_count>>>(
            x, y,
            pin2net_map,
            net_mask,
            num_nets,
            num_pins,
            inv_gamma,
            xy_max, xy_min,
            exp_xy, exp_nxy,
            exp_xy_sum, exp_nxy_sum,
            xyexp_xy_sum, xyexp_nxy_sum);
        // compute log sum exp
        int block_count_nets = (num_nets - 1 + thread_count) / thread_count;
        computeXExpSumByExpSum<<<block_count_nets, thread_count>>>(
            xyexp_xy_sum, xyexp_nxy_sum,
            exp_xy_sum, exp_nxy_sum,
            pin2net_map,
            net_mask,
            num_nets,
            partial_wl);

        // I move out the summation to use ATen
        // significant speedup is observed
        //sumArray<<<1, 1>>>(partial_wl, 2*num_nets, wl);
    }
    
    return 0;
}


#define REGISTER_KERNEL_LAUNCHER(T, V) \
    int instantiateComputeWeightedAverageWirelengthAtomicLauncher(\
            const T* x, const T* y, \
            const int* pin2net_map, \
            const unsigned char* net_mask, \
            int num_nets, \
            int num_pins, \
            const T* inv_gamma, \
            T* exp_xy, T* exp_nxy, \
            T* exp_xy_sum, T* exp_nxy_sum,\
            T* xyexp_xy_sum, T* xyexp_nxy_sum, \
            V* xy_max, V* xy_min, \
            T* partial_wl, \
            const T* grad_tensor, \
            T* grad_x_tensor, T* grad_y_tensor \
            )\
    {\
        return computeWeightedAverageWirelengthCudaAtomicLauncher(\
                x, y, \
                pin2net_map, \
                net_mask, \
                num_nets,\
                num_pins,\
                inv_gamma, \
                exp_xy, exp_nxy, \
                exp_xy_sum, exp_nxy_sum, \
                xyexp_xy_sum, xyexp_nxy_sum, \
                xy_max, xy_min, \
                partial_wl, \
                grad_tensor, \
                grad_x_tensor, grad_y_tensor  \
                );\
    }
REGISTER_KERNEL_LAUNCHER(float, int);
REGISTER_KERNEL_LAUNCHER(double, int);

DREAMPLACE_END_NAMESPACE
