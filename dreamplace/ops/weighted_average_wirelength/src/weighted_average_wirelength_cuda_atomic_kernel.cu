#include <cfloat>
#include <stdio.h>
#include "assert.h"
#include "cuda_runtime.h"
#include "utility/src/utils.cuh"
#include "weighted_average_wirelength/src/functional_cuda.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T, typename V>
int computeWeightedAverageWirelengthCudaAtomicLauncher(
    const T *pos, // x then y
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
    V *xy_max, V *xy_min,
    T *partial_wl, // wirelength of each net
    const T *grad_tensor,
    T *grad_x_tensor, T *grad_y_tensor // the gradient is partial total wirelength to partial pin position
)
{
    int thread_count = 64;
    int block_count_pins = (num_pins - 1 + thread_count) / thread_count;
    int block_count_nets = (num_nets - 1 + thread_count) / thread_count;
    dim3 block_size(thread_count, 2, 1);

    const T* x = pos;
    const T* y = pos + num_pins;

    if (grad_tensor)
    {
        // computeWeightedAverageWirelengthGradInterleavePinByPin<<<block_count_pins, block_size>>>(
        computeWeightedAverageWirelengthGradPinByPin<<<block_count_pins, thread_count>>>(
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
        #if 0
        // compute max and min in one kernel (pin by pin)
        // computeMaxMinInterleavePinByPin<<<block_count_pins, block_size>>>(
        computeMaxMinPinByPin<<<block_count_pins, thread_count>>>(
            x, y,
            pin2net_map,
            net_mask,
            num_pins,
            num_nets,
            xy_max,
            xy_min);
        #else
        // compute max and min in one kernel (net by net)
        computeMaxMinInterleaveNetByNet<<<block_count_nets, block_size>>>(
        // computeMaxMinNetByNet<<<block_count_nets, thread_count>>>(
            x, y,
            flat_netpin,
            netpin_start,
            net_mask,
            num_nets,
            xy_max,
            xy_min);
        #endif

        #if 1
        // compute plus-minus exp, sum of plus-minus exp, sum of x*exp in one CUDA kernels (pin by pin)
        // corresponding to the plus and minus a b c kernels in the DREAMPlace paper
        // computeABCKernelsInterleavePinByPin<<<block_count_pins, block_size>>>(
        computeABCKernelsPinByPin<<<block_count_pins, thread_count>>>(
            // pos,
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
        
        // compute partial wirelength
        computeXExpSumByExpSumXY<<<block_count_nets, thread_count>>>(
            xyexp_xy_sum, xyexp_nxy_sum,
            exp_xy_sum, exp_nxy_sum,
            pin2net_map,
            net_mask,
            num_nets,
            partial_wl);
        #else
        // compute plus-minus exp, sum of plus-minus exp, sum of x*exp in one CUDA kernels (net by net)
        // corresponding to the plus and minus a b c kernels in the DREAMPlace paper
        // compute partial wirelength at the same time
        // computeABCKernelsInterleaveAndWLNetByNet<<<block_count_nets, block_size>>>(
        computeABCKernelsAndWLNetByNet<<<block_count_nets, thread_count>>>(
            pos,
            flat_netpin,
            netpin_start,
            net_mask,
            num_nets,
            num_pins,
            inv_gamma,
            xy_max, xy_min,
            exp_xy, exp_nxy,
            exp_xy_sum, exp_nxy_sum,
            xyexp_xy_sum, xyexp_nxy_sum,
            partial_wl);
        
        #endif
        
        // Yibo: move out the summation to use ATen
        // significant speedup is observed
        //sumArray<<<1, 1>>>(partial_wl, num_nets, wl);
    }

    return 0;
}


#define REGISTER_KERNEL_LAUNCHER(T, V)                             \
    template int computeWeightedAverageWirelengthCudaAtomicLauncher<T, V>( \
        const T *pos,                                              \
        const int *pin2net_map,                                    \
        const int *flat_netpin,                                    \
        const int *netpin_start,                                   \
        const unsigned char *net_mask,                             \
        int num_nets,                                              \
        int num_pins,                                              \
        const T *inv_gamma,                                        \
        T *exp_xy, T *exp_nxy,                                     \
        T *exp_xy_sum, T *exp_nxy_sum,                             \
        T *xyexp_xy_sum, T *xyexp_nxy_sum,                         \
        V *xy_max, V *xy_min,                                      \
        T* partial_wl,                                             \
        const T *grad_tensor,                                      \
        T *grad_x_tensor, T *grad_y_tensor); 

REGISTER_KERNEL_LAUNCHER(float, int);
REGISTER_KERNEL_LAUNCHER(double, int);

DREAMPLACE_END_NAMESPACE
