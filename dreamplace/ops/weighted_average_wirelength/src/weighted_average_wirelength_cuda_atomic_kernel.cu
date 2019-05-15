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
    const T *gamma,
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
    int block_count_pins = (num_pins - 1 + thread_count) / thread_count; // separate x and y
    int block_count_nets = (num_nets - 1 + thread_count) / thread_count;

    cudaError_t status;
    cudaStream_t stream_y_exp;
    status = cudaStreamCreate(&stream_y_exp);
    if (status != cudaSuccess)
    {
        printf("cudaStreamCreate failed for stream_y_exp\n");
        fflush(stdout);
        return 1;
    }

    if (grad_tensor)
    {
        computeWeightedAverageWirelengthGrad<<<block_count_pins, thread_count>>>(
            x,
            exp_xy, exp_nxy,
            exp_xy_sum, exp_nxy_sum,
            xyexp_xy_sum, xyexp_nxy_sum,
            pin2net_map,
            net_mask,
            num_nets,
            num_pins,
            gamma,
            grad_tensor,
            grad_x_tensor);
        computeWeightedAverageWirelengthGrad<<<block_count_pins, thread_count, 0, stream_y_exp>>>(
            y,
            exp_xy + num_pins, exp_nxy + num_pins,
            exp_xy_sum + num_nets, exp_nxy_sum + num_nets,
            xyexp_xy_sum + num_nets, xyexp_nxy_sum + num_nets,
            pin2net_map,
            net_mask,
            num_nets,
            num_pins,
            gamma,
            grad_tensor,
            grad_y_tensor);
    }
    else
    {
        // compute max and min in one kernel
        computeMaxMin<<<block_count_pins, thread_count>>>(
            x,
            pin2net_map,
            net_mask,
            num_pins,
            xy_max,
            xy_min);
        computeMaxMin<<<block_count_pins, thread_count, 0, stream_y_exp>>>(
            y,
            pin2net_map,
            net_mask,
            num_pins,
            xy_max + num_nets,
            xy_min + num_nets);
        // compute plus-minus exp, sum of plus-minus exp, sum of x*exp in one CUDA kernels
        // corresponding to the plus and minus a b c kernels in the DREAMPlace paper
        computeABCKernels<<<block_count_pins, thread_count>>>(
            x,
            pin2net_map,
            net_mask,
            num_nets,
            num_pins,
            gamma,
            xy_max, xy_min,
            exp_xy, exp_nxy,
            exp_xy_sum, exp_nxy_sum,
            xyexp_xy_sum, xyexp_nxy_sum);
        computeABCKernels<<<block_count_pins, thread_count, 0, stream_y_exp>>>(
            y,
            pin2net_map,
            net_mask,
            num_nets,
            num_pins,
            gamma,
            xy_max + num_nets, xy_min + num_nets,
            exp_xy + num_pins, exp_nxy + num_pins,
            exp_xy_sum + num_nets, exp_nxy_sum + num_nets,
            xyexp_xy_sum + num_nets, xyexp_nxy_sum + num_nets);
        // compute log sum exp
        computeXExpSumByExpSum<<<block_count_nets, thread_count>>>(
            xyexp_xy_sum, xyexp_nxy_sum,
            exp_xy_sum, exp_nxy_sum,
            pin2net_map,
            net_mask,
            num_nets,
            gamma,
            partial_wl);
        computeXExpSumByExpSum<<<block_count_nets, thread_count, 0, stream_y_exp>>>(
            xyexp_xy_sum + num_nets, xyexp_nxy_sum + num_nets,
            exp_xy_sum + num_nets, exp_nxy_sum + num_nets,
            pin2net_map,
            net_mask,
            num_nets,
            gamma,
            partial_wl + 2 * num_nets);

        // I move out the summation to use ATen
        // significant speedup is observed
        //sumArray<<<1, 1>>>(partial_wl, 2*num_nets, wl);
    }

    /* destroy stream */
    status = cudaStreamDestroy(stream_y_exp);
    if (status != cudaSuccess)
    {
        printf("stream_y_exp destroy failed\n");
        fflush(stdout);
        return 1;
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
            const T* gamma, \
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
                gamma, \
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
