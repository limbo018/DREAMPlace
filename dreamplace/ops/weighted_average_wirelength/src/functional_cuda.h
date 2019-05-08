/*
 * @Author      : undefined
 * @Date: 2019-05-07 20:14:28
 * @LastEditTime: 2019-05-07 20:21:23
 */
/**
 * @file   functional_cuda.h
 * @author Yibo Lin
 * @date   Nov 2018
 */

#ifndef GPUPLACE_WEIGHTED_AVERAGE_WIRELENGTH_FUNCTIONAL_CUDA_H
#define GPUPLACE_WEIGHTED_AVERAGE_WIRELENGTH_FUNCTIONAL_CUDA_H

// V has to be int, or long long int
template <typename T, typename V>
__global__ void computeMax(
    const T *x,
    const int *pin2net_map,
    const unsigned char *net_mask,
    int num_pins,
    V *x_max)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_pins)
    {
        int net_id = pin2net_map[i];
        if (net_id >= 0 || net_mask[net_id])
        {
            atomicMax(&x_max[net_id], (V)(x[i]));
        }
    }
}

// V has to be int, or long long int
template <typename T, typename V>
__global__ void computeMin(
    const T *x,
    const int *pin2net_map,
    const unsigned char *net_mask,
    int num_pins,
    V *x_min)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_pins)
    {
        int net_id = pin2net_map[i];
        if (net_id >= 0 || net_mask[net_id])
        {
            atomicMin(&x_min[net_id], (V)(x[i]));
        }
    }
}

// V has to be int, or long long int
template <typename T, typename V>
__global__ void computeMaxMin(
    const T *x,
    const int *pin2net_map,
    const unsigned char *net_mask,
    int num_pins,
    V *x_max,
    V *x_min)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_pins)
    {
        int net_id = pin2net_map[i];
        if (net_id >= 0 || net_mask[net_id])
        {
            atomicMax(&x_max[net_id], (V)(x[i]));
            atomicMin(&x_min[net_id], (V)(x[i]));
        }
    }
}

template <typename T, typename V>
__global__ void computeExp(
    const T *x,
    const int *pin2net_map,
    const unsigned char *net_mask,
    int num_nets,
    int num_pins,
    const T *gamma,
    V *x_max,
    T *exp_x)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_pins)
    {
        int net_id = pin2net_map[i];
        if (net_id >= 0 || net_mask[net_id])
        {
            exp_x[i] = exp((x[i] - x_max[net_id]) / (*gamma));
        }
    }
}

template <typename T, typename V>
__global__ void computeNegExp(
    const T *x,
    const int *pin2net_map,
    const unsigned char *net_mask,
    int num_nets,
    int num_pins,
    const T *gamma,
    V *x_min,
    T *exp_nx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_pins)
    {
        int net_id = pin2net_map[i];
        if (net_id >= 0 || net_mask[net_id])
        {
            exp_nx[i] = exp(-(x[i] - x_min[net_id]) / (*gamma));
        }
    }
}

template <typename T>
__global__ void computeExpSum(
    const T *exp_x,
    const int *pin2net_map,
    const unsigned char *net_mask,
    int num_nets,
    int num_pins,
    T *exp_x_sum)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_pins)
    {
        int net_id = pin2net_map[i];
        if (net_id >= 0 || net_mask[net_id])
        {
            atomicAdd(&exp_x_sum[net_id], exp_x[i]);
        }
    }
}

template <typename T>
__global__ void computeXExpSum(
    const T *x,
    const T *exp_x,
    const int *pin2net_map,
    const unsigned char *net_mask,
    int num_nets,
    int num_pins,
    T *xexp_x_sum)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_pins)
    {
        int net_id = pin2net_map[i];
        if (net_id >= 0 || net_mask[net_id])
        {
            atomicAdd(&xexp_x_sum[net_id], x[i] * exp_x[i]);
        }
    }
}

template <typename T, typename V>
__global__ void computeABCKernels(
    const T *x,
    const int *pin2net_map,
    const unsigned char *net_mask,
    int num_nets,
    int num_pins,
    const T *gamma,
    V *x_max, V *x_min,
    T *exp_x, T *exp_nx,
    T *exp_x_sum, T *exp_nx_sum,
    T *xexp_x_sum, T *xexp_nx_sum)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_pins)
    {
        int net_id = pin2net_map[i];
        if (net_id >= 0 || net_mask[net_id])
        {
            exp_x[i] = exp((x[i] - x_max[net_id]) / (*gamma));
            exp_nx[i] = exp(-(x[i] - x_min[net_id]) / (*gamma));

            atomicAdd(&exp_x_sum[net_id], exp_x[i]);
            atomicAdd(&exp_nx_sum[net_id], exp_nx[i]);
            atomicAdd(&xexp_x_sum[net_id], x[i] * exp_x[i]);
            atomicAdd(&xexp_nx_sum[net_id], x[i] * exp_nx[i]);
        }
    }
}

template <typename T>
__global__ void computeXExpSumByExpSum(
    const T *xexp_x_sum,
    const T *exp_x_sum,
    const int *pin2net_map,
    const unsigned char *net_mask,
    int num_nets,
    const T *gamma,
    T *partial_wl)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_nets)
    {
        if (net_mask[i])
        {
            partial_wl[i] = xexp_x_sum[i] / exp_x_sum[i];
        }
    }
}

template <typename T>
__global__ void computeXNegExpSumByNegExpSum(
    const T *xexp_nx_sum,
    const T *exp_nx_sum,
    const int *pin2net_map,
    const unsigned char *net_mask,
    int num_nets,
    const T *gamma,
    T *partial_wl)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_nets)
    {
        if (net_mask[i])
        {
            partial_wl[i] = -xexp_nx_sum[i] / exp_nx_sum[i];
        }
    }
}

template <typename T>
__global__ void computeXExpSumByExpSum(
    const T *xexp_x_sum, const T *xexp_nx_sum,
    const T *exp_x_sum, const T *exp_nx_sum,
    const int *pin2net_map,
    const unsigned char *net_mask,
    int num_nets,
    const T *gamma,
    T *partial_wl)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_nets)
    {
        if (net_mask[i])
        {
            partial_wl[i] = xexp_x_sum[i] / exp_x_sum[i];
            partial_wl[i + +num_nets] = -xexp_nx_sum[i] / exp_nx_sum[i];
        }
    }
}

template <typename T>
__global__ void computeWeightedAverageWirelengthGrad(
    const T *x,
    const T *exp_x, const T *exp_nx,
    const T *exp_x_sum, const T *exp_nx_sum,
    const T *xexp_x_sum, const T *xexp_nx_sum,
    const int *pin2net_map,
    const unsigned char *net_mask,
    int num_nets,
    int num_pins,
    const T *gamma,
    const T *grad_tensor,
    T *grad_x_tensor)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_pins)
    {
        int net_id = pin2net_map[i];
        if (net_id >= 0 || net_mask[net_id])
        {
            T gamma_inv = 1.0/(*gamma);
            grad_x_tensor[i] = (\
                    ( (1+gamma_inv*x[i])*exp_x_sum[net_id] - gamma_inv*xexp_x_sum[net_id] ) / (exp_x_sum[net_id]*exp_x_sum[net_id]) * exp_x[i] \
                    - ( (1-gamma_inv*x[i])*exp_nx_sum[net_id] + gamma_inv*xexp_nx_sum[net_id] ) / (exp_nx_sum[net_id]*exp_nx_sum[net_id]) * exp_nx[i] \
                    ) * (*grad_tensor);
        }
    }
}

#endif
