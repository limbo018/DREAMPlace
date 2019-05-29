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
        if (net_mask[net_id])
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
        if (net_mask[net_id])
        {
            atomicMin(&x_min[net_id], (V)(x[i]));
        }
    }
}

// V has to be int, or long long int
template <typename T, typename V>
__global__ void computeMaxMin(
    const T *x, const T *y,
    const int *pin2net_map,
    const unsigned char *net_mask,
    int num_pins,
    int num_nets,
    V *x_max,
    V *x_min)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_pins)
    {
        int net_id = pin2net_map[i];
        if (net_mask[net_id])
        {
            atomicMax(&x_max[net_id], (V)(x[i]));
            atomicMin(&x_min[net_id], (V)(x[i]));

            net_id += num_nets;
            atomicMax(&x_max[net_id], (V)(y[i]));
            atomicMin(&x_min[net_id], (V)(y[i]));
        }
    }
}

// V has to be int, or long long int
template <typename T, typename V>
__global__ void computeMaxMinInterleave(
    const T *x, const T *y,
    const int *pin2net_map,
    const unsigned char *net_mask,
    int num_pins,
    int num_nets,
    V *x_max,
    V *x_min)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_pins)
    {
        int net_id = pin2net_map[i];
        if (net_mask[net_id])
        {
            net_id += threadIdx.y * num_nets;
            int pin_id = i + threadIdx.y * num_pins;
            
            atomicMax(&x_max[net_id], (V)(x[pin_id]));
            atomicMin(&x_min[net_id], (V)(x[pin_id]));
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
    const T *inv_gamma,
    V *x_max,
    T *exp_x)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_pins)
    {
        int net_id = pin2net_map[i];
        if (net_mask[net_id])
        {
            exp_x[i] = exp((x[i] - x_max[net_id]) * (*inv_gamma));
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
    const T *inv_gamma,
    V *x_min,
    T *exp_nx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_pins)
    {
        int net_id = pin2net_map[i];
        if (net_mask[net_id])
        {
            exp_nx[i] = exp(-(x[i] - x_min[net_id]) * (*inv_gamma));
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
        if (net_mask[net_id])
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
        if (net_mask[net_id])
        {
            atomicAdd(&xexp_x_sum[net_id], x[i] * exp_x[i]);
        }
    }
}

template <typename T, typename V>
__global__ void computeABCKernels(
    const T *x, const T *y,
    const int *pin2net_map,
    const unsigned char *net_mask,
    int num_nets,
    int num_pins,
    const T *inv_gamma,
    V *x_max, V *x_min,
    T *exp_x, T *exp_nx,
    T *exp_x_sum, T *exp_nx_sum,
    T *xexp_x_sum, T *xexp_nx_sum)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_pins)
    {
        int net_id = pin2net_map[i];
        if (net_mask[net_id])
        {
            exp_x[i] = exp((x[i] - x_max[net_id]) * (*inv_gamma));
            exp_nx[i] = exp((x_min[net_id] - x[i]) * (*inv_gamma));

            atomicAdd(&exp_x_sum[net_id], exp_x[i]);
            atomicAdd(&exp_nx_sum[net_id], exp_nx[i]);
            atomicAdd(&xexp_x_sum[net_id], x[i] * exp_x[i]);
            atomicAdd(&xexp_nx_sum[net_id], x[i] * exp_nx[i]);

            net_id += num_nets;
            int pin_id = i + num_pins;
            exp_x[pin_id] = exp((y[i] - x_max[net_id]) * (*inv_gamma));
            exp_nx[pin_id] = exp((x_min[net_id] - y[i]) * (*inv_gamma));

            atomicAdd(&exp_x_sum[net_id], exp_x[pin_id]);
            atomicAdd(&exp_nx_sum[net_id], exp_nx[pin_id]);
            atomicAdd(&xexp_x_sum[net_id], y[i] * exp_x[pin_id]);
            atomicAdd(&xexp_nx_sum[net_id], y[i] * exp_nx[pin_id]);
        }
    }
}

template <typename T, typename V>
__global__ void computeABCKernelsInterleave(
    const T *x, const T *y,
    const int *pin2net_map,
    const unsigned char *net_mask,
    int num_nets,
    int num_pins,
    const T *inv_gamma,
    V *x_max, V *x_min,
    T *exp_x, T *exp_nx,
    T *exp_x_sum, T *exp_nx_sum,
    T *xexp_x_sum, T *xexp_nx_sum)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_pins)
    {
        int net_id = pin2net_map[i];
        if (net_mask[net_id])
        {
            net_id += threadIdx.y * num_nets;
            int pin_id = i + threadIdx.y * num_pins;

            exp_x[pin_id] = exp((x[pin_id] - x_max[net_id]) * (*inv_gamma));
            exp_nx[pin_id] = exp((x_min[net_id] - x[pin_id]) * (*inv_gamma));

            atomicAdd(&exp_x_sum[net_id], exp_x[pin_id]);
            atomicAdd(&exp_nx_sum[net_id], exp_nx[pin_id]);
            atomicAdd(&xexp_x_sum[net_id], x[pin_id] * exp_x[pin_id]);
            atomicAdd(&xexp_nx_sum[net_id], x[pin_id] * exp_nx[pin_id]);
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
__global__ void computeWeightedAverageWirelengthGrad(
    const T *x, const T *y,
    const T *exp_x, const T *exp_nx,
    const T *exp_x_sum, const T *exp_nx_sum,
    const T *xexp_x_sum, const T *xexp_nx_sum,
    const int *pin2net_map,
    const unsigned char *net_mask,
    int num_nets,
    int num_pins,
    const T *inv_gamma,
    const T *grad_tensor,
    T *grad_x_tensor, T* grad_y_tensor)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_pins)
    {
        int net_id = pin2net_map[i];
        if (net_mask[net_id])
        {
            T gamma_inv = (*inv_gamma);

            grad_x_tensor[i] = (*grad_tensor) * (
                  ((1+gamma_inv*x[i])*exp_x_sum[net_id]  - gamma_inv*xexp_x_sum[net_id])  / (exp_x_sum[net_id]*exp_x_sum[net_id])   * exp_x[i] 
                - ((1-gamma_inv*x[i])*exp_nx_sum[net_id] + gamma_inv*xexp_nx_sum[net_id]) / (exp_nx_sum[net_id]*exp_nx_sum[net_id]) * exp_nx[i] 
                );
            
            net_id += num_nets;
            int pin_id = i + num_pins;
            grad_y_tensor[i] = (*grad_tensor) * (
                  ((1+gamma_inv*y[i])*exp_x_sum[net_id]  - gamma_inv*xexp_x_sum[net_id])  / (exp_x_sum[net_id]*exp_x_sum[net_id])   * exp_x[pin_id]
                - ((1-gamma_inv*y[i])*exp_nx_sum[net_id] + gamma_inv*xexp_nx_sum[net_id]) / (exp_nx_sum[net_id]*exp_nx_sum[net_id]) * exp_nx[pin_id] 
                );
        }
    }
}

template <typename T>
__global__ void computeWeightedAverageWirelengthGradInterleave(
    const T *x, const T *y,
    const T *exp_x, const T *exp_nx,
    const T *exp_x_sum, const T *exp_nx_sum,
    const T *xexp_x_sum, const T *xexp_nx_sum,
    const int *pin2net_map,
    const unsigned char *net_mask,
    int num_nets,
    int num_pins,
    const T *inv_gamma,
    const T *grad_tensor,
    T *grad_x_tensor, T* grad_y_tensor)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_pins)
    {
        int net_id = pin2net_map[i];
        if (net_mask[net_id])
        {
            T gamma_inv = (*inv_gamma);
            net_id += threadIdx.y * num_nets;
            int pin_id = i + threadIdx.y * num_pins;
            
            grad_x_tensor[pin_id] = (*grad_tensor) * (
                  ((1+gamma_inv*x[pin_id])*exp_x_sum[net_id]  - gamma_inv*xexp_x_sum[net_id])  / (exp_x_sum[net_id]*exp_x_sum[net_id])   * exp_x[pin_id] 
                - ((1-gamma_inv*x[pin_id])*exp_nx_sum[net_id] + gamma_inv*xexp_nx_sum[net_id]) / (exp_nx_sum[net_id]*exp_nx_sum[net_id]) * exp_nx[pin_id] 
                );
        }
    }
}

#endif
