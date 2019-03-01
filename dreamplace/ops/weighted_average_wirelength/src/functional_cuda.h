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
        const T* x, 
        const int* pin2net_map, 
        const unsigned char* net_mask, 
        int num_nets,
        int num_pins, 
        V* x_max
        )
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_pins; i += blockDim.x * gridDim.x) 
    {
        int net_id = pin2net_map[i];
        if (net_id >= 0 || net_mask[net_id])
        {
            atomicMax(&x_max[net_id], (V)(x[i]));
            __syncthreads();
        }
    }
}

// V has to be int, or long long int 
template <typename T, typename V>
__global__ void computeMin(
        const T* x, 
        const int* pin2net_map, 
        const unsigned char* net_mask, 
        int num_nets,
        int num_pins, 
        V* x_min
        )
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_pins; i += blockDim.x * gridDim.x) 
    {
        int net_id = pin2net_map[i];
        if (net_id >= 0 || net_mask[net_id])
        {
            atomicMin(&x_min[net_id], (V)(x[i]));
            __syncthreads();
        }
    }
}

template <typename T, typename V>
__global__ void computeExp(
        const T* x, 
        const int* pin2net_map, 
        const unsigned char* net_mask, 
        int num_nets,
        int num_pins, 
        const T* gamma, 
        V* x_max, 
        T* exp_x
        )
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_pins; i += blockDim.x * gridDim.x) 
    {
        int net_id = pin2net_map[i]; 
        if (net_id >= 0 || net_mask[net_id])
        {
            exp_x[i] = exp((x[i]-x_max[net_id])/(*gamma)); 
        }
    }
}

template <typename T, typename V>
__global__ void computeNegExp(
        const T* x, 
        const int* pin2net_map, 
        const unsigned char* net_mask, 
        int num_nets,
        int num_pins, 
        const T* gamma, 
        V* x_min, 
        T* exp_nx
        )
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_pins; i += blockDim.x * gridDim.x) 
    {
        int net_id = pin2net_map[i]; 
        if (net_id >= 0 || net_mask[net_id])
        {
            exp_nx[i] = exp(-(x[i]-x_min[net_id])/(*gamma)); 
        }
    }
}

template <typename T>
__global__ void computeXExpSumByExpSum(
        const T* xexp_x_sum, 
        const T* exp_x_sum, 
        const int* pin2net_map, 
        const unsigned char* net_mask, 
        int num_nets,
        const T* gamma, 
        T* partial_wl 
        )
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_nets; i += blockDim.x * gridDim.x) 
    {
        if (net_mask[i])
        {
            partial_wl[i] = xexp_x_sum[i]/exp_x_sum[i]; 
        }
    }
}

template <typename T>
__global__ void computeXNegExpSumByNegExpSum(
        const T* xexp_nx_sum, 
        const T* exp_nx_sum, 
        const int* pin2net_map, 
        const unsigned char* net_mask, 
        int num_nets,
        const T* gamma, 
        T* partial_wl 
        )
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_nets; i += blockDim.x * gridDim.x) 
    {
        if (net_mask[i])
        {
            partial_wl[i] = -xexp_nx_sum[i]/exp_nx_sum[i]; 
        }
    }
}

template <typename T>
__global__ void computeWeightedAverageWirelengthGrad(
        const T* x, 
        const T* exp_x, const T* exp_nx, 
        const T* exp_x_sum, const T* exp_nx_sum, 
        const T* xexp_x_sum, const T* xexp_nx_sum, 
        const int* pin2net_map, 
        const unsigned char* net_mask, 
        int num_nets,
        int num_pins, 
        const T* gamma, 
        const T* grad_tensor, 
        T* grad_x_tensor
        )
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_pins; i += blockDim.x * gridDim.x) 
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
