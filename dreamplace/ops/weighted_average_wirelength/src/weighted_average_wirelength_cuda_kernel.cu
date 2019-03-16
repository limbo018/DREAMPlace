#include <cfloat>
#include <stdio.h>
#include "assert.h"
#include "cuda_runtime.h"
#include "utility/src/print.h"
#include "utility/src/Msg.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
__global__ void fillArray(T* x, const int n, const T v)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) 
    {
        x[i] = v; 
    }
}

template <typename T>
__global__ void computeWeightedAverageWirelength(
        const T* x, const T* y, 
        const int* flat_netpin, 
        const int* netpin_start, 
        const unsigned char* net_mask, 
        int num_nets,
        const T* gamma, 
        T* partial_wl 
        )
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < 2*num_nets; i += blockDim.x * gridDim.x) 
    {
        int ii = i>>1; 
        const T* values = (i&1)? y : x;
        T xexp_x_sum = 0; 
        T xexp_nx_sum = 0; 
        T exp_x_sum = 0; 
        T exp_nx_sum = 0; 
        //int degree = netpin_start[ii+1]-netpin_start[ii];
        if (!net_mask[ii])
        {
            partial_wl[i] = 0; 
            continue; 
        }
        T x_max = -FLT_MAX; 
        T x_min = FLT_MAX; 
        for (int j = netpin_start[ii]; j < netpin_start[ii+1]; ++j)
        {
            T xx = values[flat_netpin[j]]; 
            x_max = max(xx, x_max); 
            x_min = min(xx, x_min); 
        }

        for (int j = netpin_start[ii]; j < netpin_start[ii+1]; ++j)
        {
            T xx = values[flat_netpin[j]]; 
            T exp_x = exp((xx-x_max)/(*gamma)); 
            T exp_nx = exp(-(xx-x_min)/(*gamma)); 

            xexp_x_sum += xx*exp_x; 
            xexp_nx_sum += xx*exp_nx; 
            exp_x_sum += exp_x; 
            exp_nx_sum += exp_nx; 
        }
        partial_wl[i] = xexp_x_sum/exp_x_sum - xexp_nx_sum/exp_nx_sum; 
    }
}

template <typename T>
__global__ void computeWeightedAverageWirelengthGrad(
        const T* x, const T* y, 
        const int* flat_netpin, 
        const int* netpin_start, 
        const unsigned char* net_mask, 
        int num_nets,
        const T* gamma, 
        const T* grad_tensor, 
        T* grad_x_tensor, T* grad_y_tensor 
        )
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < 2*num_nets; i += blockDim.x * gridDim.x) 
    {
        int ii = i/2; 
        const T* values = (i&1)? y : x;
        T* grads = (i&1)? grad_y_tensor : grad_x_tensor; 
        T xexp_x_sum = 0; 
        T xexp_nx_sum = 0; 
        T exp_x_sum = 0; 
        T exp_nx_sum = 0; 
        T xx = 0; 
        //int degree = netpin_start[ii+1]-netpin_start[ii];
        if (!net_mask[ii])
        {
            continue; 
        }

        T x_max = -FLT_MAX; 
        T x_min = FLT_MAX; 
        for (int j = netpin_start[ii]; j < netpin_start[ii+1]; ++j)
        {
            xx = values[flat_netpin[j]]; 
            x_max = max(xx, x_max); 
            x_min = min(xx, x_min); 
        }

        for (int j = netpin_start[ii]; j < netpin_start[ii+1]; ++j)
        {
            xx = values[flat_netpin[j]]; 
            T exp_x = exp((xx-x_max)/(*gamma)); 
            T exp_nx = exp(-(xx-x_min)/(*gamma)); 

            xexp_x_sum += xx*exp_x; 
            xexp_nx_sum += xx*exp_nx; 
            exp_x_sum += exp_x; 
            exp_nx_sum += exp_nx; 
        }
        T b_x = 1.0/((*gamma)*exp_x_sum);
        T a_x = (1.0 - b_x*xexp_x_sum)/exp_x_sum; 
        T b_nx = -1.0/((*gamma)*exp_nx_sum);
        T a_nx = (1.0 - b_nx*xexp_nx_sum)/exp_nx_sum; 
        for (int j = netpin_start[ii]; j < netpin_start[ii+1]; ++j)
        {
            // for x 
            xx = values[flat_netpin[j]]; 
            T exp_x = exp((xx-x_max)/(*gamma)); 
            T exp_nx = exp(-(xx-x_min)/(*gamma)); 
            T xexp_x = xx*exp_x; 
            T xexp_nx = xx*exp_nx;

            grads[flat_netpin[j]] = ((a_x*exp_x + b_x*xexp_x) - (a_nx*exp_nx + b_nx*xexp_nx))*(*grad_tensor); 
            //grads[flat_netpin[j]] = ( (1+1/(*gamma)*xx)/exp_x_sum - 1/(*gamma)*xexp_x_sum/(exp_x_sum*exp_x_sum) ) * exp_x \
            //                        - ( (1-1/(*gamma)*xx)/exp_nx_sum + 1/(*gamma)*xexp_nx_sum/(exp_nx_sum*exp_nx_sum) ) * exp_nx ; 
        }
    }
}

template <typename T>
int computeWeightedAverageWirelengthCudaLauncher(
        const T* x, const T* y, 
        const int* flat_netpin, 
        const int* netpin_start, 
        const unsigned char* net_mask, 
        int num_nets,
        const T* gamma, 
        T* partial_wl,
        const T* grad_tensor, 
        T* grad_x_tensor, T* grad_y_tensor  
        )
{
    int thread_count = 1024; 
    int block_count = 32; // separate x and y

    if (grad_tensor)
    {
        computeWeightedAverageWirelengthGrad<<<block_count, thread_count>>>(
                x, y, 
                flat_netpin, 
                netpin_start, 
                net_mask, 
                num_nets,
                gamma, 
                grad_tensor, 
                grad_x_tensor, grad_y_tensor
                );
    }
    else
    {
        computeWeightedAverageWirelength<<<block_count, thread_count>>>(
                x, y, 
                flat_netpin, 
                netpin_start, 
                net_mask, 
                num_nets,
                gamma, 
                partial_wl
                );
    }

    return 0; 
}


#define REGISTER_KERNEL_LAUNCHER(T) \
    int instantiateComputeWeightedAverageWirelengthLauncher(\
            const T* x, const T* y, \
            const int* flat_netpin, \
            const int* netpin_start, \
            const unsigned char* net_mask, \
            int num_nets,\
            const T* gamma, \
            T* partial_wl,\
            const T* grad_tensor, \
            T* grad_x_tensor, T* grad_y_tensor  \
            )\
    {\
        return computeWeightedAverageWirelengthCudaLauncher(\
                x, y, \
                flat_netpin, \
                netpin_start, \
                net_mask, \
                num_nets,\
                gamma, \
                partial_wl,\
                grad_tensor, \
                grad_x_tensor, grad_y_tensor  \
                );\
    }
REGISTER_KERNEL_LAUNCHER(float);
REGISTER_KERNEL_LAUNCHER(double);

DREAMPLACE_END_NAMESPACE
