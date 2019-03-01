#include <cfloat>
#include <stdio.h>
#include "assert.h"
#include "cuda_runtime.h"
#include "print.h"
#include "functional_cuda.h"
#include "csrmv.h"

template <typename T>
__global__ void multiply(const T* a, const T* b, int n, T* c)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) 
    {
        c[i] = a[i]*b[i]; 
    }
}

/// compute summation of values correlated to each pin for each net 
template <typename T>
void computeNetSum(
        T** x, // length of batch x #pins 
        const int* flat_netpin, // JA
        const int* netpin_start, // IA
        const T* netpin_values, // A
        int num_nets,
        int num_pins, 
        int num_batch, 
        T** net_sum_x // length of batch x #nets 
        )
{
    // ------------------ Prepare Data for GPU sparse matrix multiplication ------------
    cusparseStatus_t status;
    cusparseHandle_t handle=0;
    cusparseMatDescr_t descr=0;

    /* initialize cusparse library */
    assert( cusparseCreate(&handle) == CUSPARSE_STATUS_SUCCESS );
    assert( cusparseCreateMatDescr(&descr) == CUSPARSE_STATUS_SUCCESS );
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

    const T alpha = 1.0; 
    const T beta = 0.0; 

    cudaDeviceSynchronize(); 

    //cudaEvent_t start, stop;
    //cudaEventCreate(&start);
    //cudaEventCreate(&stop);
    //cudaEventRecord(start);

    for (int i = 0; i < num_batch; ++i)
    {
        /* exercise Level 2 routines (csrmv) */ 
        /* Multiply to get sum of pins for each net */
        status = csrmv<T>(
                handle, 
                CUSPARSE_OPERATION_NON_TRANSPOSE, 
                num_nets, 
                num_pins, 
                num_pins, 
                &alpha, 
                descr, 
                netpin_values, 
                netpin_start, 
                flat_netpin, 
                x[i], 
                &beta, 
                net_sum_x[i]
                );
        if (status != CUSPARSE_STATUS_SUCCESS)
        {
            printf("[E] cusparse csrmv failed for batch %d\n", i);
            exit(-1); 
        }
    }
    cudaDeviceSynchronize(); 

    //cudaEventRecord(stop);
    //cudaEventSynchronize(stop);

    //float milliseconds = 0;
    //cudaEventElapsedTime(&milliseconds, start, stop);
    //std::cout << "Net Sum : " << milliseconds << " milli sec" << std::endl;
}

template <typename T, typename V>
int computeWeightedAverageWirelengthCudaSparseLauncher(
        const T* x, const T* y, 
        const int* flat_netpin, 
        const int* netpin_start, 
        const T* netpin_values, 
        const int* pin2net_map, 
        const unsigned char* net_mask, 
        int num_nets,
        int num_pins, 
        const T* gamma, 
        T* exp_xy, T* exp_nxy, 
        T* exp_xy_sum, T* exp_nxy_sum, 
        T* xyexp_xy_sum, T* xyexp_nxy_sum, 
        V* xy_max, V* xy_min, 
        T* partial_wl, // wirelength of each net 
        const T* grad_tensor, 
        T* grad_x_tensor, T* grad_y_tensor // the gradient is partial total wirelength to partial pin position  
        )
{
    int thread_count = 1024; 
    int block_count = 32; // separate x and y

    cudaError_t status; 
    cudaStream_t stream_x_exp; 
    cudaStream_t stream_nx_exp; 
    cudaStream_t stream_y_exp; 
    cudaStream_t stream_ny_exp; 
    status = cudaStreamCreate(&stream_x_exp);
    if (status != cudaSuccess)
    {
        printf("cudaStreamCreate failed for stream_x_exp\n");
        fflush(stdout);
        return 1; 
    }
    status = cudaStreamCreate(&stream_y_exp);
    if (status != cudaSuccess)
    {
        printf("cudaStreamCreate failed for stream_y_exp\n");
        fflush(stdout);
        return 1; 
    }

    if (grad_tensor)
    {
        computeWeightedAverageWirelengthGrad<<<block_count, thread_count, 0, stream_x_exp>>>(
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
                grad_x_tensor
                );
        computeWeightedAverageWirelengthGrad<<<block_count, thread_count, 0, stream_y_exp>>>(
                y, 
                exp_xy+num_pins, exp_nxy+num_pins, 
                exp_xy_sum+num_nets, exp_nxy_sum+num_nets, 
                xyexp_xy_sum+num_nets, xyexp_nxy_sum+num_nets, 
                pin2net_map, 
                net_mask, 
                num_nets,
                num_pins, 
                gamma, 
                grad_tensor, 
                grad_y_tensor
                );
    }
    else
    {
        T* xyexp_xy = nullptr; 
        T* xyexp_nxy = nullptr; 
        status = cudaMalloc((void**)&xyexp_xy, 2*num_pins*sizeof(T));
        if (status != cudaSuccess)
        {
            printf("cudaMalloc failed for xyexp_xy\n");
            fflush(stdout);
            return 1; 
        }
        status = cudaMalloc((void**)&xyexp_nxy, 2*num_pins*sizeof(T));
        if (status != cudaSuccess)
        {
            printf("cudaMalloc failed for xyexp_nxy\n");
            fflush(stdout);
            return 1; 
        }

        status = cudaStreamCreate(&stream_nx_exp);
        if (status != cudaSuccess)
        {
            printf("cudaStreamCreate failed for stream_nx_exp\n");
            fflush(stdout);
            return 1; 
        }
        status = cudaStreamCreate(&stream_ny_exp);
        if (status != cudaSuccess)
        {
            printf("cudaStreamCreate failed for stream_ny_exp\n");
            fflush(stdout);
            return 1; 
        }

        // compute max/min 
        computeMax<<<block_count, thread_count, 0, stream_x_exp>>>(
                x, 
                pin2net_map, 
                net_mask, 
                num_nets, 
                num_pins, 
                xy_max
                );
        computeMin<<<block_count, thread_count, 0, stream_nx_exp>>>(
                x, 
                pin2net_map, 
                net_mask, 
                num_nets, 
                num_pins, 
                xy_min
                );
        computeMax<<<block_count, thread_count, 0, stream_y_exp>>>(
                y, 
                pin2net_map, 
                net_mask, 
                num_nets, 
                num_pins, 
                xy_max+num_nets
                );
        computeMin<<<block_count, thread_count, 0, stream_ny_exp>>>(
                y, 
                pin2net_map, 
                net_mask, 
                num_nets, 
                num_pins, 
                xy_min+num_nets
                );

        // compute exp and negative exp 
        computeExp<<<block_count, thread_count, 0, stream_x_exp>>>(
                x, 
                pin2net_map, 
                net_mask, 
                num_nets,
                num_pins, 
                gamma, 
                xy_max, 
                exp_xy
                );
        computeNegExp<<<block_count, thread_count, 0, stream_nx_exp>>>(
                x, 
                pin2net_map, 
                net_mask, 
                num_nets,
                num_pins, 
                gamma, 
                xy_min, 
                exp_nxy
                );
        computeExp<<<block_count, thread_count, 0, stream_y_exp>>>(
                y, 
                pin2net_map, 
                net_mask, 
                num_nets,
                num_pins, 
                gamma, 
                xy_max+num_nets, 
                exp_xy+num_pins
                );
        computeNegExp<<<block_count, thread_count, 0, stream_ny_exp>>>(
                y, 
                pin2net_map, 
                net_mask, 
                num_nets,
                num_pins, 
                gamma, 
                xy_min+num_nets, 
                exp_nxy+num_pins
                );

        // compute x*exp and x*negative exp 
        multiply<<<block_count, thread_count, 0, stream_x_exp>>>(
                x, 
                exp_xy, 
                num_pins, 
                xyexp_xy
                );
        multiply<<<block_count, thread_count, 0, stream_nx_exp>>>(
                x, 
                exp_nxy, 
                num_pins, 
                xyexp_nxy
                );
        multiply<<<block_count, thread_count, 0, stream_y_exp>>>(
                y, 
                exp_xy+num_pins, 
                num_pins, 
                xyexp_xy+num_pins
                );
        multiply<<<block_count, thread_count, 0, stream_ny_exp>>>(
                y, 
                exp_nxy+num_pins, 
                num_pins, 
                xyexp_nxy+num_pins
                );

        // compute exp sum 
        // compute x exp sum 
        T** pin_value_arrays = new T* [8]; 
        pin_value_arrays[0] = exp_xy; 
        pin_value_arrays[1] = exp_xy+num_pins; 
        pin_value_arrays[2] = exp_nxy; 
        pin_value_arrays[3] = exp_nxy+num_pins; 
        pin_value_arrays[4] = xyexp_xy; 
        pin_value_arrays[5] = xyexp_xy+num_pins; 
        pin_value_arrays[6] = xyexp_nxy; 
        pin_value_arrays[7] = xyexp_nxy+num_pins; 
        T** net_sum_x_arrays = new T* [8];
        net_sum_x_arrays[0] = exp_xy_sum;
        net_sum_x_arrays[1] = exp_xy_sum+num_nets;
        net_sum_x_arrays[2] = exp_nxy_sum;
        net_sum_x_arrays[3] = exp_nxy_sum+num_nets;
        net_sum_x_arrays[4] = xyexp_xy_sum;
        net_sum_x_arrays[5] = xyexp_xy_sum+num_nets;
        net_sum_x_arrays[6] = xyexp_nxy_sum;
        net_sum_x_arrays[7] = xyexp_nxy_sum+num_nets;
        computeNetSum(
                pin_value_arrays, 
                flat_netpin, 
                netpin_start, 
                netpin_values, 
                num_nets, 
                num_pins, 
                8, 
                net_sum_x_arrays
                );
        delete [] pin_value_arrays; 
        delete [] net_sum_x_arrays; 

        // compute log sum exp 
        computeXExpSumByExpSum<<<block_count, thread_count, 0, stream_x_exp>>>(
                xyexp_xy_sum, 
                exp_xy_sum, 
                pin2net_map, 
                net_mask, 
                num_nets,
                gamma, 
                partial_wl
                );
        computeXNegExpSumByNegExpSum<<<block_count, thread_count, 0, stream_nx_exp>>>(
                xyexp_nxy_sum, 
                exp_nxy_sum, 
                pin2net_map, 
                net_mask, 
                num_nets,
                gamma, 
                partial_wl+num_nets
                );

        computeXExpSumByExpSum<<<block_count, thread_count, 0, stream_y_exp>>>(
                xyexp_xy_sum+num_nets, 
                exp_xy_sum+num_nets, 
                pin2net_map, 
                net_mask, 
                num_nets,
                gamma, 
                partial_wl+2*num_nets
                );
        computeXNegExpSumByNegExpSum<<<block_count, thread_count, 0, stream_ny_exp>>>(
                xyexp_nxy_sum+num_nets, 
                exp_nxy_sum+num_nets, 
                pin2net_map, 
                net_mask, 
                num_nets,
                gamma, 
                partial_wl+3*num_nets
                );

        // I move out the summation to use ATen 
        // significant speedup is observed 
        //sumArray<<<1, 1>>>(partial_wl, 2*num_nets, wl);

        status = cudaStreamDestroy(stream_nx_exp); 
        stream_nx_exp = 0;
        if (status != cudaSuccess) 
        {
            printf("stream_nx_exp destroy failed\n");
            fflush(stdout);
            return 1;
        }   
        status = cudaStreamDestroy(stream_ny_exp); 
        stream_ny_exp = 0; 
        if (status != cudaSuccess) 
        {
            printf("stream_ny_exp destroy failed\n");
            fflush(stdout);
            return 1;
        }   

        cudaFree(xyexp_xy);
        if (status != cudaSuccess)
        {
            printf("cudaFree failed for xyexp_xy\n");
            fflush(stdout);
            return 1; 
        }
        cudaFree(xyexp_nxy);
        if (status != cudaSuccess)
        {
            printf("cudaFree failed for xyexp_nxy\n");
            fflush(stdout);
            return 1; 
        }
    }

    /* destroy stream */
    status = cudaStreamDestroy(stream_x_exp); 
    stream_x_exp = 0;
    if (status != cudaSuccess) 
    {
        printf("stream_x_exp destroy failed\n");
        fflush(stdout);
        return 1;
    }   
    status = cudaStreamDestroy(stream_y_exp); 
    stream_y_exp = 0; 
    if (status != cudaSuccess) 
    {
        printf("stream_y_exp destroy failed\n");
        fflush(stdout);
        return 1;
    }   

    return 0; 
}


#define REGISTER_KERNEL_LAUNCHER(T, V) \
    int instantiateComputeWeightedAverageWirelengthSparseLauncher(\
            const T* x, const T* y, \
            const int* flat_netpin, \
            const int* netpin_start, \
            const T* netpin_values, \
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
        return computeWeightedAverageWirelengthCudaSparseLauncher(\
                x, y, \
                flat_netpin, \
                netpin_start, \
                netpin_values, \
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
