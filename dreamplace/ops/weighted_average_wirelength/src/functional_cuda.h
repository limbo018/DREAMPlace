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
__global__ void computeMaxMinPinByPin(
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
__global__ void computeMaxMinInterleavePinByPin(
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

// V has to be int, or long long int
template <typename T, typename V>
__global__ void computeMaxMinNetByNet(
    const T *x, const T *y,
    const int *flat_netpin,
    const int *netpin_start,
    const unsigned char *net_mask,
    int num_nets,
    V *x_max_ptr,
    V *x_min_ptr)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_nets && net_mask[i])
    {
        const int x_index = i;
        const int y_index = i + num_nets;

        V x_max = x_max_ptr[x_index];
        V x_min = x_min_ptr[x_index];
        V y_max = x_max_ptr[y_index];
        V y_min = x_min_ptr[y_index];

        for (int j = netpin_start[i]; j < netpin_start[i + 1]; ++j)
        {
            T xx = x[flat_netpin[j]];
            x_max = max((V)xx, x_max);
            x_min = min((V)xx, x_min);

            T yy = y[flat_netpin[j]];
            y_max = max((V)yy, y_max);
            y_min = min((V)yy, y_min);
        }

        x_max_ptr[x_index] = x_max;
        x_min_ptr[x_index] = x_min;
        x_max_ptr[y_index] = y_max;
        x_min_ptr[y_index] = y_min;
    }
}

// V has to be int, or long long int
template <typename T, typename V>
__global__ void computeMaxMinInterleaveNetByNet(
    const T *x, const T *y,
    const int *flat_netpin,
    const int *netpin_start,
    const unsigned char *net_mask,
    int num_nets,
    V *pos_max_ptr,
    V *pos_min_ptr)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_nets && net_mask[i])
    {
        const T *pos_ptr = threadIdx.y == 0 ? x : y;
        const int net_id = i + threadIdx.y * num_nets;
        V pos_max = pos_max_ptr[net_id];
        V pos_min = pos_min_ptr[net_id];

        for (int j = netpin_start[i]; j < netpin_start[i + 1]; ++j)
        {
            T pos = pos_ptr[flat_netpin[j]];
            pos_max = max((V)pos, pos_max);
            pos_min = min((V)pos, pos_min);
        }
        pos_max_ptr[net_id] = pos_max;
        pos_min_ptr[net_id] = pos_min;
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
__global__ void computeABCKernelsPinByPin(
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
__global__ void computeABCKernelsInterleavePinByPin(
    const T *x,
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

template <typename T, typename V>
__global__ void computeABCKernelsNetByNet(
    const T *x,
    const int *flat_netpin,
    const int *netpin_start,
    const unsigned char *net_mask,
    int num_nets,
    int num_pins,
    const T *inv_gamma,
    V *x_max, V *x_min,
    T *exp_xy, T *exp_nxy,
    T *exp_xy_sum, T *exp_nxy_sum,
    T *xyexp_xy_sum, T *xyexp_nxy_sum)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_nets && net_mask[i])
    {
        int x_index = i;
        int y_index = i + num_nets;
        for (int j = netpin_start[i]; j < netpin_start[i + 1]; ++j)
        {
            int pin_id = flat_netpin[j];
            exp_xy[pin_id] = exp((x[pin_id] - x_max[x_index]) * (*inv_gamma));
            exp_nxy[pin_id] = exp((x_min[x_index] - x[pin_id]) * (*inv_gamma));
            exp_xy_sum[x_index] += exp_xy[pin_id];
            exp_nxy_sum[x_index] += exp_nxy[pin_id];
            xyexp_xy_sum[x_index] += x[pin_id] * exp_xy[pin_id];
            xyexp_nxy_sum[x_index] += x[pin_id] * exp_nxy[pin_id];

            pin_id += num_pins;
            exp_xy[pin_id] = exp((x[pin_id] - x_max[y_index]) * (*inv_gamma));
            exp_nxy[pin_id] = exp((x_min[y_index] - x[pin_id]) * (*inv_gamma));
            exp_xy_sum[y_index] += exp_xy[pin_id];
            exp_nxy_sum[y_index] += exp_nxy[pin_id];
            xyexp_xy_sum[y_index] += x[pin_id] * exp_xy[pin_id];
            xyexp_nxy_sum[y_index] += x[pin_id] * exp_nxy[pin_id];
        }
    }
}

template <typename T, typename V>
__global__ void computeABCKernelsInterleaveNetByNet(
    const T *x,
    const int *flat_netpin,
    const int *netpin_start,
    const unsigned char *net_mask,
    int num_nets,
    int num_pins,
    const T *inv_gamma,
    V *x_max, V *x_min,
    T *exp_xy, T *exp_nxy,
    T *exp_xy_sum, T *exp_nxy_sum,
    T *xyexp_xy_sum, T *xyexp_nxy_sum)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_nets && net_mask[i])
    {
        int net_id = i + threadIdx.y * num_nets;
        for (int j = netpin_start[i]; j < netpin_start[i + 1]; ++j)
        {
            int pin_id = flat_netpin[j] + threadIdx.y * num_pins;

            exp_xy[pin_id] = exp((x[pin_id] - x_max[net_id]) * (*inv_gamma));
            exp_nxy[pin_id] = exp((x_min[net_id] - x[pin_id]) * (*inv_gamma));
            exp_xy_sum[net_id] += exp_xy[pin_id];
            exp_nxy_sum[net_id] += exp_nxy[pin_id];
            xyexp_xy_sum[net_id] += x[pin_id] * exp_xy[pin_id];
            xyexp_nxy_sum[net_id] += x[pin_id] * exp_nxy[pin_id];
        }
    }
}

template <typename T, typename V>
__global__ void computeABCKernelsAndWLNetByNet(
    const T *x,
    const int *flat_netpin,
    const int *netpin_start,
    const unsigned char *net_mask,
    int num_nets,
    int num_pins,
    const T *inv_gamma,
    V *x_max, V *x_min,
    T *exp_xy, T *exp_nxy,
    T *exp_xy_sum, T *exp_nxy_sum,
    T *xyexp_xy_sum, T *xyexp_nxy_sum,
    T * partial_wl)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_nets && net_mask[i])
    {
        int x_index = i;
        int y_index = i + num_nets;
        for (int j = netpin_start[i]; j < netpin_start[i + 1]; ++j)
        {
            int pin_id = flat_netpin[j];
            exp_xy[pin_id] = exp((x[pin_id] - x_max[x_index]) * (*inv_gamma));
            exp_nxy[pin_id] = exp((x_min[x_index] - x[pin_id]) * (*inv_gamma));
            exp_xy_sum[x_index] += exp_xy[pin_id];
            exp_nxy_sum[x_index] += exp_nxy[pin_id];
            xyexp_xy_sum[x_index] += x[pin_id] * exp_xy[pin_id];
            xyexp_nxy_sum[x_index] += x[pin_id] * exp_nxy[pin_id];

            pin_id += num_pins;
            exp_xy[pin_id] = exp((x[pin_id] - x_max[y_index]) * (*inv_gamma));
            exp_nxy[pin_id] = exp((x_min[y_index] - x[pin_id]) * (*inv_gamma));
            exp_xy_sum[y_index] += exp_xy[pin_id];
            exp_nxy_sum[y_index] += exp_nxy[pin_id];
            xyexp_xy_sum[y_index] += x[pin_id] * exp_xy[pin_id];
            xyexp_nxy_sum[y_index] += x[pin_id] * exp_nxy[pin_id];
        }
        partial_wl[i] = xyexp_xy_sum[x_index] / exp_xy_sum[x_index] - xyexp_nxy_sum[x_index] / exp_nxy_sum[x_index] +
                        xyexp_xy_sum[y_index] / exp_xy_sum[y_index] - xyexp_nxy_sum[y_index] / exp_nxy_sum[y_index];
    }
}

template <typename T, typename V>
__global__ void computeABCKernelsInterleaveAndWLNetByNet(
    const T *x,
    const int *flat_netpin,
    const int *netpin_start,
    const unsigned char *net_mask,
    int num_nets,
    int num_pins,
    const T *inv_gamma,
    V *x_max, V *x_min,
    T *exp_xy, T *exp_nxy,
    T *exp_xy_sum, T *exp_nxy_sum,
    T *xyexp_xy_sum, T *xyexp_nxy_sum,
    T *partial_wl)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_nets && net_mask[i])
    {
        int net_id = i + threadIdx.y * num_nets;
        for (int j = netpin_start[i]; j < netpin_start[i + 1]; ++j)
        {
            int pin_id = flat_netpin[j] + threadIdx.y * num_pins;

            exp_xy[pin_id] = exp((x[pin_id] - x_max[net_id]) * (*inv_gamma));
            exp_nxy[pin_id] = exp((x_min[net_id] - x[pin_id]) * (*inv_gamma));
            exp_xy_sum[net_id] += exp_xy[pin_id];
            exp_nxy_sum[net_id] += exp_nxy[pin_id];
            xyexp_xy_sum[net_id] += x[pin_id] * exp_xy[pin_id];
            xyexp_nxy_sum[net_id] += x[pin_id] * exp_nxy[pin_id];
        }
        atomicAdd(&partial_wl[i], xyexp_xy_sum[net_id] / exp_xy_sum[net_id] - xyexp_nxy_sum[net_id] / exp_nxy_sum[net_id]);
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
__global__ void computeXExpSumByExpSumXY(
    const T *xexp_x_sum, const T *xexp_nx_sum,
    const T *exp_x_sum, const T *exp_nx_sum,
    const int *pin2net_map,
    const unsigned char *net_mask,
    int num_nets,
    T *partial_wl)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_nets && net_mask[i])
    {
        T wl_x = xexp_x_sum[i] / exp_x_sum[i] - xexp_nx_sum[i] / exp_nx_sum[i];
        int y_index = i + num_nets;
        T wl_y = xexp_x_sum[y_index] / exp_x_sum[y_index] - xexp_nx_sum[y_index] / exp_nx_sum[y_index];

        partial_wl[i] = wl_x + wl_y;
    }
}

template <typename T>
__global__ void computeWeightedAverageWirelength(
    const T *x, const T *y,
    const int *flat_netpin,
    const int *netpin_start,
    const unsigned char *net_mask,
    int num_nets,
    int num_pins,
    const T *inv_gamma,
    T *exp_xy, T *exp_nxy,
    T *exp_xy_sum, T *exp_nxy_sum,
    T *xyexp_xy_sum, T *xyexp_nxy_sum,
    T *partial_wl)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_nets && net_mask[i])
    {
        const int x_index = i;
        const int y_index = i + num_nets;

        T x_max = -FLT_MAX;
        T x_min = FLT_MAX;
        T y_max = -FLT_MAX;
        T y_min = FLT_MAX;

        for (int j = netpin_start[i]; j < netpin_start[i + 1]; ++j)
        {
            T xx = x[flat_netpin[j]];
            x_max = max(xx, x_max);
            x_min = min(xx, x_min);

            T yy = y[flat_netpin[j]];
            y_max = max(yy, y_max);
            y_min = min(yy, y_min);
        }

        for (int j = netpin_start[i]; j < netpin_start[i + 1]; ++j)
        {
            int pin_id = flat_netpin[j];
            exp_xy[pin_id] = exp((x[pin_id] - x_max) * (*inv_gamma));
            exp_nxy[pin_id] = exp((x_min - x[pin_id]) * (*inv_gamma));
            exp_xy_sum[x_index] += exp_xy[pin_id];
            exp_nxy_sum[x_index] += exp_nxy[pin_id];
            xyexp_xy_sum[x_index] += x[pin_id] * exp_xy[pin_id];
            xyexp_nxy_sum[x_index] += x[pin_id] * exp_nxy[pin_id];

            pin_id += num_pins;
            exp_xy[pin_id] = exp((x[pin_id] - y_max) * (*inv_gamma));
            exp_nxy[pin_id] = exp((y_min - x[pin_id]) * (*inv_gamma));
            exp_xy_sum[y_index] += exp_xy[pin_id];
            exp_nxy_sum[y_index] += exp_nxy[pin_id];
            xyexp_xy_sum[y_index] += x[pin_id] * exp_xy[pin_id];
            xyexp_nxy_sum[y_index] += x[pin_id] * exp_nxy[pin_id];
        }

        partial_wl[i] = xyexp_xy_sum[x_index] / exp_xy_sum[x_index] - xyexp_nxy_sum[x_index] / exp_nxy_sum[x_index] +
                        xyexp_xy_sum[y_index] / exp_xy_sum[y_index] - xyexp_nxy_sum[y_index] / exp_nxy_sum[y_index];
    }
}

template <typename T>
__global__ void computeWeightedAverageWirelengthGradPinByPin(
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
    T *grad_x_tensor, T *grad_y_tensor)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_pins)
    {
        int net_id = pin2net_map[i];
        if (net_mask[net_id])
        {
            grad_x_tensor[i] = (*grad_tensor) *
                               (((1 + (*inv_gamma) * x[i]) * exp_x_sum[net_id] - (*inv_gamma) * xexp_x_sum[net_id]) / (exp_x_sum[net_id] * exp_x_sum[net_id]) * exp_x[i] - ((1 - (*inv_gamma) * x[i]) * exp_nx_sum[net_id] + (*inv_gamma) * xexp_nx_sum[net_id]) / (exp_nx_sum[net_id] * exp_nx_sum[net_id]) * exp_nx[i]);

            net_id += num_nets;
            int pin_id = i + num_pins;
            grad_y_tensor[i] = (*grad_tensor) *
                               (((1 + (*inv_gamma) * y[i]) * exp_x_sum[net_id] - (*inv_gamma) * xexp_x_sum[net_id]) / (exp_x_sum[net_id] * exp_x_sum[net_id]) * exp_x[pin_id] - ((1 - (*inv_gamma) * y[i]) * exp_nx_sum[net_id] + (*inv_gamma) * xexp_nx_sum[net_id]) / (exp_nx_sum[net_id] * exp_nx_sum[net_id]) * exp_nx[pin_id]);
        }
    }
}

template <typename T>
__global__ void computeWeightedAverageWirelengthGradInterleavePinByPin(
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
    T *grad_x_tensor, T *grad_y_tensor)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_pins)
    {
        int net_id = pin2net_map[i];
        if (net_mask[net_id])
        {
            net_id += threadIdx.y * num_nets;
            int pin_id = i + threadIdx.y * num_pins;

            grad_x_tensor[pin_id] = (*grad_tensor) *
                                    (((1 + (*inv_gamma) * x[pin_id]) * exp_x_sum[net_id] - (*inv_gamma) * xexp_x_sum[net_id]) / (exp_x_sum[net_id] * exp_x_sum[net_id]) * exp_x[pin_id] - ((1 - (*inv_gamma) * x[pin_id]) * exp_nx_sum[net_id] + (*inv_gamma) * xexp_nx_sum[net_id]) / (exp_nx_sum[net_id] * exp_nx_sum[net_id]) * exp_nx[pin_id]);
        }
    }
}

template <typename T>
__global__ void computeWeightedAverageWirelengthGradNetByNet(
    const T *x, const T *y,
    const T *exp_x, const T *exp_nx,
    const T *exp_x_sum, const T *exp_nx_sum,
    const T *xexp_x_sum, const T *xexp_nx_sum,
    const int *flat_netpin,
    const int *netpin_start,
    const unsigned char *net_mask,
    int num_nets,
    int num_pins,
    const T *inv_gamma,
    const T *grad_tensor,
    T *grad_x_tensor, T *grad_y_tensor)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_nets && net_mask[i])
    {
        int x_index = i;
        int y_index = i + num_nets;
        for (int j = netpin_start[i]; j < netpin_start[i + 1]; ++j)
        {
            int pin_id = flat_netpin[j];
            grad_x_tensor[pin_id] = (*grad_tensor) *
                                    (((1 + (*inv_gamma) * x[pin_id]) * exp_x_sum[x_index] - (*inv_gamma) * xexp_x_sum[x_index]) / (exp_x_sum[x_index] * exp_x_sum[x_index]) * exp_x[pin_id] - ((1 - (*inv_gamma) * x[pin_id]) * exp_nx_sum[x_index] + (*inv_gamma) * xexp_nx_sum[x_index]) / (exp_nx_sum[x_index] * exp_nx_sum[x_index]) * exp_nx[pin_id]);

            int pin_id_y = pin_id + num_pins;
            grad_y_tensor[pin_id] = (*grad_tensor) *
                                    (((1 + (*inv_gamma) * y[pin_id]) * exp_x_sum[y_index] - (*inv_gamma) * xexp_x_sum[y_index]) / (exp_x_sum[y_index] * exp_x_sum[y_index]) * exp_x[pin_id_y] - ((1 - (*inv_gamma) * y[pin_id]) * exp_nx_sum[y_index] + (*inv_gamma) * xexp_nx_sum[y_index]) / (exp_nx_sum[y_index] * exp_nx_sum[y_index]) * exp_nx[pin_id_y]);
        }
    }
}

template <typename T>
__global__ void computeWeightedAverageWirelengthGradInterleaveNetByNet(
    const T *x, const T *y,
    const T *exp_x, const T *exp_nx,
    const T *exp_x_sum, const T *exp_nx_sum,
    const T *xexp_x_sum, const T *xexp_nx_sum,
    const int *flat_netpin,
    const int *netpin_start,
    const unsigned char *net_mask,
    int num_nets,
    int num_pins,
    const T *inv_gamma,
    const T *grad_tensor,
    T *grad_x_tensor, T *grad_y_tensor)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_nets && net_mask[i])
    {
        int net_id = i + threadIdx.y * num_nets;
        for (int j = netpin_start[i]; j < netpin_start[i + 1]; ++j)
        {
            int pin_id = flat_netpin[j] + threadIdx.y * num_pins;
            grad_x_tensor[pin_id] = (*grad_tensor) *
                                    (((1 + (*inv_gamma) * x[pin_id]) * exp_x_sum[net_id] - (*inv_gamma) * xexp_x_sum[net_id]) / (exp_x_sum[net_id] * exp_x_sum[net_id]) * exp_x[pin_id] - ((1 - (*inv_gamma) * x[pin_id]) * exp_nx_sum[net_id] + (*inv_gamma) * xexp_nx_sum[net_id]) / (exp_nx_sum[net_id] * exp_nx_sum[net_id]) * exp_nx[pin_id]);
        }
    }
}

#endif
