#ifndef GPUPLACE_WEIGHTED_AVERAGE_WIRELENGTH_FUNCTIONAL_H
#define GPUPLACE_WEIGHTED_AVERAGE_WIRELENGTH_FUNCTIONAL_H

#include "utility/src/torch.h"
#include "utility/src/utils.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
void integrateNetWeightsLauncher(
    const int *flat_netpin,
    const int *netpin_start,
    const unsigned char *net_mask,
    const T *net_weights,
    T *grad_x_tensor, T *grad_y_tensor,
    int num_nets,
    int num_threads)
{
    int chunk_size = DREAMPLACE_STD_NAMESPACE::max(int(num_nets / num_threads / 16), 1);
#pragma omp parallel for num_threads(num_threads) schedule(dynamic, chunk_size)
    for (int net_id = 0; net_id < num_nets; ++net_id)
    {
        if (net_mask[net_id])
        {
            T weight = net_weights[net_id];
            for (int j = netpin_start[net_id]; j < netpin_start[net_id + 1]; ++j)
            {
                int pin_id = flat_netpin[j];
                grad_x_tensor[pin_id] *= weight;
                grad_y_tensor[pin_id] *= weight;
            }
        }
    }
}

// V has to be int, or long long int
template <typename T, typename V>
void computeMaxMinNetByNet(
    const T *x, const T *y,
    const int *flat_netpin,
    const int *netpin_start,
    const unsigned char *net_mask,
    int num_nets,
    V *x_max_ptr,
    V *x_min_ptr,
    int num_threads)
{
    int chunk_size = DREAMPLACE_STD_NAMESPACE::max(int(num_nets / num_threads / 16), 1);
#pragma omp parallel for num_threads(num_threads) schedule(dynamic, chunk_size)
    for (int i = 0; i < num_nets; ++i)
    {
        if (net_mask[i])
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
                x_max = DREAMPLACE_STD_NAMESPACE::max((V)xx, x_max);
                x_min = DREAMPLACE_STD_NAMESPACE::min((V)xx, x_min);

                T yy = y[flat_netpin[j]];
                y_max = DREAMPLACE_STD_NAMESPACE::max((V)yy, y_max);
                y_min = DREAMPLACE_STD_NAMESPACE::min((V)yy, y_min);
            }

            x_max_ptr[x_index] = x_max;
            x_min_ptr[x_index] = x_min;
            x_max_ptr[y_index] = y_max;
            x_min_ptr[y_index] = y_min;
        }
    }
}

template <typename T, typename V>
void computeABCKernelsPinByPin(
    const T *x, const T *y,
    const int *pin2net_map,
    const unsigned char *net_mask,
    int num_nets,
    int num_pins,
    const T *inv_gamma,
    V *x_max, V *x_min,
    T *exp_x, T *exp_nx,
    T *exp_x_sum, T *exp_nx_sum,
    T *xexp_x_sum, T *xexp_nx_sum,
    int num_threads)
{
    int chunk_size = DREAMPLACE_STD_NAMESPACE::max(int(num_pins / num_threads / 16), 1);
#pragma omp parallel for num_threads(num_threads) schedule(dynamic, chunk_size)
    for (int i = 0; i < num_pins; ++i)
    {
        int net_id = pin2net_map[i];
        if (net_mask[net_id])
        {
            exp_x[i] = exp((x[i] - x_max[net_id]) * (*inv_gamma));
            exp_nx[i] = exp((x_min[net_id] - x[i]) * (*inv_gamma));

#pragma omp atomic
            exp_x_sum[net_id] += exp_x[i];
#pragma omp atomic
            exp_nx_sum[net_id] += exp_nx[i];
#pragma omp atomic
            xexp_x_sum[net_id] += x[i] * exp_x[i];
#pragma omp atomic
            xexp_nx_sum[net_id] += x[i] * exp_nx[i];

            net_id += num_nets;
            int pin_id = i + num_pins;
            exp_x[pin_id] = exp((y[i] - x_max[net_id]) * (*inv_gamma));
            exp_nx[pin_id] = exp((x_min[net_id] - y[i]) * (*inv_gamma));

#pragma omp atomic
            exp_x_sum[net_id] += exp_x[pin_id];
#pragma omp atomic
            exp_nx_sum[net_id] += exp_nx[pin_id];
#pragma omp atomic
            xexp_x_sum[net_id] += y[i] * exp_x[pin_id];
#pragma omp atomic
            xexp_nx_sum[net_id] += y[i] * exp_nx[pin_id];
        }
    }
}

template <typename T>
void computeXExpSumByExpSumXY(
    const T *xexp_x_sum, const T *xexp_nx_sum,
    const T *exp_x_sum, const T *exp_nx_sum,
    const int *pin2net_map,
    const unsigned char *net_mask,
    int num_nets,
    T *partial_wl,
    int num_threads)
{
    int chunk_size = DREAMPLACE_STD_NAMESPACE::max(int(num_nets / num_threads / 16), 1);
#pragma omp parallel for num_threads(num_threads) schedule(dynamic, chunk_size)
    for (int i = 0; i < num_nets; ++i)
    {
        if (net_mask[i])
        {
            T wl_x = xexp_x_sum[i] / exp_x_sum[i] - xexp_nx_sum[i] / exp_nx_sum[i];
            int y_index = i + num_nets;
            T wl_y = xexp_x_sum[y_index] / exp_x_sum[y_index] - xexp_nx_sum[y_index] / exp_nx_sum[y_index];

            partial_wl[i] = wl_x + wl_y;
        }
    }
}

template <typename T>
void computeWeightedAverageWirelengthGradPinByPin(
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
    T *grad_x_tensor, T *grad_y_tensor,
    int num_threads)
{
    int chunk_size = DREAMPLACE_STD_NAMESPACE::max(int(num_pins / num_threads / 16), 1);
#pragma omp parallel for num_threads(num_threads) schedule(dynamic, chunk_size)
    for (int i = 0; i < num_pins; ++i)
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

DREAMPLACE_END_NAMESPACE

#endif
