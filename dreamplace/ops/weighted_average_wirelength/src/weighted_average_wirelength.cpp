/**
 * @file   src/weighted_average_wirelength.cpp
 * @author Yibo Lin
 * @date   Jun 2018
 * @brief  Compute weighted-average wirelength and gradient according to e-place
 */
#include "utility/src/torch.h"
#include "utility/src/Msg.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
int computeWeightedAverageWirelengthLauncher(
        const T* x, const T* y, 
        const int* flat_netpin, 
        const int* netpin_start, 
        const unsigned char* net_mask, 
        int num_nets, 
        const T* gamma, 
        T* wl, 
        const T* grad_tensor, 
        int num_threads, 
        T* grad_x_tensor, T* grad_y_tensor 
        );

/// @brief add net weights to gradient 
template <typename T>
void integrateNetWeightsLauncher(
        const int* flat_netpin, 
        const int* netpin_start, 
        const unsigned char* net_mask, 
        const T* net_weights, 
        T* grad_x_tensor, T* grad_y_tensor, 
        int num_nets, 
        int num_threads
        );

#define CHECK_FLAT(x) AT_ASSERTM(!x.is_cuda() && x.ndimension() == 1, #x " must be a flat tensor on CPU")
#define CHECK_EVEN(x) AT_ASSERTM((x.numel()&1) == 0, #x " must have even number of elements")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")

/// @brief Compute weighted-average wirelength according to e-place 
///     \sum(x*exp(x/gamma)) / \sum(exp(x/gamma)) - \sum(x*exp(-x/gamma)) / \sum(exp(-x/gamma))
/// @param pos cell locations, array of x locations and then y locations 
/// @param flat_netpin similar to the JA array in CSR format, which is flattened from the net2pin map (array of array)
/// @param netpin_start similar to the IA array in CSR format, IA[i+1]-IA[i] is the number of pins in each net, the length of IA is number of nets + 1
/// @param net_mask an array to record whether compute the where for a net or not 
/// @param net_weights weight of nets 
/// @param gamma a scalar tensor for the parameter in the equation 
at::Tensor weighted_average_wirelength_forward(
        at::Tensor pos,
        at::Tensor flat_netpin,
        at::Tensor netpin_start, 
        at::Tensor net_mask, 
        at::Tensor net_weights, 
        at::Tensor gamma, 
        int num_threads
        ) 
{
    CHECK_FLAT(pos); 
    CHECK_EVEN(pos);
    CHECK_CONTIGUOUS(pos);
    CHECK_FLAT(flat_netpin);
    CHECK_CONTIGUOUS(flat_netpin);
    CHECK_FLAT(netpin_start);
    CHECK_CONTIGUOUS(netpin_start);

    int num_nets = netpin_start.numel()-1; 
    at::Tensor wl = at::zeros({num_nets}, pos.options());

    AT_DISPATCH_FLOATING_TYPES(pos.type(), "computeWeightedAverageWirelengthLauncher", [&] {
            computeWeightedAverageWirelengthLauncher<scalar_t>(
                    pos.data<scalar_t>(), pos.data<scalar_t>()+pos.numel()/2, 
                    flat_netpin.data<int>(), 
                    netpin_start.data<int>(), 
                    net_mask.data<unsigned char>(), 
                    num_nets, 
                    gamma.data<scalar_t>(), 
                    wl.data<scalar_t>(), 
                    nullptr, 
                    num_threads, 
                    nullptr, nullptr
                    );
            });
    return wl.mul_(net_weights).sum(); 
}

/// @brief Compute gradient 
/// @param grad_pos input gradient from backward propagation 
/// @param pos locations of pins 
/// @param flat_netpin similar to the JA array in CSR format, which is flattened from the net2pin map (array of array)
/// @param netpin_start similar to the IA array in CSR format, IA[i+1]-IA[i] is the number of pins in each net, the length of IA is number of nets + 1
/// @param net_mask an array to record whether compute the where for a net or not 
/// @param net_weights weight of nets 
/// @param gamma a scalar tensor for the parameter in the equation 
at::Tensor weighted_average_wirelength_backward(
        at::Tensor grad_pos, 
        at::Tensor pos,
        at::Tensor flat_netpin,
        at::Tensor netpin_start, 
        at::Tensor net_mask, 
        at::Tensor net_weights, 
        at::Tensor gamma, 
        int num_threads
        ) 
{
    CHECK_FLAT(pos); 
    CHECK_EVEN(pos);
    CHECK_CONTIGUOUS(pos);
    CHECK_FLAT(flat_netpin);
    CHECK_CONTIGUOUS(flat_netpin);
    CHECK_FLAT(netpin_start);
    CHECK_CONTIGUOUS(netpin_start);
    at::Tensor grad_out = at::zeros_like(pos);

    AT_DISPATCH_FLOATING_TYPES(pos.type(), "computeWeightedAverageWirelengthLauncher", [&] {
            computeWeightedAverageWirelengthLauncher<scalar_t>(
                    pos.data<scalar_t>(), pos.data<scalar_t>()+pos.numel()/2, 
                    flat_netpin.data<int>(), 
                    netpin_start.data<int>(), 
                    net_mask.data<unsigned char>(), 
                    netpin_start.numel()-1, 
                    gamma.data<scalar_t>(), 
                    nullptr, 
                    grad_pos.data<scalar_t>(), 
                    num_threads, 
                    grad_out.data<scalar_t>(), grad_out.data<scalar_t>()+pos.numel()/2
                    );
            integrateNetWeightsLauncher<scalar_t>(
                flat_netpin.data<int>(), 
                netpin_start.data<int>(), 
                net_mask.data<unsigned char>(), 
                net_weights.data<scalar_t>(), 
                grad_out.data<scalar_t>(), grad_out.data<scalar_t>()+pos.numel()/2, 
                netpin_start.numel()-1, 
                num_threads
                );
            });
    return grad_out; 
}

template <typename T>
int computeWeightedAverageWirelengthLauncher(
        const T* x, const T* y, 
        const int* flat_netpin, 
        const int* netpin_start, 
        const unsigned char* net_mask, 
        int num_nets, 
        const T* gamma, 
        T* wl,
        const T* grad_tensor, 
        int num_threads, 
        T* grad_x_tensor, T* grad_y_tensor 
        )
{
#pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < num_nets; ++i)
    {
        T xexp_x_sum = 0; 
        T xexp_nx_sum = 0; 
        T exp_x_sum = 0; 
        T exp_nx_sum = 0; 

        T yexp_y_sum = 0; 
        T yexp_ny_sum = 0; 
        T exp_y_sum = 0; 
        T exp_ny_sum = 0; 

        //int degree = netpin_start[i+1]-netpin_start[i]; 
        if (!net_mask[i])
        {
            continue; 
        }
        T x_max = -std::numeric_limits<T>::max(); 
        T x_min = std::numeric_limits<T>::max(); 
        T y_max = -std::numeric_limits<T>::max(); 
        T y_min = std::numeric_limits<T>::max(); 
        for (int j = netpin_start[i]; j < netpin_start[i+1]; ++j)
        {
            T xx = x[flat_netpin[j]]; 
            x_max = std::max(xx, x_max); 
            x_min = std::min(xx, x_min); 
            T yy = y[flat_netpin[j]]; 
            y_max = std::max(yy, y_max); 
            y_min = std::min(yy, y_min); 
        }
        for (int j = netpin_start[i]; j < netpin_start[i+1]; ++j)
        {
            // for x 
            T xx = x[flat_netpin[j]]; 
            T exp_x = exp((xx-x_max)/(*gamma)); 
            T exp_nx = exp(-(xx-x_min)/(*gamma)); 

            xexp_x_sum += xx*exp_x; 
            xexp_nx_sum += xx*exp_nx; 
            exp_x_sum += exp_x; 
            exp_nx_sum += exp_nx; 

            // for y 
            T yy = y[flat_netpin[j]]; 
            T exp_y = exp((yy-y_max)/(*gamma)); 
            T exp_ny = exp(-(yy-y_min)/(*gamma)); 

            yexp_y_sum += yy*exp_y; 
            yexp_ny_sum += yy*exp_ny; 
            exp_y_sum += exp_y; 
            exp_ny_sum += exp_ny; 
        }
        if (grad_tensor) // gradient 
        {
            T b_x = 1.0/((*gamma)*exp_x_sum);
            T a_x = (1.0 - b_x*xexp_x_sum)/exp_x_sum; 
            T b_nx = -1.0/((*gamma)*exp_nx_sum);
            T a_nx = (1.0 - b_nx*xexp_nx_sum)/exp_nx_sum; 

            T b_y = 1.0/((*gamma)*exp_y_sum);
            T a_y = (1.0 - b_y*yexp_y_sum)/exp_y_sum; 
            T b_ny = -1.0/((*gamma)*exp_ny_sum);
            T a_ny = (1.0 - b_ny*yexp_ny_sum)/exp_ny_sum; 
            for (int j = netpin_start[i]; j < netpin_start[i+1]; ++j)
            {
                // for x 
                T xx = x[flat_netpin[j]]; 
                T exp_x = exp((xx-x_max)/(*gamma)); 
                T exp_nx = exp(-(xx-x_min)/(*gamma)); 
                T xexp_x = xx*exp_x; 
                T xexp_nx = xx*exp_nx;

                grad_x_tensor[flat_netpin[j]] = ((a_x*exp_x + b_x*xexp_x) - (a_nx*exp_nx + b_nx*xexp_nx))*(*grad_tensor); 

                // for y 
                T yy = y[flat_netpin[j]]; 
                T exp_y = exp((yy-y_max)/(*gamma)); 
                T exp_ny = exp(-(yy-y_min)/(*gamma)); 
                T yexp_y = yy*exp_y; 
                T yexp_ny = yy*exp_ny;

                grad_y_tensor[flat_netpin[j]] = ((a_y*exp_y + b_y*yexp_y) - (a_ny*exp_ny + b_ny*yexp_ny))*(*grad_tensor); 
            }
        }
        else // wirelength 
        {
            T wl_x = xexp_x_sum/exp_x_sum - xexp_nx_sum/exp_nx_sum;
            T wl_y = yexp_y_sum/exp_y_sum - yexp_ny_sum/exp_ny_sum; 
            //printf("wl_x = %g, wl_y = %g\n", wl_x, wl_y);
            wl[i] = (wl_x + wl_y); 
        }
    }

    return 0; 
}

template <typename T>
void integrateNetWeightsLauncher(
        const int* flat_netpin, 
        const int* netpin_start, 
        const unsigned char* net_mask, 
        const T* net_weights, 
        T* grad_x_tensor, T* grad_y_tensor, 
        int num_nets, 
        int num_threads
        )
{
#pragma omp parallel for num_threads(num_threads)
    for (int net_id = 0; net_id < num_nets; ++net_id)
    {
        if (net_mask[net_id])
        {
            T weight = net_weights[net_id]; 
            for (int j = netpin_start[net_id]; j < netpin_start[net_id+1]; ++j)
            {
                int pin_id = flat_netpin[j]; 
                grad_x_tensor[pin_id] *= weight; 
                grad_y_tensor[pin_id] *= weight; 
            }
        }
    }
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &DREAMPLACE_NAMESPACE::weighted_average_wirelength_forward, "WeightedAverageWirelength forward");
  m.def("backward", &DREAMPLACE_NAMESPACE::weighted_average_wirelength_backward, "WeightedAverageWirelength backward");
}
