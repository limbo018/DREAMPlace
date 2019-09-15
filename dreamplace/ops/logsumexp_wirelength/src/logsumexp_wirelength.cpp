/**
 * @file   logsumexp_wirelength.cpp
 * @author Yibo Lin
 * @date   Jun 2018
 * @brief  Compute log-sum-exp wirelength and gradient according to NTUPlace3 
 */
#include <cfloat>
#include "utility/src/torch.h"
#include "utility/src/Msg.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
int computeLogSumExpWirelengthLauncher(
        const T* x, const T* y, 
        const int* flat_netpin, 
        const int* netpin_start, 
        const unsigned char* net_mask, 
        int num_nets, 
        int num_pins, 
        const T* gamma, 
        T* exp_xy, T* exp_nxy, 
        T* exp_xy_sum, T* exp_nxy_sum,
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

/// @brief Compute log-sum-exp wirelength according to NTUPlace3 
///     gamma * (log(\sum exp(x_i/gamma)) + log(\sum exp(-x_i/gamma)))
/// @param pos cell locations, array of x locations and then y locations 
/// @param flat_netpin similar to the JA array in CSR format, which is flattened from the net2pin map (array of array)
/// @param netpin_start similar to the IA array in CSR format, IA[i+1]-IA[i] is the number of pins in each net, the length of IA is number of nets + 1
/// @param net_weights weight of nets 
/// @param net_mask an array to record whether compute the where for a net or not 
/// @param gamma a scalar tensor for the parameter in the equation 
std::vector<at::Tensor> logsumexp_wirelength_forward(
        at::Tensor pos,
        at::Tensor flat_netpin,
        at::Tensor netpin_start, 
        at::Tensor net_weights, 
        at::Tensor net_mask, 
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
    CHECK_FLAT(net_weights); 
    CHECK_CONTIGUOUS(net_weights); 
    CHECK_FLAT(net_mask);
    CHECK_CONTIGUOUS(net_mask); 

    int num_nets = netpin_start.numel()-1;
    at::Tensor wl = at::zeros({num_nets}, pos.type());
    at::Tensor exp_xy = at::zeros_like(pos);
    at::Tensor exp_nxy = at::zeros_like(pos);
    at::Tensor exp_xy_sum = at::zeros({2*(netpin_start.numel()-1)}, pos.type());
    at::Tensor exp_nxy_sum = at::zeros({2*(netpin_start.numel()-1)}, pos.type());

    AT_DISPATCH_FLOATING_TYPES(pos.type().scalarType(), "computeLogSumExpWirelengthLauncher", [&] {
            computeLogSumExpWirelengthLauncher<scalar_t>(
                    pos.data<scalar_t>(), pos.data<scalar_t>()+pos.numel()/2, 
                    flat_netpin.data<int>(), 
                    netpin_start.data<int>(), 
                    net_mask.data<unsigned char>(), 
                    num_nets, 
                    flat_netpin.numel(), 
                    gamma.data<scalar_t>(), 
                    exp_xy.data<scalar_t>(), exp_nxy.data<scalar_t>(), 
                    exp_xy_sum.data<scalar_t>(), exp_nxy_sum.data<scalar_t>(),
                    wl.data<scalar_t>(), 
                    nullptr, 
                    num_threads, 
                    nullptr, nullptr
                    );
            });

    if (net_weights.numel())
    {
        wl.mul_(net_weights);
    }

    return {wl.sum(), exp_xy, exp_nxy, exp_xy_sum, exp_nxy_sum}; 
}

/// @brief Compute gradient 
/// @param grad_pos input gradient from back-propagation 
/// @param pos locations of pins 
/// @param exp_xy array of exp(x/gamma) and then exp(y/gamma)
/// @param exp_nxy array of exp(-x/gamma) and then exp(-y/gamma)
/// @param exp_xy_sum array of \sum(exp(x/gamma)) for each net and then \sum(exp(y/gamma))
/// @param exp_nxy_sum array of \sum(exp(-x/gamma)) for each net and then \sum(exp(-y/gamma))
/// @param flat_netpin similar to the JA array in CSR format, which is flattened from the net2pin map (array of array)
/// @param netpin_start similar to the IA array in CSR format, IA[i+1]-IA[i] is the number of pins in each net, the length of IA is number of nets + 1
/// @param net_weights weight of nets 
/// @param net_mask an array to record whether compute the where for a net or not 
/// @param gamma a scalar tensor for the parameter in the equation 
at::Tensor logsumexp_wirelength_backward(
        at::Tensor grad_pos, 
        at::Tensor pos,
        at::Tensor exp_xy, at::Tensor exp_nxy, 
        at::Tensor exp_xy_sum, at::Tensor exp_nxy_sum, 
        at::Tensor flat_netpin,
        at::Tensor netpin_start, 
        at::Tensor net_weights, 
        at::Tensor net_mask, 
        at::Tensor gamma, // a scalar tensor 
        int num_threads
        ) 
{
    CHECK_FLAT(pos); 
    CHECK_EVEN(pos);
    CHECK_CONTIGUOUS(pos);
    CHECK_FLAT(exp_xy); 
    CHECK_EVEN(exp_xy);
    CHECK_CONTIGUOUS(exp_xy);
    CHECK_FLAT(exp_nxy); 
    CHECK_EVEN(exp_nxy);
    CHECK_CONTIGUOUS(exp_nxy);
    CHECK_FLAT(exp_xy_sum); 
    CHECK_EVEN(exp_xy_sum);
    CHECK_CONTIGUOUS(exp_xy_sum);
    CHECK_FLAT(exp_nxy_sum); 
    CHECK_EVEN(exp_nxy_sum);
    CHECK_CONTIGUOUS(exp_nxy_sum);
    CHECK_FLAT(flat_netpin);
    CHECK_CONTIGUOUS(flat_netpin);
    CHECK_FLAT(netpin_start);
    CHECK_CONTIGUOUS(netpin_start);
    CHECK_FLAT(net_weights); 
    CHECK_CONTIGUOUS(net_weights); 
    CHECK_FLAT(net_mask);
    CHECK_CONTIGUOUS(net_mask); 

    at::Tensor grad_out = at::zeros_like(pos);

    AT_DISPATCH_FLOATING_TYPES(pos.type().scalarType(), "computeLogSumExpWirelengthLauncher", [&] {
            computeLogSumExpWirelengthLauncher<scalar_t>(
                    pos.data<scalar_t>(), pos.data<scalar_t>()+pos.numel()/2, 
                    flat_netpin.data<int>(), 
                    netpin_start.data<int>(), 
                    net_mask.data<unsigned char>(), 
                    netpin_start.numel()-1, 
                    flat_netpin.numel(), 
                    gamma.data<scalar_t>(), 
                    exp_xy.data<scalar_t>(), exp_nxy.data<scalar_t>(), 
                    exp_xy_sum.data<scalar_t>(), exp_nxy_sum.data<scalar_t>(),
                    nullptr, 
                    grad_pos.data<scalar_t>(), 
                    num_threads, 
                    grad_out.data<scalar_t>(), grad_out.data<scalar_t>()+pos.numel()/2
                    );
            if (net_weights.numel())
            {
                integrateNetWeightsLauncher<scalar_t>(
                    flat_netpin.data<int>(), 
                    netpin_start.data<int>(), 
                    net_mask.data<unsigned char>(), 
                    net_weights.data<scalar_t>(), 
                    grad_out.data<scalar_t>(), grad_out.data<scalar_t>()+pos.numel()/2, 
                    netpin_start.numel()-1, 
                    num_threads
                    );
            }
            });
    return grad_out; 
}

template <typename T>
int computeLogSumExpWirelengthLauncher(
        const T* x, const T* y, 
        const int* flat_netpin, 
        const int* netpin_start, 
        const unsigned char* net_mask, 
        int num_nets, 
        int num_pins, 
        const T* gamma, 
        T* exp_xy, T* exp_nxy, 
        T* exp_xy_sum, T* exp_nxy_sum,
        T* wl,
        const T* grad_tensor, 
        int num_threads, 
        T* grad_x_tensor, T* grad_y_tensor 
        )
{
    T tol = 80; // tolerance to trigger numeric adjustment, which may cause precision loss  
    if (grad_tensor)
    {
#pragma omp parallel for num_threads(num_threads)
        for (int i = 0; i < num_pins; ++i)
        {
            grad_x_tensor[i] = 0; 
            grad_y_tensor[i] = 0; 
        }
    }
#pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < num_nets; ++i)
    {
        if (!net_mask[i])
        {
            continue; 
        }
        for (int k = 0; k < 2; ++k)
        {
            const T* xy = (k)? y : x; 
            T* grad = (k)? grad_y_tensor : grad_x_tensor; 

            if (grad_tensor) // gradient 
            {
                T reciprocal_exp_xy_sum = 1.0/exp_xy_sum[i+k*num_nets];
                T reciprocal_exp_nxy_sum = 1.0/exp_nxy_sum[i+k*num_nets];
                for (int j = netpin_start[i]; j < netpin_start[i+1]; ++j)
                {
                    // I assume one pin will only appear in one net 
                    // for x 
                    grad[flat_netpin[j]] = (exp_xy[flat_netpin[j]+k*netpin_start[num_nets]]*reciprocal_exp_xy_sum - exp_nxy[flat_netpin[j]+k*netpin_start[num_nets]]*reciprocal_exp_nxy_sum)*(*grad_tensor); 
                }
            }
            else // wirelength 
            {
                exp_xy_sum[i+k*num_nets] = 0; 
                exp_nxy_sum[i+k*num_nets] = 0; 

                T xy_max = -std::numeric_limits<T>::max(); // maximum x to resolve numerical overflow
                T xy_min = std::numeric_limits<T>::max(); // minimum x to resolve numerical overflow

                for (int j = netpin_start[i]; j < netpin_start[i+1]; ++j)
                {
                    // for x 
                    T xx = xy[flat_netpin[j]]; 
                    xy_max = std::max(xy_max, xx); 
                    xy_min = std::min(xy_min, xx); 
                }
                if (xy_max < tol*(*gamma))
                {
                    xy_max = 0; 
                }
                if (xy_min > -tol*(*gamma))
                {
                    xy_min = 0; 
                }
                for (int j = netpin_start[i]; j < netpin_start[i+1]; ++j)
                {
                    // for x 
                    T xx = xy[flat_netpin[j]]; 
                    exp_xy[flat_netpin[j]+k*netpin_start[num_nets]] = exp((xx-xy_max)/(*gamma)); 
                    exp_nxy[flat_netpin[j]+k*netpin_start[num_nets]] = exp(-(xx-xy_min)/(*gamma)); 

                    exp_xy_sum[i+k*num_nets] += exp_xy[flat_netpin[j]+k*netpin_start[num_nets]]; 
                    exp_nxy_sum[i+k*num_nets] += exp_nxy[flat_netpin[j]+k*netpin_start[num_nets]]; 
                }

                T log_exp_xy_sum = log(exp_xy_sum[i+k*num_nets])*(*gamma) + xy_max; 
                T log_exp_nxy_sum = log(exp_nxy_sum[i+k*num_nets])*(*gamma) - xy_min; 

                T wl_xy = log_exp_xy_sum + log_exp_nxy_sum;
                wl[i] += wl_xy; 
            }
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
  m.def("forward", &DREAMPLACE_NAMESPACE::logsumexp_wirelength_forward, "LogSumExpWirelength forward");
  m.def("backward", &DREAMPLACE_NAMESPACE::logsumexp_wirelength_backward, "LogSumExpWirelength backward");
}
