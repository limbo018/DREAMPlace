/**
 * @file   hpwl_cuda.cpp
 * @author Yibo Lin
 * @date   Jun 2018
 * @brief  Compute log-sum-exp wirelength and gradient according to NTUPlace3 
 */
#include "utility/src/torch.h"
#include "utility/src/Msg.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
int computeLogSumExpWirelengthCudaLauncher(
        const T* x, const T* y, 
        const int* flat_netpin, 
        const int* netpin_start, 
        const unsigned char* net_mask, 
        int num_nets,
        const T* gamma, 
        const T* inv_gamma, 
        T* partial_wl,
        T* grad_intermediate_x, T* grad_intermediate_y
        );

/// @brief add net weights to gradient 
template <typename T>
void integrateNetWeightsCudaLauncher(
        const int* pin2net_map, 
        const unsigned char* net_mask, 
        const T* net_weights, 
        T* grad_x_tensor, T* grad_y_tensor, 
        int num_pins
        );

#define CHECK_FLAT(x) AT_ASSERTM(x.is_cuda() && x.ndimension() == 1, #x "must be a flat tensor on GPU")
#define CHECK_EVEN(x) AT_ASSERTM((x.numel()&1) == 0, #x "must have even number of elements")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x "must be contiguous")

/// @brief Compute log-sum-exp wirelength according to NTUPlace3 
///     gamma * (log(\sum exp(x_i/gamma)) + log(\sum exp(-x_i/gamma)))
/// @param pos cell locations, array of x locations and then y locations 
/// @param flat_netpin similar to the JA array in CSR format, which is flattened from the net2pin map (array of array)
/// @param netpin_start similar to the IA array in CSR format, IA[i+1]-IA[i] is the number of pins in each net, the length of IA is number of nets + 1
/// @param netpin_values similar to the value array in CSR format, a dummy array of all ones
/// @param pin2net_map pin2net map 
/// @param net_weights weight of nets 
/// @param net_mask an array to record whether compute the where for a net or not 
/// @param gamma a scalar tensor for the parameter in the equation 
std::vector<at::Tensor> logsumexp_wirelength_forward(
        at::Tensor pos,
        at::Tensor flat_netpin,
        at::Tensor netpin_start,
        at::Tensor pin2net_map,
        at::Tensor net_weights,
        at::Tensor net_mask,
        at::Tensor gamma // a scalar tensor 
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

    int num_nets = netpin_start.numel() - 1;
    int num_pins = pos.numel() / 2;
    
    // x, y interleave 
    at::Tensor partial_wl = at::zeros({num_nets, 2}, pos.options());
    // timed with grad_in yet 
    at::Tensor grad_intermediate = at::zeros_like(pos);
    auto inv_gamma = 1.0 / gamma;

    DREAMPLACE_DISPATCH_FLOATING_TYPES(pos.type(), "computeLogSumExpWirelengthCudaLauncher", [&] {
        computeLogSumExpWirelengthCudaLauncher<scalar_t>(
            pos.data<scalar_t>(), pos.data<scalar_t>() + num_pins,
            flat_netpin.data<int>(),
            netpin_start.data<int>(),
            net_mask.data<unsigned char>(),
            num_nets,
            gamma.data<scalar_t>(), 
            inv_gamma.data<scalar_t>(),
            partial_wl.data<scalar_t>(),
            grad_intermediate.data<scalar_t>(), grad_intermediate.data<scalar_t>() + num_pins
            );
        if (net_weights.numel())
        {
            partial_wl.mul_(net_weights.view({num_nets, 1}));
        }
    });

    auto wl = partial_wl.sum();
    return {wl, grad_intermediate};
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
/// @param pin2net_map pin2net map 
/// @param net_weights weight of nets 
/// @param net_mask an array to record whether compute the where for a net or not 
/// @param gamma a scalar tensor for the parameter in the equation 
at::Tensor logsumexp_wirelength_backward(
        at::Tensor grad_pos,
        at::Tensor pos,
        at::Tensor grad_intermediate, 
        at::Tensor flat_netpin,
        at::Tensor netpin_start,
        at::Tensor pin2net_map,
        at::Tensor net_weights,
        at::Tensor net_mask,
        at::Tensor gamma
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
    CHECK_FLAT(pin2net_map);
    CHECK_CONTIGUOUS(pin2net_map);
    CHECK_FLAT(grad_intermediate);
    CHECK_EVEN(grad_intermediate);
    CHECK_CONTIGUOUS(grad_intermediate);

    at::Tensor grad_out = grad_intermediate.mul_(grad_pos);
    //int num_nets = netpin_start.numel() - 1;
    int num_pins = pos.numel() / 2;

    DREAMPLACE_DISPATCH_FLOATING_TYPES(pos.type(), "computeLogSumExpWirelengthCudaLauncher", [&] {
        if (net_weights.numel())
        {
            integrateNetWeightsCudaLauncher(
                pin2net_map.data<int>(),
                net_mask.data<unsigned char>(),
                net_weights.data<scalar_t>(),
                grad_out.data<scalar_t>(), grad_out.data<scalar_t>() + num_pins,
                num_pins);
        }
    });
    return grad_out;
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &DREAMPLACE_NAMESPACE::logsumexp_wirelength_forward, "LogSumExpWirelength forward (CUDA)");
  m.def("backward", &DREAMPLACE_NAMESPACE::logsumexp_wirelength_backward, "LogSumExpWirelength backward (CUDA)");
}
