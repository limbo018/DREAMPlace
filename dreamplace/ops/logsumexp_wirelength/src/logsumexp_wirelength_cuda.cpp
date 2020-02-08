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
        const T* netpin_values, 
        const unsigned char* net_mask, 
        int num_nets, 
        int num_pins, 
        const T* gamma, 
        T* exp_xy, T* exp_nxy, 
        T* exp_xy_sum, T* exp_nxy_sum,
        T* partial_wl, // wirelength of each net 
        const T* grad_tensor, 
        T* grad_x_tensor, T* grad_y_tensor // the gradient is partial total wirelength to partial pin position  
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
        at::Tensor netpin_values, // all ones 
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

    int num_nets = netpin_start.numel()-1;
    // log-sum-exp for x, log-sum-exp for -x, log-sum-exp for y, log-sum-exp for -y 
    at::Tensor partial_wl = at::zeros({4, num_nets}, pos.type());
    at::Tensor exp_xy = at::zeros_like(pos);
    at::Tensor exp_nxy = at::zeros_like(pos);
    at::Tensor exp_xy_sum = at::zeros({2*num_nets}, pos.type());
    at::Tensor exp_nxy_sum = at::zeros({2*num_nets}, pos.type());
    if (netpin_values.numel() == 0)
    {
        netpin_values = at::ones({flat_netpin.numel()}, pos.type());
    }

    DREAMPLACE_DISPATCH_FLOATING_TYPES(pos.type(), "computeLogSumExpWirelengthCudaLauncher", [&] {
            computeLogSumExpWirelengthCudaLauncher<scalar_t>(
                    DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t), DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t)+pos.numel()/2, 
                    DREAMPLACE_TENSOR_DATA_PTR(flat_netpin, int), 
                    DREAMPLACE_TENSOR_DATA_PTR(netpin_start, int), 
                    DREAMPLACE_TENSOR_DATA_PTR(netpin_values, scalar_t), 
                    DREAMPLACE_TENSOR_DATA_PTR(net_mask, unsigned char), 
                    num_nets, 
                    flat_netpin.numel(),  
                    DREAMPLACE_TENSOR_DATA_PTR(gamma, scalar_t), 
                    DREAMPLACE_TENSOR_DATA_PTR(exp_xy, scalar_t), DREAMPLACE_TENSOR_DATA_PTR(exp_nxy, scalar_t), 
                    DREAMPLACE_TENSOR_DATA_PTR(exp_xy_sum, scalar_t), DREAMPLACE_TENSOR_DATA_PTR(exp_nxy_sum, scalar_t),
                    DREAMPLACE_TENSOR_DATA_PTR(partial_wl, scalar_t), 
                    nullptr, 
                    nullptr, nullptr
                    );
            });
    if (net_weights.numel())
    {
        partial_wl.mul_(net_weights.view({1, num_nets}));
    }
    // significant speedup is achieved by using summation in ATen 
    auto wl = partial_wl.sum(); 
    return {wl, exp_xy, exp_nxy, exp_xy_sum, exp_nxy_sum}; 
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
        at::Tensor exp_xy, at::Tensor exp_nxy, 
        at::Tensor exp_xy_sum, at::Tensor exp_nxy_sum, 
        at::Tensor flat_netpin,
        at::Tensor netpin_start, 
        at::Tensor netpin_values, // all ones 
        at::Tensor pin2net_map, 
        at::Tensor net_weights, 
        at::Tensor net_mask, 
        at::Tensor gamma // a scalar tensor 
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
    int num_nets = netpin_start.numel()-1;
    int num_pins = pos.numel()/2;

    DREAMPLACE_DISPATCH_FLOATING_TYPES(pos.type(), "computeLogSumExpWirelengthCudaLauncher", [&] {
            computeLogSumExpWirelengthCudaLauncher<scalar_t>(
                    DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t), DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t)+num_pins, 
                    DREAMPLACE_TENSOR_DATA_PTR(flat_netpin, int), 
                    DREAMPLACE_TENSOR_DATA_PTR(netpin_start, int), 
                    nullptr, 
                    DREAMPLACE_TENSOR_DATA_PTR(net_mask, unsigned char), 
                    num_nets, 
                    flat_netpin.numel(),  
                    DREAMPLACE_TENSOR_DATA_PTR(gamma, scalar_t), 
                    DREAMPLACE_TENSOR_DATA_PTR(exp_xy, scalar_t), DREAMPLACE_TENSOR_DATA_PTR(exp_nxy, scalar_t), 
                    DREAMPLACE_TENSOR_DATA_PTR(exp_xy_sum, scalar_t), DREAMPLACE_TENSOR_DATA_PTR(exp_nxy_sum, scalar_t),
                    nullptr, 
                    DREAMPLACE_TENSOR_DATA_PTR(grad_pos, scalar_t), 
                    DREAMPLACE_TENSOR_DATA_PTR(grad_out, scalar_t), DREAMPLACE_TENSOR_DATA_PTR(grad_out, scalar_t)+num_pins
                    );
            if (net_weights.numel())
            {
                integrateNetWeightsCudaLauncher(
                        DREAMPLACE_TENSOR_DATA_PTR(pin2net_map, int), 
                        DREAMPLACE_TENSOR_DATA_PTR(net_mask, unsigned char), 
                        DREAMPLACE_TENSOR_DATA_PTR(net_weights, scalar_t), 
                        DREAMPLACE_TENSOR_DATA_PTR(grad_out, scalar_t), DREAMPLACE_TENSOR_DATA_PTR(grad_out, scalar_t)+num_pins,
                        num_pins
                        );
            }
            });
    return grad_out; 
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &DREAMPLACE_NAMESPACE::logsumexp_wirelength_forward, "LogSumExpWirelength forward (CUDA)");
  m.def("backward", &DREAMPLACE_NAMESPACE::logsumexp_wirelength_backward, "LogSumExpWirelength backward (CUDA)");
}
