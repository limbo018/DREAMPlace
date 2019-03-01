/**
 * @file   hpwl_cuda.cpp
 * @author Yibo Lin
 * @date   Jun 2018
 */
#include <torch/torch.h>
#include <limits>

/// @brief Compute weighted average wirelength and gradient. 
/// WL = \sum_i x_i*exp(x_i/gamma) / \sum_i exp(x_i/gamma) - \sum_i x_i*exp(-x_i/gamma) / \sum_i x_i*exp(-x_i/gamma), 
/// where x_i is pin location.
/// 
/// @param x x location of pins. 
/// @param y y location of pins. 
/// @param flat_netpin consists pins of each net, pins belonging to the same net are abutting to each other. 
/// @param netpin_start bookmark for the starting index of each net in flat_netpin. The length is number of nets. The last entry equals to the number of pins. 
/// @param net_mask whether compute the wirelength for a net or not 
/// @param num_nets number of nets. 
/// @param gamma gamma coefficient in weighted average wirelength. 
/// @param partial_wl wirelength in x and y directions of each net. The first half is the wirelength in x direction, and the second half is the wirelength in y direction. 
/// @param grad_tensor back-propagated gradient from previous stage.
/// @param grad_x_tensor gradient in x direction.
/// @param grad_y_tensor gradient in y direction.
/// @return 0 if successfully done.
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
        );

#define CHECK_FLAT(x) AT_ASSERTM(x.is_cuda() && x.ndimension() == 1, #x "must be a flat tensor on GPU")
#define CHECK_EVEN(x) AT_ASSERTM((x.numel()&1) == 0, #x "must have even number of elements")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x "must be contiguous")

/// @brief Compute weighted average wirelength and gradient. 
/// WL = \sum_i x_i*exp(x_i/gamma) / \sum_i exp(x_i/gamma) - \sum_i x_i*exp(-x_i/gamma) / \sum_i x_i*exp(-x_i/gamma), 
/// where x_i is pin location.
/// 
/// @param pos location of pins, x array followed by y array. 
/// @param flat_netpin consists pins of each net, pins belonging to the same net are abutting to each other. 
/// @param netpin_start bookmark for the starting index of each net in flat_netpin. The length is number of nets. The last entry equals to the number of pins. 
/// @param net_mask whether compute the wirelength for a net or not 
/// @param gamma gamma coefficient in weighted average wirelength. 
/// @return total wirelength cost.
at::Tensor weighted_average_wirelength_forward(
        at::Tensor pos,
        at::Tensor flat_netpin,
        at::Tensor netpin_start, 
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
    at::Tensor partial_wl = at::zeros_like(pos);

    AT_DISPATCH_FLOATING_TYPES(pos.type(), "computeWeightedAverageWirelengthCudaLauncher", [&] {
            computeWeightedAverageWirelengthCudaLauncher<scalar_t>(
                    pos.data<scalar_t>(), pos.data<scalar_t>()+pos.numel()/2, 
                    flat_netpin.data<int>(), 
                    netpin_start.data<int>(), 
                    net_mask.data<unsigned char>(), 
                    netpin_start.numel()-1, 
                    gamma.data<scalar_t>(), 
                    partial_wl.data<scalar_t>(), 
                    nullptr, 
                    nullptr, nullptr
                    );
            });

    auto wl = partial_wl.sum();
    return wl; 
}

at::Tensor weighted_average_wirelength_backward(
        at::Tensor grad_pos, 
        at::Tensor pos,
        at::Tensor flat_netpin,
        at::Tensor netpin_start, 
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
    at::Tensor grad_out = at::zeros_like(pos);

    AT_DISPATCH_FLOATING_TYPES(pos.type(), "computeWeightedAverageWirelengthCudaLauncher", [&] {
            computeWeightedAverageWirelengthCudaLauncher<scalar_t>(
                    pos.data<scalar_t>(), pos.data<scalar_t>()+pos.numel()/2, 
                    flat_netpin.data<int>(), 
                    netpin_start.data<int>(), 
                    net_mask.data<unsigned char>(), 
                    netpin_start.numel()-1, 
                    gamma.data<scalar_t>(), 
                    nullptr, 
                    grad_pos.data<scalar_t>(), 
                    grad_out.data<scalar_t>(), grad_out.data<scalar_t>()+pos.numel()/2
                    );
            });
    return grad_out; 
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &weighted_average_wirelength_forward, "WeightedAverageWirelength forward (CUDA)");
  m.def("backward", &weighted_average_wirelength_backward, "WeightedAverageWirelength backward (CUDA)");
}
