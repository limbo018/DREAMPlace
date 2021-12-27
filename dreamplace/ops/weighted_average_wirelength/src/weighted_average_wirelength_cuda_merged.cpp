/**
 * @file   weighted_average_wirelength_cuda_merged.cpp
 * @author Yibo Lin
 * @date   Sep 2019
 * @brief  Compute weighted-average wirelength and gradient according to e-place
 */
#include "utility/src/torch.h"
#include "utility/src/utils.h"

DREAMPLACE_BEGIN_NAMESPACE

/// @brief Compute weighted average wirelength and gradient.
/// WL = \sum_i x_i*exp(x_i/gamma) / \sum_i exp(x_i/gamma) - \sum_i
/// x_i*exp(-x_i/gamma) / \sum_i x_i*exp(-x_i/gamma), where x_i is pin location.
///
/// @param x x location of pins.
/// @param y y location of pins.
/// @param flat_netpin consists pins of each net, pins belonging to the same net
/// are abutting to each other.
/// @param netpin_start bookmark for the starting index of each net in
/// flat_netpin. The length is number of nets. The last entry equals to the
/// number of pins.
/// @param net_mask whether compute the wirelength for a net or not
/// @param net_weights weight of nets
/// @param num_nets number of nets.
/// @param inv_gamma the inverse number of gamma coefficient in weighted average
/// wirelength.
/// @param partial_wl wirelength in x and y directions of each net. The first
/// half is the wirelength in x direction, and the second half is the wirelength
/// in y direction.
/// @param grad_tensor back-propagated gradient from previous stage.
/// @param grad_x_tensor gradient in x direction.
/// @param grad_y_tensor gradient in y direction.
/// @return 0 if successfully done.
template <typename T>
int computeWeightedAverageWirelengthCudaMergedLauncher(
    const T* x, const T* y, const int* flat_netpin, const int* netpin_start,
    const unsigned char* net_mask, int num_nets, const T* inv_gamma,
    T* partial_wl, T* grad_intermediate_x, T* grad_intermediate_y);

/// @brief add net weights to gradient
template <typename T>
void integrateNetWeightsCudaLauncher(const int* pin2net_map,
                                     const unsigned char* net_mask,
                                     const T* net_weights, T* grad_x_tensor,
                                     T* grad_y_tensor, int num_pins);

/// @brief Compute weighted average wirelength and gradient.
/// WL = \sum_i x_i*exp(x_i/gamma) / \sum_i exp(x_i/gamma) - \sum_i
/// x_i*exp(-x_i/gamma) / \sum_i x_i*exp(-x_i/gamma), where x_i is pin location.
///
/// @param pos location of pins, x array followed by y array.
/// @param flat_netpin consists pins of each net, pins belonging to the same net
/// are abutting to each other.
/// @param netpin_start bookmark for the starting index of each net in
/// flat_netpin. The length is number of nets. The last entry equals to the
/// number of pins.
/// @param net_weights weight of nets
/// @param net_mask whether compute the wirelength for a net or not
/// @param inv_gamma the inverse number of gamma coefficient in weighted average
/// wirelength.
/// @return total wirelength cost.
std::vector<at::Tensor> weighted_average_wirelength_forward(
    at::Tensor pos, at::Tensor flat_netpin, at::Tensor netpin_start,
    at::Tensor pin2net_map, at::Tensor net_weights, at::Tensor net_mask,
    at::Tensor inv_gamma) {
  CHECK_FLAT_CUDA(pos);
  CHECK_EVEN(pos);
  CHECK_CONTIGUOUS(pos);
  CHECK_FLAT_CUDA(flat_netpin);
  CHECK_CONTIGUOUS(flat_netpin);
  CHECK_FLAT_CUDA(netpin_start);
  CHECK_CONTIGUOUS(netpin_start);
  CHECK_FLAT_CUDA(net_weights);
  CHECK_CONTIGUOUS(net_weights);
  CHECK_FLAT_CUDA(net_mask);
  CHECK_CONTIGUOUS(net_mask);
  CHECK_FLAT_CUDA(pin2net_map);
  CHECK_CONTIGUOUS(pin2net_map);

  int num_nets = netpin_start.numel() - 1;
  int num_pins = pos.numel() / 2;

  // x, y interleave
  at::Tensor partial_wl = at::zeros({num_nets, 2}, pos.options());
  // timed with grad_in yet
  at::Tensor grad_intermediate = at::zeros_like(pos);

  DREAMPLACE_DISPATCH_FLOATING_TYPES(
      pos, "computeWeightedAverageWirelengthCudaMergedLauncher", [&] {
        computeWeightedAverageWirelengthCudaMergedLauncher<scalar_t>(
            DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + num_pins,
            DREAMPLACE_TENSOR_DATA_PTR(flat_netpin, int),
            DREAMPLACE_TENSOR_DATA_PTR(netpin_start, int),
            DREAMPLACE_TENSOR_DATA_PTR(net_mask, unsigned char), num_nets,
            DREAMPLACE_TENSOR_DATA_PTR(inv_gamma, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(partial_wl, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(grad_intermediate, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(grad_intermediate, scalar_t) + num_pins);
        if (net_weights.numel()) {
          partial_wl.mul_(net_weights.view({num_nets, 1}));
        }
      });

  auto wl = partial_wl.sum();
  return {wl, grad_intermediate};
}

/// @brief Compute gradient
/// @param grad_pos input gradient from backward propagation
/// @param pos locations of pins
/// @param flat_netpin similar to the JA array in CSR format, which is flattened
/// from the net2pin map (array of array)
/// @param netpin_start similar to the IA array in CSR format, IA[i+1]-IA[i] is
/// the number of pins in each net, the length of IA is number of nets + 1
/// @param net_weights weight of nets
/// @param net_mask an array to record whether compute the where for a net or
/// not
/// @param inv_gamma a scalar tensor for the parameter in the equation
at::Tensor weighted_average_wirelength_backward(
    at::Tensor grad_pos, at::Tensor pos, at::Tensor grad_intermediate,
    at::Tensor flat_netpin, at::Tensor netpin_start, at::Tensor pin2net_map,
    at::Tensor net_weights, at::Tensor net_mask, at::Tensor inv_gamma) {
  CHECK_FLAT_CUDA(pos);
  CHECK_EVEN(pos);
  CHECK_CONTIGUOUS(pos);
  CHECK_FLAT_CUDA(flat_netpin);
  CHECK_CONTIGUOUS(flat_netpin);
  CHECK_FLAT_CUDA(netpin_start);
  CHECK_CONTIGUOUS(netpin_start);
  CHECK_FLAT_CUDA(net_weights);
  CHECK_CONTIGUOUS(net_weights);
  CHECK_FLAT_CUDA(net_mask);
  CHECK_CONTIGUOUS(net_mask);
  CHECK_FLAT_CUDA(pin2net_map);
  CHECK_CONTIGUOUS(pin2net_map);
  CHECK_FLAT_CUDA(grad_intermediate);
  CHECK_EVEN(grad_intermediate);
  CHECK_CONTIGUOUS(grad_intermediate);

  at::Tensor grad_out = grad_intermediate.mul_(grad_pos);
  // int num_nets = netpin_start.numel() - 1;
  int num_pins = pos.numel() / 2;

  DREAMPLACE_DISPATCH_FLOATING_TYPES(
      pos, "computeWeightedAverageWirelengthCudaMergedLauncher", [&] {
        if (net_weights.numel()) {
          integrateNetWeightsCudaLauncher(
              DREAMPLACE_TENSOR_DATA_PTR(pin2net_map, int),
              DREAMPLACE_TENSOR_DATA_PTR(net_mask, unsigned char),
              DREAMPLACE_TENSOR_DATA_PTR(net_weights, scalar_t),
              DREAMPLACE_TENSOR_DATA_PTR(grad_out, scalar_t),
              DREAMPLACE_TENSOR_DATA_PTR(grad_out, scalar_t) + num_pins,
              num_pins);
        }
      });
  return grad_out;
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &DREAMPLACE_NAMESPACE::weighted_average_wirelength_forward,
        "WeightedAverageWirelength forward (CUDA)");
  m.def("backward", &DREAMPLACE_NAMESPACE::weighted_average_wirelength_backward,
        "WeightedAverageWirelength backward (CUDA)");
}
