/**
 * @file   weighted_average_wirelength_cuda_atomic.cpp
 * @author Yibo Lin
 * @date   Aug 2018
 */
#include "utility/src/torch.h"
#include "utility/src/utils.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T, typename V>
int computeWeightedAverageWirelengthCudaAtomicLauncher(
    const T *pos, const int *pin2net_map, const int *flat_netpin,
    const int *netpin_start, const unsigned char *net_mask, int num_nets,
    int num_pins, const T *inv_gamma, T *exp_xy, T *exp_nxy, T *exp_xy_sum,
    T *exp_nxy_sum, T *xyexp_xy_sum, T *xyexp_nxy_sum, V *xy_max, V *xy_min,
    T *partial_wl,  // wirelength of each net
    const T *grad_tensor, T *grad_x_tensor,
    T *grad_y_tensor  // the gradient is partial total wirelength to partial pin
                      // position
);

/// @brief add net weights to gradient
template <typename T>
void integrateNetWeightsCudaLauncher(const int *pin2net_map,
                                     const unsigned char *net_mask,
                                     const T *net_weights, T *grad_x_tensor,
                                     T *grad_y_tensor, int num_pins);

typedef int V;

/// @brief Compute weighted average wirelength and gradient.
/// WL = \sum_i x_i*exp(x_i/gamma) / \sum_i exp(x_i/gamma) - \sum_i
/// x_i*exp(-x_i/gamma) / \sum_i x_i*exp(-x_i/gamma), where x_i is pin location.
///
/// @param pos location of pins, x array followed by y array.
/// @param pin2net_map map pin to net
/// @param net_weights weight of nets
/// @param net_mask whether compute the wirelength for a net or not
/// @param inv_gamma inverse of gamma coefficient in weighted average
/// wirelength.
/// @return total wirelength cost.
std::vector<at::Tensor> weighted_average_wirelength_atomic_forward(
    at::Tensor pos, at::Tensor pin2net_map, at::Tensor flat_netpin,
    at::Tensor netpin_start, at::Tensor net_weights, at::Tensor net_mask,
    at::Tensor inv_gamma) {
  CHECK_FLAT_CUDA(pos);
  CHECK_EVEN(pos);
  CHECK_CONTIGUOUS(pos);
  CHECK_FLAT_CUDA(pin2net_map);
  CHECK_CONTIGUOUS(pin2net_map);
  CHECK_FLAT_CUDA(flat_netpin);
  CHECK_CONTIGUOUS(flat_netpin);
  CHECK_FLAT_CUDA(netpin_start);
  CHECK_CONTIGUOUS(netpin_start);
  CHECK_FLAT_CUDA(net_mask);
  CHECK_CONTIGUOUS(net_mask);
  CHECK_FLAT_CUDA(net_weights);
  CHECK_CONTIGUOUS(net_weights);

  int num_nets = net_mask.numel();
  int num_pins = pin2net_map.numel();

  // log-sum-exp for x, log-sum-exp for -x, log-sum-exp for y, log-sum-exp for
  // -y
  at::Tensor exp_xy = at::empty_like(pos);
  at::Tensor exp_nxy = at::empty_like(pos);
  at::Tensor exp_xy_sum = at::zeros({2, num_nets}, pos.options());
  at::Tensor exp_nxy_sum = at::zeros({2, num_nets}, pos.options());
  at::Tensor xyexp_xy_sum = at::zeros({2, num_nets}, pos.options());
  at::Tensor xyexp_nxy_sum = at::zeros({2, num_nets}, pos.options());
  at::Tensor partial_wl = at::zeros({num_nets}, pos.options());

  // it is ok for xy_max and xy_min to be integer
  // we do not really need accurate max/min, just some values to scale x/y
  // therefore, there is no need to scale xy_max and xy_min to improve accuracy
  at::Tensor xy_max = at::full({2, num_nets}, std::numeric_limits<V>::min(),
                               at::CUDA(at::kInt));
  at::Tensor xy_min = at::full({2, num_nets}, std::numeric_limits<V>::max(),
                               at::CUDA(at::kInt));

  DREAMPLACE_DISPATCH_FLOATING_TYPES(
      pos, "computeWeightedAverageWirelengthCudaAtomicLauncher", [&] {
        computeWeightedAverageWirelengthCudaAtomicLauncher<scalar_t, V>(
            DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t),  // x then y
            DREAMPLACE_TENSOR_DATA_PTR(pin2net_map, int),
            DREAMPLACE_TENSOR_DATA_PTR(flat_netpin, int),
            DREAMPLACE_TENSOR_DATA_PTR(netpin_start, int),
            DREAMPLACE_TENSOR_DATA_PTR(net_mask, unsigned char), num_nets,
            num_pins, DREAMPLACE_TENSOR_DATA_PTR(inv_gamma, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(exp_xy, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(exp_nxy, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(exp_xy_sum, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(exp_nxy_sum, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(xyexp_xy_sum, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(xyexp_nxy_sum, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(xy_max, V),
            DREAMPLACE_TENSOR_DATA_PTR(xy_min, V),
            DREAMPLACE_TENSOR_DATA_PTR(partial_wl, scalar_t), nullptr, nullptr,
            nullptr);
      });

  if (net_weights.numel()) {
    partial_wl.mul_(net_weights.view({num_nets}));
  }
  // significant speedup is achieved by using summation in ATen
  auto wl = partial_wl.sum();
  return {wl,          exp_xy,       exp_nxy,      exp_xy_sum,
          exp_nxy_sum, xyexp_xy_sum, xyexp_nxy_sum};
}

/// @brief Compute gradient
/// @param grad_pos input gradient from backward propagation
/// @param pos locations of pins
/// @param exp_xy array of exp(x/gamma) and then exp(y/gamma)
/// @param exp_nxy array of exp(-x/gamma) and then exp(-y/gamma)
/// @param exp_xy_sum array of \sum(exp(x/gamma)) for each net and then
/// \sum(exp(y/gamma))
/// @param exp_nxy_sum array of \sum(exp(-x/gamma)) for each net and then
/// \sum(exp(-y/gamma))
/// @param xyexp_xy_sum array of \sum(x*exp(x/gamma)) for each net and then
/// \sum(y*exp(y/gamma))
/// @param xyexp_nxy_sum array of \sum(x*exp(-x/gamma)) for each net and then
/// \sum(y*exp(-y/gamma))
/// @param pin2net_map map pin to net
/// @param net_weights weight of nets
/// @param net_mask an array to record whether compute the where for a net or
/// not
/// @param inv_gamma inverse of gamma, a scalar tensor for the parameter in the
/// equation
at::Tensor weighted_average_wirelength_atomic_backward(
    at::Tensor grad_pos, at::Tensor pos, at::Tensor exp_xy, at::Tensor exp_nxy,
    at::Tensor exp_xy_sum, at::Tensor exp_nxy_sum, at::Tensor xyexp_xy_sum,
    at::Tensor xyexp_nxy_sum, at::Tensor pin2net_map, at::Tensor flat_netpin,
    at::Tensor netpin_start, at::Tensor net_weights, at::Tensor net_mask,
    at::Tensor inv_gamma) {
  CHECK_FLAT_CUDA(pos);
  CHECK_EVEN(pos);
  CHECK_CONTIGUOUS(pos);
  CHECK_FLAT_CUDA(exp_xy);
  CHECK_EVEN(exp_xy);
  CHECK_CONTIGUOUS(exp_xy);
  CHECK_FLAT_CUDA(exp_nxy);
  CHECK_EVEN(exp_nxy);
  CHECK_CONTIGUOUS(exp_nxy);
  CHECK_FLAT_CUDA(exp_xy_sum);
  CHECK_EVEN(exp_xy_sum);
  CHECK_CONTIGUOUS(exp_xy_sum);
  CHECK_FLAT_CUDA(exp_nxy_sum);
  CHECK_EVEN(exp_nxy_sum);
  CHECK_CONTIGUOUS(exp_nxy_sum);
  CHECK_FLAT_CUDA(xyexp_xy_sum);
  CHECK_EVEN(xyexp_xy_sum);
  CHECK_CONTIGUOUS(xyexp_xy_sum);
  CHECK_FLAT_CUDA(xyexp_nxy_sum);
  CHECK_EVEN(xyexp_nxy_sum);
  CHECK_CONTIGUOUS(xyexp_nxy_sum);
  CHECK_FLAT_CUDA(pin2net_map);
  CHECK_CONTIGUOUS(pin2net_map);
  CHECK_FLAT_CUDA(flat_netpin);
  CHECK_CONTIGUOUS(flat_netpin);
  CHECK_FLAT_CUDA(netpin_start);
  CHECK_CONTIGUOUS(netpin_start);
  CHECK_FLAT_CUDA(net_mask);
  CHECK_CONTIGUOUS(net_mask);
  CHECK_FLAT_CUDA(net_weights);
  CHECK_CONTIGUOUS(net_weights);

  at::Tensor grad_out = at::zeros_like(pos);

  int num_nets = net_mask.numel();
  int num_pins = pin2net_map.numel();

  DREAMPLACE_DISPATCH_FLOATING_TYPES(
      pos, "computeWeightedAverageWirelengthCudaAtomicLauncher", [&] {
        computeWeightedAverageWirelengthCudaAtomicLauncher<scalar_t, V>(
            DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t),  // x then y
            DREAMPLACE_TENSOR_DATA_PTR(pin2net_map, int),
            DREAMPLACE_TENSOR_DATA_PTR(flat_netpin, int),
            DREAMPLACE_TENSOR_DATA_PTR(netpin_start, int),
            DREAMPLACE_TENSOR_DATA_PTR(net_mask, unsigned char), num_nets,
            num_pins, DREAMPLACE_TENSOR_DATA_PTR(inv_gamma, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(exp_xy, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(exp_nxy, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(exp_xy_sum, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(exp_nxy_sum, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(xyexp_xy_sum, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(xyexp_nxy_sum, scalar_t), nullptr,
            nullptr, nullptr, DREAMPLACE_TENSOR_DATA_PTR(grad_pos, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(grad_out, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(grad_out, scalar_t) + num_pins);
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
  m.def("forward",
        &DREAMPLACE_NAMESPACE::weighted_average_wirelength_atomic_forward,
        "WeightedAverageWirelength forward (CUDA)");
  m.def("backward",
        &DREAMPLACE_NAMESPACE::weighted_average_wirelength_atomic_backward,
        "WeightedAverageWirelength backward (CUDA)");
}
