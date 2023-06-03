/**
 * @file   logsumexp_wirelength.cpp
 * @author Yibo Lin
 * @date   Jun 2018
 * @brief  Compute log-sum-exp wirelength and gradient according to NTUPlace3
 */
#include <cfloat>
#include "utility/src/torch.h"
#include "utility/src/utils.h"
#include "weighted_average_wirelength/src/functional.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
void computeLogSumExpWirelengthLauncher(
    const T *x, const T *y, const int *flat_netpin, const int *netpin_start,
    const unsigned char *net_mask, int num_nets, const T *gamma,
    const T *inv_gamma, T *partial_wl, T *grad_intermediate_x,
    T *grad_intermediate_y, int num_threads) {
  int chunk_size =
      DREAMPLACE_STD_NAMESPACE::max(int(num_nets / num_threads / 16), 1);
#pragma omp parallel for num_threads(num_threads) schedule(dynamic, chunk_size)
  for (int i = 0; i < num_nets; ++i) {
    if (net_mask[i]) {
      T x_max = -std::numeric_limits<T>::max();
      T x_min = std::numeric_limits<T>::max();
      T y_max = -std::numeric_limits<T>::max();
      T y_min = std::numeric_limits<T>::max();
      for (int j = netpin_start[i]; j < netpin_start[i + 1]; ++j) {
        T xx = x[flat_netpin[j]];
        x_max = DREAMPLACE_STD_NAMESPACE::max(xx, x_max);
        x_min = DREAMPLACE_STD_NAMESPACE::min(xx, x_min);
        T yy = y[flat_netpin[j]];
        y_max = DREAMPLACE_STD_NAMESPACE::max(yy, y_max);
        y_min = DREAMPLACE_STD_NAMESPACE::min(yy, y_min);
      }

      T exp_x_sum = 0;
      T exp_nx_sum = 0;

      T exp_y_sum = 0;
      T exp_ny_sum = 0;

      for (int j = netpin_start[i]; j < netpin_start[i + 1]; ++j) {
        T xx = x[flat_netpin[j]];
        T exp_x = exp((xx - x_max) * (*inv_gamma));
        T exp_nx = exp((x_min - xx) * (*inv_gamma));

        exp_x_sum += exp_x;
        exp_nx_sum += exp_nx;

        T yy = y[flat_netpin[j]];
        T exp_y = exp((yy - y_max) * (*inv_gamma));
        T exp_ny = exp((y_min - yy) * (*inv_gamma));

        exp_y_sum += exp_y;
        exp_ny_sum += exp_ny;
      }

      partial_wl[i] = (log(exp_x_sum) + log(exp_nx_sum)) * (*gamma) + x_max -
                      x_min + (log(exp_y_sum) + log(exp_ny_sum)) * (*gamma) +
                      y_max - y_min;

      T reciprocal_exp_x_sum = 1.0 / exp_x_sum;
      T reciprocal_exp_nx_sum = 1.0 / exp_nx_sum;
      T reciprocal_exp_y_sum = 1.0 / exp_y_sum;
      T reciprocal_exp_ny_sum = 1.0 / exp_ny_sum;
      for (int j = netpin_start[i]; j < netpin_start[i + 1]; ++j) {
        int jj = flat_netpin[j];

        T xx = x[jj];
        T exp_x = exp((xx - x_max) * (*inv_gamma));
        T exp_nx = exp((x_min - xx) * (*inv_gamma));
        grad_intermediate_x[jj] =
            (exp_x * reciprocal_exp_x_sum - exp_nx * reciprocal_exp_nx_sum);

        T yy = y[jj];
        T exp_y = exp((yy - y_max) * (*inv_gamma));
        T exp_ny = exp((y_min - yy) * (*inv_gamma));
        grad_intermediate_y[jj] =
            (exp_y * reciprocal_exp_y_sum - exp_ny * reciprocal_exp_ny_sum);
      }
    }
  }
}

/// @brief Compute log-sum-exp wirelength according to NTUPlace3
///     gamma * (log(\sum exp(x_i/gamma)) + log(\sum exp(-x_i/gamma)))
/// @param pos cell locations, array of x locations and then y locations
/// @param flat_netpin similar to the JA array in CSR format, which is flattened
/// from the net2pin map (array of array)
/// @param netpin_start similar to the IA array in CSR format, IA[i+1]-IA[i] is
/// the number of pins in each net, the length of IA is number of nets + 1
/// @param net_weights weight of nets
/// @param net_mask an array to record whether compute the where for a net or
/// not
/// @param gamma a scalar tensor for the parameter in the equation
std::vector<at::Tensor> logsumexp_wirelength_forward(
    at::Tensor pos, at::Tensor flat_netpin, at::Tensor netpin_start,
    at::Tensor pin2net_map, at::Tensor net_weights, at::Tensor net_mask,
    at::Tensor gamma  // a scalar tensor
) {
  CHECK_FLAT_CPU(pos);
  CHECK_EVEN(pos);
  CHECK_CONTIGUOUS(pos);
  CHECK_FLAT_CPU(flat_netpin);
  CHECK_CONTIGUOUS(flat_netpin);
  CHECK_FLAT_CPU(netpin_start);
  CHECK_CONTIGUOUS(netpin_start);
  CHECK_FLAT_CPU(net_weights);
  CHECK_CONTIGUOUS(net_weights);
  CHECK_FLAT_CPU(net_mask);
  CHECK_CONTIGUOUS(net_mask);
  CHECK_FLAT_CPU(pin2net_map);
  CHECK_CONTIGUOUS(pin2net_map);

  int num_nets = netpin_start.numel() - 1;
  int num_pins = pos.numel() / 2;

  // x, y interleave
  at::Tensor partial_wl = at::zeros({num_nets}, pos.options());
  // timed with grad_in yet
  at::Tensor grad_intermediate = at::zeros_like(pos);
  auto inv_gamma = 1.0 / gamma;

  DREAMPLACE_DISPATCH_FLOATING_TYPES(
      pos, "computeLogSumExpWirelengthLauncher", [&] {
        computeLogSumExpWirelengthLauncher<scalar_t>(
            DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + num_pins,
            DREAMPLACE_TENSOR_DATA_PTR(flat_netpin, int),
            DREAMPLACE_TENSOR_DATA_PTR(netpin_start, int),
            DREAMPLACE_TENSOR_DATA_PTR(net_mask, unsigned char), num_nets,
            DREAMPLACE_TENSOR_DATA_PTR(gamma, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(inv_gamma, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(partial_wl, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(grad_intermediate, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(grad_intermediate, scalar_t) + num_pins,
            at::get_num_threads());
        if (net_weights.numel()) {
          partial_wl.mul_(net_weights);
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
/// @param exp_xy_sum array of \sum(exp(x/gamma)) for each net and then
/// \sum(exp(y/gamma))
/// @param exp_nxy_sum array of \sum(exp(-x/gamma)) for each net and then
/// \sum(exp(-y/gamma))
/// @param flat_netpin similar to the JA array in CSR format, which is flattened
/// from the net2pin map (array of array)
/// @param netpin_start similar to the IA array in CSR format, IA[i+1]-IA[i] is
/// the number of pins in each net, the length of IA is number of nets + 1
/// @param net_weights weight of nets
/// @param net_mask an array to record whether compute the where for a net or
/// not
/// @param gamma a scalar tensor for the parameter in the equation
at::Tensor logsumexp_wirelength_backward(
    at::Tensor grad_pos, at::Tensor pos, at::Tensor grad_intermediate,
    at::Tensor flat_netpin, at::Tensor netpin_start, at::Tensor pin2net_map,
    at::Tensor net_weights, at::Tensor net_mask,
    at::Tensor gamma  // a scalar tensor
) {
  CHECK_FLAT_CPU(pos);
  CHECK_EVEN(pos);
  CHECK_CONTIGUOUS(pos);
  CHECK_FLAT_CPU(flat_netpin);
  CHECK_CONTIGUOUS(flat_netpin);
  CHECK_FLAT_CPU(netpin_start);
  CHECK_CONTIGUOUS(netpin_start);
  CHECK_FLAT_CPU(net_weights);
  CHECK_CONTIGUOUS(net_weights);
  CHECK_FLAT_CPU(net_mask);
  CHECK_CONTIGUOUS(net_mask);
  CHECK_FLAT_CPU(pin2net_map);
  CHECK_CONTIGUOUS(pin2net_map);
  CHECK_FLAT_CPU(grad_intermediate);
  CHECK_EVEN(grad_intermediate);
  CHECK_CONTIGUOUS(grad_intermediate);

  at::Tensor grad_out = grad_intermediate.mul_(grad_pos);

  DREAMPLACE_DISPATCH_FLOATING_TYPES(
      pos, "computeLogSumExpWirelengthLauncher", [&] {
        if (net_weights.numel()) {
          integrateNetWeightsLauncher<scalar_t>(
              DREAMPLACE_TENSOR_DATA_PTR(flat_netpin, int),
              DREAMPLACE_TENSOR_DATA_PTR(netpin_start, int),
              DREAMPLACE_TENSOR_DATA_PTR(net_mask, unsigned char),
              DREAMPLACE_TENSOR_DATA_PTR(net_weights, scalar_t),
              DREAMPLACE_TENSOR_DATA_PTR(grad_out, scalar_t),
              DREAMPLACE_TENSOR_DATA_PTR(grad_out, scalar_t) + pos.numel() / 2,
              netpin_start.numel() - 1, at::get_num_threads());
        }
      });
  return grad_out;
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &DREAMPLACE_NAMESPACE::logsumexp_wirelength_forward,
        "LogSumExpWirelength forward");
  m.def("backward", &DREAMPLACE_NAMESPACE::logsumexp_wirelength_backward,
        "LogSumExpWirelength backward");
}
