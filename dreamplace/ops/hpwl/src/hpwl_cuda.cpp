/**
 * @file   hpwl_cuda.cpp
 * @author Yibo Lin
 * @date   Jun 2018
 * @brief  Compute half-perimeter wirelength
 */
#include "utility/src/torch.h"
#include "utility/src/utils.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
int computeHPWLCudaLauncher(const T* x, const T* y, const int* flat_netpin,
                            const int* netpin_start,
                            const unsigned char* net_mask, int num_nets,
                            T* partial_wl);

/// @brief Compute half-perimeter wirelength
/// @param pos cell locations, array of x locations and then y locations
/// @param flat_netpin similar to the JA array in CSR format, which is flattened
/// from the net2pin map (array of array)
/// @param netpin_start similar to the IA array in CSR format, IA[i+1]-IA[i] is
/// the number of pins in each net, the length of IA is number of nets + 1
/// @param net_weights weight of nets
/// @param net_mask an array to record whether compute the where for a net or
/// not
at::Tensor hpwl_forward(at::Tensor pos, at::Tensor flat_netpin,
                        at::Tensor netpin_start, at::Tensor net_weights,
                        at::Tensor net_mask) {
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

  // x then y
  int num_nets = net_mask.numel();
  at::Tensor partial_wl = at::zeros({2, num_nets}, pos.options());

  DREAMPLACE_DISPATCH_FLOATING_TYPES(
      pos, "computeHPWLCudaLauncher", [&] {
        computeHPWLCudaLauncher<scalar_t>(
            DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + pos.numel() / 2,
            DREAMPLACE_TENSOR_DATA_PTR(flat_netpin, int),
            DREAMPLACE_TENSOR_DATA_PTR(netpin_start, int),
            DREAMPLACE_TENSOR_DATA_PTR(net_mask, unsigned char), num_nets,
            DREAMPLACE_TENSOR_DATA_PTR(partial_wl, scalar_t));
      });
  // std::cout << "partial_hpwl = \n" << partial_wl << "\n";

  if (net_weights.numel()) {
    partial_wl.mul_(net_weights.view({1, num_nets}));
  }
  return partial_wl.sum();
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &DREAMPLACE_NAMESPACE::hpwl_forward, "HPWL forward (CUDA)");
}
