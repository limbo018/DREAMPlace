/**
 * @file   hpwl.cpp
 * @author Yibo Lin
 * @date   Jun 2018
 * @brief  Compute half-perimeter wirelength
 */
#include "utility/src/torch.h"
#include "utility/src/utils.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
int computeHPWLLauncher(const T* x, const T* y, const int* flat_netpin,
                        const int* netpin_start, const unsigned char* net_mask,
                        int num_nets, int num_threads, T* hpwl);

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

  int num_nets = netpin_start.numel() - 1;
  at::Tensor hpwl = at::zeros(num_nets, pos.options());
  DREAMPLACE_DISPATCH_FLOATING_TYPES(pos, "computeHPWLLauncher", [&] {
    computeHPWLLauncher<scalar_t>(
        DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + pos.numel() / 2,
        DREAMPLACE_TENSOR_DATA_PTR(flat_netpin, int),
        DREAMPLACE_TENSOR_DATA_PTR(netpin_start, int),
        DREAMPLACE_TENSOR_DATA_PTR(net_mask, unsigned char), num_nets,
        at::get_num_threads(), DREAMPLACE_TENSOR_DATA_PTR(hpwl, scalar_t));
  });
  if (net_weights.numel()) {
    hpwl.mul_(net_weights);
  }
  return hpwl.sum();
}

template <typename T>
int computeHPWLLauncher(const T* x, const T* y, const int* flat_netpin,
                        const int* netpin_start, const unsigned char* net_mask,
                        int num_nets, int num_threads, T* hpwl) {
#pragma omp parallel for num_threads(num_threads)
  for (int i = 0; i < num_nets; ++i) {
    T max_x = -std::numeric_limits<T>::max();
    T min_x = std::numeric_limits<T>::max();
    T max_y = -std::numeric_limits<T>::max();
    T min_y = std::numeric_limits<T>::max();

    // ignore large degree nets
    if (net_mask[i]) {
      for (int j = netpin_start[i]; j < netpin_start[i + 1]; ++j) {
        min_x = std::min(min_x, x[flat_netpin[j]]);
        max_x = std::max(max_x, x[flat_netpin[j]]);
        min_y = std::min(min_y, y[flat_netpin[j]]);
        max_y = std::max(max_y, y[flat_netpin[j]]);
      }
      hpwl[i] = max_x - min_x + max_y - min_y;
    }
  }

  return 0;
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &DREAMPLACE_NAMESPACE::hpwl_forward, "HPWL forward");
}
