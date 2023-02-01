#include "utility/src/torch.h"
#include "utility/src/utils.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
int computePWSLauncher(const T* net_weights, const int* flat_nodepin,
                       const int* nodepin_start, const int* pin2net_map,
                       int num_physical_nodes, int num_threads, T* node_weights);

/// @brief Compute node weights: the sum of all related net weights.
at::Tensor pws_forward(at::Tensor net_weights, at::Tensor flat_nodepin,
                       at::Tensor nodepin_start, at::Tensor pin2net_map,
                       int num_nodes) {
  CHECK_FLAT_CPU(net_weights);
  CHECK_CONTIGUOUS(net_weights);
  CHECK_FLAT_CPU(flat_nodepin);
  CHECK_CONTIGUOUS(flat_nodepin);
  CHECK_FLAT_CPU(nodepin_start);
  CHECK_CONTIGUOUS(nodepin_start);
  CHECK_FLAT_CPU(pin2net_map);
  CHECK_CONTIGUOUS(pin2net_map);

  int num_physical_nodes = nodepin_start.numel() - 1;
  at::Tensor node_weights = at::zeros(num_nodes, net_weights.options());
  DREAMPLACE_DISPATCH_FLOATING_TYPES(net_weights, "computePWSLauncher", [&] {
    computePWSLauncher<scalar_t>(
        DREAMPLACE_TENSOR_DATA_PTR(net_weights, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(flat_nodepin, int),
        DREAMPLACE_TENSOR_DATA_PTR(nodepin_start, int),
        DREAMPLACE_TENSOR_DATA_PTR(pin2net_map, int),
        num_physical_nodes, at::get_num_threads(),
        DREAMPLACE_TENSOR_DATA_PTR(node_weights, scalar_t));
  });
  return node_weights;
}

template <typename T>
int computePWSLauncher(const T* net_weights, const int* flat_nodepin,
                       const int* nodepin_start, const int* pin2net_map,
                       int num_physical_nodes, int num_threads, T* node_weights) {
#pragma omp parallel for num_threads(num_threads)
  for (int i = 0; i < num_physical_nodes; ++i) {
    for (int j = nodepin_start[i]; j < nodepin_start[i + 1]; ++j) {
      node_weights[i] += net_weights[pin2net_map[flat_nodepin[j]]];
    }
  }
  return 0;
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &DREAMPLACE_NAMESPACE::pws_forward, "PWS forward");
}
