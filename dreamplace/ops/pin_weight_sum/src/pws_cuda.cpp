#include "utility/src/torch.h"
#include "utility/src/utils.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
int computePWSCudaLauncher(const T* net_weights, const int* flat_nodepin,
                           const int* nodepin_start, const int* pin2net_map,
                           int num_physical_nodes, T* node_weights);

/// @brief Compute node weights: the sum of all related net weights.
at::Tensor pws_forward(at::Tensor net_weights, at::Tensor flat_nodepin,
                       at::Tensor nodepin_start, at::Tensor pin2net_map,
                       int num_nodes) {
  CHECK_FLAT_CUDA(net_weights);
  CHECK_CONTIGUOUS(net_weights);
  CHECK_FLAT_CUDA(flat_nodepin);
  CHECK_CONTIGUOUS(flat_nodepin);
  CHECK_FLAT_CUDA(nodepin_start);
  CHECK_CONTIGUOUS(nodepin_start);
  CHECK_FLAT_CUDA(pin2net_map);
  CHECK_CONTIGUOUS(pin2net_map);

  int num_physical_nodes = nodepin_start.numel() - 1;
  at::Tensor node_weights = at::zeros(num_nodes, net_weights.options());
  DREAMPLACE_DISPATCH_FLOATING_TYPES(net_weights, "computePWSCudaLauncher", [&] {
    computePWSCudaLauncher<scalar_t>(
        DREAMPLACE_TENSOR_DATA_PTR(net_weights, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(flat_nodepin, int),
        DREAMPLACE_TENSOR_DATA_PTR(nodepin_start, int),
        DREAMPLACE_TENSOR_DATA_PTR(pin2net_map, int),
        num_physical_nodes,
        DREAMPLACE_TENSOR_DATA_PTR(node_weights, scalar_t));
  });
  return node_weights;
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &DREAMPLACE_NAMESPACE::pws_forward, "PWS forward (CUDA)");
}
