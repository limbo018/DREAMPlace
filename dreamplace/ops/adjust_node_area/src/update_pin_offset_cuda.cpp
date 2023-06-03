#include "utility/src/torch.h"
#include "utility/src/utils.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
void updatePinOffsetCudaLauncher(const T *node_size_x, const T *node_size_y,
                                 const int *flat_node2pin_start_map,
                                 const int *flat_node2pin_map,
                                 const T *node_ratios, const int num_nodes,
                                 T *pin_offset_x, T *pin_offset_y);

void update_pin_offset_forward(at::Tensor node_size_x, at::Tensor node_size_y,
                               at::Tensor flat_node2pin_start_map,
                               at::Tensor flat_node2pin_map,
                               at::Tensor node_ratios, int num_movable_nodes,
                               at::Tensor pin_offset_x,
                               at::Tensor pin_offset_y) {
  CHECK_FLAT_CUDA(flat_node2pin_start_map);
  CHECK_CONTIGUOUS(flat_node2pin_start_map);

  CHECK_FLAT_CUDA(flat_node2pin_map);
  CHECK_CONTIGUOUS(flat_node2pin_map);

  CHECK_FLAT_CUDA(node_ratios);
  CHECK_CONTIGUOUS(node_ratios);

  CHECK_FLAT_CUDA(pin_offset_x);
  CHECK_CONTIGUOUS(pin_offset_x);

  CHECK_FLAT_CUDA(pin_offset_y);
  CHECK_CONTIGUOUS(pin_offset_y);

  DREAMPLACE_DISPATCH_FLOATING_TYPES(
      pin_offset_x, "updatePinOffsetCudaLauncher", [&] {
        updatePinOffsetCudaLauncher<scalar_t>(
            DREAMPLACE_TENSOR_DATA_PTR(node_size_x, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(node_size_y, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(flat_node2pin_start_map, int),
            DREAMPLACE_TENSOR_DATA_PTR(flat_node2pin_map, int),
            DREAMPLACE_TENSOR_DATA_PTR(node_ratios, scalar_t),
            num_movable_nodes,
            DREAMPLACE_TENSOR_DATA_PTR(pin_offset_x, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(pin_offset_y, scalar_t));
      });
}

DREAMPLACE_END_NAMESPACE
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &DREAMPLACE_NAMESPACE::update_pin_offset_forward,
        "Update pin offset with cell scaling");
}
