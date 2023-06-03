#include "utility/src/torch.h"
#include "utility/src/utils.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
void updatePinOffset(const T *node_size_x, const T *node_size_y,
                     const int *flat_node2pin_start_map,
                     const int *flat_node2pin_map, const T *node_ratios,
                     const int num_nodes, T *pin_offset_x, T *pin_offset_y,
                     const int num_threads) {
#pragma omp parallel for num_threads(num_threads)
  for (int i = 0; i < num_nodes; ++i) {
    T ratio = (node_ratios[i] - 1) / 2;
    T sx = node_size_x[i];
    T sy = node_size_y[i];

    int start = flat_node2pin_start_map[i];
    int end = flat_node2pin_start_map[i + 1];
    for (int j = start; j < end; ++j) {
      int pin_id = flat_node2pin_map[j];
      pin_offset_x[pin_id] += ratio * sx;
      pin_offset_y[pin_id] += ratio * sy;
    }
  }
}

void update_pin_offset_forward(at::Tensor node_size_x, at::Tensor node_size_y,
                               at::Tensor flat_node2pin_start_map,
                               at::Tensor flat_node2pin_map,
                               at::Tensor node_ratios, int num_movable_nodes,
                               at::Tensor pin_offset_x,
                               at::Tensor pin_offset_y) {
  CHECK_FLAT_CPU(flat_node2pin_start_map);
  CHECK_CONTIGUOUS(flat_node2pin_start_map);

  CHECK_FLAT_CPU(flat_node2pin_map);
  CHECK_CONTIGUOUS(flat_node2pin_map);

  CHECK_FLAT_CPU(node_ratios);
  CHECK_CONTIGUOUS(node_ratios);

  CHECK_FLAT_CPU(pin_offset_x);
  CHECK_CONTIGUOUS(pin_offset_x);

  CHECK_FLAT_CPU(pin_offset_y);
  CHECK_CONTIGUOUS(pin_offset_y);

  DREAMPLACE_DISPATCH_FLOATING_TYPES(
      pin_offset_x, "updatePinOffset", [&] {
        updatePinOffset<scalar_t>(
            DREAMPLACE_TENSOR_DATA_PTR(node_size_x, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(node_size_y, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(flat_node2pin_start_map, int),
            DREAMPLACE_TENSOR_DATA_PTR(flat_node2pin_map, int),
            DREAMPLACE_TENSOR_DATA_PTR(node_ratios, scalar_t),
            num_movable_nodes,
            DREAMPLACE_TENSOR_DATA_PTR(pin_offset_x, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(pin_offset_y, scalar_t),
            at::get_num_threads());
      });
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &DREAMPLACE_NAMESPACE::update_pin_offset_forward,
        "Update pin offset with cell scaling");
}
