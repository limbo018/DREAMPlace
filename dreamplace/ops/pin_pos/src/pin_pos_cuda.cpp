/**
 * @file   pin_pos_cuda.cpp
 * @author Xiaohan Gao
 * @date   Sep 2019
 * @brief  Given cell locations, compute pin locations on CPU
 */

#include "utility/src/torch.h"
#include "utility/src/utils.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
int computePinPosCudaLauncher(const T* x, const T* y, const T* pin_offset_x,
                              const T* pin_offset_y, const long* pin2node_map,
                              const int* flat_node2pin_map,
                              const int* flat_node2pin_start_map, int num_pins,
                              T* pin_x, T* pin_y);

template <typename T>
int computePinPosGradCudaLauncher(
    const T* grad_out_x, const T* grad_out_y, const T* x, const T* y,
    const T* pin_offset_x, const T* pin_offset_y, const long* pin2node_map,
    const int* flat_node2pin_map, const int* flat_node2pin_start_map,
    int num_nodes, int num_pins, T* grad, T* grad_y);

at::Tensor pin_pos_forward(at::Tensor pos, at::Tensor pin_offset_x,
                           at::Tensor pin_offset_y, at::Tensor pin2node_map,
                           at::Tensor flat_node2pin_map,
                           at::Tensor flat_node2pin_start_map) {
  CHECK_FLAT_CUDA(pos);
  CHECK_EVEN(pos);
  CHECK_CONTIGUOUS(pos);

  auto out = at::zeros(pin_offset_x.numel() * 2, pos.options());
  int num_nodes = pos.numel() / 2;
  int num_pins = pin_offset_x.numel();

  DREAMPLACE_DISPATCH_FLOATING_TYPES(
      pos, "computePinPosCudaLauncher", [&] {
        computePinPosCudaLauncher<scalar_t>(
            DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + num_nodes,
            DREAMPLACE_TENSOR_DATA_PTR(pin_offset_x, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(pin_offset_y, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(pin2node_map, long),
            DREAMPLACE_TENSOR_DATA_PTR(flat_node2pin_map, int),
            DREAMPLACE_TENSOR_DATA_PTR(flat_node2pin_start_map, int), num_pins,
            DREAMPLACE_TENSOR_DATA_PTR(out, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(out, scalar_t) + num_pins);
      });

  return out;
}

at::Tensor pin_pos_backward(at::Tensor grad_out, at::Tensor pos,
                            at::Tensor pin_offset_x, at::Tensor pin_offset_y,
                            at::Tensor pin2node_map,
                            at::Tensor flat_node2pin_map,
                            at::Tensor flat_node2pin_start_map,
                            int num_physical_nodes) {
  CHECK_FLAT_CUDA(pos);
  CHECK_EVEN(pos);
  CHECK_CONTIGUOUS(pos);
  CHECK_FLAT_CUDA(grad_out);
  CHECK_EVEN(grad_out);
  CHECK_CONTIGUOUS(grad_out);

  auto out = at::zeros_like(pos);
  int num_nodes = pos.numel() / 2;
  int num_pins = pin_offset_x.numel();

  DREAMPLACE_DISPATCH_FLOATING_TYPES(
      pos, "computePinPosGradCudaLauncher", [&] {
        computePinPosGradCudaLauncher<scalar_t>(
            DREAMPLACE_TENSOR_DATA_PTR(grad_out, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(grad_out, scalar_t) + num_pins,
            DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + num_nodes,
            DREAMPLACE_TENSOR_DATA_PTR(pin_offset_x, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(pin_offset_y, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(pin2node_map, long),
            DREAMPLACE_TENSOR_DATA_PTR(flat_node2pin_map, int),
            DREAMPLACE_TENSOR_DATA_PTR(flat_node2pin_start_map, int),
            num_physical_nodes, num_pins,
            DREAMPLACE_TENSOR_DATA_PTR(out, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(out, scalar_t) + num_nodes);
      });

  return out;
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &DREAMPLACE_NAMESPACE::pin_pos_forward, "PinPos forward");
  m.def("backward", &DREAMPLACE_NAMESPACE::pin_pos_backward, "PinPos backward");
}
