/**
 * @file   pin_pos_cuda_segment.cpp
 * @author Xiaohan Gao
 * @date   Sep 2019
 * @brief  Given cell locations, compute pin locations on CPU 
 */

#include "utility/src/torch.h"
#include "utility/src/Msg.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
int computePinPosCudaSegmentLauncher(
	const T* x, const T* y,
	const T* pin_offset_x,
	const T* pin_offset_y,
	const long* pin2node_map,
	const int* flat_node2pin_map,
	const int* flat_node2pin_start_map,
	int num_pins,
	T* pin_x, T* pin_y
	);

template <typename T>
int computePinPosGradCudaSegmentLauncher(
	const T* grad_out_x, const T* grad_out_y,
	const T* x, const T* y,
	const T* pin_offset_x,
	const T* pin_offset_y,
	const long* pin2node_map,
	const int* flat_node2pin_map,
	const int* flat_node2pin_start_map,
	int num_nodes,
	int num_pins,
	T* grad, T* grad_y, 
    T* grad_perm_buf ///< 2*num_pins 
	);

#define CHECK_FLAT(x) AT_ASSERTM(x.is_cuda() && x.ndimension() == 1, #x "must be a flat tensor on GPU")
#define CHECK_EVEN(x) AT_ASSERTM((x.numel()&1) == 0, #x "must have even number of elements")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x "must be contiguous")

at::Tensor pin_pos_forward(
	at::Tensor pos,
	at::Tensor pin_offset_x,
	at::Tensor pin_offset_y,
	at::Tensor pin2node_map,
	at::Tensor flat_node2pin_map,
	at::Tensor flat_node2pin_start_map
	)
{
    CHECK_FLAT(pos);
    CHECK_EVEN(pos);
    CHECK_CONTIGUOUS(pos);

    auto out = at::zeros(pin_offset_x.numel()*2, pos.options());
    int num_nodes = pos.numel()/2;
    int num_pins = pin_offset_x.numel();

    DREAMPLACE_DISPATCH_FLOATING_TYPES(pos.type(), "computePinPosCudaSegmentLauncher", [&] {
            computePinPosCudaSegmentLauncher<scalar_t>(
                    pos.data<scalar_t>(), pos.data<scalar_t>()+num_nodes, 
                    pin_offset_x.data<scalar_t>(), 
                    pin_offset_y.data<scalar_t>(), 
                    pin2node_map.data<long>(), 
                    flat_node2pin_map.data<int>(), 
                    flat_node2pin_start_map.data<int>(),  
                    num_pins, 
                    out.data<scalar_t>(), out.data<scalar_t>()+num_pins
                    );
            });

    return out;
}

at::Tensor pin_pos_backward(
	at::Tensor grad_out,
	at::Tensor pos,
	at::Tensor pin_offset_x,
	at::Tensor pin_offset_y,
	at::Tensor pin2node_map,
	at::Tensor flat_node2pin_map,
	at::Tensor flat_node2pin_start_map,
	int num_physical_nodes
	)
{
    CHECK_FLAT(pos); 
    CHECK_EVEN(pos);
    CHECK_CONTIGUOUS(pos);
    CHECK_FLAT(grad_out);
    CHECK_EVEN(grad_out);
    CHECK_CONTIGUOUS(grad_out);

    auto out = at::zeros_like(pos);
    int num_nodes = pos.numel()/2;
    int num_pins = pin_offset_x.numel();
    auto grad_perm_buf = at::empty({2*num_pins}, pos.options());

    DREAMPLACE_DISPATCH_FLOATING_TYPES(pos.type(), "computePinPosGradCudaSegmentLauncher", [&] {
            computePinPosGradCudaSegmentLauncher<scalar_t>(
                    grad_out.data<scalar_t>(), grad_out.data<scalar_t>()+num_pins, 
                    pos.data<scalar_t>(), pos.data<scalar_t>()+num_nodes, 
                    pin_offset_x.data<scalar_t>(), 
                    pin_offset_y.data<scalar_t>(), 
                    pin2node_map.data<long>(), 
                    flat_node2pin_map.data<int>(), 
                    flat_node2pin_start_map.data<int>(),  
                    num_physical_nodes, 
                    num_pins, 
                    out.data<scalar_t>(), out.data<scalar_t>()+num_nodes, 
                    grad_perm_buf.data<scalar_t>()
                    );
            });

    return out;
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &DREAMPLACE_NAMESPACE::pin_pos_forward, "PinPos forward");
  m.def("backward", &DREAMPLACE_NAMESPACE::pin_pos_backward, "PinPos backward");
}
