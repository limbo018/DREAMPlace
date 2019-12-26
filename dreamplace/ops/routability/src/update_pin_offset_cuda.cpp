#include "utility/src/torch.h"
#include "utility/src/Msg.h"

DREAMPLACE_BEGIN_NAMESPACE

#define CHECK_FLAT(x) AT_ASSERTM(x.is_cuda() && x.ndimension() == 1, #x "must be a flat tensor on GPU")
#define CHECK_EVEN(x) AT_ASSERTM((x.numel() & 1) == 0, #x "must have even number of elements")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x "must be contiguous")

template <typename T>
void updatePinOffsetCudaLauncher(
    const int num_nodes,
    const int num_movable_nodes,
    const int num_filler_nodes,
    const int *flat_node2pin_start_map,
    const int *flat_node2pin_map,
    const T *movable_nodes_ratio,
    const T filler_nodes_ratio,
    T *pin_offset_x, T *pin_offset_y);

void update_pin_offset(
    int num_nodes,
    int num_movable_nodes,
    int num_filler_nodes,
    at::Tensor flat_node2pin_start_map,
    at::Tensor flat_node2pin_map,
    at::Tensor movable_nodes_ratio,
    double filler_nodes_ratio,
    at::Tensor pin_offset_x,
    at::Tensor pin_offset_y)
{
    CHECK_FLAT(flat_node2pin_start_map);
    CHECK_CONTIGUOUS(flat_node2pin_start_map);

    CHECK_FLAT(flat_node2pin_map);
    CHECK_CONTIGUOUS(flat_node2pin_map);

    CHECK_FLAT(movable_nodes_ratio);
    CHECK_CONTIGUOUS(movable_nodes_ratio);

    CHECK_FLAT(pin_offset_x);
    CHECK_CONTIGUOUS(pin_offset_x);

    CHECK_FLAT(pin_offset_y);
    CHECK_CONTIGUOUS(pin_offset_y);

    DREAMPLACE_DISPATCH_FLOATING_TYPES(pin_offset_x.type(), "updatePinOffsetCudaLauncher", [&] {
        updatePinOffsetCudaLauncher<scalar_t>(
            num_nodes,
            num_movable_nodes,
            num_filler_nodes,
            flat_node2pin_start_map.data<int>(),
            flat_node2pin_map.data<int>(),
            movable_nodes_ratio.data<scalar_t>(),
            filler_nodes_ratio,
            pin_offset_x.data<scalar_t>(), pin_offset_y.data<scalar_t>());
    });
}

DREAMPLACE_END_NAMESPACE
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("update_pin_offset", &DREAMPLACE_NAMESPACE::update_pin_offset, "update pin offset");
}
