#include "utility/src/torch.h"
#include "utility/src/Msg.h"

DREAMPLACE_BEGIN_NAMESPACE

#define CHECK_FLAT(x) AT_ASSERTM(x.is_cuda() && x.ndimension() == 1, #x "must be a flat tensor on GPU")
#define CHECK_EVEN(x) AT_ASSERTM((x.numel() & 1) == 0, #x "must have even number of elements")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x "must be contiguous")

template <typename T>
void updatePinOffsetCudaLauncher(
    const int *flat_node2pin_start_map,
    const int *flat_node2pin_map,
    const T *node_ratios,
    const int num_nodes,
    T *pin_offset_x, T *pin_offset_y
    );

void update_pin_offset_forward(
    at::Tensor flat_node2pin_start_map,
    at::Tensor flat_node2pin_map,
    at::Tensor node_ratios,
    int num_movable_nodes,
    at::Tensor pin_offset_x,
    at::Tensor pin_offset_y
    )
{
    CHECK_FLAT(flat_node2pin_start_map);
    CHECK_CONTIGUOUS(flat_node2pin_start_map);

    CHECK_FLAT(flat_node2pin_map);
    CHECK_CONTIGUOUS(flat_node2pin_map);

    CHECK_FLAT(node_ratios);
    CHECK_CONTIGUOUS(node_ratios);

    CHECK_FLAT(pin_offset_x);
    CHECK_CONTIGUOUS(pin_offset_x);

    CHECK_FLAT(pin_offset_y);
    CHECK_CONTIGUOUS(pin_offset_y);

    DREAMPLACE_DISPATCH_FLOATING_TYPES(pin_offset_x.type(), "updatePinOffsetCudaLauncher", [&] {
        updatePinOffsetCudaLauncher<scalar_t>(
            flat_node2pin_start_map.data<int>(),
            flat_node2pin_map.data<int>(),
            node_ratios.data<scalar_t>(),
            num_movable_nodes,
            pin_offset_x.data<scalar_t>(), pin_offset_y.data<scalar_t>()
            );
    });
}

DREAMPLACE_END_NAMESPACE
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &DREAMPLACE_NAMESPACE::update_pin_offset_forward, "Update pin offset with cell scaling");
}
