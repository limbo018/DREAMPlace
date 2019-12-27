#include "utility/src/torch.h"
#include "utility/src/Msg.h"
#include "utility/src/utils.h"

DREAMPLACE_BEGIN_NAMESPACE

#define CHECK_FLAT(x) AT_ASSERTM(!x.is_cuda() && x.ndimension() == 1, #x "must be a flat tensor on CPU")
#define CHECK_EVEN(x) AT_ASSERTM((x.numel() & 1) == 0, #x "must have even number of elements")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x "must be contiguous")

template <typename T>
void updatePinOffset(
    const int *flat_node2pin_start_map,
    const int *flat_node2pin_map,
    const T *node_ratios,
    const int num_nodes,
    T *pin_offset_x, T *pin_offset_y,
    const int num_threads)
{
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < num_nodes; ++i)
    {
        T ratio = node_ratios[i];

        int start = flat_node2pin_start_map[i];
        int end = flat_node2pin_start_map[i + 1];
        for (int j = start; j < end; ++j)
        {
            int pin_id = flat_node2pin_map[j];
            pin_offset_x[pin_id] *= ratio;
            pin_offset_y[pin_id] *= ratio;
        }
    }
}

void update_pin_offset_forward(
    at::Tensor flat_node2pin_start_map,
    at::Tensor flat_node2pin_map,
    at::Tensor node_ratios,
    int num_movable_nodes,
    at::Tensor pin_offset_x,
    at::Tensor pin_offset_y,
    int num_threads)
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

    DREAMPLACE_DISPATCH_FLOATING_TYPES(pin_offset_x.type(), "updatePinOffset", [&] {
        updatePinOffset<scalar_t>(
            flat_node2pin_start_map.data<int>(),
            flat_node2pin_map.data<int>(),
            node_ratios.data<scalar_t>(),
            num_movable_nodes,
            pin_offset_x.data<scalar_t>(), pin_offset_y.data<scalar_t>(),
            num_threads
            );
    });
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &DREAMPLACE_NAMESPACE::update_pin_offset_forward, "Update pin offset with cell scaling");
}
