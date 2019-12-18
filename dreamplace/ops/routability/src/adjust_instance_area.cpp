#include "utility/src/torch.h"
#include "utility/src/Msg.h"
#include "utility/src/utils.h"

DREAMPLACE_BEGIN_NAMESPACE

#define CHECK_FLAT(x) AT_ASSERTM(!x.is_cuda() && x.ndimension() == 1, #x "must be a flat tensor on CPU")
#define CHECK_EVEN(x) AT_ASSERTM((x.numel() & 1) == 0, #x "must have even number of elements")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x "must be contiguous")

#define route_area_adjust_stop_ratio (0.01)
#define area_adjust_stop_ratio (0.01)
template <typename T>
int updatePinOffset(
    const int num_nodes,
    const int num_movable_nodes,
    const int num_filler_nodes,
    const int* flat_node2pin_start_map,
    const int* flat_node2pin_map,
    const T* movable_nodes_ratio,
    const T* filler_nodes_ratio,
    T* pin_offset_x, T* pin_offset_y,
    const int num_threads)
{
{
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < num_movable_nodes; ++i)
    {   
        T ratio = movable_nodes_ratio[i];
        
        int start = flat_node2pin_start_map[i]; 
        int end = flat_node2pin_start_map[i+1];
        for (int j = start; j < end; ++j)
        {
            int pin_id = flat_node2pin_map[j]; 
            pin_offset_x[pin_id] *= ratio;
            pin_offset_y[pin_id] *= ratio;
        }
    }

    #pragma omp parallel for num_threads(num_threads)
    for (int i = num_nodes - num_filler_nodes; i < num_nodes; ++i)
    {   
        int start = flat_node2pin_start_map[i]; 
        int end = flat_node2pin_start_map[i+1];
        for (int j = start; j < end; ++j)
        {
            int pin_id = flat_node2pin_map[j]; 
            pin_offset_x[pin_id] *= *filler_nodes_ratio;
            pin_offset_y[pin_id] *= *filler_nodes_ratio;
        }
    }

    return 0; 
}
}

bool adjust_instance_area(
    at::Tensor pos,
    at::Tensor pin_pos,
    at::Tensor node_size_x,
    at::Tensor node_size_y,
    at::Tensor netpin_start,
    at::Tensor flat_netpin,
    int num_nodes,
    int num_movable_nodes,
    int num_filler_nodes,
    at::Tensor instance_route_area,
    at::Tensor max_total_area,
    int num_threads,
    bool* adjust_area_flag,
    bool* adjust_route_area_flag,
    at::Tensor pin_offset_x,
    at::Tensor pin_offset_y,
    at::Tensor flat_node2pin_start_map,
    at::Tensor flat_node2pin_map)
{
    CHECK_FLAT(instance_route_area);
    CHECK_CONTIGUOUS(instance_route_area);

    CHECK_FLAT(pos);
    CHECK_EVEN(pos);
    CHECK_CONTIGUOUS(pos);

    CHECK_FLAT(pin_pos);
    CHECK_EVEN(pin_pos);
    CHECK_CONTIGUOUS(pin_pos);

    CHECK_FLAT(netpin_start);
    CHECK_CONTIGUOUS(netpin_start);

    CHECK_FLAT(flat_netpin);
    CHECK_CONTIGUOUS(flat_netpin);

    CHECK_FLAT(node_size_x);
    CHECK_CONTIGUOUS(node_size_x);

    CHECK_FLAT(node_size_y);
    CHECK_CONTIGUOUS(node_size_y);

    CHECK_FLAT(max_total_area);
    CHECK_CONTIGUOUS(max_total_area);

    at::Tensor node_size_x_movable = at::narrow(node_size_x, 0, 0, num_movable_nodes);
    at::Tensor node_size_y_movable = at::narrow(node_size_y, 0, 0, num_movable_nodes);
    at::Tensor node_size_x_filler = at::narrow(node_size_x, 0, num_nodes - num_filler_nodes, num_filler_nodes);
    at::Tensor node_size_y_filler = at::narrow(node_size_y, 0, num_nodes - num_filler_nodes, num_filler_nodes);
    
    at::Tensor old_movable_area = node_size_x_movable.mul(node_size_y_movable);
    at::Tensor old_sum_movable_area = old_movable_area.sum();

    // compute final areas
    at::Tensor route_area_increment = at::relu(instance_route_area - old_movable_area);
    at::Tensor route_area_increment_sum = route_area_increment.sum();

    // check whether the total area is larger than the max area requirement
    // If yes, scale the extra area to meet the requirement
    // We assume the total base area is no greater than the max area requirement
    at::Tensor scale_factor = at::clamp((max_total_area - old_sum_movable_area) / route_area_increment_sum, 0, 1);

    // compute the adjusted area increment following scaling factor
    at::Tensor movable_area_increment = scale_factor * route_area_increment;
    at::Tensor movable_area_increment_sum = scale_factor * route_area_increment_sum;
    
    // set the areas of movable instance as base_area + scaled extra area
    at::Tensor new_movable_area = old_movable_area + movable_area_increment;

    *adjust_route_area_flag = *adjust_route_area_flag && 
                              (route_area_increment_sum / old_sum_movable_area > route_area_adjust_stop_ratio).data<bool>();
    *adjust_area_flag = *adjust_area_flag && 
                        *adjust_route_area_flag &&
                        (movable_area_increment_sum / old_sum_movable_area > area_adjust_stop_ratio).data<bool>();
    if (!(*adjust_area_flag))
    {
        return false;
    }

    // adjust the size of movable nodes
    at::Tensor movable_nodes_ratio = at::sqrt(new_movable_area / old_movable_area);
    node_size_x_movable.mul_(movable_nodes_ratio);
    node_size_y_movable.mul_(movable_nodes_ratio);
    
    // scale the filler instance areas to make the total area meets the max area requirement
    at::Tensor old_sum_filler_area = node_size_x_filler.mul(node_size_y_filler).sum();
    at::Tensor new_sum_filler_area = at::relu(max_total_area - old_sum_movable_area - movable_area_increment_sum);
    at::Tensor filler_nodes_ratio = at::sqrt(new_sum_filler_area / old_sum_filler_area);
    node_size_x_filler.mul_(filler_nodes_ratio);
    node_size_y_filler.mul_(filler_nodes_ratio);

    // Call the cpp kernel launcher
    DREAMPLACE_DISPATCH_FLOATING_TYPES(pin_pos.type(), "updatePinOffset", [&] {
        updatePinOffset<scalar_t>(
            num_nodes,
            num_movable_nodes,
            num_filler_nodes,
            flat_node2pin_start_map.data<int>(),
            flat_node2pin_map.data<int>(),
            movable_nodes_ratio.data<scalar_t>(),
            filler_nodes_ratio.data<scalar_t>(),
            pin_offset_x.data<scalar_t>(), 
            pin_offset_y.data<scalar_t>(),
            num_threads);
    });
}

DREAMPLACE_END_NAMESPACE
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("adjust_instance_area", &DREAMPLACE_NAMESPACE::adjust_instance_area, "adjust instance area");
}
