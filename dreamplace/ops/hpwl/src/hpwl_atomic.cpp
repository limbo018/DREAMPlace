/**
 * @file   hpwl_atomic.cpp
 * @author Yibo Lin
 * @date   Mar 2019
 * @brief  Compute half-perimeter wirelength to mimic a parallel atomic implementation
 */
#include "utility/src/torch.h"
#include "utility/src/Msg.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
int computeHPWLAtomicLauncher(
        const T* x, const T* y, 
        const int* pin2net_map, 
        const unsigned char* net_mask, 
        int num_nets, 
        int num_pins, 
        T* partial_hpwl_max, 
        T* partial_hpwl_min
        );

#define CHECK_FLAT(x) AT_ASSERTM(!x.is_cuda() && x.ndimension() == 1, #x "must be a flat tensor on CPU")
#define CHECK_EVEN(x) AT_ASSERTM((x.numel()&1) == 0, #x "must have even number of elements")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x "must be contiguous")

/// @brief Compute half-perimeter wirelength 
/// @param pos cell locations, array of x locations and then y locations 
/// @param pin2net_map map pin to net 
/// @param net_mask an array to record whether compute the where for a net or not 
at::Tensor hpwl_atomic_forward(
        at::Tensor pos,
        at::Tensor pin2net_map, 
        at::Tensor net_mask, 
        at::Tensor net_weights) 
{
    CHECK_FLAT(pos); 
    CHECK_EVEN(pos);
    CHECK_CONTIGUOUS(pos);
    CHECK_FLAT(pin2net_map);
    CHECK_CONTIGUOUS(pin2net_map);

    int num_nets = net_mask.numel();
    // x then y 
    at::Tensor partial_hpwl_max = at::zeros({2, num_nets}, pos.type()); 
    at::Tensor partial_hpwl_min = at::zeros({2, num_nets}, pos.type()); 

    AT_DISPATCH_FLOATING_TYPES(pos.type(), "computeHPWLAtomicLauncher", [&] {
            partial_hpwl_max[0].masked_fill_(net_mask, std::numeric_limits<scalar_t>::min());
            partial_hpwl_max[1].masked_fill_(net_mask, std::numeric_limits<scalar_t>::min());
            partial_hpwl_min[0].masked_fill_(net_mask, std::numeric_limits<scalar_t>::max());
            partial_hpwl_min[1].masked_fill_(net_mask, std::numeric_limits<scalar_t>::max());
            computeHPWLAtomicLauncher<scalar_t>(
                    pos.data<scalar_t>(), pos.data<scalar_t>()+pos.numel()/2, 
                    pin2net_map.data<int>(), 
                    net_mask.data<unsigned char>(), 
                    num_nets, 
                    pin2net_map.numel(), 
                    partial_hpwl_max.data<scalar_t>(), 
                    partial_hpwl_min.data<scalar_t>()
                    );
            });

    //std::cout << "partial_hpwl_max = " << partial_hpwl_max << "\n";
    //std::cout << "partial_hpwl_min = " << partial_hpwl_min << "\n";
    //std::cout << "partial_hpwl = \n" << (partial_hpwl_max-partial_hpwl_min)._cast_double().mul(1.0/1000) << "\n";

    auto hpwl = (partial_hpwl_max-partial_hpwl_min).mul_(net_weights.view({1, num_nets})).sum();

    return hpwl; 
}

template <typename T>
int computeHPWLAtomicLauncher(
        const T* x, const T* y, 
        const int* pin2net_map, 
        const unsigned char* net_mask, 
        int num_nets, 
        int num_pins, 
        T* partial_hpwl_max, 
        T* partial_hpwl_min
        )
{
    T* partial_hpwl_x_max = partial_hpwl_max;
    T* partial_hpwl_x_min = partial_hpwl_min;
    T* partial_hpwl_y_max = partial_hpwl_max+num_nets;
    T* partial_hpwl_y_min = partial_hpwl_min+num_nets;
    for (int i = 0; i < num_pins; ++i)
    {
        int net_id = pin2net_map[i];
        if (net_mask[net_id])
        {
            partial_hpwl_x_max[net_id] = std::max(partial_hpwl_x_max[net_id], x[i]); 
            partial_hpwl_x_min[net_id] = std::min(partial_hpwl_x_min[net_id], x[i]); 
            partial_hpwl_y_max[net_id] = std::max(partial_hpwl_y_max[net_id], y[i]); 
            partial_hpwl_y_min[net_id] = std::min(partial_hpwl_y_min[net_id], y[i]); 
        }
    }

    return 0; 
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &DREAMPLACE_NAMESPACE::hpwl_atomic_forward, "HPWL forward");
}
