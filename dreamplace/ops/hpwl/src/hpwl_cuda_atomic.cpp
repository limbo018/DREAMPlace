/**
 * @file   hpwl_cuda_atomic.cpp
 * @author Yibo Lin
 * @date   Jul 2018
 * @brief  Compute half-perimeter wirelength to mimic a parallel atomic implementation
 */
#include "utility/src/torch.h"
#include "utility/src/Msg.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
int computeHPWLCudaAtomicLauncher(
        const T* x, const T* y, 
        const int* pin2net_map, 
        const unsigned char* net_mask, 
        int num_nets, 
        int num_pins, 
        T* partial_hpwl_max, 
        T* partial_hpwl_min
        );

#define CHECK_FLAT(x) AT_ASSERTM(x.is_cuda() && x.ndimension() == 1, #x "must be a flat tensor on GPU")
#define CHECK_EVEN(x) AT_ASSERTM((x.numel()&1) == 0, #x "must have even number of elements")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x "must be contiguous")

/// @brief Compute half-perimeter wirelength 
/// @param pos cell locations, array of x locations and then y locations 
/// @param pin2net_map map pin to net 
/// @param net_weights weight of nets 
/// @param net_mask an array to record whether compute the where for a net or not 
at::Tensor hpwl_atomic_forward(
        at::Tensor pos,
        at::Tensor pin2net_map, 
        at::Tensor net_weights,
        at::Tensor net_mask
        ) 
{
    typedef int T; 

    CHECK_FLAT(pos); 
    CHECK_EVEN(pos);
    CHECK_CONTIGUOUS(pos);
    CHECK_FLAT(pin2net_map);
    CHECK_CONTIGUOUS(pin2net_map);
    CHECK_FLAT(net_weights); 
    CHECK_CONTIGUOUS(net_weights);
    CHECK_FLAT(net_mask);
    CHECK_CONTIGUOUS(net_mask);

    int num_nets = net_mask.numel();
    // x then y 
    at::Tensor scaled_pos = at::_cast_Int(pos.mul(1000), false);
    at::Tensor partial_hpwl_max = at::zeros({2, num_nets}, scaled_pos.type()); 
    at::Tensor partial_hpwl_min = at::zeros({2, num_nets}, scaled_pos.type()); 
    partial_hpwl_max[0].masked_fill_(net_mask.to(at::ScalarType::Bool), std::numeric_limits<T>::min());
    partial_hpwl_max[1].masked_fill_(net_mask.to(at::ScalarType::Bool), std::numeric_limits<T>::min());
    partial_hpwl_min[0].masked_fill_(net_mask.to(at::ScalarType::Bool), std::numeric_limits<T>::max());
    partial_hpwl_min[1].masked_fill_(net_mask.to(at::ScalarType::Bool), std::numeric_limits<T>::max());

    computeHPWLCudaAtomicLauncher<T>(
            scaled_pos.data<T>(), scaled_pos.data<T>()+scaled_pos.numel()/2, 
            pin2net_map.data<int>(), 
            net_mask.data<unsigned char>(), 
            num_nets, 
            pin2net_map.numel(), 
            partial_hpwl_max.data<T>(), 
            partial_hpwl_min.data<T>()
            );

    //std::cout << "partial_hpwl_max = " << partial_hpwl_max << "\n";
    //std::cout << "partial_hpwl_min = " << partial_hpwl_min << "\n";
    //std::cout << "partial_hpwl = \n" << (partial_hpwl_max-partial_hpwl_min)._cast_double().mul(1.0/1000) << "\n";

    auto delta = partial_hpwl_max-partial_hpwl_min;

    at::Tensor hpwl; 
    switch (pos.scalar_type())
    {
        case at::ScalarType::Double:
            hpwl = at::_cast_Double(delta, false); 
            break; 
        case at::ScalarType::Float:
            hpwl = at::_cast_Float(delta, false);
            break; 
        default:
            AT_ERROR("hpwl_atomic_forward", " not implemented for '", at::toString(pos.scalar_type()), "'"); 
    }

    if (net_weights.numel())
    {
        hpwl.mul_(net_weights.view({1, num_nets}));
    }
    return hpwl.sum().mul_(1.0/1000); 
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &DREAMPLACE_NAMESPACE::hpwl_atomic_forward, "HPWL forward (CUDA)");
}