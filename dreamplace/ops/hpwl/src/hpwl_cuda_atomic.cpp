/**
 * @file   hpwl_cuda_atomic.cpp
 * @author Yibo Lin
 * @date   Jul 2018
 */
#include <torch/torch.h>
#include <limits>

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

at::Tensor hpwl_atomic_forward(
        at::Tensor pos,
        at::Tensor pin2net_map, 
        at::Tensor net_mask) 
{
    typedef int T; 

    CHECK_FLAT(pos); 
    CHECK_EVEN(pos);
    CHECK_CONTIGUOUS(pos);
    CHECK_FLAT(pin2net_map);
    CHECK_CONTIGUOUS(pin2net_map);

    AT_ASSERTM(pos.is_cuda() && pos.ndimension() == 1 && (pos.numel()&1) == 0, "pos must be a flat tensor on GPU");

    int num_nets = net_mask.numel();
    // x then y 
    at::Tensor scaled_pos = pos.mul(1000)._cast_Int();
    at::Tensor partial_hpwl_max = at::zeros({2, num_nets}, scaled_pos.type()); 
    at::Tensor partial_hpwl_min = at::zeros({2, num_nets}, scaled_pos.type()); 
    partial_hpwl_max[0].masked_fill_(net_mask, std::numeric_limits<T>::min());
    partial_hpwl_max[1].masked_fill_(net_mask, std::numeric_limits<T>::min());
    partial_hpwl_min[0].masked_fill_(net_mask, std::numeric_limits<T>::max());
    partial_hpwl_min[1].masked_fill_(net_mask, std::numeric_limits<T>::max());

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

    auto hpwl = (partial_hpwl_max-partial_hpwl_min)._cast_Long().sum()._cast_Double().mul(1.0/1000);

    const at::Type& the_type = pos.type();
    switch (the_type.scalarType())
    {
        case at::ScalarType::Double:
            return hpwl; 
        case at::ScalarType::Float:
            return hpwl._cast_Float(); 
        default:
            AT_ERROR("hpwl_atomic_forward", " not implemented for '", the_type.toString(), "'"); 
    }

    return hpwl; 
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &hpwl_atomic_forward, "HPWL forward (CUDA)");
  //m.def("backward", &hpwl_backward, "HPWL backward (CUDA)");
}
