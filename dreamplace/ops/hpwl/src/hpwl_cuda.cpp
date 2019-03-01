/**
 * @file   hpwl_cuda.cpp
 * @author Yibo Lin
 * @date   Jun 2018
 */
#include <torch/torch.h>
#include <limits>

template <typename T>
int computeHPWLCudaLauncher(
        const T* x, const T* y, 
        const int* flat_netpin, 
        const int* netpin_start, 
        const int ignore_net_degree, 
        int num_nets, 
        T* partial_wl 
        );

#define CHECK_FLAT(x) AT_ASSERTM(x.is_cuda() && x.ndimension() == 1, #x "must be a flat tensor on GPU")
#define CHECK_EVEN(x) AT_ASSERTM((x.numel()&1) == 0, #x "must have even number of elements")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x "must be contiguous")

at::Tensor hpwl_forward(
        at::Tensor pos,
        at::Tensor flat_netpin,
        at::Tensor netpin_start, 
        int ignore_net_degree) 
{
    CHECK_FLAT(pos); 
    CHECK_EVEN(pos);
    CHECK_CONTIGUOUS(pos);
    CHECK_FLAT(flat_netpin);
    CHECK_CONTIGUOUS(flat_netpin);
    CHECK_FLAT(netpin_start);
    CHECK_CONTIGUOUS(netpin_start);

    AT_ASSERTM(pos.is_cuda() && pos.ndimension() == 1 && (pos.numel()&1) == 0, "pos must be a flat tensor on GPU");

    // x then y 
    at::Tensor partial_wl = at::zeros({2, netpin_start.numel()-1}, pos.type()); 

    AT_DISPATCH_FLOATING_TYPES(pos.type(), "computeHPWLCudaLauncher", [&] {
            computeHPWLCudaLauncher<scalar_t>(
                    pos.data<scalar_t>(), pos.data<scalar_t>()+pos.numel()/2, 
                    flat_netpin.data<int>(), 
                    netpin_start.data<int>(), 
                    ignore_net_degree, 
                    netpin_start.numel()-1, 
                    partial_wl.data<scalar_t>()
                    );
            });
    //std::cout << "partial_hpwl = \n" << partial_wl << "\n";

    auto hpwl = at::sum(partial_wl);

    return hpwl; 
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &hpwl_forward, "HPWL forward (CUDA)");
  //m.def("backward", &hpwl_backward, "HPWL backward (CUDA)");
}
