/**
 * @file   src/hpwl.cpp
 * @author Yibo Lin
 * @date   Jun 2018
 */
#include <torch/torch.h>
#include <limits>

template <typename T>
int computeHPWLLauncher(
        const T* x, const T* y, 
        const int* flat_netpin, 
        const int* netpin_start, 
        const int ignore_net_degree, 
        int num_nets, 
        T* hpwl 
        );

#define CHECK_FLAT(x) AT_ASSERTM(!x.is_cuda() && x.ndimension() == 1, #x "must be a flat tensor on CPU")
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

    at::Tensor hpwl = at::zeros(1, pos.type()); 
    AT_DISPATCH_FLOATING_TYPES(pos.type(), "computeHPWLLauncher", [&] {
            computeHPWLLauncher<scalar_t>(
                    pos.data<scalar_t>(), pos.data<scalar_t>()+pos.numel()/2, 
                    flat_netpin.data<int>(), 
                    netpin_start.data<int>(), 
                    ignore_net_degree, 
                    netpin_start.numel()-1, 
                    hpwl.data<scalar_t>()
                    );
            });
    return hpwl; 
}

template <typename T>
int computeHPWLLauncher(
        const T* x, const T* y, 
        const int* flat_netpin, 
        const int* netpin_start, 
        const int ignore_net_degree, 
        int num_nets, 
        T* hpwl 
        )
{
    *hpwl = 0; 
    for (int i = 0; i < num_nets; ++i)
    {
        T max_x = -std::numeric_limits<T>::max();
        T min_x = std::numeric_limits<T>::max();
        T max_y = -std::numeric_limits<T>::max();
        T min_y = std::numeric_limits<T>::max();

        // ignore large degree nets 
        if (netpin_start[i+1]-netpin_start[i] >= ignore_net_degree)
            continue; 

        for (int j = netpin_start[i]; j < netpin_start[i+1]; ++j)
        {
            min_x = std::min(min_x, x[flat_netpin[j]]);
            max_x = std::max(max_x, x[flat_netpin[j]]);
            min_y = std::min(min_y, y[flat_netpin[j]]);
            max_y = std::max(max_y, y[flat_netpin[j]]);
        }
        *hpwl += max_x-min_x + max_y-min_y; 
    }

    return 0; 
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &hpwl_forward, "HPWL forward");
}
