/**
 * @file   hpwl_cuda.cpp
 * @author Yibo Lin
 * @date   Jun 2018
 */
#include <torch/torch.h>
#include <limits>

template <typename T>
int computeLogSumExpWirelengthCudaLauncher(
        const T* x, const T* y, 
        const int* flat_netpin, 
        const int* netpin_start, 
        const T* netpin_values, 
        const int ignore_net_degree, 
        int num_nets, 
        int num_pins, 
        const T* gamma, 
        T* exp_xy, T* exp_nxy, 
        T* exp_xy_sum, T* exp_nxy_sum,
        T* partial_wl, // wirelength of each net 
        const T* grad_tensor, 
        T* grad_x_tensor, T* grad_y_tensor // the gradient is partial total wirelength to partial pin position  
        );

#define CHECK_FLAT(x) AT_ASSERTM(x.is_cuda() && x.ndimension() == 1, #x "must be a flat tensor on GPU")
#define CHECK_EVEN(x) AT_ASSERTM((x.numel()&1) == 0, #x "must have even number of elements")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x "must be contiguous")

std::vector<at::Tensor> logsumexp_wirelength_forward(
        at::Tensor pos,
        at::Tensor flat_netpin,
        at::Tensor netpin_start, 
        at::Tensor netpin_values, // all ones 
        at::Tensor gamma, // a scalar tensor 
        int ignore_net_degree) 
{
    CHECK_FLAT(pos); 
    CHECK_EVEN(pos);
    CHECK_CONTIGUOUS(pos);
    CHECK_FLAT(flat_netpin);
    CHECK_CONTIGUOUS(flat_netpin);
    CHECK_FLAT(netpin_start);
    CHECK_CONTIGUOUS(netpin_start);
    // log-sum-exp for x, log-sum-exp for -x, log-sum-exp for y, log-sum-exp for -y 
    at::Tensor partial_wl = at::zeros({4, netpin_start.numel()-1}, pos.type());
    at::Tensor exp_xy = at::zeros_like(pos);
    at::Tensor exp_nxy = at::zeros_like(pos);
    at::Tensor exp_xy_sum = at::zeros({2*(netpin_start.numel()-1)}, pos.type());
    at::Tensor exp_nxy_sum = at::zeros({2*(netpin_start.numel()-1)}, pos.type());
    if (netpin_values.numel() == 0)
    {
        netpin_values = at::ones({flat_netpin.numel()}, pos.type());
    }

    AT_DISPATCH_FLOATING_TYPES(pos.type(), "computeLogSumExpWirelengthCudaLauncher", [&] {
            computeLogSumExpWirelengthCudaLauncher<scalar_t>(
                    pos.data<scalar_t>(), pos.data<scalar_t>()+pos.numel()/2, 
                    flat_netpin.data<int>(), 
                    netpin_start.data<int>(), 
                    netpin_values.data<scalar_t>(), 
                    ignore_net_degree, 
                    netpin_start.numel()-1, 
                    flat_netpin.numel(),  
                    gamma.data<scalar_t>(), 
                    exp_xy.data<scalar_t>(), exp_nxy.data<scalar_t>(), 
                    exp_xy_sum.data<scalar_t>(), exp_nxy_sum.data<scalar_t>(),
                    partial_wl.data<scalar_t>(), 
                    nullptr, 
                    nullptr, nullptr
                    );
            });
    // significant speedup is achieved by using summation in ATen 
    auto wl = at::sum(partial_wl); 
    return {wl, exp_xy, exp_nxy, exp_xy_sum, exp_nxy_sum}; 
}

at::Tensor logsumexp_wirelength_backward(
        at::Tensor grad_pos, 
        at::Tensor pos,
        at::Tensor exp_xy, at::Tensor exp_nxy, 
        at::Tensor exp_xy_sum, at::Tensor exp_nxy_sum, 
        at::Tensor flat_netpin,
        at::Tensor netpin_start, 
        at::Tensor netpin_values, // all ones 
        at::Tensor gamma, // a scalar tensor 
        int ignore_net_degree) 
{
    CHECK_FLAT(pos); 
    CHECK_EVEN(pos);
    CHECK_CONTIGUOUS(pos);
    CHECK_FLAT(exp_xy); 
    CHECK_EVEN(exp_xy);
    CHECK_CONTIGUOUS(exp_xy);
    CHECK_FLAT(exp_nxy); 
    CHECK_EVEN(exp_nxy);
    CHECK_CONTIGUOUS(exp_nxy);
    CHECK_FLAT(exp_xy_sum); 
    CHECK_EVEN(exp_xy_sum);
    CHECK_CONTIGUOUS(exp_xy_sum);
    CHECK_FLAT(exp_nxy_sum); 
    CHECK_EVEN(exp_nxy_sum);
    CHECK_CONTIGUOUS(exp_nxy_sum);
    CHECK_FLAT(flat_netpin);
    CHECK_CONTIGUOUS(flat_netpin);
    CHECK_FLAT(netpin_start);
    CHECK_CONTIGUOUS(netpin_start);
    at::Tensor grad_out = at::zeros_like(pos);

    AT_DISPATCH_FLOATING_TYPES(pos.type(), "computeLogSumExpWirelengthCudaLauncher", [&] {
            computeLogSumExpWirelengthCudaLauncher<scalar_t>(
                    pos.data<scalar_t>(), pos.data<scalar_t>()+pos.numel()/2, 
                    flat_netpin.data<int>(), 
                    netpin_start.data<int>(), 
                    nullptr, 
                    ignore_net_degree, 
                    netpin_start.numel()-1, 
                    flat_netpin.numel(),  
                    gamma.data<scalar_t>(), 
                    exp_xy.data<scalar_t>(), exp_nxy.data<scalar_t>(), 
                    exp_xy_sum.data<scalar_t>(), exp_nxy_sum.data<scalar_t>(),
                    nullptr, 
                    grad_pos.data<scalar_t>(), 
                    grad_out.data<scalar_t>(), grad_out.data<scalar_t>()+pos.numel()/2
                    );
            });
    return grad_out; 
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &logsumexp_wirelength_forward, "LogSumExpWirelength forward (CUDA)");
  m.def("backward", &logsumexp_wirelength_backward, "LogSumExpWirelength backward (CUDA)");
}
