/**
 * @file   src/logsumexp_wirelength.cpp
 * @author Yibo Lin
 * @date   Jun 2018
 */
#include <torch/torch.h>
#include <cfloat>

template <typename T>
int computeLogSumExpWirelengthLauncher(
        const T* x, const T* y, 
        const int* flat_netpin, 
        const int* netpin_start, 
        const int ignore_net_degree, 
        int num_nets, 
        int num_pins, 
        const T* gamma, 
        T* exp_xy, T* exp_nxy, 
        T* exp_xy_sum, T* exp_nxy_sum,
        T* wl, 
        const T* grad_tensor, 
        T* grad_x_tensor, T* grad_y_tensor 
        );

#define CHECK_FLAT(x) AT_ASSERTM(!x.is_cuda() && x.ndimension() == 1, #x " must be a flat tensor on CPU")
#define CHECK_EVEN(x) AT_ASSERTM((x.numel()&1) == 0, #x " must have even number of elements")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")

std::vector<at::Tensor> logsumexp_wirelength_forward(
        at::Tensor pos,
        at::Tensor flat_netpin,
        at::Tensor netpin_start, 
        at::Tensor netpin_values, // all ones, not used 
        at::Tensor gamma, // a scalar tensor 
        int ignore_net_degree 
        ) 
{
    CHECK_FLAT(pos); 
    CHECK_EVEN(pos);
    CHECK_CONTIGUOUS(pos);
    CHECK_FLAT(flat_netpin);
    CHECK_CONTIGUOUS(flat_netpin);
    CHECK_FLAT(netpin_start);
    CHECK_CONTIGUOUS(netpin_start);
    at::Tensor wl = at::zeros({1}, pos.type());
    at::Tensor exp_xy = at::zeros_like(pos);
    at::Tensor exp_nxy = at::zeros_like(pos);
    at::Tensor exp_xy_sum = at::zeros({2*(netpin_start.numel()-1)}, pos.type());
    at::Tensor exp_nxy_sum = at::zeros({2*(netpin_start.numel()-1)}, pos.type());

    AT_DISPATCH_FLOATING_TYPES(pos.type(), "computeLogSumExpWirelengthLauncher", [&] {
            computeLogSumExpWirelengthLauncher<scalar_t>(
                    pos.data<scalar_t>(), pos.data<scalar_t>()+pos.numel()/2, 
                    flat_netpin.data<int>(), 
                    netpin_start.data<int>(), 
                    ignore_net_degree, 
                    netpin_start.numel()-1, 
                    flat_netpin.numel(), 
                    gamma.data<scalar_t>(), 
                    exp_xy.data<scalar_t>(), exp_nxy.data<scalar_t>(), 
                    exp_xy_sum.data<scalar_t>(), exp_nxy_sum.data<scalar_t>(),
                    wl.data<scalar_t>(), 
                    nullptr, 
                    nullptr, nullptr
                    );
            });

    return {wl, exp_xy, exp_nxy, exp_xy_sum, exp_nxy_sum}; 
}

at::Tensor logsumexp_wirelength_backward(
        at::Tensor grad_pos, 
        at::Tensor pos,
        at::Tensor exp_xy, at::Tensor exp_nxy, 
        at::Tensor exp_xy_sum, at::Tensor exp_nxy_sum, 
        at::Tensor flat_netpin,
        at::Tensor netpin_start, 
        at::Tensor netpin_values, // all ones, not used 
        at::Tensor gamma, // a scalar tensor 
        int ignore_net_degree
        ) 
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

    AT_DISPATCH_FLOATING_TYPES(pos.type(), "computeLogSumExpWirelengthLauncher", [&] {
            computeLogSumExpWirelengthLauncher<scalar_t>(
                    pos.data<scalar_t>(), pos.data<scalar_t>()+pos.numel()/2, 
                    flat_netpin.data<int>(), 
                    netpin_start.data<int>(), 
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

template <typename T>
int computeLogSumExpWirelengthLauncher(
        const T* x, const T* y, 
        const int* flat_netpin, 
        const int* netpin_start, 
        const int ignore_net_degree, 
        int num_nets, 
        int num_pins, 
        const T* gamma, 
        T* exp_xy, T* exp_nxy, 
        T* exp_xy_sum, T* exp_nxy_sum,
        T* wl,
        const T* grad_tensor, 
        T* grad_x_tensor, T* grad_y_tensor 
        )
{
    T tol = 80; // tolerance to trigger numeric adjustment, which may cause precision loss  
    if (grad_tensor)
    {
        for (int i = 0; i < num_pins; ++i)
        {
            grad_x_tensor[i] = 0; 
            grad_y_tensor[i] = 0; 
        }
    }
    else 
    {
        *wl = 0; 
    }
    for (int i = 0; i < num_nets; ++i)
    {
        int degree = netpin_start[i+1]-netpin_start[i];
        if (degree < 2 || degree >= ignore_net_degree)
            continue; 
        for (int k = 0; k < 2; ++k)
        {
            const T* xy = (k)? y : x; 
            T* grad = (k)? grad_y_tensor : grad_x_tensor; 

            if (grad_tensor) // gradient 
            {
                T reciprocal_exp_xy_sum = 1.0/exp_xy_sum[i+k*num_nets];
                T reciprocal_exp_nxy_sum = 1.0/exp_nxy_sum[i+k*num_nets];
                for (int j = netpin_start[i]; j < netpin_start[i+1]; ++j)
                {
                    // I assume one pin will only appear in one net 
                    // for x 
                    grad[flat_netpin[j]] = (exp_xy[flat_netpin[j]+k*netpin_start[num_nets]]*reciprocal_exp_xy_sum - exp_nxy[flat_netpin[j]+k*netpin_start[num_nets]]*reciprocal_exp_nxy_sum)*(*grad_tensor); 
                }
            }
            else // wirelength 
            {
                exp_xy_sum[i+k*num_nets] = 0; 
                exp_nxy_sum[i+k*num_nets] = 0; 

                T xy_max = -FLT_MAX; // maximum x to resolve numerical overflow
                T xy_min = FLT_MAX; // minimum x to resolve numerical overflow

                for (int j = netpin_start[i]; j < netpin_start[i+1]; ++j)
                {
                    // for x 
                    T xx = xy[flat_netpin[j]]; 
                    xy_max = std::max(xy_max, xx); 
                    xy_min = std::min(xy_min, xx); 
                }
                if (xy_max < tol*(*gamma))
                {
                    xy_max = 0; 
                }
                if (xy_min > -tol*(*gamma))
                {
                    xy_min = 0; 
                }
                for (int j = netpin_start[i]; j < netpin_start[i+1]; ++j)
                {
                    // for x 
                    T xx = xy[flat_netpin[j]]; 
                    exp_xy[flat_netpin[j]+k*netpin_start[num_nets]] = exp((xx-xy_max)/(*gamma)); 
                    exp_nxy[flat_netpin[j]+k*netpin_start[num_nets]] = exp(-(xx-xy_min)/(*gamma)); 

                    exp_xy_sum[i+k*num_nets] += exp_xy[flat_netpin[j]+k*netpin_start[num_nets]]; 
                    exp_nxy_sum[i+k*num_nets] += exp_nxy[flat_netpin[j]+k*netpin_start[num_nets]]; 
                }

                T log_exp_xy_sum = log(exp_xy_sum[i+k*num_nets])*(*gamma) + xy_max; 
                T log_exp_nxy_sum = log(exp_nxy_sum[i+k*num_nets])*(*gamma) - xy_min; 

                T wl_xy = log_exp_xy_sum + log_exp_nxy_sum;
                *wl += wl_xy; 
            }
        }
    }

    return 0; 
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &logsumexp_wirelength_forward, "LogSumExpWirelength forward");
  m.def("backward", &logsumexp_wirelength_backward, "LogSumExpWirelength backward");
}
