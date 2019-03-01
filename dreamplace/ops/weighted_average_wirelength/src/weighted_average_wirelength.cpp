/**
 * @file   src/weighted_average_wirelength.cpp
 * @author Yibo Lin
 * @date   Jun 2018
 */
#include <torch/torch.h>
#include <limits>

template <typename T>
int computeWeightedAverageWirelengthLauncher(
        const T* x, const T* y, 
        const int* flat_netpin, 
        const int* netpin_start, 
        const unsigned char* net_mask, 
        int num_nets, 
        const T* gamma, 
        T* wl, 
        const T* grad_tensor, 
        T* grad_x_tensor, T* grad_y_tensor 
        );

#define CHECK_FLAT(x) AT_ASSERTM(!x.is_cuda() && x.ndimension() == 1, #x " must be a flat tensor on CPU")
#define CHECK_EVEN(x) AT_ASSERTM((x.numel()&1) == 0, #x " must have even number of elements")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")

at::Tensor weighted_average_wirelength_forward(
        at::Tensor pos,
        at::Tensor flat_netpin,
        at::Tensor netpin_start, 
        at::Tensor net_mask, 
        at::Tensor gamma
        ) 
{
    CHECK_FLAT(pos); 
    CHECK_EVEN(pos);
    CHECK_CONTIGUOUS(pos);
    CHECK_FLAT(flat_netpin);
    CHECK_CONTIGUOUS(flat_netpin);
    CHECK_FLAT(netpin_start);
    CHECK_CONTIGUOUS(netpin_start);
    at::Tensor wl = at::zeros({1}, pos.options());

    AT_DISPATCH_FLOATING_TYPES(pos.type(), "computeWeightedAverageWirelengthLauncher", [&] {
            computeWeightedAverageWirelengthLauncher<scalar_t>(
                    pos.data<scalar_t>(), pos.data<scalar_t>()+pos.numel()/2, 
                    flat_netpin.data<int>(), 
                    netpin_start.data<int>(), 
                    net_mask.data<unsigned char>(), 
                    netpin_start.numel()-1, 
                    gamma.data<scalar_t>(), 
                    wl.data<scalar_t>(), 
                    nullptr, 
                    nullptr, nullptr
                    );
            });
    return wl; 
}

at::Tensor weighted_average_wirelength_backward(
        at::Tensor grad_pos, 
        at::Tensor pos,
        at::Tensor flat_netpin,
        at::Tensor netpin_start, 
        at::Tensor net_mask, 
        at::Tensor gamma
        ) 
{
    CHECK_FLAT(pos); 
    CHECK_EVEN(pos);
    CHECK_CONTIGUOUS(pos);
    CHECK_FLAT(flat_netpin);
    CHECK_CONTIGUOUS(flat_netpin);
    CHECK_FLAT(netpin_start);
    CHECK_CONTIGUOUS(netpin_start);
    at::Tensor grad_out = at::zeros_like(pos);

    AT_DISPATCH_FLOATING_TYPES(pos.type(), "computeWeightedAverageWirelengthLauncher", [&] {
            computeWeightedAverageWirelengthLauncher<scalar_t>(
                    pos.data<scalar_t>(), pos.data<scalar_t>()+pos.numel()/2, 
                    flat_netpin.data<int>(), 
                    netpin_start.data<int>(), 
                    net_mask.data<unsigned char>(), 
                    netpin_start.numel()-1, 
                    gamma.data<scalar_t>(), 
                    nullptr, 
                    grad_pos.data<scalar_t>(), 
                    grad_out.data<scalar_t>(), grad_out.data<scalar_t>()+pos.numel()/2
                    );
            });
    return grad_out; 
}

template <typename T>
int computeWeightedAverageWirelengthLauncher(
        const T* x, const T* y, 
        const int* flat_netpin, 
        const int* netpin_start, 
        const unsigned char* net_mask, 
        int num_nets, 
        const T* gamma, 
        T* wl,
        const T* grad_tensor, 
        T* grad_x_tensor, T* grad_y_tensor 
        )
{
    if (!grad_tensor)
    {
        *wl = 0; 
    }
    for (int i = 0; i < num_nets; ++i)
    {
        T xexp_x_sum = 0; 
        T xexp_nx_sum = 0; 
        T exp_x_sum = 0; 
        T exp_nx_sum = 0; 

        T yexp_y_sum = 0; 
        T yexp_ny_sum = 0; 
        T exp_y_sum = 0; 
        T exp_ny_sum = 0; 

        //int degree = netpin_start[i+1]-netpin_start[i]; 
        if (!net_mask[i])
        {
            continue; 
        }
        for (int j = netpin_start[i]; j < netpin_start[i+1]; ++j)
        {
            // for x 
            T xx = x[flat_netpin[j]]; 
            T exp_x = exp(xx/(*gamma)); 
            T exp_nx = exp(-xx/(*gamma)); 

            xexp_x_sum += xx*exp_x; 
            xexp_nx_sum += xx*exp_nx; 
            exp_x_sum += exp_x; 
            exp_nx_sum += exp_nx; 

            // for y 
            T yy = y[flat_netpin[j]]; 
            T exp_y = exp(yy/(*gamma)); 
            T exp_ny = exp(-yy/(*gamma)); 

            yexp_y_sum += yy*exp_y; 
            yexp_ny_sum += yy*exp_ny; 
            exp_y_sum += exp_y; 
            exp_ny_sum += exp_ny; 
        }
        if (grad_tensor) // gradient 
        {
            T b_x = 1.0/((*gamma)*exp_x_sum);
            T a_x = (1.0 - b_x*xexp_x_sum)/exp_x_sum; 
            T b_nx = -1.0/((*gamma)*exp_nx_sum);
            T a_nx = (1.0 - b_nx*xexp_nx_sum)/exp_nx_sum; 

            T b_y = 1.0/((*gamma)*exp_y_sum);
            T a_y = (1.0 - b_y*yexp_y_sum)/exp_y_sum; 
            T b_ny = -1.0/((*gamma)*exp_ny_sum);
            T a_ny = (1.0 - b_ny*yexp_ny_sum)/exp_ny_sum; 
            for (int j = netpin_start[i]; j < netpin_start[i+1]; ++j)
            {
                // for x 
                T xx = x[flat_netpin[j]]; 
                T exp_x = exp(xx/(*gamma)); 
                T exp_nx = exp(-xx/(*gamma)); 
                T xexp_x = xx*exp_x; 
                T xexp_nx = xx*exp_nx;

                grad_x_tensor[flat_netpin[j]] = ((a_x*exp_x + b_x*xexp_x) - (a_nx*exp_nx + b_nx*xexp_nx))*(*grad_tensor); 

                // for y 
                T yy = y[flat_netpin[j]]; 
                T exp_y = exp(yy/(*gamma)); 
                T exp_ny = exp(-yy/(*gamma)); 
                T yexp_y = yy*exp_y; 
                T yexp_ny = yy*exp_ny;

                grad_y_tensor[flat_netpin[j]] = ((a_y*exp_y + b_y*yexp_y) - (a_ny*exp_ny + b_ny*yexp_ny))*(*grad_tensor); 
            }
        }
        else // wirelength 
        {
            T wl_x = xexp_x_sum/exp_x_sum - xexp_nx_sum/exp_nx_sum;
            T wl_y = yexp_y_sum/exp_y_sum - yexp_ny_sum/exp_ny_sum; 
            //printf("wl_x = %g, wl_y = %g\n", wl_x, wl_y);
            *wl += wl_x + wl_y; 
        }
    }

    return 0; 
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &weighted_average_wirelength_forward, "WeightedAverageWirelength forward");
  m.def("backward", &weighted_average_wirelength_backward, "WeightedAverageWirelength backward");
}
