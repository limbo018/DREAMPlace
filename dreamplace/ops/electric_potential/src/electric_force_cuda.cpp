/**
 * @file   electric_force_cuda.cpp
 * @author Yibo Lin
 * @date   Aug 2018
 */
#include <torch/torch.h>
#include <limits>

template <typename T>
int computeElectricForceCudaLauncher(
        int num_bins_x, int num_bins_y, 
        int num_movable_impacted_bins_x, int num_movable_impacted_bins_y, 
        int num_filler_impacted_bins_x, int num_filler_impacted_bins_y, 
        const T* field_map_x_tensor, const T* field_map_y_tensor, 
        const T* x_tensor, const T* y_tensor, 
        const T* node_size_x_tensor, const T* node_size_y_tensor, 
        const T* bin_center_x_tensor, const T* bin_center_y_tensor, 
        T xl, T yl, T xh, T yh, 
        T bin_size_x, T bin_size_y, 
        int num_nodes, int num_movable_nodes, int num_filler_nodes, 
        T* grad_x_tensor, T* grad_y_tensor
        );

#define CHECK_FLAT(x) AT_ASSERTM(x.is_cuda() && x.ndimension() == 1, #x "must be a flat tensor on CPU")
#define CHECK_EVEN(x) AT_ASSERTM((x.numel()&1) == 0, #x "must have even number of elements")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x "must be contiguous")

// compute electric force for movable and filler cells 
at::Tensor electric_force(
        at::Tensor grad_pos,
        int num_bins_x, int num_bins_y, 
        int num_movable_impacted_bins_x, int num_movable_impacted_bins_y, 
        int num_filler_impacted_bins_x, int num_filler_impacted_bins_y, 
        at::Tensor field_map_x, at::Tensor field_map_y, 
        at::Tensor pos, 
        at::Tensor node_size_x, at::Tensor node_size_y, 
        at::Tensor bin_center_x, at::Tensor bin_center_y, 
        double xl, double yl, double xh, double yh, 
        double bin_size_x, double bin_size_y, 
        int num_movable_nodes, 
        int num_filler_nodes
        ) 
{
    CHECK_FLAT(pos); 
    CHECK_EVEN(pos);
    CHECK_CONTIGUOUS(pos);
    CHECK_FLAT(field_map_x);
    CHECK_CONTIGUOUS(field_map_x);
    CHECK_FLAT(field_map_y);
    CHECK_CONTIGUOUS(field_map_y);

    at::Tensor grad_out = at::zeros_like(pos);
    int num_nodes = pos.numel()/2; 

    AT_DISPATCH_FLOATING_TYPES(pos.type(), "computeElectricForceCudaLauncher", [&] {
            computeElectricForceCudaLauncher<scalar_t>(
                    num_bins_x, num_bins_y, 
                    num_movable_impacted_bins_x, num_movable_impacted_bins_y, 
                    num_filler_impacted_bins_x, num_filler_impacted_bins_y, 
                    field_map_x.data<scalar_t>(), field_map_y.data<scalar_t>(), 
                    pos.data<scalar_t>(), pos.data<scalar_t>()+num_nodes, 
                    node_size_x.data<scalar_t>(), node_size_y.data<scalar_t>(), 
                    bin_center_x.data<scalar_t>(), bin_center_y.data<scalar_t>(), 
                    xl, yl, xh, yh, 
                    bin_size_x, bin_size_y, 
                    num_nodes, num_movable_nodes, num_filler_nodes, 
                    grad_out.data<scalar_t>(), grad_out.data<scalar_t>()+num_nodes
                    );
            });

    return grad_out.mul_(grad_pos); 
}
