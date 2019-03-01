/**
 * @file   electric_force.cpp
 * @author Yibo Lin
 * @date   Aug 2018
 */
#include <torch/torch.h>
#include <limits>

template <typename T>
int computeElectricForceLauncher(
        int num_bins_x, int num_bins_y, 
        int num_impacted_bins_x, int num_impacted_bins_y, 
        const T* field_map_x_tensor, const T* field_map_y_tensor, 
        const T* x_tensor, const T* y_tensor, 
        const T* node_size_x_tensor, const T* node_size_y_tensor, 
        const T* bin_center_x_tensor, const T* bin_center_y_tensor, 
        T xl, T yl, T xh, T yh, 
        T bin_size_x, T bin_size_y, 
        int num_nodes, 
        T* grad_x_tensor, T* grad_y_tensor
        );

#define CHECK_FLAT(x) AT_ASSERTM(!x.is_cuda() && x.ndimension() == 1, #x "must be a flat tensor on CPU")
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

    at::Tensor grad_out = at::zeros_like(pos);
    int num_nodes = pos.numel()/2; 

    AT_DISPATCH_FLOATING_TYPES(pos.type(), "computeElectricForceLauncher", [&] {
            computeElectricForceLauncher<scalar_t>(
                    num_bins_x, num_bins_y, 
                    num_movable_impacted_bins_x, num_movable_impacted_bins_y, 
                    field_map_x.data<scalar_t>(), field_map_y.data<scalar_t>(), 
                    pos.data<scalar_t>(), pos.data<scalar_t>()+num_nodes, 
                    node_size_x.data<scalar_t>(), node_size_y.data<scalar_t>(), 
                    bin_center_x.data<scalar_t>(), bin_center_y.data<scalar_t>(), 
                    xl, yl, xh, yh, 
                    bin_size_x, bin_size_y, 
                    num_movable_nodes,
                    grad_out.data<scalar_t>(), grad_out.data<scalar_t>()+num_nodes
                    );
            });

    if (num_filler_nodes)
    {
        AT_DISPATCH_FLOATING_TYPES(pos.type(), "computeElectricForceLauncher", [&] {
                computeElectricForceLauncher<scalar_t>(
                        num_bins_x, num_bins_y, 
                        num_filler_impacted_bins_x, num_filler_impacted_bins_y, 
                        field_map_x.data<scalar_t>(), field_map_y.data<scalar_t>(), 
                        pos.data<scalar_t>()+num_nodes-num_filler_nodes, pos.data<scalar_t>()+num_nodes*2-num_filler_nodes, 
                        node_size_x.data<scalar_t>()+num_nodes-num_filler_nodes, node_size_y.data<scalar_t>()+num_nodes-num_filler_nodes, 
                        bin_center_x.data<scalar_t>(), bin_center_y.data<scalar_t>(), 
                        xl, yl, xh, yh, 
                        bin_size_x, bin_size_y, 
                        num_filler_nodes,
                        grad_out.data<scalar_t>()+num_nodes-num_filler_nodes, grad_out.data<scalar_t>()+num_nodes*2-num_filler_nodes
                        );
                });
    }

    return grad_out.mul_(grad_pos); 
}

#define SQRT2 1.4142135623730950488016887242096980785696718753769480731766797379907324784621

template <typename T>
int computeElectricForceLauncher(
        int num_bins_x, int num_bins_y, 
        int num_impacted_bins_x, int num_impacted_bins_y, 
        const T* field_map_x_tensor, const T* field_map_y_tensor, 
        const T* x_tensor, const T* y_tensor, 
        const T* node_size_x_tensor, const T* node_size_y_tensor, 
        const T* bin_center_x_tensor, const T* bin_center_y_tensor, 
        T xl, T yl, T xh, T yh, 
        T bin_size_x, T bin_size_y, 
        int num_nodes, 
        T* grad_x_tensor, T* grad_y_tensor
        )
{
    // density_map_tensor should be initialized outside 

    auto computeDensityFunc = [](T x, T node_size, T bin_center, T bin_size){
        //return std::max(T(0.0), min(x+node_size, bin_center+bin_size/2) - std::max(x, bin_center-bin_size/2));
        // Yibo: cannot understand why negative overlap is allowed in RePlAce 
        return std::min(x+node_size, bin_center+bin_size/2) - std::max(x, bin_center-bin_size/2);
    };
    for (int i = 0; i < num_nodes; ++i)
    {
        // stretch node size to bin size 
        T node_size_x = std::max((T)(bin_size_x*SQRT2), node_size_x_tensor[i]); 
        T node_size_y = std::max((T)(bin_size_y*SQRT2), node_size_y_tensor[i]); 
        T node_x = x_tensor[i]+node_size_x_tensor[i]/2-node_size_x/2;
        T node_y = y_tensor[i]+node_size_y_tensor[i]/2-node_size_y/2;
        T ratio = (node_size_x_tensor[i]*node_size_y_tensor[i]/(node_size_x*node_size_y));

        // Yibo: looks very weird implementation, but this is how RePlAce implements it 
        // the common practice should be floor 
        int bin_index_xl = round((node_x-xl)/bin_size_x);
        int bin_index_xh = round(((node_x+node_size_x-xl)/bin_size_x))+1; // exclusive 
        bin_index_xl = std::max(bin_index_xl, 0); 
        bin_index_xh = std::min(bin_index_xh, num_bins_x);
        //int bin_index_xh = bin_index_xl+num_impacted_bins_x; 

        // Yibo: looks very weird implementation, but this is how RePlAce implements it 
        // the common practice should be floor 
        int bin_index_yl = round((node_y-yl)/bin_size_y);
        int bin_index_yh = round(((node_y+node_size_y-yl)/bin_size_y))+1; // exclusive 
        bin_index_yl = std::max(bin_index_yl, 0); 
        bin_index_yh = std::min(bin_index_yh, num_bins_y);
        //int bin_index_yh = bin_index_yl+num_impacted_bins_y; 

        grad_x_tensor[i] = 0; 
        grad_y_tensor[i] = 0; 
        // update density potential map 
        for (int k = bin_index_xl; k < bin_index_xh; ++k)
        {
            T px = computeDensityFunc(node_x, node_size_x, bin_center_x_tensor[k], bin_size_x);
            for (int h = bin_index_yl; h < bin_index_yh; ++h)
            {
                T py = computeDensityFunc(node_y, node_size_y, bin_center_y_tensor[h], bin_size_y);

                T area = px*py*ratio; 

                grad_x_tensor[i] += area*field_map_x_tensor[k*num_bins_y+h];
                grad_y_tensor[i] += area*field_map_y_tensor[k*num_bins_y+h];
            }
        }
    }

    return 0; 
}

