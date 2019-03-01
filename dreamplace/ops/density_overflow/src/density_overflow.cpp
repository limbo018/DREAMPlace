/**
 * @file   src/density_overflow.cpp
 * @author Yibo Lin
 * @date   Jun 2018
 */
#include <torch/torch.h>
#include <limits>

template <typename T>
int computeDensityOverflowMapLauncher(
        const T* x_tensor, const T* y_tensor, 
        const T* node_size_x_tensor, const T* node_size_y_tensor, 
        const T* bin_center_x_tensor, const T* bin_center_y_tensor, 
        const int num_nodes, 
        const int num_bins_x, const int num_bins_y, 
        const T xl, const T yl, const T xh, const T yh, 
        const T bin_size_x, const T bin_size_y, 
        const T target_density, 
        T* density_map_tensor, T* density_cost_tensor
        );

#define CHECK_FLAT(x) AT_ASSERTM(!x.is_cuda() && x.ndimension() == 1, #x "must be a flat tensor on CPU")
#define CHECK_EVEN(x) AT_ASSERTM((x.numel()&1) == 0, #x "must have even number of elements")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x "must be contiguous")

std::vector<at::Tensor> density_overflow_forward(
        at::Tensor pos,
        at::Tensor node_size_x,
        at::Tensor node_size_y,
        at::Tensor bin_center_x, 
        at::Tensor bin_center_y, 
        at::Tensor initial_density_map, 
        double target_density, 
        double xl, 
        double yl, 
        double xh, 
        double yh, 
        double bin_size_x, 
        double bin_size_y, 
        int num_movable_nodes, 
        int num_filler_nodes
        ) 
{
    CHECK_FLAT(pos); 
    CHECK_EVEN(pos);
    CHECK_CONTIGUOUS(pos);

    int num_bins_x = int(ceil((xh-xl)/bin_size_x));
    int num_bins_y = int(ceil((yh-yl)/bin_size_y));
    at::Tensor density_map = initial_density_map.clone();
    at::Tensor density_cost = at::zeros({1}, pos.options());
    double density_area = target_density*bin_size_x*bin_size_y;

    // Call the cuda kernel launcher
    AT_DISPATCH_FLOATING_TYPES(pos.type(), "computeDensityOverflowMapLauncher", [&] {
            computeDensityOverflowMapLauncher<scalar_t>(
                    pos.data<scalar_t>(), pos.data<scalar_t>()+pos.numel()/2, 
                    node_size_x.data<scalar_t>(), node_size_y.data<scalar_t>(), 
                    bin_center_x.data<scalar_t>(), bin_center_y.data<scalar_t>(), 
                    num_movable_nodes, // only compute that for movable cells 
                    num_bins_x, num_bins_y, 
                    xl, yl, xh, yh, 
                    bin_size_x, bin_size_y, 
                    density_area, 
                    density_map.data<scalar_t>(), 
                    density_cost.data<scalar_t>()
                    );
            });
    if (num_filler_nodes)
    {
        AT_DISPATCH_FLOATING_TYPES(pos.type(), "computeDensityOverflowMapLauncher", [&] {
                computeDensityOverflowMapLauncher<scalar_t>(
                        pos.data<scalar_t>()+pos.numel()/2-num_filler_nodes, pos.data<scalar_t>()+pos.numel()-num_filler_nodes, 
                        node_size_x.data<scalar_t>()+pos.numel()/2-num_filler_nodes, node_size_y.data<scalar_t>()+pos.numel()/2-num_filler_nodes, 
                        bin_center_x.data<scalar_t>(), bin_center_y.data<scalar_t>(), 
                        num_filler_nodes, // only compute that for movable cells 
                        num_bins_x, num_bins_y, 
                        xl, yl, xh, yh, 
                        bin_size_x, bin_size_y, 
                        density_area, 
                        density_map.data<scalar_t>(), 
                        density_cost.data<scalar_t>()
                        );
                });
    }

    auto max_density = density_map.max().div(bin_size_x*bin_size_y);

    return {density_cost, density_map, max_density};
}

at::Tensor fixed_density_overflow_map(
        at::Tensor pos,
        at::Tensor node_size_x,
        at::Tensor node_size_y,
        at::Tensor bin_center_x, 
        at::Tensor bin_center_y, 
        double xl, 
        double yl, 
        double xh, 
        double yh, 
        double bin_size_x, 
        double bin_size_y, 
        int num_movable_nodes, 
        int num_terminals
        ) 
{
    CHECK_FLAT(pos); 
    CHECK_EVEN(pos);
    CHECK_CONTIGUOUS(pos);

    int num_bins_x = int(ceil((xh-xl)/bin_size_x));
    int num_bins_y = int(ceil((yh-yl)/bin_size_y));
    at::Tensor density_map = at::zeros({num_bins_x, num_bins_y}, pos.options());

    if (num_terminals)
    {
        // Call the cuda kernel launcher
        AT_DISPATCH_FLOATING_TYPES(pos.type(), "computeDensityOverflowMapLauncher", [&] {
                computeDensityOverflowMapLauncher<scalar_t>(
                        pos.data<scalar_t>()+num_movable_nodes, pos.data<scalar_t>()+pos.numel()/2+num_movable_nodes, 
                        node_size_x.data<scalar_t>()+num_movable_nodes, node_size_y.data<scalar_t>()+num_movable_nodes, 
                        bin_center_x.data<scalar_t>(), bin_center_y.data<scalar_t>(), 
                        num_terminals, 
                        num_bins_x, num_bins_y, 
                        xl, yl, xh, yh, 
                        bin_size_x, bin_size_y, 
                        0, 
                        density_map.data<scalar_t>(), 
                        nullptr
                        );
                });
    }

    return density_map;
}
template <typename T>
int computeDensityOverflowMapLauncher(
        const T* x_tensor, const T* y_tensor, 
        const T* node_size_x_tensor, const T* node_size_y_tensor, 
        const T* bin_center_x_tensor, const T* bin_center_y_tensor, 
        const int num_nodes, 
        const int num_bins_x, const int num_bins_y, 
        const T xl, const T yl, const T xh, const T yh, 
        const T bin_size_x, const T bin_size_y, 
        const T target_area, 
        T* density_map_tensor, T* density_cost_tensor
        )
{
    // density_map_tensor and density_cost_tensor should be initialized outside 
    
    // density overflow function 
    auto computeDensityOverflowFunc = [](T x, T node_size, T bin_center, T bin_size){
        return std::max(T(0.0), std::min(x+node_size, bin_center+bin_size/2) - std::max(x, bin_center-bin_size/2));
    };
    for (int i = 0; i < num_nodes; ++i)
    {
        // x direction 
        int bin_index_xl = int((x_tensor[i]-xl)/bin_size_x);
        int bin_index_xh = int(ceil((x_tensor[i]-xl+node_size_x_tensor[i])/bin_size_x))+1; // exclusive 
        bin_index_xl = std::max(bin_index_xl, 0); 
        bin_index_xh = std::min(bin_index_xh, num_bins_x);

        // y direction 
        int bin_index_yl = int((y_tensor[i]-yl-2*bin_size_y)/bin_size_y);
        int bin_index_yh = int(ceil((y_tensor[i]-yl+node_size_y_tensor[i]+2*bin_size_y)/bin_size_y))+1; // exclusive 
        bin_index_yl = std::max(bin_index_yl, 0); 
        bin_index_yh = std::min(bin_index_yh, num_bins_y);

        for (int k = bin_index_xl; k < bin_index_xh; ++k)
        {
            T px = computeDensityOverflowFunc(x_tensor[i], node_size_x_tensor[i], bin_center_x_tensor[k], bin_size_x);
            for (int h = bin_index_yl; h < bin_index_yh; ++h)
            {
                T py = computeDensityOverflowFunc(y_tensor[i], node_size_y_tensor[i], bin_center_y_tensor[h], bin_size_y);
                //printf("px[%d, %d] = %g, py[%d, %d] = %g\n", k, h, px, k, h, py);

                // still area 
                density_map_tensor[k*num_bins_y+h] += px*py;
            }
        }
    }

    if (density_cost_tensor)
    {
        for (int i = 0; i < num_bins_x; ++i)
        {
            for (int j = 0; j < num_bins_y; ++j)
            {
                //printf("density_map[%d, %d] = %g, target_area = %g\n", i, j, density_map_tensor[i*num_bins_y+j], target_area);
                *density_cost_tensor += std::max(density_map_tensor[i*num_bins_y+j]-target_area, T(0.0));
            }
        }
        //printf("density_cost_tensor = %g\n", *density_cost_tensor);
        // convert area to density 
        //*density_cost_tensor /= bin_size_x*bin_size_y; 
    }
    return 0; 
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &density_overflow_forward, "DensityOverflow forward");
  //m.def("backward", &density_overflow_backward, "DensityOverflow backward");
  m.def("fixed_density_map", &fixed_density_overflow_map, "DensityOverflow Map for Fixed Cells");
}
