/**
 * @file   density_overflow.cpp
 * @author Yibo Lin
 * @date   Jun 2018
 * @brief  Compute density overflow on CPU 
 */
#include "utility/src/torch.h"
#include "utility/src/Msg.h"

DREAMPLACE_BEGIN_NAMESPACE

/// @brief compute density overflow map 
/// @param x_tensor cell x locations
/// @param y_tensor cell y locations 
/// @param node_size_x_tensor cell width array
/// @param node_size_y_tensor cell height array 
/// @param bin_center_x_tensor bin center x locations 
/// @param bin_center_y_tensor bin center y locations 
/// @param num_nodes number of cells 
/// @param num_bins_x number of bins in horizontal bins 
/// @param num_bins_y number of bins in vertical bins 
/// @param xl left boundary 
/// @param yl bottom boundary 
/// @param xh right boundary 
/// @param yh top boundary 
/// @param bin_size_x bin width 
/// @param bin_size_y bin height 
/// @param num_threads number of threads 
/// @param density_map_tensor 2D density map in column-major to write 
template <typename T>
int computeDensityOverflowMapLauncher(
        const T* x_tensor, const T* y_tensor, 
        const T* node_size_x_tensor, const T* node_size_y_tensor, 
        const T* bin_center_x_tensor, const T* bin_center_y_tensor, 
        const int num_nodes, 
        const int num_bins_x, const int num_bins_y, 
        const T xl, const T yl, const T xh, const T yh, 
        const T bin_size_x, const T bin_size_y, 
        const int num_threads, 
        T* density_map_tensor
        );

#define CHECK_FLAT(x) AT_ASSERTM(!x.is_cuda() && x.ndimension() == 1, #x "must be a flat tensor on CPU")
#define CHECK_EVEN(x) AT_ASSERTM((x.numel()&1) == 0, #x "must have even number of elements")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x "must be contiguous")

/// @brief Compute density overflow map. 
/// @param pos cell locations, array of x locations and then y locations 
/// @param node_size_x_tensor cell width array
/// @param node_size_y_tensor cell height array 
/// @param bin_center_x_tensor bin center x locations 
/// @param bin_center_y_tensor bin center y locations 
/// @param initial_density_map initial density map 
/// @param target_density target density 
/// @param xl left boundary 
/// @param yl bottom boundary 
/// @param xh right boundary 
/// @param yh top boundary 
/// @param bin_size_x bin width 
/// @param bin_size_y bin height 
/// @param num_movable_nodes number of movable cells 
/// @param num_filler_nodes number of filler cells 
/// @return density overflow map, total density overflow, maximum density 
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
        int num_filler_nodes, 
        int num_threads
        ) 
{
    CHECK_FLAT(pos); 
    CHECK_EVEN(pos);
    CHECK_CONTIGUOUS(pos);

    int num_bins_x = int(ceil((xh-xl)/bin_size_x));
    int num_bins_y = int(ceil((yh-yl)/bin_size_y));
    at::Tensor density_map = initial_density_map.clone();
    double density_area = target_density*bin_size_x*bin_size_y;

    DREAMPLACE_DISPATCH_FLOATING_TYPES(pos.type(), "computeDensityOverflowMapLauncher", [&] {
            computeDensityOverflowMapLauncher<scalar_t>(
                    DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t), DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t)+pos.numel()/2, 
                    DREAMPLACE_TENSOR_DATA_PTR(node_size_x, scalar_t), DREAMPLACE_TENSOR_DATA_PTR(node_size_y, scalar_t), 
                    DREAMPLACE_TENSOR_DATA_PTR(bin_center_x, scalar_t), DREAMPLACE_TENSOR_DATA_PTR(bin_center_y, scalar_t), 
                    num_movable_nodes, // only compute that for movable cells 
                    num_bins_x, num_bins_y, 
                    xl, yl, xh, yh, 
                    bin_size_x, bin_size_y, 
                    num_threads, 
                    DREAMPLACE_TENSOR_DATA_PTR(density_map, scalar_t)
                    );
            });
    if (num_filler_nodes)
    {
        DREAMPLACE_DISPATCH_FLOATING_TYPES(pos.type(), "computeDensityOverflowMapLauncher", [&] {
                computeDensityOverflowMapLauncher<scalar_t>(
                        DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t)+pos.numel()/2-num_filler_nodes, DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t)+pos.numel()-num_filler_nodes, 
                        DREAMPLACE_TENSOR_DATA_PTR(node_size_x, scalar_t)+pos.numel()/2-num_filler_nodes, DREAMPLACE_TENSOR_DATA_PTR(node_size_y, scalar_t)+pos.numel()/2-num_filler_nodes, 
                        DREAMPLACE_TENSOR_DATA_PTR(bin_center_x, scalar_t), DREAMPLACE_TENSOR_DATA_PTR(bin_center_y, scalar_t), 
                        num_filler_nodes, // only compute that for movable cells 
                        num_bins_x, num_bins_y, 
                        xl, yl, xh, yh, 
                        bin_size_x, bin_size_y, 
                        num_threads, 
                        DREAMPLACE_TENSOR_DATA_PTR(density_map, scalar_t)
                        );
                });
    }

    auto density_cost = (density_map-density_area).clamp_min(0.0).sum();
    auto max_density = density_map.max().div(bin_size_x*bin_size_y);

    return {density_cost, density_map, max_density};
}

/// @brief Compute the density overflow for fixed cells. 
/// This map can be used as the initial density map since it only needs to be computed once.  
/// @param pos cell locations, array of x locations and then y locations 
/// @param node_size_x_tensor cell width array
/// @param node_size_y_tensor cell height array 
/// @param bin_center_x_tensor bin center x locations 
/// @param bin_center_y_tensor bin center y locations 
/// @param xl left boundary 
/// @param yl bottom boundary 
/// @param xh right boundary 
/// @param yh top boundary 
/// @param bin_size_x bin width 
/// @param bin_size_y bin height 
/// @param num_movable_nodes number of movable cells 
/// @param num_terminals number of fixed cells 
/// @return a density map for fixed cells 
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
        int num_terminals, 
        int num_threads
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
        DREAMPLACE_DISPATCH_FLOATING_TYPES(pos.type(), "computeDensityOverflowMapLauncher", [&] {
                computeDensityOverflowMapLauncher<scalar_t>(
                        DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t)+num_movable_nodes, DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t)+pos.numel()/2+num_movable_nodes, 
                        DREAMPLACE_TENSOR_DATA_PTR(node_size_x, scalar_t)+num_movable_nodes, DREAMPLACE_TENSOR_DATA_PTR(node_size_y, scalar_t)+num_movable_nodes, 
                        DREAMPLACE_TENSOR_DATA_PTR(bin_center_x, scalar_t), DREAMPLACE_TENSOR_DATA_PTR(bin_center_y, scalar_t), 
                        num_terminals, 
                        num_bins_x, num_bins_y, 
                        xl, yl, xh, yh, 
                        bin_size_x, bin_size_y, 
                        num_threads, 
                        DREAMPLACE_TENSOR_DATA_PTR(density_map, scalar_t)
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
        int num_threads, 
        T* density_map_tensor
        )
{
    // density_map_tensor should be initialized outside 
    
    // density overflow function 
    auto computeDensityOverflowFunc = [](T x, T node_size, T bin_center, T bin_size){
        return std::max(T(0.0), std::min(x+node_size, bin_center+bin_size/2) - std::max(x, bin_center-bin_size/2));
    };
#pragma omp parallel for num_threads(num_threads)
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
                T& density = density_map_tensor[k*num_bins_y+h];
#pragma omp atomic 
                density += px*py;
            }
        }
    }

    return 0; 
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &DREAMPLACE_NAMESPACE::density_overflow_forward, "DensityOverflow forward");
  //m.def("backward", &DREAMPLACE_NAMESPACE::density_overflow_backward, "DensityOverflow backward");
  m.def("fixed_density_map", &DREAMPLACE_NAMESPACE::fixed_density_overflow_map, "DensityOverflow Map for Fixed Cells");
}
