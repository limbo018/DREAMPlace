/**
 * @file   density_overflow_cuda_by_node.cpp
 * @author Yibo Lin
 * @date   Nov 2018
 * @brief  Compute density overflow with cell-by-cell parallelization on CUDA 
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
/// @param density_map_tensor 2D density map in column-major to write 
template <typename T>
int computeDensityOverflowMapCudaByNodeLauncher(
        const T* x_tensor, const T* y_tensor, 
        const T* node_size_x_tensor, const T* node_size_y_tensor, 
        const T* bin_center_x_tensor, const T* bin_center_y_tensor, 
        const int num_nodes, 
        const int num_bins_x, const int num_bins_y, 
        const T xl, const T yl, const T xh, const T yh, 
        const T bin_size_x, const T bin_size_y, 
        T* density_map_tensor
        );

#define CHECK_FLAT(x) AT_ASSERTM(x.is_cuda() && x.ndimension() == 1, #x "must be a flat tensor on GPU")
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
        int num_filler_nodes
        ) 
{
    CHECK_FLAT(pos); 
    CHECK_EVEN(pos);
    CHECK_CONTIGUOUS(pos);

    int num_bins_x = int(ceil((xh-xl)/bin_size_x));
    int num_bins_y = int(ceil((yh-yl)/bin_size_y));
    at::Tensor density_map = initial_density_map.clone(); 
    double density_area = target_density*bin_size_x*bin_size_y;
    int num_nodes = pos.numel()/2; 

    // Call the cuda kernel launcher
    AT_DISPATCH_FLOATING_TYPES(pos.type().scalarType(), "computeDensityOverflowMapCudaByNodeLauncher", [&] {
            computeDensityOverflowMapCudaByNodeLauncher<scalar_t>(
                    pos.data<scalar_t>(), pos.data<scalar_t>()+num_nodes, 
                    node_size_x.data<scalar_t>(), node_size_y.data<scalar_t>(), 
                    bin_center_x.data<scalar_t>(), bin_center_y.data<scalar_t>(), 
                    num_movable_nodes, // only need to compute for movable nodes 
                    num_bins_x, num_bins_y, 
                    xl, yl, xh, yh, 
                    bin_size_x, bin_size_y, 
                    density_map.data<scalar_t>()
                    );
            });
    if (num_filler_nodes)
    {
        AT_DISPATCH_FLOATING_TYPES(pos.type().scalarType(), "computeDensityOverflowMapCudaByNodeLauncher", [&] {
                computeDensityOverflowMapCudaByNodeLauncher<scalar_t>(
                        pos.data<scalar_t>()+num_nodes-num_filler_nodes, pos.data<scalar_t>()+num_nodes*2-num_filler_nodes, 
                        node_size_x.data<scalar_t>()+num_nodes-num_filler_nodes, node_size_y.data<scalar_t>()+num_nodes-num_filler_nodes, 
                        bin_center_x.data<scalar_t>(), bin_center_y.data<scalar_t>(), 
                        num_filler_nodes, // only need to compute for filler nodes 
                        num_bins_x, num_bins_y, 
                        xl, yl, xh, yh, 
                        bin_size_x, bin_size_y, 
                        density_map.data<scalar_t>()
                        );
                });
    }

    // max(0, density-density_area)  
    auto delta = (density_map-density_area).clamp_min(0); 
    auto density_cost = at::sum(delta);

    auto max_density = density_map.max().div(bin_size_x*bin_size_y);

    return {density_cost, density_map, max_density}; 
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &DREAMPLACE_NAMESPACE::density_overflow_forward, "DensityOverflow forward (CUDA)");
  //m.def("backward", &DREAMPLACE_NAMESPACE::density_overflow_backward, "DensityOverflow backward (CUDA)");
}
