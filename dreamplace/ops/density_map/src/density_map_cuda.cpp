/**
 * @file   density_map.cpp
 * @author Yibo Lin
 * @date   Jun 2018
 * @brief  Compute density map on GPU 
 */
#include "utility/src/torch.h"
#include "utility/src/Msg.h"

DREAMPLACE_BEGIN_NAMESPACE

/// @brief compute density map 
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
int computeDensityMapCudaLauncher(
        const T* x_tensor, const T* y_tensor, 
        const T* node_size_x_tensor, const T* node_size_y_tensor, 
        const T* bin_center_x_tensor, const T* bin_center_y_tensor, 
        const int num_nodes, 
        const int num_bins_x, const int num_bins_y, 
        const T xl, const T yl, const T xh, const T yh, 
        const T bin_size_x, const T bin_size_y, 
        T* density_map_tensor
        );

/// @brief same function with different API 
/// @param boxes an array of boxes to distribute 
template <typename T>
int computeDensityMapCudaLauncher(
        const T* boxes, 
        const T* bin_center_x_tensor, const T* bin_center_y_tensor, 
        const int num_boxes, 
        const int num_bins_x, const int num_bins_y, 
        const T xl, const T yl, const T xh, const T yh, 
        const T bin_size_x, const T bin_size_y, 
        T* density_map_tensor
        );

#define CHECK_FLAT(x) AT_ASSERTM(x.is_cuda() && x.ndimension() == 1, #x "must be a flat tensor on GPU")
#define CHECK_EVEN(x) AT_ASSERTM((x.numel()&1) == 0, #x "must have even number of elements")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x "must be contiguous")

/// @brief Compute density map. 
/// @param pos cell locations, array of x locations and then y locations 
/// @param node_size_x_tensor cell width array
/// @param node_size_y_tensor cell height array 
/// @param bin_center_x_tensor bin center x locations 
/// @param bin_center_y_tensor bin center y locations 
/// @param initial_density_map initial density map 
/// @param xl left boundary 
/// @param yl bottom boundary 
/// @param xh right boundary 
/// @param yh top boundary 
/// @param bin_size_x bin width 
/// @param bin_size_y bin height 
/// @param num_movable_nodes number of movable cells 
/// @param num_filler_nodes number of filler cells 
/// @return density map
at::Tensor density_map_forward(
        at::Tensor pos,
        at::Tensor node_size_x,
        at::Tensor node_size_y,
        at::Tensor bin_center_x, 
        at::Tensor bin_center_y, 
        at::Tensor initial_density_map, 
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

    DREAMPLACE_DISPATCH_FLOATING_TYPES(pos.type(), "computeDensityMapCudaLauncher", [&] {
            computeDensityMapCudaLauncher<scalar_t>(
                    pos.data<scalar_t>(), pos.data<scalar_t>()+pos.numel()/2, 
                    node_size_x.data<scalar_t>(), node_size_y.data<scalar_t>(), 
                    bin_center_x.data<scalar_t>(), bin_center_y.data<scalar_t>(), 
                    num_movable_nodes, // only compute that for movable cells 
                    num_bins_x, num_bins_y, 
                    xl, yl, xh, yh, 
                    bin_size_x, bin_size_y, 
                    density_map.data<scalar_t>()
                    );
            });
    if (num_filler_nodes)
    {
        DREAMPLACE_DISPATCH_FLOATING_TYPES(pos.type(), "computeDensityMapCudaLauncher", [&] {
                computeDensityMapCudaLauncher<scalar_t>(
                        pos.data<scalar_t>()+pos.numel()/2-num_filler_nodes, pos.data<scalar_t>()+pos.numel()-num_filler_nodes, 
                        node_size_x.data<scalar_t>()+pos.numel()/2-num_filler_nodes, node_size_y.data<scalar_t>()+pos.numel()/2-num_filler_nodes, 
                        bin_center_x.data<scalar_t>(), bin_center_y.data<scalar_t>(), 
                        num_filler_nodes, // only compute that for movable cells 
                        num_bins_x, num_bins_y, 
                        xl, yl, xh, yh, 
                        bin_size_x, bin_size_y, 
                        density_map.data<scalar_t>()
                        );
                });
    }

    return density_map; 
}

/// @brief Compute the density overflow for fixed cells. 
/// This map can be used as the initial density map since it only needs to be computed once.  
/// @param flat_fixed_node_boxes flat array of boxes of fixed cells 
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
at::Tensor fixed_density_map(
        at::Tensor flat_fixed_node_boxes, 
        at::Tensor bin_center_x, 
        at::Tensor bin_center_y, 
        double xl, 
        double yl, 
        double xh, 
        double yh, 
        double bin_size_x, 
        double bin_size_y
        ) 
{
    int num_bins_x = int(ceil((xh-xl)/bin_size_x));
    int num_bins_y = int(ceil((yh-yl)/bin_size_y));
    int num_boxes = flat_fixed_node_boxes.numel() / 4; 
    at::Tensor density_map = at::zeros({num_bins_x, num_bins_y}, flat_fixed_node_boxes.options());

    if (num_boxes)
    {
        // Call the cuda kernel launcher
        DREAMPLACE_DISPATCH_FLOATING_TYPES(flat_fixed_node_boxes.type(), "computeDensityMapCudaLauncher", [&] {
                computeDensityMapCudaLauncher<scalar_t>(
                        flat_fixed_node_boxes.data<scalar_t>(), 
                        bin_center_x.data<scalar_t>(), bin_center_y.data<scalar_t>(), 
                        num_boxes, 
                        num_bins_x, num_bins_y, 
                        xl, yl, xh, yh, 
                        bin_size_x, bin_size_y, 
                        density_map.data<scalar_t>()
                        );
                });
    }

    return density_map;
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &DREAMPLACE_NAMESPACE::density_map_forward, "DensityMap forward (CUDA)");
  m.def("fixed_density_map", &DREAMPLACE_NAMESPACE::fixed_density_map, "DensityMap Map for Fixed Cells (CUDA)");
}
