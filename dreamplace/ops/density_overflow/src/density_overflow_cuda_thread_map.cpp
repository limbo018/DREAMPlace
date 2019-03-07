/**
 * @file   density_overflow_cuda_by_node.cpp
 * @author Yibo Lin
 * @date   Jun 2018
 * @brief  Compute density overflow with cell2bin parallelization on CUDA 
 */
#include <torch/torch.h>
#include <limits>

/// @brief compute density overflow map 
/// @param x_tensor cell x locations
/// @param y_tensor cell y locations 
/// @param node_size_x_tensor cell width array
/// @param node_size_y_tensor cell height array 
/// @param bin_center_x_tensor bin center x locations 
/// @param bin_center_y_tensor bin center y locations 
/// @param thread2node_map map a thread to a cell 
/// @param thread2bin_x_map map a thread to a horizontal bin index 
/// @param thread2bin_y_map map a thread to a vertical bin index 
/// @param num_threads number of threads 
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
int computeDensityOverflowMapCudaThreadMapLauncher(
        const T* x_tensor, const T* y_tensor, 
        const T* node_size_x_tensor, const T* node_size_y_tensor, 
        const T* bin_center_x_tensor, const T* bin_center_y_tensor, 
        const int* thread2node_map, const int* thread2bin_x_map, const int* thread2bin_y_map, 
        const int num_threads, 
        const int num_nodes, 
        const int num_bins_x, const int num_bins_y, 
        const T xl, const T yl, const T xh, const T yh, 
        const T bin_size_x, const T bin_size_y, 
        T* density_map_tensor
        );

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
/// @param num_impacted_bins_x number of maximum impacted bins given any cell in horizontal direction 
/// @param num_impacted_bins_y number of maximum impacted bins given any cell in vertical direction 
/// @param xl left boundary 
/// @param yl bottom boundary 
/// @param xh right boundary 
/// @param yh top boundary 
/// @param bin_size_x bin width 
/// @param bin_size_y bin height 
/// @param density_map_tensor 2D density map in column-major to write 
template <typename T>
int computeDensityOverflowMapCudaLauncher(
        const T* x_tensor, const T* y_tensor, 
        const T* node_size_x_tensor, const T* node_size_y_tensor, 
        const T* bin_center_x_tensor, const T* bin_center_y_tensor, 
        const int num_nodes, 
        const int num_bins_x, const int num_bins_y, 
        const int num_impacted_bins_x, const int num_impacted_bins_y, 
        const T xl, const T yl, const T xh, const T yh, 
        const T bin_size_x, const T bin_size_y, 
        T* density_map_tensor
        );

#define CHECK_FLAT(x) AT_ASSERTM(x.is_cuda() && x.ndimension() == 1, #x "must be a flat tensor on GPU")
#define CHECK_EVEN(x) AT_ASSERTM((x.numel()&1) == 0, #x "must have even number of elements")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x "must be contiguous")
#define CHECK_CPU(x) AT_ASSERTM(!x.is_cuda(), #x "must be a tensor on CPU")

/// @brief Compute density overflow map. 
/// @param pos cell locations, array of x locations and then y locations 
/// @param node_size_x_tensor cell width array
/// @param node_size_y_tensor cell height array 
/// @param bin_center_x_tensor bin center x locations 
/// @param bin_center_y_tensor bin center y locations 
/// @param initial_density_map initial density map 
/// @param thread2node_map map a thread to a cell 
/// @param thread2bin_x_map map a thread to a horizontal bin index 
/// @param thread2bin_y_map map a thread to a vertical bin index 
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
        at::Tensor thread2node_map, 
        at::Tensor thread2bin_x_map, 
        at::Tensor thread2bin_y_map, 
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
    AT_DISPATCH_FLOATING_TYPES(pos.type(), "computeDensityOverflowMapCudaThreadMapLauncher", [&] {
            computeDensityOverflowMapCudaThreadMapLauncher<scalar_t>(
                    pos.data<scalar_t>(), pos.data<scalar_t>()+pos.numel()/2, 
                    node_size_x.data<scalar_t>(), node_size_y.data<scalar_t>(), 
                    bin_center_x.data<scalar_t>(), bin_center_y.data<scalar_t>(), 
                    thread2node_map.data<int>(), thread2bin_x_map.data<int>(), thread2bin_y_map.data<int>(), 
                    thread2node_map.numel(), 
                    num_nodes, 
                    num_bins_x, num_bins_y, 
                    xl, yl, xh, yh, 
                    bin_size_x, bin_size_y, 
                    density_map.data<scalar_t>()
                    );
            });

    // max(0, density-density_area)  
    auto delta = (density_map-density_area).clamp_min(0); 
    auto density_cost = at::sum(delta);

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
        int num_impacted_bins_x, int num_impacted_bins_y
        ) 
{
    CHECK_FLAT(pos); 
    CHECK_EVEN(pos);
    CHECK_CONTIGUOUS(pos);

    int num_bins_x = int(ceil((xh-xl)/bin_size_x));
    int num_bins_y = int(ceil((yh-yl)/bin_size_y));
    at::Tensor density_map = at::zeros({num_bins_x, num_bins_y}, pos.options());

    if (num_terminals && num_impacted_bins_x && num_impacted_bins_y)
    {
        // Call the cuda kernel launcher
        AT_DISPATCH_FLOATING_TYPES(pos.type(), "computeDensityOverflowMapCudaLauncher", [&] {
                computeDensityOverflowMapCudaLauncher<scalar_t>(
                        pos.data<scalar_t>()+num_movable_nodes, pos.data<scalar_t>()+pos.numel()/2+num_movable_nodes, 
                        node_size_x.data<scalar_t>()+num_movable_nodes, node_size_y.data<scalar_t>()+num_movable_nodes, 
                        bin_center_x.data<scalar_t>(), bin_center_y.data<scalar_t>(), 
                        num_terminals, 
                        num_bins_x, num_bins_y, 
                        num_impacted_bins_x, num_impacted_bins_y, 
                        xl, yl, xh, yh, 
                        bin_size_x, bin_size_y, 
                        density_map.data<scalar_t>()
                        );
                });
    }

    return density_map; 
}

/// @brief Construct thread map 
/// @param node_size_x must be on CPU 
/// @param node_size_y must be on CPU 
/// @param xl left boundary 
/// @param yl bottom boundary 
/// @param xh right boundary 
/// @param yh top boundary 
/// @param bin_size_x bin width 
/// @param bin_size_y bin height 
/// @param num_movable_nodes number of movable cells 
/// @param num_filler_nodes number of filler cells 
/// @return {thread2node_map, thread2bin_x_map, thread2bin_y_map} on CPU 
std::vector<at::Tensor> thread_map(
        at::Tensor node_size_x, 
        at::Tensor node_size_y, 
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
    CHECK_CPU(node_size_x); 
    CHECK_CONTIGUOUS(node_size_x);
    CHECK_CPU(node_size_y); 
    CHECK_CONTIGUOUS(node_size_y);

    int num_bins_x = int(ceil((xh-xl)/bin_size_x));
    int num_bins_y = int(ceil((yh-yl)/bin_size_y));
    int num_nodes = node_size_x.numel(); 
    int thread_count = 0; 

    AT_DISPATCH_FLOATING_TYPES(node_size_x.type(), "thread_count", [&] {
            auto node_size_x_accessor = node_size_x.accessor<scalar_t, 1>(); 
            auto node_size_y_accessor = node_size_x.accessor<scalar_t, 1>(); 

            // compute total number of threads 
            for (int i = 0; i < num_nodes; ++i)
            {
                // skip fixed cells 
                if (i >= num_movable_nodes && i < num_nodes-num_filler_nodes)
                {
                    continue; 
                }
                int num_impacted_bins_x = ceil((node_size_x_accessor[i]+2*bin_size_x) / bin_size_x); 
                num_impacted_bins_x = std::min(num_impacted_bins_x, num_bins_x);
                int num_impacted_bins_y = ceil((node_size_y_accessor[i]+2*bin_size_y) / bin_size_y); 
                num_impacted_bins_y = std::min(num_impacted_bins_y, num_bins_y);
                thread_count += num_impacted_bins_x*num_impacted_bins_y; 
            }
    });

    // allocate memory for thread map on CPU 
    auto thread2node_map = at::zeros(thread_count, torch::CPU(at::kInt));
    auto thread2bin_x_map = at::zeros(thread_count, torch::CPU(at::kInt));
    auto thread2bin_y_map = at::zeros(thread_count, torch::CPU(at::kInt));

    AT_DISPATCH_FLOATING_TYPES(node_size_x.type(), "thread_map", [&] {
            auto node_size_x_accessor = node_size_x.accessor<scalar_t, 1>(); 
            auto node_size_y_accessor = node_size_x.accessor<scalar_t, 1>(); 

            thread_count = 0; 
            for (int i = 0; i < num_nodes; ++i)
            {
                // skip fixed cells 
                if (i >= num_movable_nodes && i < num_nodes-num_filler_nodes)
                {
                    continue; 
                }
                int num_impacted_bins_x = ceil((node_size_x_accessor[i]+2*bin_size_x) / bin_size_x); 
                num_impacted_bins_x = std::min(num_impacted_bins_x, num_bins_x);
                int num_impacted_bins_y = ceil((node_size_y_accessor[i]+2*bin_size_y) / bin_size_y); 
                num_impacted_bins_y = std::min(num_impacted_bins_y, num_bins_y);

                for (int k = 0; k < num_impacted_bins_x; ++k)
                {
                    for (int h = 0; h < num_impacted_bins_y; ++h)
                    {
                        thread2node_map[thread_count] = i;
                        thread2bin_x_map[thread_count] = k; 
                        thread2bin_y_map[thread_count] = h; 
                        ++thread_count; 
                    }
                }
            }
    });
    return {thread2node_map, thread2bin_x_map, thread2bin_y_map}; 
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &density_overflow_forward, "DensityOverflow forward (CUDA)");
  //m.def("backward", &density_overflow_backward, "DensityOverflow backward (CUDA)");
  m.def("fixed_density_map", &fixed_density_overflow_map, "DensityOverflow Map for Fixed Cells (CUDA)");
  m.def("thread_map", &thread_map, "Thread Map to Cell and Bin offset for DensityOverflow (CUDA)");
}
