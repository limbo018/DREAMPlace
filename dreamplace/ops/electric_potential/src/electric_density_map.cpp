/**
 * @file   density_map.cpp
 * @author Yibo Lin
 * @date   Aug 2018
 */
#include <torch/torch.h>
#include <limits>

#define SQRT2 1.414213562

// The triangular density model from e-place 
// The impact of a cell to bins is extended to two neighboring bins 
template <typename T>
int computeTriangleDensityMapLauncher(
        const T* x_tensor, const T* y_tensor, 
        const T* node_size_x_tensor, const T* node_size_y_tensor, 
        const T* bin_center_x_tensor, const T* bin_center_y_tensor, 
        const int num_nodes, 
        const int num_bins_x, const int num_bins_y, 
        const T xl, const T yl, const T xh, const T yh, 
        const T bin_size_x, const T bin_size_y, 
        T* density_map_tensor
        );

// The exact density model
// Compute the exact overlap area for density 
template <typename T>
int computeExactDensityMapLauncher(
        const T* x_tensor, const T* y_tensor, 
        const T* node_size_x_tensor, const T* node_size_y_tensor, 
        const T* bin_center_x_tensor, const T* bin_center_y_tensor, 
        const int num_nodes, 
        const int num_bins_x, const int num_bins_y, 
        const T xl, const T yl, const T xh, const T yh, 
        const T bin_size_x, const T bin_size_y, 
        bool fixed_node_flag, 
        T* density_map_tensor
        );

#define CHECK_FLAT(x) AT_ASSERTM(!x.is_cuda() && x.ndimension() == 1, #x "must be a flat tensor on CPU")
#define CHECK_EVEN(x) AT_ASSERTM((x.numel()&1) == 0, #x "must have even number of elements")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x "must be contiguous")

// compute density map for movable and filler cells 
at::Tensor density_map(
        at::Tensor pos,
        at::Tensor node_size_x, at::Tensor node_size_y,
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
        int padding, 
        at::Tensor padding_mask, 
        int num_bins_x, int num_bins_y, 
        int num_movable_impacted_bins_x, int num_movable_impacted_bins_y, 
        int num_filler_impacted_bins_x, int num_filler_impacted_bins_y
        ) 
{
    CHECK_FLAT(pos); 
    CHECK_EVEN(pos);
    CHECK_CONTIGUOUS(pos);

    at::Tensor density_map = initial_density_map.clone();
    int num_nodes = pos.numel()/2; 

    // Call the cuda kernel launcher
    AT_DISPATCH_FLOATING_TYPES(pos.type(), "computeTriangleDensityMapLauncher", [&] {
            computeTriangleDensityMapLauncher<scalar_t>(
                    pos.data<scalar_t>(), pos.data<scalar_t>()+num_nodes, 
                    node_size_x.data<scalar_t>(), node_size_y.data<scalar_t>(), 
                    bin_center_x.data<scalar_t>(), bin_center_y.data<scalar_t>(), 
                    num_movable_nodes, 
                    num_bins_x, num_bins_y, 
                    xl, yl, xh, yh, 
                    bin_size_x, bin_size_y, 
                    //false, 
                    density_map.data<scalar_t>()
                    );
            });

    if (num_filler_nodes)
    {
        AT_DISPATCH_FLOATING_TYPES(pos.type(), "computeTriangleDensityMapLauncher", [&] {
                computeTriangleDensityMapLauncher<scalar_t>(
                        pos.data<scalar_t>()+num_nodes-num_filler_nodes, pos.data<scalar_t>()+num_nodes*2-num_filler_nodes, 
                        node_size_x.data<scalar_t>()+num_nodes-num_filler_nodes, node_size_y.data<scalar_t>()+num_nodes-num_filler_nodes, 
                        bin_center_x.data<scalar_t>(), bin_center_y.data<scalar_t>(), 
                        num_filler_nodes, 
                        num_bins_x, num_bins_y, 
                        xl, yl, xh, yh, 
                        bin_size_x, bin_size_y, 
                        //false, 
                        density_map.data<scalar_t>()
                        );
                });
    }

    // set padding density 
    if (padding > 0)
    {
        density_map.masked_fill_(padding_mask, at::Scalar(target_density*bin_size_x*bin_size_y));
    }

    return density_map;
}

// compute density map for fixed cells 
at::Tensor fixed_density_map(
        at::Tensor pos,
        at::Tensor node_size_x, at::Tensor node_size_y,
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
        int num_bins_x, int num_bins_y,
        int num_fixed_impacted_bins_x, int num_fixed_impacted_bins_y
        ) 
{
    CHECK_FLAT(pos); 
    CHECK_EVEN(pos);
    CHECK_CONTIGUOUS(pos);

    at::Tensor density_map = at::zeros({num_bins_x, num_bins_y}, pos.type());

    int num_nodes = pos.numel()/2; 

    // Call the cuda kernel launcher
    if (num_terminals && num_fixed_impacted_bins_x && num_fixed_impacted_bins_y)
    {
        AT_DISPATCH_FLOATING_TYPES(pos.type(), "computeExactDensityMapLauncher", [&] {
                computeExactDensityMapLauncher<scalar_t>(
                        pos.data<scalar_t>()+num_movable_nodes, pos.data<scalar_t>()+num_nodes+num_movable_nodes, 
                        node_size_x.data<scalar_t>()+num_movable_nodes, node_size_y.data<scalar_t>()+num_movable_nodes, 
                        bin_center_x.data<scalar_t>(), bin_center_y.data<scalar_t>(), 
                        num_terminals, 
                        num_bins_x, num_bins_y, 
                        xl, yl, xh, yh, 
                        bin_size_x, bin_size_y, 
                        true, 
                        density_map.data<scalar_t>()
                        );
                });
    }

    return density_map;
}

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
        );

template <typename T>
int computeTriangleDensityMapLauncher(
        const T* x_tensor, const T* y_tensor, 
        const T* node_size_x_tensor, const T* node_size_y_tensor, 
        const T* bin_center_x_tensor, const T* bin_center_y_tensor, 
        const int num_nodes, 
        const int num_bins_x, const int num_bins_y, 
        const T xl, const T yl, const T xh, const T yh, 
        const T bin_size_x, const T bin_size_y, 
        T* density_map_tensor
        )
{
    // density_map_tensor should be initialized outside 

    // density function 
    // extend a cell to have impact between (cb-wb, cb+wb)
    auto computeDensityFunc = [](T x, T node_size, T bin_center, T bin_size){
        return std::max(T(0.0), std::min(x+node_size, bin_center+bin_size/2) - std::max(x, bin_center-bin_size/2));
    };
    for (int i = 0; i < num_nodes; ++i)
    {
        // x direction 
        // stretch node size to bin size 
        T node_size_x = bin_size_x*SQRT2; 
        T node_x = x_tensor[i]+node_size_x_tensor[i]/2-node_size_x/2;
        int bin_index_xl = int((node_x-xl)/bin_size_x);
        int bin_index_xh = int(ceil((node_x+node_size_x-xl)/bin_size_x))+1; // exclusive 
        bin_index_xl = std::max(bin_index_xl, 0); 
        bin_index_xh = std::min(bin_index_xh, num_bins_x);

        // y direction 
        // stretch node size to bin size 
        T node_size_y = bin_size_y*SQRT2; 
        T node_y = y_tensor[i]+node_size_y_tensor[i]/2-node_size_y/2;
        int bin_index_yl = int((node_y-yl)/bin_size_y);
        int bin_index_yh = int(ceil((node_y+node_size_y-yl)/bin_size_y))+1; // exclusive 
        bin_index_yl = std::max(bin_index_yl, 0); 
        bin_index_yh = std::min(bin_index_yh, num_bins_y);

        for (int k = bin_index_xl; k < bin_index_xh; ++k)
        {
            T px = computeDensityFunc(node_x, node_size_x, bin_center_x_tensor[k], bin_size_x);
            for (int h = bin_index_yl; h < bin_index_yh; ++h)
            {
                // stretch node size to bin size 
                T node_size_y = bin_size_y*SQRT2; 
                T py = computeDensityFunc(node_y, node_size_y, bin_center_y_tensor[h], bin_size_y);
                //printf("node %d@(%g, %g) size (%g, %g): px[%d, %d] = %g, py[%d, %d] = %g\n", i, x_tensor[i], y_tensor[i], node_size_x_tensor[i], node_size_y_tensor[i], k, h, px, k, h, py);

                // scale the total area back to node area 
                T area = px*py*(node_size_x_tensor[i]*node_size_y_tensor[i]/(bin_size_x*bin_size_y*2));
                // still area 
                density_map_tensor[k*num_bins_y+h] += area;
            }
        }
    }

    return 0; 
}

template <typename T>
int computeExactDensityMapLauncher(
        const T* x_tensor, const T* y_tensor, 
        const T* node_size_x_tensor, const T* node_size_y_tensor, 
        const T* bin_center_x_tensor, const T* bin_center_y_tensor, 
        const int num_nodes, 
        const int num_bins_x, const int num_bins_y, 
        const T xl, const T yl, const T xh, const T yh, 
        const T bin_size_x, const T bin_size_y, 
        bool fixed_node_flag, 
        T* density_map_tensor
        )
{
    // density_map_tensor should be initialized outside 

    // density function 
    // extend a cell to have impact between (cb-wb, cb+wb)
    auto computeDensityFunc = [](T x, T node_size, T bin_center, T bin_size, T l, T h, bool flag){
        T bin_xl = bin_center-bin_size/2;
        T bin_xh = bin_center+bin_size/2;
        if (!flag) // only for movable nodes 
        {
            // if a node is out of boundary, count in the nearest bin 
            if (bin_xl <= l) // left most bin 
            {
                bin_xl = std::min(bin_xl, x); 
            }
            if (bin_xh >= h) // right most bin 
            {
                bin_xh = std::max(bin_xh, x+node_size); 
            }
        }
        return std::max(T(0.0), std::min(x+node_size, bin_xh) - std::max(x, bin_xl));
    };
    for (int i = 0; i < num_nodes; ++i)
    {
        // x direction 
        int bin_index_xl = int((x_tensor[i]-xl)/bin_size_x);
        int bin_index_xh = int(ceil((x_tensor[i]-xl+node_size_x_tensor[i])/bin_size_x))+1; // exclusive 
        bin_index_xl = std::max(bin_index_xl, 0); 
        bin_index_xh = std::min(bin_index_xh, num_bins_x);

        // y direction 
        int bin_index_yl = int((y_tensor[i]-yl)/bin_size_y);
        int bin_index_yh = int(ceil((y_tensor[i]-yl+node_size_y_tensor[i])/bin_size_y))+1; // exclusive 
        bin_index_yl = std::max(bin_index_yl, 0); 
        bin_index_yh = std::min(bin_index_yh, num_bins_y);

        for (int k = bin_index_xl; k < bin_index_xh; ++k)
        {
            T px = computeDensityFunc(x_tensor[i], node_size_x_tensor[i], bin_center_x_tensor[k], bin_size_x, xl, xh, fixed_node_flag);
            for (int h = bin_index_yl; h < bin_index_yh; ++h)
            {
                T py = computeDensityFunc(y_tensor[i], node_size_y_tensor[i], bin_center_y_tensor[h], bin_size_y, yl, yh, fixed_node_flag);
                //printf("px[%d, %d] = %g, py[%d, %d] = %g\n", k, h, px, k, h, py);

                // still area 
                density_map_tensor[k*num_bins_y+h] += px*py;
            }
        }
    }

    return 0; 
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("density_map", &density_map, "ElectricPotential Density Map");
  m.def("fixed_density_map", &fixed_density_map, "ElectricPotential Density Map for Fixed Cells");
  m.def("electric_force", &electric_force, "ElectricPotential Electric Force");
}
