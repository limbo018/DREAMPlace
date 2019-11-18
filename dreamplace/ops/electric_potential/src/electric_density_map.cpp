/**
 * @file   density_map.cpp
 * @author Yibo Lin
 * @date   Aug 2018
 * @brief  Compute density map according to e-place (http://cseweb.ucsd.edu/~jlu/papers/eplace-todaes14/paper.pdf)
 */
#include "utility/src/torch.h"
#include "utility/src/Msg.h"
#include "utility/src/utils.h"
#include "electric_potential/src/density_function.h"

DREAMPLACE_BEGIN_NAMESPACE

/// define triangle_density_function
template <typename T> 
DEFINE_TRIANGLE_DENSITY_FUNCTION(T);
/// define exact_density_function
template <typename T> 
DEFINE_EXACT_DENSITY_FUNCTION(T);

/// @brief The triangular density model from e-place.
/// The impact of a cell to bins is extended to two neighboring bins
template <typename T>
int computeTriangleDensityMapLauncher(
        const T* x_tensor, const T* y_tensor,
        const T* node_size_x_tensor, const T* node_size_y_tensor,
        const T *offset_x_tensor, const T *offset_y_tensor,
        const T* ratio_tensor,
        const T* bin_center_x_tensor, const T* bin_center_y_tensor,
        const int num_nodes,
        const int num_bins_x, const int num_bins_y,
        const T xl, const T yl, const T xh, const T yh,
        const T bin_size_x, const T bin_size_y,
        const int num_threads,
        T* density_map_tensor
        );

/// @brief The exact density model.
/// Compute the exact overlap area for density
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
        const int num_threads,
        T* density_map_tensor
        );

#define CHECK_FLAT(x) AT_ASSERTM(!x.is_cuda() && x.ndimension() == 1, #x "must be a flat tensor on CPU")
#define CHECK_EVEN(x) AT_ASSERTM((x.numel()&1) == 0, #x "must have even number of elements")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x "must be contiguous")

/// @brief compute density map for movable and filler cells
/// @param pos cell locations. The array consists of all x locations and then y locations.
/// @param node_size_x cell width array
/// @param node_size_y cell height array
/// @param bin_center_x bin center x locations
/// @param bin_center_y bin center y locations
/// @param initial_density_map initial density map for fixed cells
/// @param target_density target density
/// @param xl left boundary
/// @param yl bottom boundary
/// @param xh right boundary
/// @param yh top boundary
/// @param bin_size_x bin width
/// @param bin_size_y bin height
/// @param num_movable_nodes number of movable cells
/// @param num_filler_nodes number of filler cells
/// @param padding bin padding to boundary of placement region
/// @param padding_mask padding mask with 0 and 1 to indicate padding bins with padding regions to be 1
/// @param num_bins_x number of bins in horizontal bins
/// @param num_bins_y number of bins in vertical bins
/// @param num_movable_impacted_bins_x number of impacted bins for any movable cell in x direction
/// @param num_movable_impacted_bins_y number of impacted bins for any movable cell in y direction
/// @param num_filler_impacted_bins_x number of impacted bins for any filler cell in x direction
/// @param num_filler_impacted_bins_y number of impacted bins for any filler cell in y direction
at::Tensor density_map(
        at::Tensor pos,
        at::Tensor node_size_x_clamped, at::Tensor node_size_y_clamped,
        at::Tensor offset_x, at::Tensor offset_y,
        at::Tensor ratio,
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
        int num_filler_impacted_bins_x, int num_filler_impacted_bins_y,
        int num_threads
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
                    node_size_x_clamped.data<scalar_t>(), node_size_y_clamped.data<scalar_t>(),
                    offset_x.data<scalar_t>(), offset_y.data<scalar_t>(),
                    ratio.data<scalar_t>(),
                    bin_center_x.data<scalar_t>(), bin_center_y.data<scalar_t>(),
                    num_movable_nodes,
                    num_bins_x, num_bins_y,
                    xl, yl, xh, yh,
                    bin_size_x, bin_size_y,
                    //false,
                    num_threads,
                    density_map.data<scalar_t>()
                    );
            });

    if (num_filler_nodes)
    {
        int num_physical_nodes = num_nodes - num_filler_nodes;
        AT_DISPATCH_FLOATING_TYPES(pos.type(), "computeTriangleDensityMapLauncher", [&] {
                computeTriangleDensityMapLauncher<scalar_t>(
                        pos.data<scalar_t>()+num_physical_nodes, pos.data<scalar_t>()+num_nodes+num_physical_nodes,
                        node_size_x_clamped.data<scalar_t>()+num_physical_nodes, node_size_y_clamped.data<scalar_t>()+num_physical_nodes,
                        offset_x.data<scalar_t>()+num_physical_nodes, offset_y.data<scalar_t>()+num_physical_nodes,
                        ratio.data<scalar_t>()+num_physical_nodes,
                        bin_center_x.data<scalar_t>(), bin_center_y.data<scalar_t>(),
                        num_filler_nodes,
                        num_bins_x, num_bins_y,
                        xl, yl, xh, yh,
                        bin_size_x, bin_size_y,
                        //false,
                        num_threads,
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

/// @brief Compute density map for fixed cells
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
        int num_fixed_impacted_bins_x, int num_fixed_impacted_bins_y,
        int num_threads
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
                        num_threads,
                        density_map.data<scalar_t>()
                        );
                });

        // Fixed cells may have overlaps. We should not over-compute the density map. 
        // This is just an approximate fix. It does not guarantee the exact value in each bin. 
        density_map.clamp_max_(bin_size_x*bin_size_y);
    }

    return density_map;
}

/// @brief Compute electric force for movable and filler cells
/// @param grad_pos input gradient from backward propagation
/// @param num_bins_x number of bins in horizontal bins
/// @param num_bins_y number of bins in vertical bins
/// @param num_movable_impacted_bins_x number of impacted bins for any movable cell in x direction
/// @param num_movable_impacted_bins_y number of impacted bins for any movable cell in y direction
/// @param num_filler_impacted_bins_x number of impacted bins for any filler cell in x direction
/// @param num_filler_impacted_bins_y number of impacted bins for any filler cell in y direction
/// @param field_map_x electric field map in x direction
/// @param field_map_y electric field map in y direction
/// @param pos cell locations. The array consists of all x locations and then y locations.
/// @param node_size_x cell width array
/// @param node_size_y cell height array
/// @param bin_center_x bin center x locations
/// @param bin_center_y bin center y locations
/// @param xl left boundary
/// @param yl bottom boundary
/// @param xh right boundary
/// @param yh top boundary
/// @param bin_size_x bin width
/// @param bin_size_y bin height
/// @param num_movable_nodes number of movable cells
/// @param num_filler_nodes number of filler cells
at::Tensor electric_force(
        at::Tensor grad_pos,
        int num_bins_x, int num_bins_y,
        int num_movable_impacted_bins_x, int num_movable_impacted_bins_y,
        int num_filler_impacted_bins_x, int num_filler_impacted_bins_y,
        at::Tensor field_map_x, at::Tensor field_map_y,
        at::Tensor pos,
        at::Tensor node_size_x_clamped, at::Tensor node_size_y_clamped,
        at::Tensor offset_x, at::Tensor offset_y,
        at::Tensor ratio,
        at::Tensor bin_center_x, at::Tensor bin_center_y,
        double xl, double yl, double xh, double yh,
        double bin_size_x, double bin_size_y,
        int num_movable_nodes,
        int num_filler_nodes,
        int num_threads
        );

template <typename T>
int computeTriangleDensityMapLauncher(
        const T* x_tensor, const T* y_tensor,
        const T* node_size_x_clamped_tensor, const T* node_size_y_clamped_tensor,
        const T *offset_x_tensor, const T *offset_y_tensor,
        const T* ratio_tensor,
        const T* bin_center_x_tensor, const T* bin_center_y_tensor,
        const int num_nodes,
        const int num_bins_x, const int num_bins_y,
        const T xl, const T yl, const T xh, const T yh,
        const T bin_size_x, const T bin_size_y,
        const int num_threads,
        T* density_map_tensor
        )
{
    // density_map_tensor should be initialized outside

    T inv_bin_size_x = 1.0 / bin_size_x; 
    T inv_bin_size_y = 1.0 / bin_size_y; 
    int chunk_size = DREAMPLACE_STD_NAMESPACE::max(int(num_nodes/num_threads/16), 1);
#pragma omp parallel for num_threads(num_threads) schedule(dynamic, chunk_size)
    for (int i = 0; i < num_nodes; ++i)
    {
        // use stretched node size 
        T node_size_x = node_size_x_clamped_tensor[i];
        T node_size_y = node_size_y_clamped_tensor[i];
        T node_x = x_tensor[i] + offset_x_tensor[i];
        T node_y = y_tensor[i] + offset_y_tensor[i];
        T ratio = ratio_tensor[i];

        int bin_index_xl = int((node_x - xl) * inv_bin_size_x);
        int bin_index_xh = int(((node_x + node_size_x - xl) * inv_bin_size_x)) + 1; // exclusive
        bin_index_xl = DREAMPLACE_STD_NAMESPACE::max(bin_index_xl, 0);
        bin_index_xh = DREAMPLACE_STD_NAMESPACE::min(bin_index_xh, num_bins_x);

        int bin_index_yl = int((node_y - yl) * inv_bin_size_y);
        int bin_index_yh = int(((node_y + node_size_y - yl) * inv_bin_size_y)) + 1; // exclusive
        bin_index_yl = DREAMPLACE_STD_NAMESPACE::max(bin_index_yl, 0);
        bin_index_yh = DREAMPLACE_STD_NAMESPACE::min(bin_index_yh, num_bins_y);

        // update density potential map
        for (int k = bin_index_xl; k < bin_index_xh; ++k)
        {
            T px = triangle_density_function(node_x, node_size_x, xl, k, bin_size_x);
            T px_by_ratio = px * ratio;

            for (int h = bin_index_yl; h < bin_index_yh; ++h)
            {
                T py = triangle_density_function(node_y, node_size_y, yl, h, bin_size_y);
                T area = px_by_ratio * py;

                T& density = density_map_tensor[k * num_bins_y + h];
#pragma omp atomic
                density += area;
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
        const int num_threads,
        T* density_map_tensor
        )
{
    // density_map_tensor should be initialized outside

#pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < num_nodes; ++i)
    {
        // x direction
        int bin_index_xl = int((x_tensor[i]-xl)/bin_size_x);
        int bin_index_xh = int(ceil((x_tensor[i]-xl+node_size_x_tensor[i])/bin_size_x))+1; // exclusive
        bin_index_xl = DREAMPLACE_STD_NAMESPACE::max(bin_index_xl, 0);
        bin_index_xh = DREAMPLACE_STD_NAMESPACE::min(bin_index_xh, num_bins_x);

        // y direction
        int bin_index_yl = int((y_tensor[i]-yl)/bin_size_y);
        int bin_index_yh = int(ceil((y_tensor[i]-yl+node_size_y_tensor[i])/bin_size_y))+1; // exclusive
        bin_index_yl = DREAMPLACE_STD_NAMESPACE::max(bin_index_yl, 0);
        bin_index_yh = DREAMPLACE_STD_NAMESPACE::min(bin_index_yh, num_bins_y);

        for (int k = bin_index_xl; k < bin_index_xh; ++k)
        {
            T px = exact_density_function(x_tensor[i], node_size_x_tensor[i], bin_center_x_tensor[k], bin_size_x, xl, xh, fixed_node_flag);
            for (int h = bin_index_yl; h < bin_index_yh; ++h)
            {
                T py = exact_density_function(y_tensor[i], node_size_y_tensor[i], bin_center_y_tensor[h], bin_size_y, yl, yh, fixed_node_flag);

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
  m.def("density_map", &DREAMPLACE_NAMESPACE::density_map, "ElectricPotential Density Map");
  m.def("fixed_density_map", &DREAMPLACE_NAMESPACE::fixed_density_map, "ElectricPotential Density Map for Fixed Cells");
  m.def("electric_force", &DREAMPLACE_NAMESPACE::electric_force, "ElectricPotential Electric Force");
}
