/**
 * @file   pin_utilization_map.cpp
 * @author Zixuan Jiang, Jiaqi Gu, Yibo Lin
 * @date   Dec 2019
 * @brief  Compute the RUDY/RISA map for routing demand. 
 *         A routing/pin utilization estimator based on the following two papers
 *         "Fast and Accurate Routing Demand Estimation for efficient Routability-driven Placement", by Peter Spindler, DATE'07
 *         "RISA: Accurate and Efficient Placement Routability Modeling", by Chih-liang Eric Cheng, ICCAD'94
 */

#include "utility/src/torch.h"
#include "utility/src/Msg.h"
#include "utility/src/utils.h"

DREAMPLACE_BEGIN_NAMESPACE

#define CHECK_FLAT(x) AT_ASSERTM(!x.is_cuda() && x.ndimension() == 1, #x "must be a flat tensor on CPU")
#define CHECK_EVEN(x) AT_ASSERTM((x.numel() & 1) == 0, #x "must have even number of elements")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x "must be contiguous")

// fill the demand map pin by pin
template <typename T>
int pinDemandMapLauncher(const T *node_x, const T *node_y,
                          const T *node_size_x, const T *node_size_y,
                          const T *half_node_size_stretch_x, const T *half_node_size_stretch_y,
                          const T *pin_weights,
                          T xl, T yl, T xh, T yh,
                          T bin_size_x, T bin_size_y,
                          int num_bins_x, int num_bins_y,
                          int num_nodes,
                          int num_threads,
                          T *pin_utilization_map
                          )
{
    const T inv_bin_size_x = 1.0 / bin_size_x;
    const T inv_bin_size_y = 1.0 / bin_size_y;

    int chunk_size = DREAMPLACE_STD_NAMESPACE::max(int(num_nodes / num_threads / 16), 1);
    #pragma omp parallel for num_threads(num_threads) schedule(dynamic, chunk_size)
    for (int i = 0; i < num_nodes; ++i)
    {
        const T node_center_x = node_x[i] + node_size_x[i]/2; 
        const T node_center_y = node_y[i] + node_size_y[i]/2; 

        const T x_min = node_center_x - half_node_size_stretch_x[i];
        const T x_max = node_center_x + half_node_size_stretch_x[i];
        int bin_index_xl = int((x_min - xl) * inv_bin_size_x);
        int bin_index_xh = int((x_max - xl) * inv_bin_size_x) + 1;
        bin_index_xl = DREAMPLACE_STD_NAMESPACE::max(bin_index_xl, 0);
        bin_index_xh = DREAMPLACE_STD_NAMESPACE::min(bin_index_xh, num_bins_x);

        const T y_min = node_center_y - half_node_size_stretch_y[i];
        const T y_max = node_center_y + half_node_size_stretch_y[i];
        int bin_index_yl = int((y_min - yl) * inv_bin_size_y);
        int bin_index_yh = int((y_max - yl) * inv_bin_size_y) + 1;
        bin_index_yl = DREAMPLACE_STD_NAMESPACE::max(bin_index_yl, 0);
        bin_index_yh = DREAMPLACE_STD_NAMESPACE::min(bin_index_yh, num_bins_y);

        T density = pin_weights[i] / (half_node_size_stretch_x[i] * half_node_size_stretch_y[i] * 4);
        for (int x = bin_index_xl; x < bin_index_xh; ++x)
        {
            for (int y = bin_index_yl; y < bin_index_yh; ++y)
            {
                T overlap = (DREAMPLACE_STD_NAMESPACE::min(x_max, (x + 1) * bin_size_x) - DREAMPLACE_STD_NAMESPACE::max(x_min, x * bin_size_x)) *
                            (DREAMPLACE_STD_NAMESPACE::min(y_max, (y + 1) * bin_size_y) - DREAMPLACE_STD_NAMESPACE::max(y_min, y * bin_size_y));
                int index = x * num_bins_y + y;
                #pragma omp atomic update 
                pin_utilization_map[index] += overlap * density;
            }
        }
    }
    return 0;
}

at::Tensor pin_utilization_map_forward(
    at::Tensor pos,
    at::Tensor node_size_x,
    at::Tensor node_size_y,
    at::Tensor half_node_size_stretch_x,
    at::Tensor half_node_size_stretch_y,
    at::Tensor pin_weights,
    double xl,
    double yl,
    double xh,
    double yh,
    double bin_size_x,
    double bin_size_y,
    double unit_pin_capacity,
    double max_pin_opt_adjust_rate,
    double min_pin_opt_adjust_rate,
    int num_physical_nodes,
    int num_bins_x,
    int num_bins_y,
    int num_threads
)
{
    CHECK_FLAT(pos);
    CHECK_EVEN(pos);
    CHECK_CONTIGUOUS(pos);

    CHECK_FLAT(node_size_x);
    CHECK_CONTIGUOUS(node_size_x);
    CHECK_FLAT(node_size_y);
    CHECK_CONTIGUOUS(node_size_y);

    CHECK_FLAT(half_node_size_stretch_x);
    CHECK_CONTIGUOUS(half_node_size_stretch_x);

    CHECK_FLAT(half_node_size_stretch_y);
    CHECK_CONTIGUOUS(half_node_size_stretch_y);

    CHECK_FLAT(pin_weights);
    CHECK_CONTIGUOUS(pin_weights);

    CHECK_FLAT(node_size_x);
    CHECK_CONTIGUOUS(node_size_x);

    CHECK_FLAT(node_size_y);
    CHECK_CONTIGUOUS(node_size_y);

    at::Tensor pin_utilization_map = at::zeros({num_bins_x, num_bins_y}, pos.options());
    auto num_nodes = pos.numel()/2;

    DREAMPLACE_DISPATCH_FLOATING_TYPES(pos.type(), "pinDemandMapLauncher", [&] {
        pinDemandMapLauncher<scalar_t>(
                pos.data<scalar_t>(), pos.data<scalar_t>() + num_nodes, 
                node_size_x.data<scalar_t>(), node_size_y.data<scalar_t>(), 
                half_node_size_stretch_x.data<scalar_t>(), half_node_size_stretch_y.data<scalar_t>(),
                pin_weights.data<scalar_t>(),
                xl, yl, xh, yh,
                bin_size_x, bin_size_y,
                num_bins_x, num_bins_y,
                num_physical_nodes,
                num_threads,
                pin_utilization_map.data<scalar_t>()
                );
    });

    // convert demand to utilization in each bin
    pin_utilization_map.mul_(1 / (bin_size_x * bin_size_y * unit_pin_capacity));
    pin_utilization_map.clamp_(min_pin_opt_adjust_rate, max_pin_opt_adjust_rate);

    return pin_utilization_map; 
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &DREAMPLACE_NAMESPACE::pin_utilization_map_forward, "compute pin utilization map");
}
