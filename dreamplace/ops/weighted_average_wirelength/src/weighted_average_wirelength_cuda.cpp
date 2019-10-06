/**
 * @file   hpwl_cuda.cpp
 * @author Yibo Lin
 * @date   Jun 2018
 * @brief  Compute weighted-average wirelength and gradient according to e-place
 */
#include "utility/src/torch.h"
#include "utility/src/Msg.h"

DREAMPLACE_BEGIN_NAMESPACE

/// @brief Compute weighted average wirelength and gradient.
/// WL = \sum_i x_i*exp(x_i/gamma) / \sum_i exp(x_i/gamma) - \sum_i x_i*exp(-x_i/gamma) / \sum_i x_i*exp(-x_i/gamma),
/// where x_i is pin location.
///
/// @param x x location of pins.
/// @param y y location of pins.
/// @param flat_netpin consists pins of each net, pins belonging to the same net are abutting to each other.
/// @param netpin_start bookmark for the starting index of each net in flat_netpin. The length is number of nets. The last entry equals to the number of pins.
/// @param net_mask whether compute the wirelength for a net or not
/// @param net_weights weight of nets
/// @param num_nets number of nets.
/// @param inv_gamma the inverse number of gamma coefficient in weighted average wirelength.
/// @param partial_wl wirelength in x and y directions of each net. The first half is the wirelength in x direction, and the second half is the wirelength in y direction.
/// @param grad_tensor back-propagated gradient from previous stage.
/// @param grad_x_tensor gradient in x direction.
/// @param grad_y_tensor gradient in y direction.
/// @return 0 if successfully done.
template <typename T, typename V>
int computeWeightedAverageWirelengthCudaLauncher(
    const T *x, const T *y,
    const int *pin2net_map,
    const int *flat_netpin,
    const int *netpin_start,
    const unsigned char *net_mask,
    int num_nets,
    int num_pins,
    const T *inv_gamma,
    T *exp_xy, T *exp_nxy,
    T *exp_xy_sum, T *exp_nxy_sum,
    T *xyexp_xy_sum, T *xyexp_nxy_sum,
    V *xy_max, V *xy_min,
    T *partial_wl,
    const T *grad_tensor,
    T *grad_x_tensor, T *grad_y_tensor);

/// @brief add net weights to gradient 
template <typename T>
void integrateNetWeightsCudaLauncher(
    const int *pin2net_map,
    const unsigned char *net_mask,
    const T *net_weights,
    T *grad_x_tensor, T *grad_y_tensor,
    int num_pins);

#define CHECK_FLAT(x) AT_ASSERTM(x.is_cuda() && x.ndimension() == 1, #x "must be a flat tensor on GPU")
#define CHECK_EVEN(x) AT_ASSERTM((x.numel() & 1) == 0, #x "must have even number of elements")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x "must be contiguous")

typedef int V;

/// @brief Compute weighted average wirelength and gradient.
/// WL = \sum_i x_i*exp(x_i/gamma) / \sum_i exp(x_i/gamma) - \sum_i x_i*exp(-x_i/gamma) / \sum_i x_i*exp(-x_i/gamma),
/// where x_i is pin location.
///
/// @param pos location of pins, x array followed by y array.
/// @param flat_netpin consists pins of each net, pins belonging to the same net are abutting to each other.
/// @param netpin_start bookmark for the starting index of each net in flat_netpin. The length is number of nets. The last entry equals to the number of pins.
/// @param net_weights weight of nets
/// @param net_mask whether compute the wirelength for a net or not
/// @param inv_gamma the inverse number of gamma coefficient in weighted average wirelength.
/// @return total wirelength cost.
std::vector<at::Tensor> weighted_average_wirelength_forward(
    at::Tensor pos,
    at::Tensor flat_netpin,
    at::Tensor netpin_start,
    at::Tensor pin2net_map,
    at::Tensor net_weights,
    at::Tensor net_mask,
    at::Tensor inv_gamma)
{
    CHECK_FLAT(pos);
    CHECK_EVEN(pos);
    CHECK_CONTIGUOUS(pos);
    CHECK_FLAT(flat_netpin);
    CHECK_CONTIGUOUS(flat_netpin);
    CHECK_FLAT(netpin_start);
    CHECK_CONTIGUOUS(netpin_start);
    CHECK_FLAT(net_weights);
    CHECK_CONTIGUOUS(net_weights);
    CHECK_FLAT(net_mask);
    CHECK_CONTIGUOUS(net_mask);
    CHECK_FLAT(pin2net_map);
    CHECK_CONTIGUOUS(pin2net_map);

    int num_nets = netpin_start.numel() - 1;
    int num_pins = pos.numel() / 2;
    
    at::Tensor exp_xy = at::empty_like(pos);
    at::Tensor exp_nxy = at::empty_like(pos);
    at::Tensor exp_xy_sum = at::zeros({2, num_nets}, pos.options());
    at::Tensor exp_nxy_sum = at::zeros({2, num_nets}, pos.options());
    at::Tensor xyexp_xy_sum = at::zeros({2, num_nets}, pos.options());
    at::Tensor xyexp_nxy_sum = at::zeros({2, num_nets}, pos.options());
    at::Tensor partial_wl = at::zeros({num_nets}, pos.options());

    // it is ok for xy_max and xy_min to be integer
    // we do not really need accurate max/min, just some values to scale x/y
    // therefore, there is no need to scale xy_max and xy_min to improve accuracy
    at::Tensor xy_max = at::full({2, num_nets}, std::numeric_limits<V>::min(), at::CUDA(at::kInt));
    at::Tensor xy_min = at::full({2, num_nets}, std::numeric_limits<V>::max(), at::CUDA(at::kInt));

    DREAMPLACE_DISPATCH_FLOATING_TYPES(pos.type(), "computeWeightedAverageWirelengthCudaLauncher", [&] {
        computeWeightedAverageWirelengthCudaLauncher<scalar_t, V>(
            pos.data<scalar_t>(), pos.data<scalar_t>() + num_pins,
            nullptr,
            flat_netpin.data<int>(),
            netpin_start.data<int>(),
            net_mask.data<unsigned char>(),
            num_nets,
            num_pins,
            inv_gamma.data<scalar_t>(),
            exp_xy.data<scalar_t>(), exp_nxy.data<scalar_t>(),
            exp_xy_sum.data<scalar_t>(), exp_nxy_sum.data<scalar_t>(),
            xyexp_xy_sum.data<scalar_t>(), xyexp_nxy_sum.data<scalar_t>(),
            xy_max.data<V>(), xy_min.data<V>(),
            partial_wl.data<scalar_t>(),
            nullptr,
            nullptr, nullptr);
        if (net_weights.numel())
        {
            partial_wl.mul_(net_weights.view({num_nets}));
        }
    });

    auto wl = partial_wl.sum();
    return {wl, exp_xy, exp_nxy, exp_xy_sum, exp_nxy_sum, xyexp_xy_sum, xyexp_nxy_sum};
}

/// @brief Compute gradient
/// @param grad_pos input gradient from backward propagation
/// @param pos locations of pins
/// @param flat_netpin similar to the JA array in CSR format, which is flattened from the net2pin map (array of array)
/// @param netpin_start similar to the IA array in CSR format, IA[i+1]-IA[i] is the number of pins in each net, the length of IA is number of nets + 1
/// @param net_weights weight of nets
/// @param net_mask an array to record whether compute the where for a net or not
/// @param inv_gamma a scalar tensor for the parameter in the equation
at::Tensor weighted_average_wirelength_backward(
    at::Tensor grad_pos,
    at::Tensor pos,
    at::Tensor exp_xy, at::Tensor exp_nxy,
    at::Tensor exp_xy_sum, at::Tensor exp_nxy_sum,
    at::Tensor xyexp_xy_sum, at::Tensor xyexp_nxy_sum,
    at::Tensor flat_netpin,
    at::Tensor netpin_start,
    at::Tensor pin2net_map,
    at::Tensor net_weights,
    at::Tensor net_mask,
    at::Tensor inv_gamma)
{
    CHECK_FLAT(pos);
    CHECK_EVEN(pos);
    CHECK_CONTIGUOUS(pos);
    CHECK_FLAT(flat_netpin);
    CHECK_CONTIGUOUS(flat_netpin);
    CHECK_FLAT(netpin_start);
    CHECK_CONTIGUOUS(netpin_start);
    CHECK_FLAT(net_weights);
    CHECK_CONTIGUOUS(net_weights);
    CHECK_FLAT(net_mask);
    CHECK_CONTIGUOUS(net_mask);
    CHECK_FLAT(pin2net_map);
    CHECK_CONTIGUOUS(pin2net_map);

    at::Tensor grad_out = at::zeros_like(pos);
    int num_nets = netpin_start.numel() - 1;
    int num_pins = pos.numel() / 2;

    DREAMPLACE_DISPATCH_FLOATING_TYPES(pos.type(), "computeWeightedAverageWirelengthCudaLauncher", [&] {
        computeWeightedAverageWirelengthCudaLauncher<scalar_t, V>(
            pos.data<scalar_t>(), pos.data<scalar_t>() + num_pins,
            pin2net_map.data<int>(),
            flat_netpin.data<int>(),
            netpin_start.data<int>(),
            net_mask.data<unsigned char>(),
            num_nets,
            num_pins,
            inv_gamma.data<scalar_t>(),
            exp_xy.data<scalar_t>(), exp_nxy.data<scalar_t>(),
            exp_xy_sum.data<scalar_t>(), exp_nxy_sum.data<scalar_t>(),
            xyexp_xy_sum.data<scalar_t>(), xyexp_nxy_sum.data<scalar_t>(),
            nullptr, nullptr,
            nullptr,
            grad_pos.data<scalar_t>(),
            grad_out.data<scalar_t>(), grad_out.data<scalar_t>() + num_pins);
        if (net_weights.numel())
        {
            integrateNetWeightsCudaLauncher(
                pin2net_map.data<int>(),
                net_mask.data<unsigned char>(),
                net_weights.data<scalar_t>(),
                grad_out.data<scalar_t>(), grad_out.data<scalar_t>() + num_pins,
                num_pins);
        }
    });
    return grad_out;
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &DREAMPLACE_NAMESPACE::weighted_average_wirelength_forward, "WeightedAverageWirelength forward (CUDA)");
    m.def("backward", &DREAMPLACE_NAMESPACE::weighted_average_wirelength_backward, "WeightedAverageWirelength backward (CUDA)");
}
