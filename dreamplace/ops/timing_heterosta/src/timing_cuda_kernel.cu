#include "cuda_runtime.h"
#include "utility/src/utils.cuh"
#include <heterosta.h>
#include <cfloat>
#include <string>

DREAMPLACE_BEGIN_NAMESPACE
template<typename T>
__global__ void update_net_weights_lilith(
    STAHoldings* sta,
    int num_nets,
    int num_pins,
    const int* flat_netpin,
    const int* netpin_start,
    const int* pin_to_net_map,
    T* net_criticality,
    T* net_criticality_deltas,
    T* net_weights,
    T* net_weight_deltas,
    T momentum_decay_factor,
    T max_net_weight,
    const float* slacks,
    float wns,
    const int* degree_map,
    int ignore_net_degree) {
	int net_i = blockIdx.x * blockDim.x + threadIdx.x;
	if(net_i >= num_nets) return;

	
	float net_slack = FLT_MAX;
	int np_s = netpin_start[net_i], np_e = netpin_start[net_i + 1];
	for(int np_i = np_s; np_i < np_e; ++np_i) {
		int pin_i = flat_netpin[np_i];
		if(pin_i >= 0 && pin_i < num_pins && slacks[pin_i] < net_slack) net_slack = slacks[pin_i];
		// if(slacks_hold[pin_i] < net_slack) net_slack = slacks_hold[pin_i];
		// if(net_i == 1209074) {
		// 	printf("net %d pin{%d}[%d] slack %f\n", net_i, np_i - np_s, pin_i, slacks[pin_i]);
		// }
	}
	if(wns < 0) {
		// Decay the criticality value of the current net.
		float nc = (net_slack < 0) ? min(max(0.f, net_slack / wns), 1.0f) : 0;
		net_criticality[net_i] = pow(1 + net_criticality[net_i], momentum_decay_factor) * pow(1 + nc, 1 - momentum_decay_factor) - 1;

		// if(net_i == 1209074) {
		// 	printf("net %d slack %f decay critic nc %f net_criticality %f\n", net_i, net_slack, nc, net_criticality[net_i]);
		// }
	}

	// Update net weights - use degree_map instead of local degree calculation
	if(degree_map[net_i] <= ignore_net_degree) {
		net_weights[net_i] *= (1 + net_criticality[net_i]);
		if(net_weights[net_i] > max_net_weight) {
			net_weights[net_i] = max_net_weight;
		}
	}
}

// Helper kernel to compute worst slack for each pin
__global__ void compute_worst_slacks(
    const float (*slack_array)[2],
    float* worst_slacks,
    int num_pins) {

    int pin_i = blockIdx.x * blockDim.x + threadIdx.x;
    if(pin_i >= num_pins) return;

    // Take minimum of rise and fall slack as worst slack
    worst_slacks[pin_i] = fminf(slack_array[pin_i][0], slack_array[pin_i][1]);
}

template <typename T>
void updateNetWeightCudaLauncher(
    STAHoldings* sta,
    int num_nets,
    int num_pins,
    const int* flat_netpin,
    const int* netpin_start,
    const int* pin_to_net_map,
    T* net_criticality,
    T* net_criticality_deltas,
    T* net_weights,
    T* net_weight_deltas,
    const int* degree_map,
    T momentum_decay_factor,
    T max_net_weight,
    int ignore_net_degree)
{
    float wns = 0.0, tns = 0.0;
    // Get WNS/TNS from HeteroSTA directly to GPU memory
    bool success = heterosta_report_wns_tns_max(sta, &wns, &tns, true);

    // Allocate GPU memory for slack array
    float *d_slack_data;
    cudaMalloc(&d_slack_data, num_pins * 2 * sizeof(float));
    float (*d_slack_array)[2] = reinterpret_cast<float(*)[2]>(d_slack_data);

    // Get pin slacks from HeteroSTA directly to GPU memory
    heterosta_report_slacks_at_max(sta, d_slack_array, true);

    // Allocate GPU memory for worst slacks
    float *d_worst_slacks;
    cudaMalloc(&d_worst_slacks, num_pins * sizeof(float));

    // Compute worst slack for each pin using helper kernel
    const int slack_bs = 512;
    compute_worst_slacks<<<(num_pins + slack_bs - 1) / slack_bs, slack_bs>>>(
        d_slack_array, d_worst_slacks, num_pins);

    // Launch the main CUDA kernel
    const int bs = 512;
    update_net_weights_lilith<<<(num_nets + bs - 1) / bs, bs>>>(
        sta, num_nets, num_pins, flat_netpin, netpin_start, pin_to_net_map,
        net_criticality, net_criticality_deltas, net_weights, net_weight_deltas,
        momentum_decay_factor, max_net_weight, d_worst_slacks, wns,
        degree_map, ignore_net_degree
    );

    // Synchronize before freeing memory to ensure kernels complete
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
    }

    cudaFree(d_slack_data);
    cudaFree(d_worst_slacks);
}

// Explicit template instantiation
#define REGISTER_KERNEL_LAUNCHER(T) \
    template void updateNetWeightCudaLauncher<T>( \
        STAHoldings* sta, \
        int num_nets, \
        int num_pins, \
        const int* flat_netpin, \
        const int* netpin_start, \
        const int* pin_to_net_map, \
        T* net_criticality, \
        T* net_criticality_deltas, \
        T* net_weights, \
        T* net_weight_deltas, \
        const int* degree_map, \
        T momentum_decay_factor, \
        T max_net_weight, \
        int ignore_net_degree);

REGISTER_KERNEL_LAUNCHER(float);
REGISTER_KERNEL_LAUNCHER(double);

DREAMPLACE_END_NAMESPACE