#include <stdio.h>
#include <cub/cub.cuh>
#include <math.h>
#include <float.h>
#include "cuda_runtime.h"
#include "utility/src/print.h"
#include "utility/src/Msg.h"
#include "utility/src/limits.cuh"
#include "electric_potential/src/density_ops.cuh"

DREAMPLACE_BEGIN_NAMESPACE

#define SQRT2 1.4142135623730950488016887242096980785696718753769480731766797379907324784621

template<typename T>
__global__ void fillKeyArray(
        int num_bins, 
        int num_runs_out, 
        int *d_unique_out,
        T   *d_aggregates_out,
        T* density_map_tensor ///< must be initialized properly to support incremental update 
    )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_runs_out)
    {
        int key = d_unique_out[i];
        if (key < num_bins)
        {
            density_map_tensor[key] += d_aggregates_out[i];
        }
    }
}

template <typename T, typename DensityOp>
__global__ void computeExactDensityMapReduce(
        const T* x_tensor, const T* y_tensor, 
        const T* node_size_x_tensor, const T* node_size_y_tensor, 
        const T* bin_center_x_tensor, const T* bin_center_y_tensor, 
        const int num_nodes, 
        const int num_bins_x, const int num_bins_y, 
        const T xl, const T yl, const T xh, const T yh, 
        const T bin_size_x, const T bin_size_y, 
        const int num_impacted_bins_x, const int num_impacted_bins_y, 
        bool fixed_node_flag, 
        DensityOp computeDensityFunc, 
        T* d_value_in_device,
        int* d_keys_in_device
        ) 
{
    // rank-one update density map 
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int num_items = num_nodes*num_impacted_bins_x*num_impacted_bins_y;
    if (i < num_items)
    {
        // density overflow function 
        int node_id = i/(num_impacted_bins_x*num_impacted_bins_y);
        int residual_index = i-node_id*num_impacted_bins_x*num_impacted_bins_y;
        // x direction 
        int bin_index_xl = int((x_tensor[node_id]-xl)/bin_size_x);
        bin_index_xl = max(bin_index_xl, 0);
        int k = bin_index_xl+int(residual_index / num_impacted_bins_y); 
        if (k+1 > num_bins_x)
        {
            return; 
        }
        // y direction 
        int bin_index_yl = int((y_tensor[node_id]-yl)/bin_size_y);
        bin_index_yl = max(bin_index_yl, 0);
        int h = bin_index_yl+(residual_index % num_impacted_bins_y); 
        if (h+1 > num_bins_y)
        {
            return; 
        }

        T px = computeDensityFunc(x_tensor[node_id], node_size_x_tensor[node_id], bin_center_x_tensor[k], bin_size_x, xl, xh, fixed_node_flag);
        T py = computeDensityFunc(y_tensor[node_id], node_size_y_tensor[node_id], bin_center_y_tensor[h], bin_size_y, yl, yh, fixed_node_flag);
        // still area 
        d_keys_in_device[i] = k*num_bins_y+h;
        d_value_in_device[i] = px*py;
    }
}

template <typename T, typename DensityOp>
__global__ void computeTriangleDensityMapCudaReduce(
        const T* x_tensor, const T* y_tensor, 
        const T* node_size_x_tensor, const T* node_size_y_tensor, 
        const T* bin_center_x_tensor, const T* bin_center_y_tensor, 
        const int num_nodes, 
        const int num_bins_x, const int num_bins_y, 
        const T xl, const T yl, const T xh, const T yh, 
        const T bin_size_x, const T bin_size_y, 
        const int num_impacted_bins_x, const int num_impacted_bins_y, 
        DensityOp computeDensityFunc, 
        T* d_value_in,
        int* d_keys_in
        ) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_nodes)
    {
        // stretch node size to bin size 
        T node_size_x = max(bin_size_x*SQRT2, node_size_x_tensor[i]); 
        T node_size_y = max(bin_size_y*SQRT2, node_size_y_tensor[i]); 
        T node_x = x_tensor[i]+node_size_x_tensor[i]/2-node_size_x/2;
        T node_y = y_tensor[i]+node_size_y_tensor[i]/2-node_size_y/2;
        T ratio = (node_size_x_tensor[i]*node_size_y_tensor[i]/(node_size_x*node_size_y));

        int bin_index_xl = int((node_x-xl)/bin_size_x);
        int bin_index_xh = int(((node_x+node_size_x-xl)/bin_size_x))+1; // exclusive 
        bin_index_xl = max(bin_index_xl, 0); 
        bin_index_xh = min(bin_index_xh, num_bins_x);
        //int bin_index_xh = bin_index_xl+num_impacted_bins_x; 

        int bin_index_yl = int((node_y-yl)/bin_size_y);
        int bin_index_yh = int(((node_y+node_size_y-yl)/bin_size_y))+1; // exclusive 
        bin_index_yl = max(bin_index_yl, 0); 
        bin_index_yh = min(bin_index_yh, num_bins_y);
        //int bin_index_yh = bin_index_yl+num_impacted_bins_y; 

        // update density potential map 
        int idx = i*num_impacted_bins_x*num_impacted_bins_y;
        for (int k = bin_index_xl; k < bin_index_xh; ++k)
        {
            T px = computeDensityFunc(node_x, node_size_x, bin_center_x_tensor[k], bin_size_x);
            for (int h = bin_index_yl; h < bin_index_yh; ++h)
            {
                T py = computeDensityFunc(node_y, node_size_y, bin_center_y_tensor[h], bin_size_y);

                T area = px*py*ratio; 
                
                d_value_in[idx] = area;
                d_keys_in[idx] = k*num_bins_y + h;
                ++idx; 
            }
        }
    }
}

template <typename T>
int computeTriangleDensityMapCudaReduceLauncher(
        const T* x_tensor, const T* y_tensor, 
        const T* node_size_x_tensor, const T* node_size_y_tensor, 
        const T* bin_center_x_tensor, const T* bin_center_y_tensor, 
        int num_nodes, 
        const int num_bins_x, const int num_bins_y, 
        int num_impacted_bins_x, int num_impacted_bins_y, 
        const T xl, const T yl, const T xh, const T yh, 
        const T bin_size_x, const T bin_size_y, 
        T* density_map_tensor, 
        T* values_buf, ///< max(2*num_items, num_bins+num_items)
        int* keys_buf ///< max(2*num_items, num_bins+num_items)
)
{
    int thread_count = 256; 

    int num_items = num_nodes*num_impacted_bins_x*num_impacted_bins_y;
    // must use this order, because num_bins may be larger than num_items
    T* d_values_in = values_buf+num_items;
    int* d_keys_in = keys_buf+num_items;
    T* d_values_out = values_buf;
    int* d_keys_out = keys_buf;
    // d_value_in and d_keys_in are no longer needed when using d_unique_out and d_aggregates_out
    int* d_unique_out = d_keys_in; 
    T* d_aggregates_out = d_values_in;
    int* d_num_runs_out;

    cudaMalloc((void**)& d_num_runs_out,  sizeof(int));

    computeTriangleDensityMapCudaReduce<<<(num_nodes+thread_count-1)/thread_count, thread_count>>>(
            x_tensor, y_tensor, 
            node_size_x_tensor, node_size_y_tensor, 
            bin_center_x_tensor, bin_center_y_tensor, 
            num_nodes, 
            num_bins_x, num_bins_y, 
            xl, yl, xh, yh, 
            bin_size_x, bin_size_y, 
            num_impacted_bins_x, num_impacted_bins_y, 
            TriangleDensity<T>(), 
            d_values_in,
            d_keys_in
    );

    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, num_items);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, num_items);
   
    cudaFree(d_temp_storage);

    //printArray(d_keys_in, num_items, "d_keys_in");
    //printArray(d_values_in, num_items, "d_values_in");
    //printArray(d_keys_out, num_items, "d_keys_out");
    //printArray(d_values_out, num_items, "d_values_out");

    // Determine temporary device storage requirements
    d_temp_storage = NULL;
    temp_storage_bytes = 0;
    cub::DeviceReduce::ReduceByKey(d_temp_storage, temp_storage_bytes, d_keys_out, d_unique_out, d_values_out, d_aggregates_out, d_num_runs_out, cub::Sum(), num_items);
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Run reduce-by-key
    cub::DeviceReduce::ReduceByKey(d_temp_storage, temp_storage_bytes, d_keys_out, d_unique_out, d_values_out, d_aggregates_out, d_num_runs_out, cub::Sum(), num_items);

    //printArray(d_unique_out, num_bins_x*num_bins_y, "d_unique_out"); 
    //printArray(d_aggregates_out, num_bins_x*num_bins_y, "d_aggregates_out");
    //printScalar(d_num_runs_out, "d_num_runs_out");
    
    cudaFree(d_temp_storage);

    int h_num_runs_out = 0; 
    cudaMemcpy(&h_num_runs_out, d_num_runs_out, sizeof(int), cudaMemcpyDeviceToHost);

    fillKeyArray<<<(h_num_runs_out+thread_count-1)/thread_count, thread_count>>>(
            num_bins_x*num_bins_y, 
            h_num_runs_out, 
            d_unique_out,
            d_aggregates_out,
            density_map_tensor
            );
    cudaFree(d_num_runs_out);

    return 0; 
}

template <typename T>
int computeExactDensityMapCudaReduceLauncher(
        const T* x_tensor, const T* y_tensor, 
        const T* node_size_x_tensor, const T* node_size_y_tensor, 
        const T* bin_center_x_tensor, const T* bin_center_y_tensor, 
        const int num_nodes, 
        const int num_bins_x, const int num_bins_y, 
        const int num_impacted_bins_x, const int num_impacted_bins_y, 
        const T xl, const T yl, const T xh, const T yh, 
        const T bin_size_x, const T bin_size_y, 
        bool fixed_node_flag, 
        T* density_map_tensor, 
        T* values_buf, ///< max(2*num_items, num_bins+num_items)
        int* keys_buf ///< max(2*num_items, num_bins+num_items)
        )
{
    int thread_count = 256;

    int num_items = num_nodes*num_impacted_bins_x*num_impacted_bins_y;
    // must use this order, because num_bins may be larger than num_items
    T* d_values_in = values_buf+num_items;
    int* d_keys_in = keys_buf+num_items;
    T* d_values_out = values_buf;
    int* d_keys_out = keys_buf;
    // d_value_in and d_keys_in are no longer needed when using d_unique_out and d_aggregates_out
    int* d_unique_out = d_keys_in; 
    T* d_aggregates_out = d_values_in;
    int* d_num_runs_out;

    cudaMalloc((void**)& d_num_runs_out, sizeof(int));

    computeExactDensityMapReduce<<<(num_items + thread_count - 1) / thread_count, thread_count>>>(
            x_tensor, y_tensor, 
            node_size_x_tensor, node_size_y_tensor, 
            bin_center_x_tensor, bin_center_y_tensor, 
            num_nodes, 
            num_bins_x, num_bins_y, 
            xl, yl, xh, yh, 
            bin_size_x, bin_size_y, 
            num_impacted_bins_x, num_impacted_bins_y, 
            fixed_node_flag, 
            ExactDensity<T>(), 
            d_values_in,
            d_keys_in
    );

    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, num_items);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, num_items);
   
    //printArray(d_keys_in, num_items, "d_keys_in");
    //printArray(d_values_in, num_items, "d_values_in");
    //printArray(d_keys_out, num_items, "d_keys_out");
    //printArray(d_values_out, num_items, "d_values_out");

    cudaFree(d_temp_storage);

    // Determine temporary device storage requirements
    d_temp_storage = NULL;
    temp_storage_bytes = 0;
    cub::DeviceReduce::ReduceByKey(d_temp_storage, temp_storage_bytes, d_keys_out, d_unique_out, d_values_out, d_aggregates_out, d_num_runs_out, cub::Sum(), num_items);
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Run reduce-by-key
    cub::DeviceReduce::ReduceByKey(d_temp_storage, temp_storage_bytes, d_keys_out, d_unique_out, d_values_out, d_aggregates_out, d_num_runs_out, cub::Sum(), num_items);
    
    cudaFree(d_temp_storage);

    //printArray(d_unique_out, num_bins_x*num_bins_y, "d_unique_out"); 
    //printArray(d_aggregates_out, num_bins_x*num_bins_y, "d_aggregates_out");
    //printScalar(d_num_runs_out, "d_num_runs_out");

    int h_num_runs_out = 0; 
    cudaMemcpy(&h_num_runs_out, d_num_runs_out, sizeof(int), cudaMemcpyDeviceToHost);

    fillKeyArray<<<(num_bins_x*num_bins_y+thread_count-1)/thread_count, thread_count>>>(
            num_bins_x*num_bins_y, 
            h_num_runs_out, 
            d_unique_out,
            d_aggregates_out,
            density_map_tensor
            );
    cudaFree(d_num_runs_out);
    return 0; 
}

#define REGISTER_KERNEL_LAUNCHER(T) \
    int instantiateComputeTriangleDensityMapReduceLauncher(\
            const T* x_tensor, const T* y_tensor, \
            const T* node_size_x_tensor, const T* node_size_y_tensor, \
            const T* bin_center_x_tensor, const T* bin_center_y_tensor, \
            const int num_nodes, \
            const int num_bins_x, const int num_bins_y, \
            const int num_impacted_bins_x, const int num_impacted_bins_y, \
            const T xl, const T yl, const T xh, const T yh, \
            const T bin_size_x, const T bin_size_y, \
            T* density_map_tensor, \
            T* values_buf, \
            int* keys_buf \
            )\
    { \
        return computeTriangleDensityMapCudaReduceLauncher(\
                x_tensor, y_tensor, \
                node_size_x_tensor, node_size_y_tensor, \
                bin_center_x_tensor, bin_center_y_tensor, \
                num_nodes, \
                num_bins_x, num_bins_y, \
                num_impacted_bins_x, num_impacted_bins_y, \
                xl, yl, xh, yh, \
                bin_size_x, bin_size_y, \
                density_map_tensor, \
                values_buf, \
                keys_buf \
                );\
    } \
    \
    int instantiateComputeExactDensityMapReduceLauncher(\
            const T* x_tensor, const T* y_tensor, \
            const T* node_size_x_tensor, const T* node_size_y_tensor, \
            const T* bin_center_x_tensor, const T* bin_center_y_tensor, \
            const int num_nodes, \
            const int num_bins_x, const int num_bins_y, \
            const int num_impacted_bins_x, const int num_impacted_bins_y, \
            const T xl, const T yl, const T xh, const T yh, \
            const T bin_size_x, const T bin_size_y, \
            bool fixed_node_flag, \
            T* density_map_tensor, \
            T* values_buf, \
            int* keys_buf \
            )\
    { \
        return computeExactDensityMapCudaReduceLauncher(\
                x_tensor, y_tensor, \
                node_size_x_tensor, node_size_y_tensor, \
                bin_center_x_tensor, bin_center_y_tensor, \
                num_nodes, \
                num_bins_x, num_bins_y, \
                num_impacted_bins_x, num_impacted_bins_y, \
                xl, yl, xh, yh, \
                bin_size_x, bin_size_y, \
                fixed_node_flag, \
                density_map_tensor, \
                values_buf, \
                keys_buf \
                );\
    }

REGISTER_KERNEL_LAUNCHER(float);
REGISTER_KERNEL_LAUNCHER(double);

DREAMPLACE_END_NAMESPACE
