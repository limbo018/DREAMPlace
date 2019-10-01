#include <stdio.h>
#include <cub/cub.cuh>
#include <math.h>
#include <float.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/reverse_iterator.h>
#include <thrust/device_vector.h>
#include "cuda_runtime.h"
#include "utility/src/print.h"
#include "utility/src/Msg.h"
#include "utility/src/limits.cuh"
#include "electric_potential/src/density_ops.cuh"

DREAMPLACE_BEGIN_NAMESPACE

#define SQRT2 1.4142135623730950488016887242096980785696718753769480731766797379907324784621

template<typename T, typename V>
__global__ void fillKeyArray(
        int num_bins, 
        int num_runs_out, 
        V *d_unique_out,
        T   *d_aggregates_out,
        T* density_map_tensor ///< must be initialized properly to support incremental update 
    )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_runs_out)
    {
        V key = d_unique_out[i];
        if (key < num_bins)
        {
            density_map_tensor[key] += d_aggregates_out[i];
        }
    }
}

template <typename T>
__global__ void fillDensityMap(int num_bins, const T* d_aggregates_out, T* density_map_tensor)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_bins)
    {
        T value = d_aggregates_out[i]; 
        if (value != 0)
        {
            density_map_tensor[i] += value; 
        }
    }
}

template <typename T, typename V, typename DensityOp>
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
        T* d_value_in,
        V* d_keys_in
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
        // add an offset to make keys unique 
        d_keys_in[i] = (V)(k*num_bins_y + h)*(V)num_nodes + node_id;
        d_value_in[i] = px*py;
    }
}

template <typename T, typename V, typename DensityOp>
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
        V* d_keys_in
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
                // add an offset to make keys unique 
                d_keys_in[idx] = (V)(k*num_bins_y + h)*(V)num_nodes + i;
                //d_keys_in[idx] = k*num_bins_y + h;
                ++idx; 
            }
        }
    }
}

template <typename T>
__global__ void recoverUniqueKeys(T* keys, int* num_items_valid, int num_items, int /*num_bins*/, T divisor)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_items)
    {
        T& key = keys[i];
        if (key != cuda::numeric_limits<T>::max())
        {
            key /= divisor; 
            atomicMax(num_items_valid, i+1); 
        }
    }
}

template <typename T, typename V>
__global__ void checkKeys(const T* keys, int num_items, int num_bins, const V* values)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_items)
    {
        assert(keys[i] >= 0);
        if (keys[i] >= num_bins)
        {
            assert(keys[i] == cuda::numeric_limits<T>::max());
            assert(values[i] == 0);
        }
        if (i)
        {
            assert(keys[i-1] <= keys[i]);
        }
    }
}

template <typename V>
__global__ void convertKeysToOffsetsStep1(const V* keys, int num_items, int num_items_valid, int* offsets, int num_bins)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // consider 3 cases of keys 
    // keys:    0, 2, 2, 3
    // offsets: 0, 1, 1, 3, 4
    // keys:    2, 2, 2, 3
    // offsets: 0, 0, 0, 3, 4
    // keys:    0, 1, 1, 3
    // offsets: 0, 1, 3, 3, 4
    if (i < num_items)
    {
        V key = keys[i]; 
        if (i == 0)
        {
            offsets[0] = 0; 
            // the first element may not be key = 0
            offsets[min(key, (V)num_bins)] = 0;
            // last element must be num_items_valid
            offsets[num_bins] = num_items_valid;
        }
        else 
        {
            V prev_key = keys[i-1];
            if (prev_key != key)
            {
                // use prev_key+1 because key can be max integer
                assert(prev_key < num_bins);
                offsets[prev_key+1] = i; 
            }
        }
    }
}

template <typename T>
void printAggregateSol(const int* d_offsets_out, const T* d_aggregates_out, int num_bins, const char* str)
{
    std::vector<int> h_offsets_out (num_bins+1); 
    std::vector<T> h_aggregates_out (num_bins+1); 

    std::cout << str << "[" << num_bins << "]" << "\n";
    cudaMemcpy(h_offsets_out.data(), d_offsets_out, (num_bins+1)*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_aggregates_out.data(), d_aggregates_out, num_bins*sizeof(T), cudaMemcpyDeviceToHost);
    for (int i = 0; i < num_bins; ++i)
    {
        if (h_aggregates_out[i] != 0)
        {
            UFloat<T> uf; 
            uf.f = h_aggregates_out[i]; 
            std::cout << "[" << i << "]: " << uf.u << " offset " << h_offsets_out[i] << "\n"; 
        }
    }
    std::cout << "\n";
}

template <typename T, typename V>
int computeDensityMapCudaReduceLauncher(
        const T* x_tensor, const T* y_tensor, 
        const T* node_size_x_tensor, const T* node_size_y_tensor, 
        const T* bin_center_x_tensor, const T* bin_center_y_tensor, 
        int num_nodes, 
        const int num_bins_x, const int num_bins_y, 
        int num_impacted_bins_x, int num_impacted_bins_y, 
        const T xl, const T yl, const T xh, const T yh, 
        const T bin_size_x, const T bin_size_y, 
        bool fixed_node_flag, ///< fixed node or not  
        bool triangle_or_exact_flag, ///< triangle mode or exact mode 
        T* density_map_tensor, 
        T* values_buf, ///< max(2*num_items, num_bins+1+num_items)
        V* keys_buf, ///< max(2*num_items, num_bins+1+num_items)
        int* offsets_buf ///< (num_bins+1)*2
)
{
    int thread_count = 256; 

    int num_items = num_nodes*num_impacted_bins_x*num_impacted_bins_y;
    int num_bins = num_bins_x*num_bins_y;
    // must use this order, because num_bins may be larger than num_items
    T* d_values_in = values_buf+num_items; // length of num_items
    V* d_keys_in = keys_buf+num_items; // length of num_items
    T* d_values_out = values_buf; // length of num_items
    V* d_keys_out = keys_buf; // length of num_items
    int* d_offsets_in = offsets_buf+num_bins+1; 
    int* d_offsets_out = offsets_buf; 
    // d_value_in and d_keys_in are no longer needed when using d_unique_out and d_aggregates_out
    //V* d_unique_out = d_keys_in; 
    T* d_aggregates_out = d_values_in; // length of num_bins+1

    if (triangle_or_exact_flag)
    {
        computeTriangleDensityMapCudaReduce<<<(num_nodes + thread_count - 1) / thread_count, thread_count>>>(
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
    }
    else 
    {
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
    }

    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, num_items);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, num_items);
   
    cudaFree(d_temp_storage);

    int h_num_items_valid = 0; 
    int* d_num_items_valid; 
    cudaMalloc((void**)& d_num_items_valid, sizeof(int));
    cudaMemset(d_num_items_valid, 0, sizeof(int));
    recoverUniqueKeys<<<(num_items+thread_count-1)/thread_count, thread_count>>>(d_keys_out, d_num_items_valid, num_items, num_bins, (V)num_nodes);
    checkKeys<<<(num_items+thread_count-1)/thread_count, thread_count>>>(d_keys_out, num_items, num_bins, d_values_out);
    cudaMemcpy(&h_num_items_valid, d_num_items_valid, sizeof(int), cudaMemcpyDeviceToHost);
    //printf("h_num_items_valid = %d\n", h_num_items_valid);

    //printf("fixed_node_flag = %d, triangle_or_exact_flag = %d\n", (int)fixed_node_flag, (int)triangle_or_exact_flag);
    //printIntegerArray(d_keys_in, num_items, "d_keys_in");
    //printFloatArray(d_values_in, num_items, "d_values_in");
    //printIntegerArray(d_keys_out, h_num_items_valid, "d_keys_out");
    //printFloatArray(d_values_out, h_num_items_valid, "d_values_out");

#if 0
    // reduce by key 
    {
        int* d_num_runs_out;
        cudaMalloc((void**)& d_num_runs_out,  sizeof(int));
        auto reduce_op = cub::Sum();
        //cudaMemset(d_aggregates_out, 0, (num_bins+1)*sizeof(T));
        // Determine temporary device storage requirements
        d_temp_storage = NULL;
        temp_storage_bytes = 0;
        cub::DeviceReduce::ReduceByKey(d_temp_storage, temp_storage_bytes, d_keys_out, d_unique_out, d_values_out, d_aggregates_out, d_num_runs_out, reduce_op, h_num_items_valid);
        // Allocate temporary storage
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        // Run reduce-by-key
        cub::DeviceReduce::ReduceByKey(d_temp_storage, temp_storage_bytes, d_keys_out, d_unique_out, d_values_out, d_aggregates_out, d_num_runs_out, reduce_op, h_num_items_valid);

        cudaFree(d_temp_storage);

        int h_num_runs_out = 0; 
        cudaMemcpy(&h_num_runs_out, d_num_runs_out, sizeof(int), cudaMemcpyDeviceToHost);

        printIntegerArray(d_unique_out, h_num_runs_out, "d_unique_out"); 
        printFloatArray(d_aggregates_out, h_num_runs_out, "d_aggregates_out");
        //printScalar(d_num_runs_out, "d_num_runs_out");

        cudaFree(d_num_runs_out);

        fillKeyArray<<<(h_num_runs_out+thread_count-1)/thread_count, thread_count>>>(
                num_bins, 
                h_num_runs_out, 
                d_unique_out,
                d_aggregates_out,
                density_map_tensor
                );
    }
#endif

    // convert keys to offsets 
    // assume d_offsets_in is filled with zeros 
    {
        // step 1
        convertKeysToOffsetsStep1<<<(num_items+thread_count-1)/thread_count, thread_count>>>(d_keys_out, num_items, h_num_items_valid, d_offsets_in, num_bins);
        //printIntegerArray(d_offsets_in, num_bins+1, "d_offsets_in");
        // step 2 
        auto scan_op = cub::Max();
        // Determine temporary device storage requirements for exclusive prefix scan
        d_temp_storage = NULL;
        temp_storage_bytes = 0;
        cub::DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, d_offsets_in, d_offsets_out, scan_op, num_bins+1);
        // Allocate temporary storage for exclusive prefix scan
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        // Run exclusive prefix min-scan
        cub::DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, d_offsets_in, d_offsets_out, scan_op, num_bins+1);

        cudaFree(d_temp_storage);
        //printIntegerArray(d_offsets_out, num_bins+1, "d_offsets_out");
    }

    // segment reduce 
    {
        d_temp_storage = NULL;
        temp_storage_bytes = 0;
        cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, d_values_out, d_aggregates_out,
                num_bins, d_offsets_out, d_offsets_out + 1);
        // Allocate temporary storage
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        // Run sum-reduction
        cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, d_values_out, d_aggregates_out,
                num_bins, d_offsets_out, d_offsets_out + 1);

        cudaFree(d_temp_storage);

        //printFloatArray(d_aggregates_out, num_bins, "d_aggregates_out");
        //printAggregateSol(d_offsets_out, d_aggregates_out, num_bins, "d_aggregates_out non-zero ");

        fillDensityMap<<<(num_bins+thread_count-1)/thread_count, thread_count>>>(num_bins, d_aggregates_out, density_map_tensor);
    }

    //printFloatArray(density_map_tensor, num_bins, "density_map_tensor");

    return 0; 
}

#define REGISTER_KERNEL_LAUNCHER(T, V) \
    int instantiateComputeDensityMapReduceLauncher(\
            const T* x_tensor, const T* y_tensor, \
            const T* node_size_x_tensor, const T* node_size_y_tensor, \
            const T* bin_center_x_tensor, const T* bin_center_y_tensor, \
            const int num_nodes, \
            const int num_bins_x, const int num_bins_y, \
            const int num_impacted_bins_x, const int num_impacted_bins_y, \
            const T xl, const T yl, const T xh, const T yh, \
            const T bin_size_x, const T bin_size_y, \
            bool fixed_node_flag, \
            bool triangle_or_exact_flag, \
            T* density_map_tensor, \
            T* values_buf, \
            V* keys_buf, \
            int* offsets_buf \
            )\
    { \
        return computeDensityMapCudaReduceLauncher(\
                x_tensor, y_tensor, \
                node_size_x_tensor, node_size_y_tensor, \
                bin_center_x_tensor, bin_center_y_tensor, \
                num_nodes, \
                num_bins_x, num_bins_y, \
                num_impacted_bins_x, num_impacted_bins_y, \
                xl, yl, xh, yh, \
                bin_size_x, bin_size_y, \
                fixed_node_flag, \
                triangle_or_exact_flag, \
                density_map_tensor, \
                values_buf, \
                keys_buf, \
                offsets_buf \
                );\
    } \

REGISTER_KERNEL_LAUNCHER(float, int);
REGISTER_KERNEL_LAUNCHER(float, long);
REGISTER_KERNEL_LAUNCHER(double, int);
REGISTER_KERNEL_LAUNCHER(double, long);

DREAMPLACE_END_NAMESPACE
