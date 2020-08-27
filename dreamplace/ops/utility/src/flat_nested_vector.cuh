/**
 * @file   FlatNestedVector.cuh
 * @author Yibo Lin
 * @date   Mar 2019
 */
#ifndef _DREAMPLACE_UTILITY_FLATNESTEDVECTOR_CUH
#define _DREAMPLACE_UTILITY_FLATNESTEDVECTOR_CUH

#include <cuda.h>
#include <cuda_runtime.h>
//#include <device_launch_parameters.h>
//#include <device_functions.h>

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
struct FlatNestedVector
{
    T* flat_element_map; ///< allocate on device, length of flat_dim1_start_map[size1] 
    unsigned int* flat_dim1_start_map; ///< allocate on device, length of size1+1
    unsigned int size1; ///< length in dimension 1 

    /// @brief constructor 
    __host__ FlatNestedVector()
        : flat_element_map(nullptr)
        , flat_dim1_start_map(nullptr)
        , size1(0)
    {
    }

    /// @brief initialization 
    __host__ void initialize(const std::vector<std::vector<T> >& nested_map)
    {
        // construct flat map on host 
        unsigned int num_elements = 0; 
        for (typename std::vector<std::vector<T> >::const_iterator it = nested_map.begin(); it != nested_map.end(); ++it)
        {
            num_elements += it->size(); 
        }
        std::vector<T> host_flat_element_map (num_elements); 
        std::vector<unsigned int> host_flat_dim1_start_map (nested_map.size()+1); 

        num_elements = 0; 
        for (unsigned int i = 0; i < nested_map.size(); ++i)
        {
            const std::vector<T>& vec = nested_map[i]; 
            std::copy(vec.begin(), vec.end(), host_flat_element_map.begin()+num_elements);
            host_flat_dim1_start_map[i] = num_elements; 
            num_elements += vec.size();
        }
        host_flat_dim1_start_map[nested_map.size()] = num_elements; 

        // copy to device 
        size1 = nested_map.size(); 
        allocateCopyCUDA(flat_element_map, host_flat_element_map.data(), host_flat_element_map.size()); 
        allocateCopyCUDA(flat_dim1_start_map, host_flat_dim1_start_map.data(), host_flat_dim1_start_map.size());
    }

    __host__ void destroy()
    {
        if (flat_element_map)
        {
            destroyCUDA(flat_element_map);
            destroyCUDA(flat_dim1_start_map); 
        }
    }

    /// @brief access element 
    inline __device__ const T& operator()(unsigned int i, unsigned int j) const 
    {
#ifdef DEBUG 
        if (!(i < size1 && j < size(i)))
        {
            printf("%u < %u && %u < %u\n", i, size1, j, size(i));
        }
#endif
        assert(i < size1 && j < size(i));
        return flat_element_map[flat_dim1_start_map[i]+j]; 
    }

    /// @brief access element 
    inline __device__ T& operator()(unsigned int i, unsigned int j)
    {
#ifdef DEBUG 
        if (!(i < size1 && j < size(i)))
        {
            printf("%u < %u && %u < %u\n", i, size1, j, size(i));
        }
#endif
        assert(i < size1 && j < size(i));
        return flat_element_map[flat_dim1_start_map[i]+j]; 
    }

    /// @brief access each row 
    inline __device__ const T* operator()(unsigned int i) const 
    {
#ifdef DEBUG 
        if (!(i < size1))
        {
            printf("%u < %u\n", i, size1);
        }
#endif
        assert(i < size1); 
        return flat_element_map+flat_dim1_start_map[i]; 
    }

    /// @brief access each row 
    inline __device__ T* operator()(unsigned int i) 
    {
#ifdef DEBUG 
        if (!(i < size1))
        {
            printf("%u < %u\n", i, size1);
        }
#endif
        assert(i < size1); 
        return flat_element_map+flat_dim1_start_map[i]; 
    }

    /// @brief length of each row 
    inline __device__ unsigned int size(unsigned int i) const 
    {
#ifdef DEBUG 
        if (!(i < size1))
        {
            printf("%u < %u\n", i, size1);
        }
#endif
        assert(i < size1);
        return flat_dim1_start_map[i+1]-flat_dim1_start_map[i]; 
    }

    /// @brief total number of elements 
    inline __device__ unsigned int size() const 
    {
        return flat_dim1_start_map[size1]; 
    }
};

DREAMPLACE_END_NAMESPACE

#endif
