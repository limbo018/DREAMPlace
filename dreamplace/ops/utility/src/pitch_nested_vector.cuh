/**
 * @file   pitch_nested_vector.cuh
 * @author Yibo Lin
 * @date   Mar 2019
 * @brief  Flat nested array using pitched memory 
 */
#ifndef _DREAMPLACE_UTILITY_PITCHNESTEDVECTOR_CUH
#define _DREAMPLACE_UTILITY_PITCHNESTEDVECTOR_CUH

#include <cuda.h>
#include <cuda_runtime.h>
//#include <device_launch_parameters.h>
//#include <device_functions.h>

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
struct PitchNestedVector
{
    T* flat_element_map; ///< allocate on device, length of size1*size2
    unsigned int* dim2_sizes; ///< sizes of dimension 2 
    unsigned int size1; ///< length in dimension 1 
    unsigned int size2; ///< maximum length in dimension 2
    unsigned int num_elements; ///< total number of elements 

    /// @brief constructor 
    __host__ PitchNestedVector()
        : flat_element_map(nullptr)
        , dim2_sizes(nullptr)
        , size1(0)
        , size2(0)
    {
    }

    /// @brief initialization 
    __host__ void initialize(const std::vector<std::vector<T> >& nested_map)
    {
        // construct flat map on host 
        unsigned int max_num_elements = 0;
        num_elements = 0; 
        for (typename std::vector<std::vector<T> >::const_iterator it = nested_map.begin(); it != nested_map.end(); ++it)
        {
            max_num_elements = max(max_num_elements, (unsigned int)it->size());
            num_elements += it->size();
        }
        std::vector<T> host_flat_element_map (nested_map.size()*max_num_elements, std::numeric_limits<T>::max()); 
        std::vector<unsigned int> host_dim2_sizes (nested_map.size()); 

        for (unsigned int i = 0; i < nested_map.size(); ++i)
        {
            const std::vector<T>& vec = nested_map[i]; 
            std::copy(vec.begin(), vec.end(), host_flat_element_map.begin()+max_num_elements*i);
            host_dim2_sizes[i] = vec.size(); 
        }

        // copy to device 
        size1 = nested_map.size(); 
        size2 = max_num_elements; 
        allocateCopyCUDA(flat_element_map, host_flat_element_map.data(), host_flat_element_map.size()); 
        allocateCopyCUDA(dim2_sizes, host_dim2_sizes.data(), host_dim2_sizes.size());
    }

    __host__ void destroy()
    {
        if (flat_element_map)
        {
            destroyCUDA(flat_element_map);
            destroyCUDA(dim2_sizes); 
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
        return flat_element_map[i*size2+j]; 
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
        return flat_element_map[i*size2+j]; 
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
        return flat_element_map+i*size2; 
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
        return flat_element_map+i*size2; 
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
        return dim2_sizes[i]; 
    }

    /// @brief total number of elements 
    inline __device__ unsigned int size() const 
    {
        return num_elements; 
    }
};

DREAMPLACE_END_NAMESPACE

#endif
