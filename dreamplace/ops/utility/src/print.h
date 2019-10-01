/**
 * @file   print.h
 * @author Yibo Lin
 * @date   Jun 2018
 */
#ifndef GPUPLACE_PRINT_H
#define GPUPLACE_PRINT_H

#include <iostream>

template <typename T>
struct UFloatTraits; 

template <>
struct UFloatTraits<float>
{
    typedef float float_type; 
    typedef unsigned int integer_type; 
};

template <>
struct UFloatTraits<double>
{
    typedef double float_type; 
    typedef unsigned long integer_type; 
};

template <typename T>
union UFloat
{
    typedef typename UFloatTraits<T>::float_type float_type;
    typedef typename UFloatTraits<T>::integer_type integer_type;
    float_type f; 
    integer_type u; 
};

/// @brief Print floating point array as integer for exact comparison 
template <typename T>
void printFloatArray(const T* x, const int n, const char* str)
{
    printf("%s[%d] \n", str, n); 
    T* host_x = (T*)malloc(n*sizeof(T));
    if (host_x == NULL)
    {
        printf("failed to allocate memory on CPU\n");
        return;
    }
    cudaMemcpy(host_x, x, n*sizeof(T), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; ++i)
    {
        UFloat<T> uf; 
        uf.f = host_x[i]; 
        std::cout << str << "[" << i << "]: " << uf.u << "\n"; 
    }
    std::cout << "\n";

    free(host_x);
}

/// @brief Print integer array for exact comparison
template <typename T>
void printIntegerArray(const T* x, const int n, const char* str)
{
    printf("%s[%d] \n", str, n); 
    T* host_x = (T*)malloc(n*sizeof(T));
    if (host_x == NULL)
    {
        printf("failed to allocate memory on CPU\n");
        return;
    }
    cudaMemcpy(host_x, x, n*sizeof(T), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; ++i)
    {
        std::cout << str << "[" << i << "]: " << host_x[i] << "\n";
    }
    std::cout << "\n";

    free(host_x);
}

template <typename T>
void printArray(const T* x, const int n, const char* str)
{
    printf("%s[%d] = ", str, n); 
    T* host_x = (T*)malloc(n*sizeof(T));
    if (host_x == NULL)
    {
        printf("failed to allocate memory on CPU\n");
        return;
    }
    cudaMemcpy(host_x, x, n*sizeof(T), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; ++i)
    {
        printf("%g ", double(host_x[i]));
    }
    printf("\n");

    free(host_x);
}

template <typename T>
void printScalar(const T* x, const char* str)
{
    printf("%s = ", str); 
    T* host_x = (T*)malloc(sizeof(T));
    if (host_x == NULL)
    {
        printf("failed to allocate memory on CPU\n");
        return;
    }
    cudaMemcpy(host_x, x, sizeof(T), cudaMemcpyDeviceToHost);
    printf("%g\n", double(*host_x));

    free(host_x);
}

template <typename T>
void print2DArray(const T* x, const int m, const int n, const char* str)
{
    printf("%s[%dx%d] = \n", str, m, n); 
    T* host_x = (T*)malloc(m*n*sizeof(T));
    if (host_x == NULL)
    {
        printf("failed to allocate memory on CPU\n");
        return;
    }
    cudaMemcpy(host_x, x, m*n*sizeof(T), cudaMemcpyDeviceToHost);
    for (int i = 0; i < m*n; ++i)
    {
        if (i && (i%n) == 0)
        {
            printf("\n");
        }
        printf("%g ", double(host_x[i]));
    }
    printf("\n");

    free(host_x);
}

#endif
