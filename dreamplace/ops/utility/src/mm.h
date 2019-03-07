/**
 * @file   mm.h
 * @author Yibo Lin
 * @date   Jul 2018
 */
#ifndef GPUPLACE_MM_H
#define GPUPLACE_MM_H

#include "cublas_v2.h"

// template specialization for cublasSgemm and cublasDgemm
template <typename T>
cublasStatus_t mm(
        cublasHandle_t handle,
        cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, int k,
        const T           *alpha,
        const T           *A, int lda,
        const T           *B, int ldb,
        const T           *beta,
        T           *C, int ldc
        );

template <>
cublasStatus_t mm<float>(
        cublasHandle_t handle,
        cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, int k,
        const float           *alpha,
        const float           *A, int lda,
        const float           *B, int ldb,
        const float           *beta,
        float           *C, int ldc
        )
{
    return cublasSgemm(
            handle,
            transa, transb,
            m, n, k,
            alpha,
            A, lda,
            B, ldb,
            beta,
            C, ldc
            );
}

template <>
cublasStatus_t mm<double>(
        cublasHandle_t handle,
        cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, int k,
        const double           *alpha,
        const double           *A, int lda,
        const double           *B, int ldb,
        const double           *beta,
        double           *C, int ldc
        )
{
    return cublasDgemm(
            handle,
            transa, transb,
            m, n, k,
            alpha,
            A, lda,
            B, ldb,
            beta,
            C, ldc
            );
}

#endif
