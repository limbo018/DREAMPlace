/**
 * @file   geam.h
 * @author Yibo Lin
 * @date   Jul 2018
 */
#ifndef GPUPLACE_GEAM_H
#define GPUPLACE_GEAM_H

#include "cublas_v2.h"

// template specialization for cublasSgeam and cublasDgeam
template <typename T>
cublasStatus_t geam(
        cublasHandle_t handle,
        cublasOperation_t transa, cublasOperation_t transb,
        int m, int n,
        const T           *alpha,
        const T           *A, int lda,
        const T           *beta,
        const T           *B, int ldb,
        T           *C, int ldc
        );

template <>
cublasStatus_t geam<float>(
        cublasHandle_t handle,
        cublasOperation_t transa, cublasOperation_t transb,
        int m, int n,
        const float           *alpha,
        const float           *A, int lda,
        const float           *beta,
        const float           *B, int ldb,
        float           *C, int ldc
        )
{
    return cublasSgeam(
        handle,
        transa, transb,
        m, n,
        alpha,
        A, lda,
        beta,
        B, ldb,
        C, ldc
            );
}

template <>
cublasStatus_t geam<double>(
        cublasHandle_t handle,
        cublasOperation_t transa, cublasOperation_t transb,
        int m, int n,
        const double           *alpha,
        const double           *A, int lda,
        const double           *beta,
        const double           *B, int ldb,
        double           *C, int ldc
        )
{
    return cublasDgeam(
        handle,
        transa, transb,
        m, n,
        alpha,
        A, lda,
        beta,
        B, ldb,
        C, ldc
            );
}

#endif
