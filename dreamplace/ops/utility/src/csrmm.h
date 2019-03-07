/**
 * @file   csrmm.h
 * @author Yibo Lin
 * @date   Jul 2018
 */
#ifndef GPUPLACE_CSRMM_H
#define GPUPLACE_CSRMM_H

#include "cusparse.h"

// template specialization for cusparseScsrmm and cusparseDcsrmm
template <typename T>
cusparseStatus_t csrmm(
        cusparseHandle_t handle, 
        cusparseOperation_t transA, 
        int m, 
        int n, 
        int k, 
        int nnz, 
        const T *alpha, 
        const cusparseMatDescr_t descrA, 
        const T *csrValA, 
        const int *csrRowPtrA, 
        const int *csrColIndA,
        const T *B, 
        int ldb,
        const T *beta, 
        T *C, 
        int ldc
        );

template <>
cusparseStatus_t csrmm<float>(
        cusparseHandle_t handle, 
        cusparseOperation_t transA, 
        int m, 
        int n, 
        int k, 
        int nnz, 
        const float *alpha, 
        const cusparseMatDescr_t descrA, 
        const float *csrValA, 
        const int *csrRowPtrA, 
        const int *csrColIndA,
        const float *B, 
        int ldb,
        const float *beta, 
        float *C, 
        int ldc
        )
{
    return cusparseScsrmm(
            handle, 
            transA, 
            m, 
            n, 
            k, 
            nnz, 
            alpha, 
            descrA, 
            csrValA, 
            csrRowPtrA, 
            csrColIndA,
            B, 
            ldb,
            beta, 
            C, 
            ldc
            );
}

template <>
cusparseStatus_t csrmm<double>(
        cusparseHandle_t handle, 
        cusparseOperation_t transA, 
        int m, 
        int n, 
        int k, 
        int nnz, 
        const double *alpha, 
        const cusparseMatDescr_t descrA, 
        const double *csrValA, 
        const int *csrRowPtrA, 
        const int *csrColIndA,
        const double *B, 
        int ldb,
        const double *beta, 
        double *C, 
        int ldc
        )
{
    return cusparseDcsrmm(
            handle, 
            transA, 
            m, 
            n, 
            k, 
            nnz, 
            alpha, 
            descrA, 
            csrValA, 
            csrRowPtrA, 
            csrColIndA,
            B, 
            ldb,
            beta, 
            C, 
            ldc
            );
}

// template specialization for cusparseScsrmm2 and cusparseDcsrmm2
template <typename T>
cusparseStatus_t csrmm2(
        cusparseHandle_t handle, 
        cusparseOperation_t transA, 
        cusparseOperation_t transB, 
        int m, 
        int n, 
        int k, 
        int nnz, 
        const T *alpha, 
        const cusparseMatDescr_t descrA, 
        const T *csrValA, 
        const int *csrRowPtrA, 
        const int *csrColIndA,
        const T *B, 
        int ldb,
        const T *beta, 
        T *C, 
        int ldc
        );

template <>
cusparseStatus_t csrmm2<float>(
        cusparseHandle_t handle, 
        cusparseOperation_t transA, 
        cusparseOperation_t transB, 
        int m, 
        int n, 
        int k, 
        int nnz, 
        const float *alpha, 
        const cusparseMatDescr_t descrA, 
        const float *csrValA, 
        const int *csrRowPtrA, 
        const int *csrColIndA,
        const float *B, 
        int ldb,
        const float *beta, 
        float *C, 
        int ldc
        )
{
    return cusparseScsrmm2(
            handle, 
            transA, 
            transB, 
            m, 
            n, 
            k, 
            nnz, 
            alpha, 
            descrA, 
            csrValA, 
            csrRowPtrA, 
            csrColIndA,
            B, 
            ldb,
            beta, 
            C, 
            ldc
            );
}

template <>
cusparseStatus_t csrmm2<double>(
        cusparseHandle_t handle, 
        cusparseOperation_t transA, 
        cusparseOperation_t transB, 
        int m, 
        int n, 
        int k, 
        int nnz, 
        const double *alpha, 
        const cusparseMatDescr_t descrA, 
        const double *csrValA, 
        const int *csrRowPtrA, 
        const int *csrColIndA,
        const double *B, 
        int ldb,
        const double *beta, 
        double *C, 
        int ldc
        )
{
    return cusparseDcsrmm2(
            handle, 
            transA, 
            transB, 
            m, 
            n, 
            k, 
            nnz, 
            alpha, 
            descrA, 
            csrValA, 
            csrRowPtrA, 
            csrColIndA,
            B, 
            ldb,
            beta, 
            C, 
            ldc
            );
}

#endif
