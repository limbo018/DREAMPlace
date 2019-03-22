/**
 * @file   csrmv.h
 * @author Yibo Lin
 * @date   Jul 2018
 */
#ifndef GPUPLACE_CSRMV_H
#define GPUPLACE_CSRMV_H

#include "cusparse.h"

// template specialization for cusparseScsrmv and cusparseDcsrmv
template <typename T>
cusparseStatus_t csrmv(
        cusparseHandle_t handle,
        cusparseOperation_t transA, 
        int m, 
        int n, 
        int nnz,
        const T *alpha,
        const cusparseMatDescr_t descrA, 
        const T *csrSortedValA, 
        const int *csrSortedRowPtrA, 
        const int *csrSortedColIndA, 
        const T *x, 
        const T *beta, 
        T *y
);

template <>
cusparseStatus_t csrmv<float>(
        cusparseHandle_t handle,
        cusparseOperation_t transA, 
        int m, 
        int n, 
        int nnz,
        const float *alpha,
        const cusparseMatDescr_t descrA, 
        const float *csrSortedValA, 
        const int *csrSortedRowPtrA, 
        const int *csrSortedColIndA, 
        const float *x, 
        const float *beta, 
        float *y
)
{
    return cusparseScsrmv(
            handle,
            transA, 
            m, 
            n, 
            nnz,
            alpha,
            descrA, 
            csrSortedValA, 
            csrSortedRowPtrA, 
            csrSortedColIndA, 
            x, 
            beta, 
            y
            );
}

template <>
cusparseStatus_t csrmv<double>(
        cusparseHandle_t handle,
        cusparseOperation_t transA, 
        int m, 
        int n, 
        int nnz,
        const double *alpha,
        const cusparseMatDescr_t descrA, 
        const double *csrSortedValA, 
        const int *csrSortedRowPtrA, 
        const int *csrSortedColIndA, 
        const double *x, 
        const double *beta, 
        double *y
)
{
    return cusparseDcsrmv(
            handle,
            transA, 
            m, 
            n, 
            nnz,
            alpha,
            descrA, 
            csrSortedValA, 
            csrSortedRowPtrA, 
            csrSortedColIndA, 
            x, 
            beta, 
            y
            );
}

#endif
