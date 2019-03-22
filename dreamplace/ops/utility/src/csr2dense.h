/**
 * @file   csr2dense.h
 * @author Yibo Lin
 * @date   Jul 2018
 */
#ifndef GPUPLACE_CSR2DENSE_H
#define GPUPLACE_CSR2DENSE_H

#include "cusparse.h"

// template specialization for cusparseScsr2dense and cusparseDcsr2dense
template <typename T>
cusparseStatus_t csr2dense(
        cusparseHandle_t handle,
        int m, 
        int n, 
        const cusparseMatDescr_t descrA,  
        const T *csrValA, 
        const int *csrRowPtrA, 
        const int *csrColIndA,
        T *A, 
        int lda
        );

template <>
cusparseStatus_t csr2dense<float>(
        cusparseHandle_t handle,
        int m, 
        int n, 
        const cusparseMatDescr_t descrA,  
        const float *csrValA, 
        const int *csrRowPtrA, 
        const int *csrColIndA,
        float *A, 
        int lda
        )
{
    return cusparseScsr2dense(
            handle,
            m, 
            n, 
            descrA,  
            csrValA, 
            csrRowPtrA, 
            csrColIndA,
            A, 
            lda
            );
}

template <>
cusparseStatus_t csr2dense<double>(
        cusparseHandle_t handle,
        int m, 
        int n, 
        const cusparseMatDescr_t descrA,  
        const double *csrValA, 
        const int *csrRowPtrA, 
        const int *csrColIndA,
        double *A, 
        int lda
        )
{
    return cusparseDcsr2dense(
            handle,
            m, 
            n, 
            descrA,  
            csrValA, 
            csrRowPtrA, 
            csrColIndA,
            A, 
            lda
            );
}

#endif
