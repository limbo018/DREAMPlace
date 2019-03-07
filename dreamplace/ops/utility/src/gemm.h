/**
 * @file   gemm.h
 * @author Yibo Lin
 * @date   Jun 2018
 */
#ifndef GPUPLACE_GEMM_H
#define GPUPLACE_GEMM_H

#include "cusparse.h"

// template specialization for cusparseScsrgemm and cusparseDcsrgemm
template <typename T>
cusparseStatus_t csrgemm(
        cusparseHandle_t handle,
        cusparseOperation_t transA, 
        cusparseOperation_t transB, 
        int m, 
        int n, 
        int k, 
        const cusparseMatDescr_t descrA,
        const int nnzA,      
        const T *csrSortedValA, 
        const int *csrSortedRowPtrA, 
        const int *csrSortedColIndA,
        const cusparseMatDescr_t descrB,
        const int nnzB,                                                    
        const T *csrSortedValB, 
        const int *csrSortedRowPtrB, 
        const int *csrSortedColIndB,
        const cusparseMatDescr_t descrC, 
        T *csrSortedValC, 
        const int *csrSortedRowPtrC, 
        int *csrSortedColIndC);

template <>
cusparseStatus_t csrgemm<float>(
        cusparseHandle_t handle,
        cusparseOperation_t transA, 
        cusparseOperation_t transB, 
        int m, 
        int n, 
        int k, 
        const cusparseMatDescr_t descrA,
        const int nnzA,      
        const float *csrSortedValA, 
        const int *csrSortedRowPtrA, 
        const int *csrSortedColIndA,
        const cusparseMatDescr_t descrB,
        const int nnzB,                                                    
        const float *csrSortedValB, 
        const int *csrSortedRowPtrB, 
        const int *csrSortedColIndB,
        const cusparseMatDescr_t descrC, 
        float *csrSortedValC, 
        const int *csrSortedRowPtrC, 
        int *csrSortedColIndC)
{
    return cusparseScsrgemm(
            handle,
            transA, 
            transB, 
            m, 
            n, 
            k, 
            descrA,
            nnzA,      
            csrSortedValA, 
            csrSortedRowPtrA, 
            csrSortedColIndA,
            descrB,
            nnzB,                                                    
            csrSortedValB, 
            csrSortedRowPtrB, 
            csrSortedColIndB,
            descrC, 
            csrSortedValC, 
            csrSortedRowPtrC, 
            csrSortedColIndC
            );
}

template <>
cusparseStatus_t csrgemm<double>(
        cusparseHandle_t handle,
        cusparseOperation_t transA, 
        cusparseOperation_t transB, 
        int m, 
        int n, 
        int k, 
        const cusparseMatDescr_t descrA,
        const int nnzA,      
        const double *csrSortedValA, 
        const int *csrSortedRowPtrA, 
        const int *csrSortedColIndA,
        const cusparseMatDescr_t descrB,
        const int nnzB,                                                    
        const double *csrSortedValB, 
        const int *csrSortedRowPtrB, 
        const int *csrSortedColIndB,
        const cusparseMatDescr_t descrC, 
        double *csrSortedValC, 
        const int *csrSortedRowPtrC, 
        int *csrSortedColIndC)
{
    return cusparseDcsrgemm(
            handle,
            transA, 
            transB, 
            m, 
            n, 
            k, 
            descrA,
            nnzA,      
            csrSortedValA, 
            csrSortedRowPtrA, 
            csrSortedColIndA,
            descrB,
            nnzB,                                                    
            csrSortedValB, 
            csrSortedRowPtrB, 
            csrSortedColIndB,
            descrC, 
            csrSortedValC, 
            csrSortedRowPtrC, 
            csrSortedColIndC
            );
}

#endif
