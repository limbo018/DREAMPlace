#ifndef DREAMPLACE_DCT2_FFT2_CUDA_H
#define DREAMPLACE_DCT2_FFT2_CUDA_H

#include "utility/src/torch.h"
#include "utility/src/Msg.h"

DREAMPLACE_BEGIN_NAMESPACE

#define CHECK_GPU(x) AT_ASSERTM(x.is_cuda(), #x "must be a tensor on GPU")
#define CHECK_FLAT(x) AT_ASSERTM(x.is_cuda() && x.ndimension() == 1, #x "must be a flat tensor on GPU")
#define CHECK_EVEN(x) AT_ASSERTM((x.numel() & 1) == 0, #x "must have even number of elements")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x "must be contiguous")

// dct2_fft2
void dct2_fft2_forward(
    at::Tensor x,
    at::Tensor expkM,
    at::Tensor expkN,
    at::Tensor out,
    at::Tensor buf);

template <typename T>
void dct2dPreprocessCudaLauncher(
    const T *x,
    T *y,
    const int M,
    const int N);

template <typename T>
void dct2dPostprocessCudaLauncher(
    const T *x,
    T *y,
    const int M,
    const int N,
    const T *__restrict__ expkM,
    const T *__restrict__ expkN);

// idct2_fft2
void idct2_fft2_forward(
    at::Tensor x,
    at::Tensor expkM,
    at::Tensor expkN,
    at::Tensor out,
    at::Tensor buf);

template <typename T>
void idct2_fft2PreprocessCudaLauncher(
    const T *x,
    T *y,
    const int M,
    const int N,
    const T *__restrict__ expkM,
    const T *__restrict__ expkN);

template <typename T>
void idct2_fft2PostprocessCudaLauncher(
    const T *x,
    T *y,
    const int M,
    const int N);

// idct_idxst
void idct_idxst_forward(
    at::Tensor x,
    at::Tensor expkM,
    at::Tensor expkN,
    at::Tensor out,
    at::Tensor buf);

template <typename T>
void idct_idxstPreprocessCudaLauncher(
    const T *x,
    T *y,
    const int M,
    const int N,
    const T *__restrict__ expkM,
    const T *__restrict__ expkN);

template <typename T>
void idct_idxstPostprocessCudaLauncher(
    const T *x,
    T *y,
    const int M,
    const int N);

// idxst_idct
void idxst_idct_forward(
    at::Tensor x,
    at::Tensor expkM,
    at::Tensor expkN,
    at::Tensor out,
    at::Tensor buf);

template <typename T>
void idxst_idctPreprocessCudaLauncher(
    const T *x,
    T *y,
    const int M,
    const int N,
    const T *__restrict__ expkM,
    const T *__restrict__ expkN);

template <typename T>
void idxst_idctPostprocessCudaLauncher(
    const T *x,
    T *y,
    const int M,
    const int N);

DREAMPLACE_END_NAMESPACE

#endif
