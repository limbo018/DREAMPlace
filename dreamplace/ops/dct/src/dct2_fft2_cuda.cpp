/**
 * @file   dct2_fft2_cuda.cpp
 * @author Zixuan Jiang, Jiaqi Gu
 * @date   Apr 2019
 * @brief  All the transforms in this file are implemented based on 2D FFT.
 *      Each transfrom has three steps, 1) preprocess, 2) 2d fft or 2d ifft, 3)
 * postprocess.
 */

#include "dct2_fft2_cuda.h"

DREAMPLACE_BEGIN_NAMESPACE

void dct2_fft2_forward(at::Tensor x, at::Tensor expkM, at::Tensor expkN,
                       at::Tensor out, at::Tensor buf) {
  CHECK_CUDA(x);
  CHECK_CUDA(expkM);
  CHECK_CUDA(expkN);
  CHECK_CUDA(out);
  CHECK_CUDA(buf);

  CHECK_CONTIGUOUS(x);
  CHECK_CONTIGUOUS(expkM);
  CHECK_CONTIGUOUS(expkN);
  CHECK_CONTIGUOUS(out);
  CHECK_CONTIGUOUS(buf);

  auto N = x.size(-1);
  auto M = x.numel() / N;

  DREAMPLACE_DISPATCH_FLOATING_TYPES(x, "dct2_fft2_forward", [&] {
    dct2dPreprocessCudaLauncher<scalar_t>(
        DREAMPLACE_TENSOR_DATA_PTR(x, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(out, scalar_t), M, N);

    buf = at::rfft(out, 2, false, true);

    dct2dPostprocessCudaLauncher<scalar_t>(
        DREAMPLACE_TENSOR_DATA_PTR(buf, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(out, scalar_t), M, N,
        DREAMPLACE_TENSOR_DATA_PTR(expkM, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(expkN, scalar_t));
  });
}

void idct2_fft2_forward(at::Tensor x, at::Tensor expkM, at::Tensor expkN,
                        at::Tensor out, at::Tensor buf) {
  CHECK_CUDA(x);
  CHECK_CUDA(expkM);
  CHECK_CUDA(expkN);
  CHECK_CUDA(out);
  CHECK_CUDA(buf);

  CHECK_CONTIGUOUS(x);
  CHECK_CONTIGUOUS(expkM);
  CHECK_CONTIGUOUS(expkN);
  CHECK_CONTIGUOUS(out);
  CHECK_CONTIGUOUS(buf);

  auto N = x.size(-1);
  auto M = x.numel() / N;

  DREAMPLACE_DISPATCH_FLOATING_TYPES(x, "idct2_fft2_forward", [&] {
    idct2_fft2PreprocessCudaLauncher<scalar_t>(
        DREAMPLACE_TENSOR_DATA_PTR(x, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(buf, scalar_t), M, N,
        DREAMPLACE_TENSOR_DATA_PTR(expkM, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(expkN, scalar_t));

    auto y = at::irfft(buf, 2, false, true, {{M, N}});

    idct2_fft2PostprocessCudaLauncher<scalar_t>(
        DREAMPLACE_TENSOR_DATA_PTR(y, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(out, scalar_t), M, N);
  });
}

void idct_idxst_forward(at::Tensor x, at::Tensor expkM, at::Tensor expkN,
                        at::Tensor out, at::Tensor buf) {
  CHECK_CUDA(x);
  CHECK_CUDA(expkM);
  CHECK_CUDA(expkN);
  CHECK_CUDA(out);
  CHECK_CUDA(buf);

  CHECK_CONTIGUOUS(x);
  CHECK_CONTIGUOUS(expkM);
  CHECK_CONTIGUOUS(expkN);
  CHECK_CONTIGUOUS(out);
  CHECK_CONTIGUOUS(buf);

  auto N = x.size(-1);
  auto M = x.numel() / N;

  DREAMPLACE_DISPATCH_FLOATING_TYPES(x, "idct_idxst_forward", [&] {
    idct_idxstPreprocessCudaLauncher<scalar_t>(
        DREAMPLACE_TENSOR_DATA_PTR(x, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(buf, scalar_t), M, N,
        DREAMPLACE_TENSOR_DATA_PTR(expkM, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(expkN, scalar_t));

    auto y = at::irfft(buf, 2, false, true, {{M, N}});

    idct_idxstPostprocessCudaLauncher<scalar_t>(
        DREAMPLACE_TENSOR_DATA_PTR(y, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(out, scalar_t), M, N);
  });
}

void idxst_idct_forward(at::Tensor x, at::Tensor expkM, at::Tensor expkN,
                        at::Tensor out, at::Tensor buf) {
  CHECK_CUDA(x);
  CHECK_CUDA(expkM);
  CHECK_CUDA(expkN);
  CHECK_CUDA(out);
  CHECK_CUDA(buf);

  CHECK_CONTIGUOUS(x);
  CHECK_CONTIGUOUS(expkM);
  CHECK_CONTIGUOUS(expkN);
  CHECK_CONTIGUOUS(out);
  CHECK_CONTIGUOUS(buf);

  auto N = x.size(-1);
  auto M = x.numel() / N;

  DREAMPLACE_DISPATCH_FLOATING_TYPES(x, "idxst_idct_forward", [&] {
    idxst_idctPreprocessCudaLauncher<scalar_t>(
        DREAMPLACE_TENSOR_DATA_PTR(x, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(buf, scalar_t), M, N,
        DREAMPLACE_TENSOR_DATA_PTR(expkM, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(expkN, scalar_t));

    auto y = at::irfft(buf, 2, false, true, {{M, N}});

    idxst_idctPostprocessCudaLauncher<scalar_t>(
        DREAMPLACE_TENSOR_DATA_PTR(y, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(out, scalar_t), M, N);
  });
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dct2_fft2", &DREAMPLACE_NAMESPACE::dct2_fft2_forward,
        "DCT2 FFT2D (CUDA)");
  m.def("idct2_fft2", &DREAMPLACE_NAMESPACE::idct2_fft2_forward,
        "IDCT2 FFT2D (CUDA)");
  m.def("idct_idxst", &DREAMPLACE_NAMESPACE::idct_idxst_forward,
        "IDCT IDXST FFT2D (CUDA)");
  m.def("idxst_idct", &DREAMPLACE_NAMESPACE::idxst_idct_forward,
        "IDXST IDCT FFT2D (CUDA)");
}
