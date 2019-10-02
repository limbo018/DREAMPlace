/**
 * @file   dct2_fft2.cpp
 * @author Zixuan Jiang, Jiaqi Gu
 * @date   Aug 2019
 * @brief  All the transforms in this file are implemented based on 2D FFT.
 *      Each transfrom has three steps, 1) preprocess, 2) 2d fft or 2d ifft, 3) postprocess.
 */

#include "dct/src/dct2_fft2.h"

DREAMPLACE_BEGIN_NAMESPACE

void dct2_fft2_forward(
    at::Tensor x,
    at::Tensor expkM,
    at::Tensor expkN,
    at::Tensor out,
    at::Tensor buf,
    int num_threads)
{
    CHECK_CPU(x);
    CHECK_CPU(expkM);
    CHECK_CPU(expkN);
    CHECK_CPU(out);
    CHECK_CPU(buf);

    CHECK_CONTIGUOUS(x);
    CHECK_CONTIGUOUS(expkM);
    CHECK_CONTIGUOUS(expkN);
    CHECK_CONTIGUOUS(out);
    CHECK_CONTIGUOUS(buf);

    auto N = x.size(-1);
    auto M = x.numel() / N;

    DREAMPLACE_DISPATCH_FLOATING_TYPES(x.type(), "dct2_fft2_forward", [&] {
        dct2dPreprocessCpuLauncher<scalar_t>(
            x.data<scalar_t>(),
            out.data<scalar_t>(),
            M,
            N,
            num_threads);

        buf = at::rfft(out, 2, false, true);

        dct2dPostprocessCpuLauncher<scalar_t>(
            buf.data<scalar_t>(),
            out.data<scalar_t>(),
            M,
            N,
            expkM.data<scalar_t>(),
            expkN.data<scalar_t>(),
            num_threads);
    });
}

void idct2_fft2_forward(
    at::Tensor x,
    at::Tensor expkM,
    at::Tensor expkN,
    at::Tensor out,
    at::Tensor buf,
    int num_threads)
{
    CHECK_CPU(x);
    CHECK_CPU(expkM);
    CHECK_CPU(expkN);
    CHECK_CPU(out);
    CHECK_CPU(buf);

    CHECK_CONTIGUOUS(x);
    CHECK_CONTIGUOUS(expkM);
    CHECK_CONTIGUOUS(expkN);
    CHECK_CONTIGUOUS(out);
    CHECK_CONTIGUOUS(buf);

    auto N = x.size(-1);
    auto M = x.numel() / N;

    DREAMPLACE_DISPATCH_FLOATING_TYPES(x.type(), "idct2_fft2_forward", [&] {
        idct2_fft2PreprocessCpuLauncher<scalar_t>(
            x.data<scalar_t>(),
            buf.data<scalar_t>(),
            M,
            N,
            expkM.data<scalar_t>(),
            expkN.data<scalar_t>(),
            num_threads);

        auto y = at::irfft(buf, 2, false, true, {M, N});

        idct2_fft2PostprocessCpuLauncher<scalar_t>(
            y.data<scalar_t>(),
            out.data<scalar_t>(),
            M,
            N,
            num_threads);
    });
}

void idct_idxst_forward(
    at::Tensor x,
    at::Tensor expkM,
    at::Tensor expkN,
    at::Tensor out,
    at::Tensor buf,
    int num_threads)
{
    CHECK_CPU(x);
    CHECK_CPU(expkM);
    CHECK_CPU(expkN);
    CHECK_CPU(out);
    CHECK_CPU(buf);

    CHECK_CONTIGUOUS(x);
    CHECK_CONTIGUOUS(expkM);
    CHECK_CONTIGUOUS(expkN);
    CHECK_CONTIGUOUS(out);
    CHECK_CONTIGUOUS(buf);

    auto N = x.size(-1);
    auto M = x.numel() / N;

    DREAMPLACE_DISPATCH_FLOATING_TYPES(x.type(), "idct_idxst_forward", [&] {
        idct_idxstPreprocessCpuLauncher<scalar_t>(
            x.data<scalar_t>(),
            buf.data<scalar_t>(),
            M,
            N,
            expkM.data<scalar_t>(),
            expkN.data<scalar_t>(),
            num_threads);

        auto y = at::irfft(buf, 2, false, true, {M, N});

        idct_idxstPostprocessCpuLauncher<scalar_t>(
            y.data<scalar_t>(),
            out.data<scalar_t>(),
            M,
            N,
            num_threads);
    });
}

void idxst_idct_forward(
    at::Tensor x,
    at::Tensor expkM,
    at::Tensor expkN,
    at::Tensor out,
    at::Tensor buf,
    int num_threads)
{
    CHECK_CPU(x);
    CHECK_CPU(expkM);
    CHECK_CPU(expkN);
    CHECK_CPU(out);
    CHECK_CPU(buf);

    CHECK_CONTIGUOUS(x);
    CHECK_CONTIGUOUS(expkM);
    CHECK_CONTIGUOUS(expkN);
    CHECK_CONTIGUOUS(out);
    CHECK_CONTIGUOUS(buf);

    auto N = x.size(-1);
    auto M = x.numel() / N;

    DREAMPLACE_DISPATCH_FLOATING_TYPES(x.type(), "idxst_idct_forward", [&] {
        idxst_idctPreprocessCpuLauncher<scalar_t>(
            x.data<scalar_t>(),
            buf.data<scalar_t>(),
            M,
            N,
            expkM.data<scalar_t>(),
            expkN.data<scalar_t>(),
            num_threads);

        auto y = at::irfft(buf, 2, false, true, {M, N});

        idxst_idctPostprocessCpuLauncher<scalar_t>(
            y.data<scalar_t>(),
            out.data<scalar_t>(),
            M,
            N,
            num_threads);
    });
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dct2_fft2", &DREAMPLACE_NAMESPACE::dct2_fft2_forward, "DCT2 FFT2D (CPU)");
    m.def("idct2_fft2", &DREAMPLACE_NAMESPACE::idct2_fft2_forward, "IDCT2 FFT2D (CPU)");
    m.def("idct_idxst", &DREAMPLACE_NAMESPACE::idct_idxst_forward, "IDCT IDXST FFT2D (CPU)");
    m.def("idxst_idct", &DREAMPLACE_NAMESPACE::idxst_idct_forward, "IDXST IDCT FFT2D (CPU)");
}
