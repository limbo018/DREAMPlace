// All the transforms in this file are implemented based on 2D FFT.
// Each transfrom has three steps, 1) preprocess, 2) 2d fft or 2d ifft, 3) postprocess.

#include "dct2_fft2_cuda.h"

DREAMPLACE_BEGIN_NAMESPACE

void dct2_fft2_forward(
    at::Tensor x,
    at::Tensor expkM,
    at::Tensor expkN,
    at::Tensor out,
    at::Tensor buf)
{
    CHECK_GPU(x);
    CHECK_GPU(expkM);
    CHECK_GPU(expkN);
    CHECK_GPU(out);
    CHECK_GPU(buf);

    CHECK_CONTIGUOUS(x);
    CHECK_CONTIGUOUS(expkM);
    CHECK_CONTIGUOUS(expkN);
    CHECK_CONTIGUOUS(out);
    CHECK_CONTIGUOUS(buf);

    auto N = x.size(-1);
    auto M = x.numel() / N;

    AT_DISPATCH_FLOATING_TYPES(x.type(), "dct2_fft2_forward", [&] {
        dct2dPreprocessCudaLauncher<scalar_t>(
            x.data<scalar_t>(),
            out.data<scalar_t>(),
            M,
            N);

        buf = at::rfft(out, 2, false, true);

        dct2dPostprocessCudaLauncher<scalar_t>(
            buf.data<scalar_t>(),
            out.data<scalar_t>(),
            M,
            N,
            expkM.data<scalar_t>(),
            expkN.data<scalar_t>());
    });
}

void idct2_fft2_forward(
    at::Tensor x,
    at::Tensor expkM,
    at::Tensor expkN,
    at::Tensor out,
    at::Tensor buf)
{
    CHECK_GPU(x);
    CHECK_GPU(expkM);
    CHECK_GPU(expkN);
    CHECK_GPU(out);
    CHECK_GPU(buf);

    CHECK_CONTIGUOUS(x);
    CHECK_CONTIGUOUS(expkM);
    CHECK_CONTIGUOUS(expkN);
    CHECK_CONTIGUOUS(out);
    CHECK_CONTIGUOUS(buf);

    auto N = x.size(-1);
    auto M = x.numel() / N;

    AT_DISPATCH_FLOATING_TYPES(x.type(), "idct2_fft2_forward", [&] {
        idct2_fft2PreprocessCudaLauncher<scalar_t>(
            x.data<scalar_t>(),
            buf.data<scalar_t>(),
            M,
            N,
            expkM.data<scalar_t>(),
            expkN.data<scalar_t>());

        auto y = at::irfft(buf, 2, false, true, {M, N});

        idct2_fft2PostprocessCudaLauncher<scalar_t>(
            y.data<scalar_t>(),
            out.data<scalar_t>(),
            M,
            N);
    });
}

void idct_idxst_forward(
    at::Tensor x,
    at::Tensor expkM,
    at::Tensor expkN,
    at::Tensor out,
    at::Tensor buf)
{
    CHECK_GPU(x);
    CHECK_GPU(expkM);
    CHECK_GPU(expkN);
    CHECK_GPU(out);
    CHECK_GPU(buf);

    CHECK_CONTIGUOUS(x);
    CHECK_CONTIGUOUS(expkM);
    CHECK_CONTIGUOUS(expkN);
    CHECK_CONTIGUOUS(out);
    CHECK_CONTIGUOUS(buf);

    auto N = x.size(-1);
    auto M = x.numel() / N;

    AT_DISPATCH_FLOATING_TYPES(x.type(), "idct_idxst_forward", [&] {
        idct_idxstPreprocessCudaLauncher<scalar_t>(
            x.data<scalar_t>(),
            buf.data<scalar_t>(),
            M,
            N,
            expkM.data<scalar_t>(),
            expkN.data<scalar_t>());

        auto y = at::irfft(buf, 2, false, true, {M, N});

        idct_idxstPostprocessCudaLauncher<scalar_t>(
            y.data<scalar_t>(),
            out.data<scalar_t>(),
            M,
            N);
    });
}

void idxst_idct_forward(
    at::Tensor x,
    at::Tensor expkM,
    at::Tensor expkN,
    at::Tensor out,
    at::Tensor buf)
{
    CHECK_GPU(x);
    CHECK_GPU(expkM);
    CHECK_GPU(expkN);
    CHECK_GPU(out);
    CHECK_GPU(buf);

    CHECK_CONTIGUOUS(x);
    CHECK_CONTIGUOUS(expkM);
    CHECK_CONTIGUOUS(expkN);
    CHECK_CONTIGUOUS(out);
    CHECK_CONTIGUOUS(buf);

    auto N = x.size(-1);
    auto M = x.numel() / N;

    AT_DISPATCH_FLOATING_TYPES(x.type(), "idxst_idct_forward", [&] {
        idxst_idctPreprocessCudaLauncher<scalar_t>(
            x.data<scalar_t>(),
            buf.data<scalar_t>(),
            M,
            N,
            expkM.data<scalar_t>(),
            expkN.data<scalar_t>());

        auto y = at::irfft(buf, 2, false, true, {M, N});

        idxst_idctPostprocessCudaLauncher<scalar_t>(
            y.data<scalar_t>(),
            out.data<scalar_t>(),
            M,
            N);
    });
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dct2_fft2", &DREAMPLACE_NAMESPACE::dct2_fft2_forward, "DCT2 FFT2D (CUDA)");
    m.def("idct2_fft2", &DREAMPLACE_NAMESPACE::idct2_fft2_forward, "IDCT2 FFT2D (CUDA)");
    m.def("idct_idxst", &DREAMPLACE_NAMESPACE::idct_idxst_forward, "IDCT IDXST FFT2D (CUDA)");
    m.def("idxst_idct", &DREAMPLACE_NAMESPACE::idxst_idct_forward, "IDXST IDCT FFT2D (CUDA)");
}
