#include "dct2_fft2_cuda.h"

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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("dct2_fft2", &dct2_fft2_forward, "DCT2 FFT2D");
    m.def("idct2_fft2", &idct2_fft2_forward, "IDCT2 FFT2D");
    m.def("idct_idxst", &idct_idxst_forward, "IDCT IDXST FFT2D");
    m.def("idxst_idct", &idxst_idct_forward, "IDXST IDCT FFT2D");
}
