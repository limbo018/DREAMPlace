/**
 * @file   dct_2N.cpp
 * @author Yibo Lin
 * @date   Nov 2018
 */
#include "dct.h"

DREAMPLACE_BEGIN_NAMESPACE

at::Tensor dct_2N_forward(at::Tensor x, at::Tensor expk, int num_threads) {
  CHECK_CPU(x);
  CHECK_CONTIGUOUS(x);
  CHECK_CPU(expk);
  CHECK_CONTIGUOUS(expk);

  auto N = x.size(-1);
  auto M = x.numel() / N;

  auto x_pad = at::zeros({M, 2 * N}, x.options());

  DREAMPLACE_DISPATCH_FLOATING_TYPES(x, "dct_2N_forward", [&] {
    computePad<scalar_t>(DREAMPLACE_TENSOR_DATA_PTR(x, scalar_t), M, N,
                         DREAMPLACE_TENSOR_DATA_PTR(x_pad, scalar_t),
                         num_threads);

    auto y = at::rfft(x_pad, 1, false, true);

    // re-use x_pad as output
    x_pad.resize_({M, N});
    computeMulExpk_2N(DREAMPLACE_TENSOR_DATA_PTR(y, scalar_t),
                      DREAMPLACE_TENSOR_DATA_PTR(expk, scalar_t), M, N,
                      DREAMPLACE_TENSOR_DATA_PTR(x_pad, scalar_t), num_threads);
    x_pad.mul_(1.0 / N);
  });

  return x_pad;
}

at::Tensor idct_2N_forward(at::Tensor x, at::Tensor expk, int num_threads) {
  CHECK_CPU(x);
  CHECK_CONTIGUOUS(x);
  CHECK_CPU(expk);
  CHECK_CONTIGUOUS(expk);

  auto N = x.size(-1);
  auto M = x.numel() / N;

  auto x_pad = at::zeros({M, 2 * N, 2}, x.options());

  DREAMPLACE_DISPATCH_FLOATING_TYPES(x, "idct_2N_forward", [&] {
    computeMulExpkAndPad_2N<scalar_t>(
        DREAMPLACE_TENSOR_DATA_PTR(x, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(expk, scalar_t), M, N,
        DREAMPLACE_TENSOR_DATA_PTR(x_pad, scalar_t), num_threads);

    // y is real now
    auto y = at::irfft(x_pad, 1, false, false, {2 * N});

    // reuse x_pad
    x_pad.resize_({M, N});
    computeTruncation(DREAMPLACE_TENSOR_DATA_PTR(y, scalar_t), M, N,
                      DREAMPLACE_TENSOR_DATA_PTR(x_pad, scalar_t), num_threads);
    // std::cout << "z\n" << z << "\n";

    // this is to match python implementation
    // normal way should be multiply by 0.25*N
    x_pad.mul_(N);
  });

  return x_pad;
}

at::Tensor dct2_2N_forward(at::Tensor x, at::Tensor expk0, at::Tensor expk1,
                           int num_threads) {
  CHECK_CPU(x);
  CHECK_CONTIGUOUS(x);
  CHECK_CPU(expk0);
  CHECK_CONTIGUOUS(expk0);
  CHECK_CPU(expk1);
  CHECK_CONTIGUOUS(expk1);

  // 1D DCT to columns

  // std::cout << "x\n" << x << "\n";
  auto N = x.size(-1);
  auto M = x.numel() / N;
  auto x_pad = at::zeros({M, 2 * N}, x.options());

  DREAMPLACE_DISPATCH_FLOATING_TYPES(x, "dct2_2N_forward", [&] {
    computePad<scalar_t>(DREAMPLACE_TENSOR_DATA_PTR(x, scalar_t), M, N,
                         DREAMPLACE_TENSOR_DATA_PTR(x_pad, scalar_t),
                         num_threads);

    auto y = at::rfft(x_pad, 1, false, true);

    // re-use x_pad as output
    x_pad.resize_({M, N});
    computeMulExpk_2N(DREAMPLACE_TENSOR_DATA_PTR(y, scalar_t),
                      DREAMPLACE_TENSOR_DATA_PTR(expk1, scalar_t), M, N,
                      DREAMPLACE_TENSOR_DATA_PTR(x_pad, scalar_t), num_threads);
    // x_pad.mul_(1.0/N);

    // 1D DCT to rows
    auto xt = x_pad.transpose(-2, -1).contiguous();
    // I do not want to allocate memory another time
    // must zero-out x_pad
    x_pad.resize_({N, 2 * M}).zero_();
    computePad<scalar_t>(DREAMPLACE_TENSOR_DATA_PTR(xt, scalar_t), N, M,
                         DREAMPLACE_TENSOR_DATA_PTR(x_pad, scalar_t),
                         num_threads);

    y = at::rfft(x_pad, 1, false, true);
    // y.mul_(1.0/M);

    // re-use x_reorder as output
    x_pad.resize_({N, M});
    computeMulExpk_2N(DREAMPLACE_TENSOR_DATA_PTR(y, scalar_t),
                      DREAMPLACE_TENSOR_DATA_PTR(expk0, scalar_t), N, M,
                      DREAMPLACE_TENSOR_DATA_PTR(x_pad, scalar_t), num_threads);

    x_pad.mul_(1.0 / (M * N));
    x_pad.transpose_(-2, -1);
  });

  return x_pad.contiguous();
}

at::Tensor idct2_2N_forward(at::Tensor x, at::Tensor expk0, at::Tensor expk1,
                            int num_threads) {
  CHECK_CPU(x);
  CHECK_CONTIGUOUS(x);
  CHECK_CPU(expk0);
  CHECK_CONTIGUOUS(expk0);
  CHECK_CPU(expk1);
  CHECK_CONTIGUOUS(expk1);

  auto N = x.size(-1);
  auto M = x.numel() / N;

  // 1D DCT to columns

  auto x_pad = at::zeros({M, 2 * N, 2}, x.options());

  DREAMPLACE_DISPATCH_FLOATING_TYPES(x, "idct2_2N_forward", [&] {
    computeMulExpkAndPad_2N<scalar_t>(
        DREAMPLACE_TENSOR_DATA_PTR(x, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(expk1, scalar_t), M, N,
        DREAMPLACE_TENSOR_DATA_PTR(x_pad, scalar_t), num_threads);

    // y is real now
    auto y = at::irfft(x_pad, 1, false, false, {2 * N});

    // reuse x_pad
    x_pad.resize_({M, N});
    computeTruncation(DREAMPLACE_TENSOR_DATA_PTR(y, scalar_t), M, N,
                      DREAMPLACE_TENSOR_DATA_PTR(x_pad, scalar_t), num_threads);

    // this is to match python implementation
    // normal way should be multiply by 0.25*N
    // x_pad.mul_(N);

    // 1D DCT to rows
    auto xt = x_pad.transpose(-2, -1).contiguous();
    x_pad.resize_({N, 2 * M, 2}).zero_();
    computeMulExpkAndPad_2N<scalar_t>(
        DREAMPLACE_TENSOR_DATA_PTR(xt, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(expk0, scalar_t), N, M,
        DREAMPLACE_TENSOR_DATA_PTR(x_pad, scalar_t), num_threads);

    // y is real now
    y = at::irfft(x_pad, 1, false, false, {2 * M});

    // reuse x_pad
    x_pad.resize_({N, M});
    computeTruncation(DREAMPLACE_TENSOR_DATA_PTR(y, scalar_t), N, M,
                      DREAMPLACE_TENSOR_DATA_PTR(x_pad, scalar_t), num_threads);

    // this is to match python implementation
    // normal way should be multiply by 0.25*0.25*M*N
    x_pad.mul_(M * N);
    x_pad.transpose_(-2, -1);
  });

  return x_pad.contiguous();
}

DREAMPLACE_END_NAMESPACE
