/**
 * @file   dxt.cpp
 * @author Yibo Lin
 * @date   Sep 2018
 */
#include "dct.h"

DREAMPLACE_BEGIN_NAMESPACE

at::Tensor idxct_forward(at::Tensor x, at::Tensor expk, int num_threads) {
  auto N = x.size(-1);
  auto M = x.numel() / N;

  auto z = idct_forward(x, expk, num_threads);

  // std::cout << __func__ << " z\n" << z << "\n";

  DREAMPLACE_DISPATCH_FLOATING_TYPES(x, "idxct_forward", [&] {
    addX0AndScale<scalar_t>(DREAMPLACE_TENSOR_DATA_PTR(x, scalar_t), M, N,
                            DREAMPLACE_TENSOR_DATA_PTR(z, scalar_t),
                            num_threads);
  });

  return z;
}

at::Tensor idxst_forward(at::Tensor x, at::Tensor expk, int num_threads) {
  auto N = x.size(-1);
  auto M = x.numel() / N;

  // std::cout << "x\n" << x << "\n";
  // auto x_reorder = at::empty_like(x);
  auto x_reorder = at::empty({M, N}, x.options());
  auto y = at::empty_like(x);

  DREAMPLACE_DISPATCH_FLOATING_TYPES(x, "idxst_forward", [&] {
    computeFlipAndShift<scalar_t>(
        DREAMPLACE_TENSOR_DATA_PTR(x, scalar_t), M, N,
        DREAMPLACE_TENSOR_DATA_PTR(x_reorder, scalar_t), num_threads);

    y = idct_forward(x_reorder, expk, num_threads);
    y.mul_(0.5);
    // std::cout << "y\n" << y << "\n";

    negateOddEntries<scalar_t>(DREAMPLACE_TENSOR_DATA_PTR(y, scalar_t), M, N,
                               num_threads);
    // std::cout << "z\n" << y << "\n";
  });

  return y;
}

at::Tensor idcct2_forward(at::Tensor x, at::Tensor expk0, at::Tensor expk1,
                          int num_threads) {
  CHECK_CPU(x);
  CHECK_CONTIGUOUS(x);
  CHECK_CPU(expk0);
  CHECK_CONTIGUOUS(expk0);
  CHECK_CPU(expk1);
  CHECK_CONTIGUOUS(expk1);

  auto N = x.size(-1);
  auto M = x.numel() / N;

  // idxct for rows

  // std::cout << "x\n" << x << "\n";
  // vk = 0.5*W_{4N}^{k} (c[k] - c[N-k])
  // vk is hermitian symmetric, only fill in half
  auto v = at::empty({M * N + std::max(M, N)}, x.options())
               .resize_({M, N / 2 + 1, 2});

  DREAMPLACE_DISPATCH_FLOATING_TYPES(x, "idcct2_forward", [&] {
    computeVk<scalar_t>(DREAMPLACE_TENSOR_DATA_PTR(x, scalar_t),
                        DREAMPLACE_TENSOR_DATA_PTR(expk1, scalar_t), M, N,
                        DREAMPLACE_TENSOR_DATA_PTR(v, scalar_t), num_threads);

    // std::cout << "v\n" << v << "\n";

    // y is real now
    auto y = at::irfft(v, 1, false, true, {N});

    // std::cout << "y\n" << y << "\n";

    // std::cout << "expk\n" << expk << "\n";
    v.resize_({M, N});
    computeReorderReverse(DREAMPLACE_TENSOR_DATA_PTR(y, scalar_t), M, N,
                          DREAMPLACE_TENSOR_DATA_PTR(v, scalar_t), num_threads);
    // std::cout << "z\n" << z << "\n";

    addX0AndScaleN<scalar_t>(DREAMPLACE_TENSOR_DATA_PTR(x, scalar_t), M, N,
                             DREAMPLACE_TENSOR_DATA_PTR(v, scalar_t),
                             num_threads);

    // std::cout << "z\n" << z << "\n";
    // idxct for columns

    auto xt = v.transpose(-2, -1).contiguous();

    // vk = 0.5*W_{4N}^{k} (c[k] - c[N-k])
    v.resize_({N, M / 2 + 1, 2});
    computeVk<scalar_t>(DREAMPLACE_TENSOR_DATA_PTR(xt, scalar_t),
                        DREAMPLACE_TENSOR_DATA_PTR(expk0, scalar_t), N, M,
                        DREAMPLACE_TENSOR_DATA_PTR(v, scalar_t), num_threads);

    // std::cout << __func__ << " v\n" << v << "\n";

    y = at::irfft(v, 1, false, true, {M});

    // std::cout << __func__ << " y\n" << y << "\n";

    // std::cout << "expk\n" << expk << "\n";
    v.resize_({N, M});
    computeReorderReverse(DREAMPLACE_TENSOR_DATA_PTR(y, scalar_t), N, M,
                          DREAMPLACE_TENSOR_DATA_PTR(v, scalar_t), num_threads);
    // std::cout << __func__ << " z\n" << z << "\n";

    addX0AndScaleN<scalar_t>(DREAMPLACE_TENSOR_DATA_PTR(xt, scalar_t), N, M,
                             DREAMPLACE_TENSOR_DATA_PTR(v, scalar_t),
                             num_threads);

    v.transpose_(-2, -1);
  });

  return v.contiguous();
}

at::Tensor idsct2_forward(at::Tensor x, at::Tensor expk0, at::Tensor expk1,
                          int num_threads) {
  CHECK_CPU(x);
  CHECK_CONTIGUOUS(x);
  CHECK_CPU(expk0);
  CHECK_CONTIGUOUS(expk0);
  CHECK_CPU(expk1);
  CHECK_CONTIGUOUS(expk1);

  auto N = x.size(-1);
  auto M = x.numel() / N;

  // idxct for rows

  // std::cout << "x\n" << x << "\n";
  // vk = 0.5*W_{4N}^{k} (c[k] - c[N-k])
  // vk is hermitian symmetric, only fill in half
  auto v =
      at::empty({M * N + std::max(M, N)}, x.options()).resize_({M, N / 2 + 1, 2});
  auto z = at::empty({M, N}, x.options());

  DREAMPLACE_DISPATCH_FLOATING_TYPES(x, "idsct2_forward", [&] {
    computeVk<scalar_t>(DREAMPLACE_TENSOR_DATA_PTR(x, scalar_t),
                        DREAMPLACE_TENSOR_DATA_PTR(expk1, scalar_t), M, N,
                        DREAMPLACE_TENSOR_DATA_PTR(v, scalar_t), num_threads);

    // std::cout << "v\n" << v << "\n";

    // y is real now
    auto y = at::irfft(v, 1, false, true, {N});

    // std::cout << "y\n" << y << "\n";

    // std::cout << "expk\n" << expk << "\n";
    // auto z = at::empty_like(x);
    computeReorderReverse(DREAMPLACE_TENSOR_DATA_PTR(y, scalar_t), M, N,
                          DREAMPLACE_TENSOR_DATA_PTR(z, scalar_t), num_threads);
    // std::cout << "z\n" << z << "\n";

    addX0AndScaleN<scalar_t>(DREAMPLACE_TENSOR_DATA_PTR(x, scalar_t), M, N,
                             DREAMPLACE_TENSOR_DATA_PTR(z, scalar_t),
                             num_threads);
    // std::cout << __func__ << " z\n" << z << "\n";

    // idxst for columns

    auto xt = z.transpose(-2, -1).contiguous();
    // std::cout << "x\n" << x << "\n";
    z = z.view_as(xt);
    computeFlipAndShift<scalar_t>(DREAMPLACE_TENSOR_DATA_PTR(xt, scalar_t), N,
                                  M, DREAMPLACE_TENSOR_DATA_PTR(z, scalar_t),
                                  num_threads);

    // std::cout << "x\n" << x << "\n";
    // vk = 0.5*W_{4N}^{k} (c[k] - c[N-k])
    v.resize_({N, M / 2 + 1, 2});
    computeVk<scalar_t>(DREAMPLACE_TENSOR_DATA_PTR(z, scalar_t),
                        DREAMPLACE_TENSOR_DATA_PTR(expk0, scalar_t), N, M,
                        DREAMPLACE_TENSOR_DATA_PTR(v, scalar_t), num_threads);

    // std::cout << "v\n" << v << "\n";

    y = at::irfft(v, 1, false, true, {M});

    // std::cout << "y\n" << y << "\n";

    // std::cout << "expk\n" << expk << "\n";
    computeReorderReverse(DREAMPLACE_TENSOR_DATA_PTR(y, scalar_t), N, M,
                          DREAMPLACE_TENSOR_DATA_PTR(z, scalar_t), num_threads);
    // std::cout << "z\n" << z << "\n";
    // this is to match python implementation
    // normal way should be multiply by 0.25*N
    z.mul_(0.25 * M);

    negateOddEntries<scalar_t>(DREAMPLACE_TENSOR_DATA_PTR(z, scalar_t), N, M,
                               num_threads);
    // std::cout << "z\n" << y << "\n";

    z.transpose_(-2, -1);
  });

  return z.contiguous();
}

at::Tensor idcst2_forward(at::Tensor x, at::Tensor expk0, at::Tensor expk1,
                          int num_threads) {
  CHECK_CPU(x);
  CHECK_CONTIGUOUS(x);
  CHECK_CPU(expk0);
  CHECK_CONTIGUOUS(expk0);
  CHECK_CPU(expk1);
  CHECK_CONTIGUOUS(expk1);

  auto N = x.size(-1);
  auto M = x.numel() / N;

  // idxst for rows
  // std::cout << "x\n" << x << "\n";
  // auto z = at::empty_like(x);
  auto z = at::empty({M, N}, x.options());

  DREAMPLACE_DISPATCH_FLOATING_TYPES(x, "idcst2_forward", [&] {
    computeFlipAndShift<scalar_t>(DREAMPLACE_TENSOR_DATA_PTR(x, scalar_t), M, N,
                                  DREAMPLACE_TENSOR_DATA_PTR(z, scalar_t),
                                  num_threads);

    // std::cout << "x\n" << x << "\n";
    // vk = 0.5*W_{4N}^{k} (c[k] - c[N-k])
    auto v = at::empty({M * N + std::max(M, N)}, x.options())
                 .resize_({M, N / 2 + 1, 2});
    computeVk<scalar_t>(DREAMPLACE_TENSOR_DATA_PTR(z, scalar_t),
                        DREAMPLACE_TENSOR_DATA_PTR(expk1, scalar_t), M, N,
                        DREAMPLACE_TENSOR_DATA_PTR(v, scalar_t), num_threads);

    // std::cout << "v\n" << v << "\n";

    auto y = at::irfft(v, 1, false, true, {N});

    // std::cout << "y\n" << y << "\n";

    // std::cout << "expk\n" << expk << "\n";
    computeReorderReverse(DREAMPLACE_TENSOR_DATA_PTR(y, scalar_t), M, N,
                          DREAMPLACE_TENSOR_DATA_PTR(z, scalar_t), num_threads);
    // std::cout << "z\n" << z << "\n";
    // this is to match python implementation
    // normal way should be multiply by 0.25*N
    z.mul_(0.25 * N);

    negateOddEntries<scalar_t>(DREAMPLACE_TENSOR_DATA_PTR(z, scalar_t), M, N,
                               num_threads);
    // std::cout << __func__ << " z\n" << z << "\n";

    // idxct for columns

    auto xt = z.transpose(-2, -1).contiguous();

    // std::cout << "x\n" << x << "\n";
    // vk = 0.5*W_{4N}^{k} (c[k] - c[N-k])
    // vk = 0.5*W_{4N}^{k} (c[k] - c[N-k])
    v.resize_({N, M / 2 + 1, 2});
    computeVk<scalar_t>(DREAMPLACE_TENSOR_DATA_PTR(xt, scalar_t),
                        DREAMPLACE_TENSOR_DATA_PTR(expk0, scalar_t), N, M,
                        DREAMPLACE_TENSOR_DATA_PTR(v, scalar_t), num_threads);

    // std::cout << "v\n" << v << "\n";

    y = at::irfft(v, 1, false, true, {M});

    // std::cout << "y\n" << y << "\n";

    // std::cout << "expk\n" << expk << "\n";
    z = z.view_as(xt);
    computeReorderReverse(DREAMPLACE_TENSOR_DATA_PTR(y, scalar_t), N, M,
                          DREAMPLACE_TENSOR_DATA_PTR(z, scalar_t), num_threads);
    // std::cout << "z\n" << z << "\n";
    addX0AndScaleN<scalar_t>(DREAMPLACE_TENSOR_DATA_PTR(xt, scalar_t), N, M,
                             DREAMPLACE_TENSOR_DATA_PTR(z, scalar_t),
                             num_threads);

    z.transpose_(-2, -1);
  });

  return z.contiguous();
}

at::Tensor idxst_idct_forward(at::Tensor x, at::Tensor expk0, at::Tensor expk1,
                              int num_threads) {
  CHECK_CPU(x);
  CHECK_CONTIGUOUS(x);
  CHECK_CPU(expk0);
  CHECK_CONTIGUOUS(expk0);
  CHECK_CPU(expk1);
  CHECK_CONTIGUOUS(expk1);

  auto N = x.size(-1);
  auto M = x.numel() / N;

  // idxct for rows

  // std::cout << "x\n" << x << "\n";
  // vk = 0.5*W_{4N}^{k} (c[k] - c[N-k])
  // vk is hermitian symmetric, only fill in half
  auto v =
      at::empty({M * N + std::max(M, N)}, x.options()).resize_({M, N / 2 + 1, 2});
  auto z = at::empty({M, N}, x.options());

  DREAMPLACE_DISPATCH_FLOATING_TYPES(x, "idsct2_forward", [&] {
    computeVk<scalar_t>(DREAMPLACE_TENSOR_DATA_PTR(x, scalar_t),
                        DREAMPLACE_TENSOR_DATA_PTR(expk1, scalar_t), M, N,
                        DREAMPLACE_TENSOR_DATA_PTR(v, scalar_t), num_threads);

    // std::cout << "v\n" << v << "\n";

    // y is real now
    auto y = at::irfft(v, 1, false, true, {N});

    // std::cout << "y\n" << y << "\n";

    // std::cout << "expk\n" << expk << "\n";
    // auto z = at::empty_like(x);
    computeReorderReverse(DREAMPLACE_TENSOR_DATA_PTR(y, scalar_t), M, N,
                          DREAMPLACE_TENSOR_DATA_PTR(z, scalar_t), num_threads);
    // std::cout << "z\n" << z << "\n";

    // idxst for columns

    auto xt = z.transpose(-2, -1).contiguous();
    // std::cout << "x\n" << x << "\n";
    z = z.view_as(xt);
    computeFlipAndShift<scalar_t>(DREAMPLACE_TENSOR_DATA_PTR(xt, scalar_t), N,
                                  M, DREAMPLACE_TENSOR_DATA_PTR(z, scalar_t),
                                  num_threads);

    // std::cout << "x\n" << x << "\n";
    // vk = 0.5*W_{4N}^{k} (c[k] - c[N-k])
    v.resize_({N, M / 2 + 1, 2});
    computeVk<scalar_t>(DREAMPLACE_TENSOR_DATA_PTR(z, scalar_t),
                        DREAMPLACE_TENSOR_DATA_PTR(expk0, scalar_t), N, M,
                        DREAMPLACE_TENSOR_DATA_PTR(v, scalar_t), num_threads);

    // std::cout << "v\n" << v << "\n";

    y = at::irfft(v, 1, false, true, {M});

    // std::cout << "y\n" << y << "\n";

    // std::cout << "expk\n" << expk << "\n";
    computeReorderReverse(DREAMPLACE_TENSOR_DATA_PTR(y, scalar_t), N, M,
                          DREAMPLACE_TENSOR_DATA_PTR(z, scalar_t), num_threads);
    // std::cout << "z\n" << z << "\n";
    // normalized to match dct2_fft2 implementation
    z.mul_(0.25 * M * N);

    negateOddEntries<scalar_t>(DREAMPLACE_TENSOR_DATA_PTR(z, scalar_t), N, M,
                               num_threads);
    // std::cout << "z\n" << y << "\n";

    z.transpose_(-2, -1);
  });

  return z.contiguous();
}

at::Tensor idct_idxst_forward(at::Tensor x, at::Tensor expk0, at::Tensor expk1,
                              int num_threads) {
  CHECK_CPU(x);
  CHECK_CONTIGUOUS(x);
  CHECK_CPU(expk0);
  CHECK_CONTIGUOUS(expk0);
  CHECK_CPU(expk1);
  CHECK_CONTIGUOUS(expk1);

  auto N = x.size(-1);
  auto M = x.numel() / N;

  // idxst for rows
  // std::cout << "x\n" << x << "\n";
  // auto z = at::empty_like(x);
  auto z = at::empty({M, N}, x.options());

  DREAMPLACE_DISPATCH_FLOATING_TYPES(x, "idcst2_forward", [&] {
    computeFlipAndShift<scalar_t>(DREAMPLACE_TENSOR_DATA_PTR(x, scalar_t), M, N,
                                  DREAMPLACE_TENSOR_DATA_PTR(z, scalar_t),
                                  num_threads);

    // std::cout << "x\n" << x << "\n";
    // vk = 0.5*W_{4N}^{k} (c[k] - c[N-k])
    auto v = at::empty({M * N + std::max(M, N)}, x.options())
                 .resize_({M, N / 2 + 1, 2});
    computeVk<scalar_t>(DREAMPLACE_TENSOR_DATA_PTR(z, scalar_t),
                        DREAMPLACE_TENSOR_DATA_PTR(expk1, scalar_t), M, N,
                        DREAMPLACE_TENSOR_DATA_PTR(v, scalar_t), num_threads);

    // std::cout << "v\n" << v << "\n";

    auto y = at::irfft(v, 1, false, true, {N});

    // std::cout << "y\n" << y << "\n";

    // std::cout << "expk\n" << expk << "\n";
    computeReorderReverse(DREAMPLACE_TENSOR_DATA_PTR(y, scalar_t), M, N,
                          DREAMPLACE_TENSOR_DATA_PTR(z, scalar_t), num_threads);
    // std::cout << "z\n" << z << "\n";
    // normalized to match dct2_fft2 implementation
    z.mul_(0.25 * N * M);

    negateOddEntries<scalar_t>(DREAMPLACE_TENSOR_DATA_PTR(z, scalar_t), M, N,
                               num_threads);
    // std::cout << __func__ << " z\n" << z << "\n";

    // idxct for columns

    auto xt = z.transpose(-2, -1).contiguous();

    // std::cout << "x\n" << x << "\n";
    // vk = 0.5*W_{4N}^{k} (c[k] - c[N-k])
    // vk = 0.5*W_{4N}^{k} (c[k] - c[N-k])
    v.resize_({N, M / 2 + 1, 2});
    computeVk<scalar_t>(DREAMPLACE_TENSOR_DATA_PTR(xt, scalar_t),
                        DREAMPLACE_TENSOR_DATA_PTR(expk0, scalar_t), N, M,
                        DREAMPLACE_TENSOR_DATA_PTR(v, scalar_t), num_threads);

    // std::cout << "v\n" << v << "\n";

    y = at::irfft(v, 1, false, true, {M});

    // std::cout << "y\n" << y << "\n";

    // std::cout << "expk\n" << expk << "\n";
    z = z.view_as(xt);
    computeReorderReverse(DREAMPLACE_TENSOR_DATA_PTR(y, scalar_t), N, M,
                          DREAMPLACE_TENSOR_DATA_PTR(z, scalar_t), num_threads);
    // std::cout << "z\n" << z << "\n";

    z.transpose_(-2, -1);
  });

  return z.contiguous();
}

DREAMPLACE_END_NAMESPACE
