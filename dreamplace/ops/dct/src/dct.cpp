/**
 * @file   dct.cpp
 * @author Yibo Lin
 * @date   Jun 2018
 */
#include "dct.h"

DREAMPLACE_BEGIN_NAMESPACE

at::Tensor dct_forward(at::Tensor x, at::Tensor expk, int num_threads) {
  CHECK_CPU(x);
  CHECK_CONTIGUOUS(x);
  CHECK_CPU(expk);
  CHECK_CONTIGUOUS(expk);

  auto N = x.size(-1);
  auto M = x.numel() / N;

  // std::cout << "x\n" << x << "\n";
  // auto x_reorder = at::empty_like(x);
  auto x_reorder = at::empty({M, N}, x.options());

  DREAMPLACE_DISPATCH_FLOATING_TYPES(x, "dct_forward", [&] {
    computeReorder<scalar_t>(DREAMPLACE_TENSOR_DATA_PTR(x, scalar_t), M, N,
                             DREAMPLACE_TENSOR_DATA_PTR(x_reorder, scalar_t),
                             num_threads);

    // std::cout << "x_reorder\n" << x_reorder << "\n";

    auto y = at::rfft(x_reorder, 1, false, true);
    // std::cout << "y\n" << y << "\n";

    // re-use x_reorder as output
    // std::cout << "x_reorder\n" << x_reorder << "\n";
    // std::cout << "expk\n" << expk << "\n";
    computeMulExpk(DREAMPLACE_TENSOR_DATA_PTR(y, scalar_t),
                   DREAMPLACE_TENSOR_DATA_PTR(expk, scalar_t), M, N,
                   DREAMPLACE_TENSOR_DATA_PTR(x_reorder, scalar_t),
                   num_threads);
    // std::cout << "z\n" << x_reorder << "\n";
    x_reorder.mul_(1.0 / N);
  });

  return x_reorder;
}

at::Tensor idct_forward(at::Tensor x, at::Tensor expk, int num_threads) {
  CHECK_CPU(x);
  CHECK_CONTIGUOUS(x);
  CHECK_CPU(expk);
  CHECK_CONTIGUOUS(expk);

  auto N = x.size(-1);
  auto M = x.numel() / N;

  // std::cout << "x\n" << x << "\n";
  // vk = 0.5*W_{4N}^{k} (c[k] - c[N-k])
  // vk is hermitian symmetric, only fill in half
  auto v = at::empty({M * N + std::max(M, N)}, x.options())
               .resize_({M, N / 2 + 1, 2});

  DREAMPLACE_DISPATCH_FLOATING_TYPES(x, "idct_forward", [&] {
    computeVk<scalar_t>(DREAMPLACE_TENSOR_DATA_PTR(x, scalar_t),
                        DREAMPLACE_TENSOR_DATA_PTR(expk, scalar_t), M, N,
                        DREAMPLACE_TENSOR_DATA_PTR(v, scalar_t), num_threads);

    // std::cout << __func__ << " v\n" << v << "\n";

    // y is real now
    auto y = at::irfft(v, 1, false, true, {N});

    // std::cout << __func__ << " y\n" << y << "\n";

    // std::cout << "expk\n" << expk << "\n";
    // reuse v
    v.resize_({M, N});
    computeReorderReverse(DREAMPLACE_TENSOR_DATA_PTR(y, scalar_t), M, N,
                          DREAMPLACE_TENSOR_DATA_PTR(v, scalar_t), num_threads);
    // std::cout << "z\n" << z << "\n";

    // this is to match python implementation
    // normal way should be multiply by 0.25*N
    v.mul_(0.5 * N);
  });

  return v;
}

at::Tensor dct2_forward(at::Tensor x, at::Tensor expk0, at::Tensor expk1,
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
  auto x_reorder = at::empty({M, N}, x.options());

  DREAMPLACE_DISPATCH_FLOATING_TYPES(x, "dct2_forward", [&] {
    computeReorder<scalar_t>(DREAMPLACE_TENSOR_DATA_PTR(x, scalar_t), M, N,
                             DREAMPLACE_TENSOR_DATA_PTR(x_reorder, scalar_t),
                             num_threads);

    // std::cout << "x_reorder\n" << x_reorder << "\n";

    auto y = at::rfft(x_reorder, 1, false, true);
    // y.mul_(1.0/N);
    // std::cout << "y\n" << y << "\n";

    // re-use x_reorder as output
    // std::cout << "expk1\n" << expk1 << "\n";
    computeMulExpk(DREAMPLACE_TENSOR_DATA_PTR(y, scalar_t),
                   DREAMPLACE_TENSOR_DATA_PTR(expk1, scalar_t), M, N,
                   DREAMPLACE_TENSOR_DATA_PTR(x_reorder, scalar_t),
                   num_threads);
    // std::cout << "z\n" << x_reorder << "\n";

    // 1D DCT to rows
    auto xt = x_reorder.transpose(-2, -1).contiguous();
    // std::cout << "xt\n" << xt << "\n";
    // I do not want to allocate memory another time
    // x_reorder = at::empty_like(xt);
    x_reorder = x_reorder.view_as(xt);
    computeReorder<scalar_t>(DREAMPLACE_TENSOR_DATA_PTR(xt, scalar_t), N, M,
                             DREAMPLACE_TENSOR_DATA_PTR(x_reorder, scalar_t),
                             num_threads);

    // std::cout << "x_reorder\n" << x_reorder << "\n";

    y = at::rfft(x_reorder, 1, false, true);
    // y.mul_(1.0/M);
    // std::cout << "y\n" << y << "\n";

    // re-use x_reorder as output
    // std::cout << "expk0\n" << expk0 << "\n";
    computeMulExpk(DREAMPLACE_TENSOR_DATA_PTR(y, scalar_t),
                   DREAMPLACE_TENSOR_DATA_PTR(expk0, scalar_t), N, M,
                   DREAMPLACE_TENSOR_DATA_PTR(x_reorder, scalar_t),
                   num_threads);

    x_reorder.mul_(1.0 / (M * N));
    x_reorder.transpose_(-2, -1);
  });

  return x_reorder.contiguous();
}

at::Tensor idct2_forward(at::Tensor x, at::Tensor expk0, at::Tensor expk1,
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

  // std::cout << "x\n" << x << "\n";
  // vk = 0.5*W_{4N}^{k} (c[k] - c[N-k])
  // vk is hermitian symmetric, only fill in half
  auto v = at::empty({M * N + std::max(M, N)}, x.options())
               .resize_({M, N / 2 + 1, 2});

  DREAMPLACE_DISPATCH_FLOATING_TYPES(x, "idct2_forward", [&] {
    computeVk<scalar_t>(DREAMPLACE_TENSOR_DATA_PTR(x, scalar_t),
                        DREAMPLACE_TENSOR_DATA_PTR(expk1, scalar_t), M, N,
                        DREAMPLACE_TENSOR_DATA_PTR(v, scalar_t), num_threads);

    // std::cout << "expk1\n" << expk1 << "\n";

    auto y = at::irfft(v, 1, false, true, {N});
    // y.mul_(0.25*N);

    // std::cout << "y\n" << y << "\n";

    // std::cout << "expk\n" << expk << "\n";
    // auto z = at::empty_like(x);
    // auto z = at::empty(x.options(), {M, N});
    /// reuse v
    v.resize_({M, N});
    computeReorderReverse(DREAMPLACE_TENSOR_DATA_PTR(y, scalar_t), M, N,
                          DREAMPLACE_TENSOR_DATA_PTR(v, scalar_t), num_threads);
    // std::cout << "z\n" << z << "\n";

    // 1D DCT to rows
    auto xt = v.transpose(-2, -1).contiguous();
    // std::cout << "xt\n" << xt << "\n";
    // vk = 0.5*W_{4N}^{k} (c[k] - c[N-k])
    v.resize_({N, M / 2 + 1, 2});
    computeVk<scalar_t>(DREAMPLACE_TENSOR_DATA_PTR(xt, scalar_t),
                        DREAMPLACE_TENSOR_DATA_PTR(expk0, scalar_t), N, M,
                        DREAMPLACE_TENSOR_DATA_PTR(v, scalar_t), num_threads);

    // std::cout << "expk0\n" << expk0 << "\n";
    // std::cout << "v\n" << v << "\n";

    y = at::irfft(v, 1, false, true, {M});
    // y.mul_(0.25*M);

    // std::cout << "y\n" << y << "\n";

    // I do not want to allocate memory another time
    // reuse v
    v.resize_({N, M});
    computeReorderReverse(DREAMPLACE_TENSOR_DATA_PTR(y, scalar_t), N, M,
                          DREAMPLACE_TENSOR_DATA_PTR(v, scalar_t), num_threads);
    // std::cout << "z\n" << z << "\n";

    // this is to match python implementation
    // normal way should be multiply by 0.25*0.25*M*N
    v.mul_(0.25 * M * N);
    v.transpose_(-2, -1);
  });

  return v.contiguous();
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dct", &DREAMPLACE_NAMESPACE::dct_forward, "DCT forward");
  m.def("idct", &DREAMPLACE_NAMESPACE::idct_forward, "IDCT forward");
  m.def("idxct", &DREAMPLACE_NAMESPACE::idxct_forward, "IDXCT forward");
  m.def("dct2", &DREAMPLACE_NAMESPACE::dct2_forward, "DCT2 forward");

  m.def("dst", &DREAMPLACE_NAMESPACE::dst_forward, "DST forward");
  m.def("idst", &DREAMPLACE_NAMESPACE::idst_forward, "IDST forward");

  m.def("idct2", &DREAMPLACE_NAMESPACE::idct2_forward, "IDCT2 forward");
  m.def("idxst", &DREAMPLACE_NAMESPACE::idxst_forward, "IDXST forward");

  // use idxst and idxct as kernels
  m.def("idcct2", &DREAMPLACE_NAMESPACE::idcct2_forward, "IDCCT2 forward");
  m.def("idcst2", &DREAMPLACE_NAMESPACE::idcst2_forward, "IDCST2 forward");
  m.def("idsct2", &DREAMPLACE_NAMESPACE::idsct2_forward, "IDSCT2 forward");

  // use idxst and idct as kernels
  m.def("idxst_idct", &DREAMPLACE_NAMESPACE::idxst_idct_forward,
        "IDXST(IDCT(x)) forward");
  m.def("idct_idxst", &DREAMPLACE_NAMESPACE::idct_idxst_forward,
        "IDCT(IDXST(x)) forward");

  m.def("dct_2N", &DREAMPLACE_NAMESPACE::dct_2N_forward, "DCT forward");
  m.def("idct_2N", &DREAMPLACE_NAMESPACE::idct_2N_forward, "IDCT forward");
  m.def("dct2_2N", &DREAMPLACE_NAMESPACE::dct2_2N_forward, "DCT2 forward");
  m.def("idct2_2N", &DREAMPLACE_NAMESPACE::idct2_2N_forward, "IDCT2 forward");
}
