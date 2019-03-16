/**
 * @file   dct_lee_cuda.cpp
 * @author Yibo Lin
 * @date   Oct 2018
 */

#include "dct_cuda.h"
#include "dct_lee_cuda.h"

DREAMPLACE_BEGIN_NAMESPACE

at::Tensor dct_lee_precompute_dct_cos(int N)
{
    typedef double T; 

    at::TensorOptions options (at::Backend::CUDA); 
    options.dtype(at::kDouble);
    auto out = at::empty(N, options);

    lee::precompute_dct_cos(out.data<T>(), N);

    return out; 
}

at::Tensor dct_lee_precompute_idct_cos(int N)
{
    typedef double T; 

    at::TensorOptions options (at::Backend::CUDA); 
    options.dtype(at::kDouble);
    auto out = at::empty(N, options);

    lee::precompute_idct_cos(out.data<T>(), N);

    return out; 
}

inline void dct_lee_forward(
        at::Tensor x,
        at::Tensor cos, 
        at::Tensor buf,
        at::Tensor out 
        ) 
{
    CHECK_GPU(x);
    CHECK_CONTIGUOUS(x);
    CHECK_GPU(cos);
    CHECK_CONTIGUOUS(cos);

    auto N = x.size(-1);
    auto M = x.numel()/N; 

    AT_DISPATCH_FLOATING_TYPES(x.type(), "dct_lee_forward", [&] {
            lee::dct(
                    x.data<scalar_t>(), 
                    out.data<scalar_t>(), 
                    buf.data<scalar_t>(), 
                    cos.data<scalar_t>(), 
                    M, 
                    N
                    );
            });

    out.mul_(2.0/N); 
}

inline void idct_lee_forward(
        at::Tensor x,
        at::Tensor cos, 
        at::Tensor buf, 
        at::Tensor out
        ) 
{
    CHECK_GPU(x);
    CHECK_CONTIGUOUS(x);
    CHECK_GPU(cos);
    CHECK_CONTIGUOUS(cos);

    auto N = x.size(-1);
    auto M = x.numel()/N; 

    AT_DISPATCH_FLOATING_TYPES(x.type(), "idct_lee_forward", [&] {
            lee::idct(
                    x.data<scalar_t>(), 
                    out.data<scalar_t>(), 
                    buf.data<scalar_t>(), 
                    cos.data<scalar_t>(), 
                    M, 
                    N
                    );
            });

    out.mul_(2); 
}

void dst_lee_forward(
        at::Tensor x,
        at::Tensor expk, 
        at::Tensor buf, 
        at::Tensor out
        ) 
{
    auto N = x.size(-1);
    auto M = x.numel()/N; 

    AT_DISPATCH_FLOATING_TYPES(x.type(), "dst_lee_forward", [&] {
            //std::cout << "x\n" << x << "\n";
            buf.copy_(x);
            negateOddEntriesCudaLauncher<scalar_t>(
                    buf.data<scalar_t>(), 
                    M, 
                    N
                    );

            dct_lee_forward(buf, expk, buf, out);
            //std::cout << "y\n" << y << "\n";

            computeFlipCudaLauncher<scalar_t>(
                    out.data<scalar_t>(), 
                    M, 
                    N, 
                    buf.data<scalar_t>()
                    );
            });

    out.copy_(buf);
}

void idst_lee_forward(
        at::Tensor x,
        at::Tensor expk, 
        at::Tensor buf, 
        at::Tensor out
        ) 
{
    auto N = x.size(-1);
    auto M = x.numel()/N; 

    AT_DISPATCH_FLOATING_TYPES(x.type(), "idst_lee_forward", [&] {
            //std::cout << "x\n" << x << "\n";
            computeFlipCudaLauncher<scalar_t>(
                    x.data<scalar_t>(), 
                    M, 
                    N, 
                    buf.data<scalar_t>()
                    );

            idct_lee_forward(buf, expk, buf, out);
            //std::cout << "y\n" << y << "\n";

            negateOddEntriesCudaLauncher<scalar_t>(
                    out.data<scalar_t>(), 
                    M, 
                    N
                    );
            //std::cout << "z\n" << y << "\n";
            });
}

void dct2_lee_forward(
        at::Tensor x,
        at::Tensor cos0, 
        at::Tensor cos1, 
        at::Tensor buf, 
        at::Tensor out
        ) 
{
    //CHECK_GPU(x);
    //CHECK_CONTIGUOUS(x);
    //CHECK_GPU(cos0);
    //CHECK_CONTIGUOUS(cos0);
    //CHECK_GPU(cos1);
    //CHECK_CONTIGUOUS(cos1);

    // 1D DCT to columns

    //std::cout << "x\n" << x << "\n";
    auto N = x.size(-1);
    auto M = x.numel()/N; 

    out.resize_({M, N});
    buf.resize_({M, N});

    dct_lee_forward(x, cos1, out, buf); 

    // 1D DCT to rows 
    out.resize_({N, M}); 
    out.copy_(buf.transpose(-2, -1)); 
    buf.resize_({N, M}); 

    dct_lee_forward(out, cos0, out, buf); 

    out.resize_({M, N}); 
    out.copy_(buf.transpose(-2, -1)); 
}

void idct2_lee_forward(
        at::Tensor x,
        at::Tensor cos0, 
        at::Tensor cos1, 
        at::Tensor buf, 
        at::Tensor out
        )
{
    CHECK_GPU(x);
    CHECK_CONTIGUOUS(x);
    CHECK_GPU(cos0);
    CHECK_CONTIGUOUS(cos0);
    CHECK_GPU(cos1);
    CHECK_CONTIGUOUS(cos1);

    // 1D DCT to columns

    //std::cout << "x\n" << x << "\n";
    auto N = x.size(-1);
    auto M = x.numel()/N; 

    out.resize_({M, N});
    buf.resize_({M, N});

    idct_lee_forward(x, cos1, out, buf); 

    // 1D DCT to rows 
    out.resize_({N, M}); 
    out.copy_(buf.transpose(-2, -1)); 
    buf.resize_({N, M}); 

    idct_lee_forward(out, cos0, out, buf); 

    out.resize_({M, N}); 
    out.copy_(buf.transpose(-2, -1)); 
}

void idxct_lee_forward(
        at::Tensor x,
        at::Tensor cos, 
        at::Tensor buf, 
        at::Tensor out
        ) 
{
    auto N = x.size(-1);
    auto M = x.numel()/N; 

    AT_DISPATCH_FLOATING_TYPES(x.type(), "idxct_lee_forward", [&] {
            idct_lee_forward(x, cos, buf, out);

            //std::cout << __func__ << " z\n" << z << "\n";

            addX0AndScaleCudaLauncher<scalar_t>(
                    x.data<scalar_t>(), 
                    M, 
                    N, 
                    out.data<scalar_t>()
                    );
    });
}

void idxst_lee_forward(
        at::Tensor x,
        at::Tensor cos, 
        at::Tensor buf, 
        at::Tensor out
        )
{
    auto N = x.size(-1);
    auto M = x.numel()/N; 

    AT_DISPATCH_FLOATING_TYPES(x.type(), "idxst_lee_forward", [&] {
            //std::cout << "x\n" << x << "\n";
            computeFlipAndShiftCudaLauncher<scalar_t>(
                    x.data<scalar_t>(), 
                    M, 
                    N, 
                    buf.data<scalar_t>()
                    );

            idct_lee_forward(buf, cos, buf, out);
            out.mul_(0.5);
            //std::cout << "y\n" << y << "\n";

            negateOddEntriesCudaLauncher<scalar_t>(
                    out.data<scalar_t>(), 
                    M, 
                    N
                    );
            //std::cout << "z\n" << y << "\n";
    });
}

void idcct2_lee_forward(
        at::Tensor x,
        at::Tensor cos0, 
        at::Tensor cos1, 
        at::Tensor buf0, 
        at::Tensor buf1, 
        at::Tensor out
        ) 
{
    CHECK_GPU(x);
    CHECK_CONTIGUOUS(x);
    CHECK_GPU(cos0);
    CHECK_CONTIGUOUS(cos0);
    CHECK_GPU(cos1);
    CHECK_CONTIGUOUS(cos1);

    auto N = x.size(-1);
    auto M = x.numel()/N; 

    // two buffers are required to keep make sure no additional allocation of memory 

    // idxct for rows 

    idxct_lee_forward(x, cos1, buf0, out);

    // idxct for columns
    
    buf0.resize_({N, M}); 
    buf0.copy_(out.transpose(-2, -1));
    buf1.resize_({N, M}); 
    out.resize_({N, M}); 

    idxct_lee_forward(buf0, cos0, out, buf1); 

    out.resize_({M, N}); 
    out.copy_(buf1.transpose(-2, -1));
}

void idcst2_lee_forward(
        at::Tensor x,
        at::Tensor cos0, 
        at::Tensor cos1, 
        at::Tensor buf0, 
        at::Tensor buf1, 
        at::Tensor out
        ) 
{
    CHECK_GPU(x);
    CHECK_CONTIGUOUS(x);
    CHECK_GPU(cos0);
    CHECK_CONTIGUOUS(cos0);
    CHECK_GPU(cos1);
    CHECK_CONTIGUOUS(cos1);

    auto N = x.size(-1);
    auto M = x.numel()/N; 

    // two buffers are required to keep make sure no additional allocation of memory 

    // idxst for rows 

    idxst_lee_forward(x, cos1, buf0, out);

    // idxct for columns
    
    buf0.resize_({N, M}); 
    buf0.copy_(out.transpose(-2, -1));
    buf1.resize_({N, M}); 
    out.resize_({N, M}); 

    idxct_lee_forward(buf0, cos0, out, buf1); 

    out.resize_({M, N}); 
    out.copy_(buf1.transpose(-2, -1));
}

void idsct2_lee_forward(
        at::Tensor x,
        at::Tensor cos0, 
        at::Tensor cos1, 
        at::Tensor buf0, 
        at::Tensor buf1, 
        at::Tensor out
        ) 
{
    CHECK_GPU(x);
    CHECK_CONTIGUOUS(x);
    CHECK_GPU(cos0);
    CHECK_CONTIGUOUS(cos0);
    CHECK_GPU(cos1);
    CHECK_CONTIGUOUS(cos1);

    auto N = x.size(-1);
    auto M = x.numel()/N; 

    // two buffers are required to keep make sure no additional allocation of memory 

    // idxst for rows 

    idxct_lee_forward(x, cos1, buf0, out);

    // idxct for columns
    
    buf0.resize_({N, M}); 
    buf0.copy_(out.transpose(-2, -1));
    buf1.resize_({N, M}); 
    out.resize_({N, M}); 

    idxst_lee_forward(buf0, cos0, out, buf1); 

    out.resize_({M, N}); 
    out.copy_(buf1.transpose(-2, -1));
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("precompute_dct_cos", &DREAMPLACE_NAMESPACE::dct_lee_precompute_dct_cos, "Precompute DCT cosine");
  m.def("precompute_idct_cos", &DREAMPLACE_NAMESPACE::dct_lee_precompute_idct_cos, "Precompute IDCT cosine");
  m.def("dct", &DREAMPLACE_NAMESPACE::dct_lee_forward, "DCT forward");
  m.def("idct", &DREAMPLACE_NAMESPACE::idct_lee_forward, "IDCT forward");
  m.def("idxct", &DREAMPLACE_NAMESPACE::idxct_lee_forward, "IDXCT forward");
  m.def("idxst", &DREAMPLACE_NAMESPACE::idxst_lee_forward, "IDXST forward");

  m.def("dst", &DREAMPLACE_NAMESPACE::dst_lee_forward, "DST forward");
  m.def("idst", &DREAMPLACE_NAMESPACE::idst_lee_forward, "IDST forward");

  m.def("dct2", &DREAMPLACE_NAMESPACE::dct2_lee_forward, "DCT2 forward");
  m.def("idct2", &DREAMPLACE_NAMESPACE::idct2_lee_forward, "IDCT2 forward");
  m.def("idcct2", &DREAMPLACE_NAMESPACE::idcct2_lee_forward, "IDCCT2 forward");
  m.def("idcst2", &DREAMPLACE_NAMESPACE::idcst2_lee_forward, "IDCST2 forward");
  m.def("idsct2", &DREAMPLACE_NAMESPACE::idsct2_lee_forward, "IDSCT2 forward");
}
