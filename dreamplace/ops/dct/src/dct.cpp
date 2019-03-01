/**
 * @file   dct.cpp
 * @author Yibo Lin
 * @date   Jun 2018
 */
#include "dct.h"

template <typename T>
void computeReorder(
        const T* x, 
        const int M, 
        const int N, 
        T* y
        )
{
    for (int i = 0; i < M*N; ++i) 
    {
        int ii = i%N; 

        if (ii < (N>>1))
        {
            // i*2
            //printf("x[%d] = y[%d]\n", i+ii, i);
            y[i] = x[i+ii];
        }
        else 
        {
            // (N-i)*2-1
            //printf("x[%d] = y[%d]\n", i+N*2-ii*3-1, i);
            y[i] = x[i+N*2-ii*3-1];
        }
    }
}

template <typename T>
void computeMulExpk(
        const T* x, 
        const T* expk, 
        const int M, 
        const int N, 
        T* z
        )
{
    for (int i = 0; i < M*N; ++i) 
    {
        int row = i/N; // row
        int col = i-row*N; // column
        int col_2x = (col<<1);
        int fft_onesided_size = (N>>1)+1;
        int fft_onesided_size_2x = fft_onesided_size<<1;

        if (col_2x <= N)
        {
            int j = row*fft_onesided_size_2x + col_2x;
            //printf("x[%d]*expk[%d] + x[%d]*expk[%d] = z[%d]\n", j, col_2x, j+1, col_2x+1, i);
            z[i] = x[j]*expk[col_2x] + x[j+1]*expk[col_2x+1];
        }
        else 
        {
            int j = row*fft_onesided_size_2x + (N<<1) - col_2x;
            //printf("x[%d]*expk[%d] + x[%d]*expk[%d] = z[%d]\n", j, col_2x, j+1, col_2x+1, i);
            z[i] = x[j]*expk[col_2x] - x[j+1]*expk[col_2x+1];
        }
    }
}

template <typename T>
void computeVk(
        const T* x, 
        const T* expk, 
        const int M, 
        const int N, 
        T* v
        )
{
    for (int i = 0; i < M*(N/2+1); ++i)
    {
        int ncol = N/2+1; 
        int row = i/ncol; // row
        int col = i-row*ncol; // column
        int col_2x = (col<<1);

        // real 
        T real = x[row*N+col];
        T imag = (col == 0)? 0 : -x[row*N+N-col];

        v[2*i] = real*expk[col_2x] - imag*expk[col_2x+1];
        // imag, x[N-i]
        v[2*i+1] = real*expk[col_2x+1] + imag*expk[col_2x]; 
    }
}

template <typename T>
void computeReorderReverse(
        const T* y, 
        const int M, 
        const int N, 
        T* z
        )
{
    for (int i = 0; i < M*N; ++i)
    {
        int row = i/N; // row
        int col = i-row*N; // column

        //assert((i-col*2+N-1)*2 < M*N*2);
        //printf("z[%d] = y[%d]\n", i, (col&1)? (i-col*3/2+N-1) : (i-col/2));
        //z[i] = (col&1)? y[(i-col*3/2+N-1)] : y[(i-col/2)];
        // according to the paper, it should be N - (col+1)/2 for col is odd 
        // but it seems previous implementation accidentally matches this as well 
        z[i] = (col&1)? y[(i-col) + N - (col+1)/2] : y[(i-col/2)];
    }
}

at::Tensor dct_forward(
        at::Tensor x,
        at::Tensor expk) 
{
    CHECK_CPU(x);
    CHECK_CONTIGUOUS(x);
    CHECK_CPU(expk);
    CHECK_CONTIGUOUS(expk);

    auto N = x.size(-1);
    auto M = x.numel()/N; 

    //std::cout << "x\n" << x << "\n";
    //auto x_reorder = at::empty_like(x);
    auto x_reorder = at::empty({M, N}, x.options());

    AT_DISPATCH_FLOATING_TYPES(x.type(), "dct_forward", [&] {
            computeReorder<scalar_t>(
                    x.data<scalar_t>(), 
                    M, 
                    N, 
                    x_reorder.data<scalar_t>()
                    );

            //std::cout << "x_reorder\n" << x_reorder << "\n";

            auto y = at::rfft(x_reorder, 1, false, true);
            //std::cout << "y\n" << y << "\n";

            // re-use x_reorder as output 
            //std::cout << "x_reorder\n" << x_reorder << "\n";
            //std::cout << "expk\n" << expk << "\n";
            computeMulExpk(
                    y.data<scalar_t>(), 
                    expk.data<scalar_t>(), 
                    M, 
                    N, 
                    x_reorder.data<scalar_t>()
                    );
            //std::cout << "z\n" << x_reorder << "\n";
            x_reorder.mul_(1.0/N);
    });

    return x_reorder; 
}

at::Tensor idct_forward(
        at::Tensor x,
        at::Tensor expk) 
{
    CHECK_CPU(x);
    CHECK_CONTIGUOUS(x);
    CHECK_CPU(expk);
    CHECK_CONTIGUOUS(expk);

    auto N = x.size(-1);
    auto M = x.numel()/N; 

    //std::cout << "x\n" << x << "\n";
    // vk = 0.5*W_{4N}^{k} (c[k] - c[N-k])
    // vk is hermitian symmetric, only fill in half 
    auto v = at::empty({M*N+std::max(M, N)}, x.options()).resize_({M, N/2+1, 2});

    AT_DISPATCH_FLOATING_TYPES(x.type(), "idct_forward", [&] {
            computeVk<scalar_t>(
                    x.data<scalar_t>(), 
                    expk.data<scalar_t>(), 
                    M, 
                    N, 
                    v.data<scalar_t>()
                    );

            //std::cout << __func__ << " v\n" << v << "\n";

            // y is real now 
            auto y = at::irfft(v, 1, false, true, {N});

            //std::cout << __func__ << " y\n" << y << "\n";

            //std::cout << "expk\n" << expk << "\n";
            // reuse v 
            v.resize_({M, N});
            computeReorderReverse(
                    y.data<scalar_t>(), 
                    M, 
                    N, 
                    v.data<scalar_t>()
                    );
            //std::cout << "z\n" << z << "\n";

            // this is to match python implementation 
            // normal way should be multiply by 0.25*N
            v.mul_(0.5*N); 
    });

    return v; 
}

at::Tensor dct2_forward(
        at::Tensor x,
        at::Tensor expk0, 
        at::Tensor expk1) 
{
    CHECK_CPU(x);
    CHECK_CONTIGUOUS(x);
    CHECK_CPU(expk0);
    CHECK_CONTIGUOUS(expk0);
    CHECK_CPU(expk1);
    CHECK_CONTIGUOUS(expk1);

    // 1D DCT to columns

    //std::cout << "x\n" << x << "\n";
    auto N = x.size(-1);
    auto M = x.numel()/N; 
    auto x_reorder = at::empty({M, N}, x.options());

    AT_DISPATCH_FLOATING_TYPES(x.type(), "dct2_forward", [&] {
            computeReorder<scalar_t>(
                    x.data<scalar_t>(), 
                    M, 
                    N, 
                    x_reorder.data<scalar_t>()
                    );

            //std::cout << "x_reorder\n" << x_reorder << "\n";

            auto y = at::rfft(x_reorder, 1, false, true);
            //y.mul_(1.0/N);
            //std::cout << "y\n" << y << "\n";

            // re-use x_reorder as output 
            //std::cout << "expk1\n" << expk1 << "\n";
            computeMulExpk(
                    y.data<scalar_t>(), 
                    expk1.data<scalar_t>(), 
                    M, 
                    N, 
                    x_reorder.data<scalar_t>()
                    );
            //std::cout << "z\n" << x_reorder << "\n";

            // 1D DCT to rows 
            auto xt = x_reorder.transpose(-2, -1).contiguous();
            //std::cout << "xt\n" << xt << "\n";
            // I do not want to allocate memory another time 
            //x_reorder = at::empty_like(xt);
            x_reorder = x_reorder.view_as(xt);
            computeReorder<scalar_t>(
                    xt.data<scalar_t>(), 
                    N, 
                    M, 
                    x_reorder.data<scalar_t>()
                    );

            //std::cout << "x_reorder\n" << x_reorder << "\n";

            y = at::rfft(x_reorder, 1, false, true);
            //y.mul_(1.0/M);
            //std::cout << "y\n" << y << "\n";

            // re-use x_reorder as output 
            //std::cout << "expk0\n" << expk0 << "\n";
            computeMulExpk(
                    y.data<scalar_t>(), 
                    expk0.data<scalar_t>(), 
                    N, 
                    M, 
                    x_reorder.data<scalar_t>()
                    );

            x_reorder.mul_(1.0/(M*N));
            x_reorder.transpose_(-2, -1);
    });

    return x_reorder.contiguous(); 
}

at::Tensor idct2_forward(
        at::Tensor x,
        at::Tensor expk0, 
        at::Tensor expk1) 
{
    CHECK_CPU(x);
    CHECK_CONTIGUOUS(x);
    CHECK_CPU(expk0);
    CHECK_CONTIGUOUS(expk0);
    CHECK_CPU(expk1);
    CHECK_CONTIGUOUS(expk1);

    auto N = x.size(-1);
    auto M = x.numel()/N; 

    // 1D DCT to columns

    //std::cout << "x\n" << x << "\n";
    // vk = 0.5*W_{4N}^{k} (c[k] - c[N-k])
    // vk is hermitian symmetric, only fill in half 
    auto v = at::empty({M*N+std::max(M, N)}, x.options()).resize_({M, N/2+1, 2});

    AT_DISPATCH_FLOATING_TYPES(x.type(), "idct2_forward", [&] {
            computeVk<scalar_t>(
                    x.data<scalar_t>(), 
                    expk1.data<scalar_t>(), 
                    M, 
                    N, 
                    v.data<scalar_t>()
                    );

            //std::cout << "expk1\n" << expk1 << "\n";

            auto y = at::irfft(v, 1, false, true, {N});
            //y.mul_(0.25*N);

            //std::cout << "y\n" << y << "\n";

            //std::cout << "expk\n" << expk << "\n";
            //auto z = at::empty_like(x);
            //auto z = at::empty(x.type(), {M, N});
            /// reuse v 
            v.resize_({M, N});
            computeReorderReverse(
                    y.data<scalar_t>(), 
                    M, 
                    N, 
                    v.data<scalar_t>()
                    );
            //std::cout << "z\n" << z << "\n";

            // 1D DCT to rows 
            auto xt = v.transpose(-2, -1).contiguous();
            //std::cout << "xt\n" << xt << "\n";
            // vk = 0.5*W_{4N}^{k} (c[k] - c[N-k])
            v.resize_({N, M/2+1, 2});
            computeVk<scalar_t>(
                    xt.data<scalar_t>(), 
                    expk0.data<scalar_t>(), 
                    N, 
                    M, 
                    v.data<scalar_t>()
                    );

            //std::cout << "expk0\n" << expk0 << "\n";
            //std::cout << "v\n" << v << "\n";

            y = at::irfft(v, 1, false, true, {M});
            //y.mul_(0.25*M);

            //std::cout << "y\n" << y << "\n";

            // I do not want to allocate memory another time 
            // reuse v 
            v.resize_({N, M});
            computeReorderReverse(
                    y.data<scalar_t>(), 
                    N, 
                    M, 
                    v.data<scalar_t>()
                    );
            //std::cout << "z\n" << z << "\n";

            // this is to match python implementation 
            // normal way should be multiply by 0.25*0.25*M*N
            v.mul_(0.25*M*N);
            v.transpose_(-2, -1);
    });

    return v.contiguous(); 
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dct", &dct_forward, "DCT forward");
  m.def("idct", &idct_forward, "IDCT forward");
  m.def("idxct", &idxct_forward, "IDXCT forward");
  m.def("dct2", &dct2_forward, "DCT2 forward");

  m.def("dst", &dst_forward, "DST forward");
  m.def("idst", &idst_forward, "IDST forward");

  m.def("idct2", &idct2_forward, "IDCT2 forward");
  m.def("idxst", &idxst_forward, "IDXST forward");
  m.def("idcct2", &idcct2_forward, "IDCCT2 forward");
  m.def("idcst2", &idcst2_forward, "IDCST2 forward");
  m.def("idsct2", &idsct2_forward, "IDSCT2 forward");

  m.def("dct_2N", &dct_2N_forward, "DCT forward");
  m.def("idct_2N", &idct_2N_forward, "IDCT forward");
  m.def("dct2_2N", &dct2_2N_forward, "DCT2 forward");
  m.def("idct2_2N", &idct2_2N_forward, "IDCT2 forward");
}

