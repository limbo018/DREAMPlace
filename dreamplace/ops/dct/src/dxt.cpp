/**
 * @file   dxt.cpp
 * @author Yibo Lin
 * @date   Sep 2018
 */
#include "dct.h"

at::Tensor idxct_forward(
        at::Tensor x,
        at::Tensor expk) 
{
    auto N = x.size(-1);
    auto M = x.numel()/N; 

    auto z = idct_forward(x, expk);

    //std::cout << __func__ << " z\n" << z << "\n";

    AT_DISPATCH_FLOATING_TYPES(x.type(), "idxct_forward", [&] {
            addX0AndScale<scalar_t>(
                    x.data<scalar_t>(), 
                    M, 
                    N, 
                    z.data<scalar_t>()
                    );
            });

    return z; 
}

at::Tensor idxst_forward(
        at::Tensor x,
        at::Tensor expk) 
{
    auto N = x.size(-1);
    auto M = x.numel()/N; 

    //std::cout << "x\n" << x << "\n";
    //auto x_reorder = at::empty_like(x);
    auto x_reorder = at::empty({M, N}, x.options());
    auto y = at::empty_like(x); 

    AT_DISPATCH_FLOATING_TYPES(x.type(), "idxst_forward", [&] {
            computeFlipAndShift<scalar_t>(
                    x.data<scalar_t>(), 
                    M, 
                    N, 
                    x_reorder.data<scalar_t>()
                    );

            y = idct_forward(x_reorder, expk);
            y.mul_(0.5);
            //std::cout << "y\n" << y << "\n";

            negateOddEntries<scalar_t>(
                    y.data<scalar_t>(), 
                    M, 
                    N
                    );
            //std::cout << "z\n" << y << "\n";
            });

    return y; 
}

at::Tensor idcct2_forward(
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

    // idxct for rows 

    //std::cout << "x\n" << x << "\n";
    // vk = 0.5*W_{4N}^{k} (c[k] - c[N-k])
    // vk is hermitian symmetric, only fill in half 
    auto v = at::empty({M*N+std::max(M, N)}, x.options()).resize_({M, N/2+1, 2});

    AT_DISPATCH_FLOATING_TYPES(x.type(), "idcct2_forward", [&] {
            computeVk<scalar_t>(
                    x.data<scalar_t>(), 
                    expk1.data<scalar_t>(), 
                    M, 
                    N, 
                    v.data<scalar_t>()
                    );

            //std::cout << "v\n" << v << "\n";

            // y is real now 
            auto y = at::irfft(v, 1, false, true, {N});

            //std::cout << "y\n" << y << "\n";

            //std::cout << "expk\n" << expk << "\n";
            v.resize_({M, N});
            computeReorderReverse(
                    y.data<scalar_t>(), 
                    M, 
                    N, 
                    v.data<scalar_t>()
                    );
            //std::cout << "z\n" << z << "\n";

            addX0AndScaleN<scalar_t>(
                    x.data<scalar_t>(), 
                    M, 
                    N, 
                    v.data<scalar_t>()
                    );

            //std::cout << "z\n" << z << "\n";
            // idxct for columns

            auto xt = v.transpose(-2, -1).contiguous();

            // vk = 0.5*W_{4N}^{k} (c[k] - c[N-k])
            v.resize_({N, M/2+1, 2});
            computeVk<scalar_t>(
                    xt.data<scalar_t>(), 
                    expk0.data<scalar_t>(), 
                    N, 
                    M, 
                    v.data<scalar_t>()
                    );

            //std::cout << __func__ << " v\n" << v << "\n";

            y = at::irfft(v, 1, false, true, {M});

            //std::cout << __func__ << " y\n" << y << "\n";

            //std::cout << "expk\n" << expk << "\n";
            v.resize_({N, M});
            computeReorderReverse(
                    y.data<scalar_t>(), 
                    N, 
                    M, 
                    v.data<scalar_t>()
                    );
            //std::cout << __func__ << " z\n" << z << "\n";

            addX0AndScaleN<scalar_t>(
                    xt.data<scalar_t>(), 
                    N, 
                    M, 
                    v.data<scalar_t>()
                    );

            v.transpose_(-2, -1);
    });

    return v.contiguous(); 
}

at::Tensor idsct2_forward(
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

    // idxct for rows 

    //std::cout << "x\n" << x << "\n";
    // vk = 0.5*W_{4N}^{k} (c[k] - c[N-k])
    // vk is hermitian symmetric, only fill in half 
    auto v = at::empty({M*N+std::max(M, N)}, x.type()).resize_({M, N/2+1, 2});
    auto z = at::empty({M, N}, x.options());

    AT_DISPATCH_FLOATING_TYPES(x.type(), "idsct2_forward", [&] {
            computeVk<scalar_t>(
                    x.data<scalar_t>(), 
                    expk1.data<scalar_t>(), 
                    M, 
                    N, 
                    v.data<scalar_t>()
                    );

            //std::cout << "v\n" << v << "\n";

            // y is real now 
            auto y = at::irfft(v, 1, false, true, {N});

            //std::cout << "y\n" << y << "\n";

            //std::cout << "expk\n" << expk << "\n";
            //auto z = at::empty_like(x);
            computeReorderReverse(
                    y.data<scalar_t>(), 
                    M, 
                    N, 
                    z.data<scalar_t>()
                    );
            //std::cout << "z\n" << z << "\n";

            addX0AndScaleN<scalar_t>(
                    x.data<scalar_t>(), 
                    M, 
                    N, 
                    z.data<scalar_t>()
                    );
            //std::cout << __func__ << " z\n" << z << "\n";

            // idxst for columns

            auto xt = z.transpose(-2, -1).contiguous();
            //std::cout << "x\n" << x << "\n";
            z = z.view_as(xt);
            computeFlipAndShift<scalar_t>(
                    xt.data<scalar_t>(), 
                    N, 
                    M, 
                    z.data<scalar_t>()
                    );

            //std::cout << "x\n" << x << "\n";
            // vk = 0.5*W_{4N}^{k} (c[k] - c[N-k])
            v.resize_({N, M/2+1, 2});
            computeVk<scalar_t>(
                    z.data<scalar_t>(), 
                    expk0.data<scalar_t>(), 
                    N, 
                    M, 
                    v.data<scalar_t>()
                    );

            //std::cout << "v\n" << v << "\n";

            y = at::irfft(v, 1, false, true, {M});

            //std::cout << "y\n" << y << "\n";

            //std::cout << "expk\n" << expk << "\n";
            computeReorderReverse(
                    y.data<scalar_t>(), 
                    N, 
                    M, 
                    z.data<scalar_t>()
                    );
            //std::cout << "z\n" << z << "\n";
            // this is to match python implementation 
            // normal way should be multiply by 0.25*N
            z.mul_(0.25*M); 

            negateOddEntries<scalar_t>(
                    z.data<scalar_t>(), 
                    N, 
                    M
                    );
            //std::cout << "z\n" << y << "\n";

            z.transpose_(-2, -1);
    });

    return z.contiguous(); 
}

at::Tensor idcst2_forward(
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

    // idxst for rows 
    //std::cout << "x\n" << x << "\n";
    //auto z = at::empty_like(x);
    auto z = at::empty({M, N}, x.options());

    AT_DISPATCH_FLOATING_TYPES(x.type(), "idcst2_forward", [&] {
            computeFlipAndShift<scalar_t>(
                    x.data<scalar_t>(), 
                    M, 
                    N, 
                    z.data<scalar_t>()
                    );

            //std::cout << "x\n" << x << "\n";
            // vk = 0.5*W_{4N}^{k} (c[k] - c[N-k])
            auto v = at::empty({M*N+std::max(M, N)}, x.options()).resize_({M, N/2+1, 2});
            computeVk<scalar_t>(
                    z.data<scalar_t>(), 
                    expk1.data<scalar_t>(), 
                    M, 
                    N, 
                    v.data<scalar_t>()
                    );

            //std::cout << "v\n" << v << "\n";

            auto y = at::irfft(v, 1, false, true, {N});

            //std::cout << "y\n" << y << "\n";

            //std::cout << "expk\n" << expk << "\n";
            computeReorderReverse(
                    y.data<scalar_t>(), 
                    M, 
                    N, 
                    z.data<scalar_t>()
                    );
            //std::cout << "z\n" << z << "\n";
            // this is to match python implementation 
            // normal way should be multiply by 0.25*N
            z.mul_(0.25*N); 

            negateOddEntries<scalar_t>(
                    z.data<scalar_t>(), 
                    M, 
                    N
                    );
            //std::cout << __func__ << " z\n" << z << "\n";

            // idxct for columns

            auto xt = z.transpose(-2, -1).contiguous();

            //std::cout << "x\n" << x << "\n";
            // vk = 0.5*W_{4N}^{k} (c[k] - c[N-k])
            // vk = 0.5*W_{4N}^{k} (c[k] - c[N-k])
            v.resize_({N, M/2+1, 2});
            computeVk<scalar_t>(
                    xt.data<scalar_t>(), 
                    expk0.data<scalar_t>(), 
                    N, 
                    M, 
                    v.data<scalar_t>()
                    );

            //std::cout << "v\n" << v << "\n";

            y = at::irfft(v, 1, false, true, {M});

            //std::cout << "y\n" << y << "\n";

            //std::cout << "expk\n" << expk << "\n";
            z = z.view_as(xt);
            computeReorderReverse(
                    y.data<scalar_t>(), 
                    N, 
                    M, 
                    z.data<scalar_t>()
                    );
            //std::cout << "z\n" << z << "\n";
            addX0AndScaleN<scalar_t>(
                    xt.data<scalar_t>(), 
                    N, 
                    M, 
                    z.data<scalar_t>()
                    );

            z.transpose_(-2, -1);
    });

    return z.contiguous(); 
}
