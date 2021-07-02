/**
 * @file   dst_cuda.cpp
 * @author Yibo Lin
 * @date   Sep 2018
 */
#include "dct_cuda.h"

DREAMPLACE_BEGIN_NAMESPACE

at::Tensor dst_forward(
        at::Tensor x,
        at::Tensor expk) 
{
    auto N = x.size(-1);
    auto M = x.numel()/N; 

    //std::cout << "x\n" << x << "\n";
    auto x_reorder = x.clone();

    DREAMPLACE_DISPATCH_FLOATING_TYPES(x, "dst_forward", [&] {
            negateOddEntriesCudaLauncher<scalar_t>(
                    DREAMPLACE_TENSOR_DATA_PTR(x_reorder, scalar_t), 
                    M, 
                    N
                    );

            auto y = dct_forward(x_reorder, expk);
            //std::cout << "y\n" << y << "\n";

            computeFlipCudaLauncher<scalar_t>(
                    DREAMPLACE_TENSOR_DATA_PTR(y, scalar_t), 
                    M, 
                    N, 
                    DREAMPLACE_TENSOR_DATA_PTR(x_reorder, scalar_t)
                    );
            //std::cout << "z\n" << y << "\n";
            });

    return x_reorder; 
}

at::Tensor idst_forward(
        at::Tensor x,
        at::Tensor expk) 
{
    auto N = x.size(-1);
    auto M = x.numel()/N; 

    //std::cout << "x\n" << x << "\n";
    auto x_reorder = at::empty_like(x);
    auto y = at::empty_like(x);

    DREAMPLACE_DISPATCH_FLOATING_TYPES(x, "idst_forward", [&] {
            computeFlipCudaLauncher<scalar_t>(
                    DREAMPLACE_TENSOR_DATA_PTR(x, scalar_t), 
                    M, 
                    N, 
                    DREAMPLACE_TENSOR_DATA_PTR(x_reorder, scalar_t)
                    );

            y = idct_forward(x_reorder, expk);
            //std::cout << "y\n" << y << "\n";

            negateOddEntriesCudaLauncher<scalar_t>(
                    DREAMPLACE_TENSOR_DATA_PTR(y, scalar_t), 
                    M, 
                    N
                    );
            //std::cout << "z\n" << y << "\n";
            });

    return y; 
}

DREAMPLACE_END_NAMESPACE
