##
# @file   dct2_fft2.py
# @author Zixuan Jiang, Jiaqi Gu
# @date   Jun 2018
# @brief  Implement 2d dct, 2d idct, idxst(idct(x)), idct(idxst(x)) based on 2d fft
#

import numpy as np
import torch
from torch.autograd import Function
from torch import nn

from dreamplace.ops.dct.discrete_spectral_transform import get_exact_expk as precompute_expk

import dreamplace.ops.dct.dct2_fft2_cpp as dct2_fft2_cpp
import dreamplace.configure as configure
if configure.compile_configurations["CUDA_FOUND"] == "TRUE": 
    import dreamplace.ops.dct.dct2_fft2_cuda as dct2_fft2_cuda


class DCT2Function(Function):
    @staticmethod
    def forward(ctx, x, expkM, expkN, out, buf):
        if x.is_cuda:
            dct2_fft2_cuda.dct2_fft2(x, expkM, expkN, out, buf)
        else:
            dct2_fft2_cpp.dct2_fft2(x, expkM, expkN, out, buf, torch.get_num_threads())
        return out


class DCT2(nn.Module):
    def __init__(self, expkM=None, expkN=None):
        super(DCT2, self).__init__()

        self.expkM = expkM
        self.expkN = expkN
        self.out = None
        self.buf = None

    def forward(self, x):
        M = x.size(-2)
        N = x.size(-1)
        if self.expkM is None or self.expkM.size(-2) != M or self.expkM.dtype != x.dtype:
            self.expkM = precompute_expk(M, dtype=x.dtype, device=x.device)
        if self.expkN is None or self.expkN.size(-2) != N or self.expkN.dtype != x.dtype:
            self.expkN = precompute_expk(N, dtype=x.dtype, device=x.device)
        if self.out is None:
            self.out = torch.empty(M, N, dtype=x.dtype, device=x.device)
            self.buf = torch.empty(M, N // 2 + 1, 2, dtype=x.dtype, device=x.device)

        return DCT2Function.apply(x, self.expkM, self.expkN, self.out, self.buf)


class IDCT2Function(Function):
    @staticmethod
    def forward(ctx, x, expkM, expkN, out, buf):
        if x.is_cuda:
            dct2_fft2_cuda.idct2_fft2(x, expkM, expkN, out, buf)
        else:
            dct2_fft2_cpp.idct2_fft2(x, expkM, expkN, out, buf, torch.get_num_threads())
        return out


class IDCT2(nn.Module):
    def __init__(self, expkM=None, expkN=None):
        super(IDCT2, self).__init__()

        self.expkM = expkM
        self.expkN = expkN
        self.out = None
        self.buf = None

    def forward(self, x):
        M = x.size(-2)
        N = x.size(-1)
        if self.expkM is None or self.expkM.size(-2) != M or self.expkM.dtype != x.dtype:
            self.expkM = precompute_expk(M, dtype=x.dtype, device=x.device)
        if self.expkN is None or self.expkN.size(-2) != N or self.expkN.dtype != x.dtype:
            self.expkN = precompute_expk(N, dtype=x.dtype, device=x.device)
        if self.out is None:
            self.out = torch.empty(M, N, dtype=x.dtype, device=x.device)
            self.buf = torch.empty(M, N // 2 + 1, 2, dtype=x.dtype, device=x.device)

        return IDCT2Function.apply(x, self.expkM, self.expkN, self.out, self.buf)


class IDCT_IDXSTFunction(Function):
    @staticmethod
    def forward(ctx, x, expkM, expkN, out, buf):
        if x.is_cuda:
            dct2_fft2_cuda.idct_idxst(x, expkM, expkN, out, buf)
        else:
            dct2_fft2_cpp.idct_idxst(x, expkM, expkN, out, buf, torch.get_num_threads())
        return out


class IDCT_IDXST(nn.Module):
    def __init__(self, expkM=None, expkN=None):
        super(IDCT_IDXST, self).__init__()

        self.expkM = expkM
        self.expkN = expkN
        self.out = None
        self.buf = None

    def forward(self, x):
        M = x.size(-2)
        N = x.size(-1)
        if self.expkM is None or self.expkM.size(-2) != M or self.expkM.dtype != x.dtype:
            self.expkM = precompute_expk(M, dtype=x.dtype, device=x.device)
        if self.expkN is None or self.expkN.size(-2) != N or self.expkN.dtype != x.dtype:
            self.expkN = precompute_expk(N, dtype=x.dtype, device=x.device)
        if self.out is None:
            self.out = torch.empty(M, N, dtype=x.dtype, device=x.device)
            self.buf = torch.empty(M, N // 2 + 1, 2, dtype=x.dtype, device=x.device)

        return IDCT_IDXSTFunction.apply(x, self.expkM, self.expkN, self.out, self.buf)


class IDXST_IDCTFunction(Function):
    @staticmethod
    def forward(ctx, x, expkM, expkN, out, buf):
        if x.is_cuda:
            dct2_fft2_cuda.idxst_idct(x, expkM, expkN, out, buf)
        else:
            dct2_fft2_cpp.idxst_idct(x, expkM, expkN, out, buf, torch.get_num_threads())
        return out


class IDXST_IDCT(nn.Module):
    def __init__(self, expkM=None, expkN=None):
        super(IDXST_IDCT, self).__init__()

        self.expkM = expkM
        self.expkN = expkN
        self.out = None
        self.buf = None

    def forward(self, x):
        M = x.size(-2)
        N = x.size(-1)
        if self.expkM is None or self.expkM.size(-2) != M or self.expkM.dtype != x.dtype:
            self.expkM = precompute_expk(M, dtype=x.dtype, device=x.device)
        if self.expkN is None or self.expkN.size(-2) != N or self.expkN.dtype != x.dtype:
            self.expkN = precompute_expk(N, dtype=x.dtype, device=x.device)
        if self.out is None:
            self.out = torch.empty(M, N, dtype=x.dtype, device=x.device)
            self.buf = torch.empty(M, N // 2 + 1, 2, dtype=x.dtype, device=x.device)

        return IDXST_IDCTFunction.apply(x, self.expkM, self.expkN, self.out, self.buf)
