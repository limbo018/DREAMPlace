##
# @file   dct.py
# @author Zixuan Jiang, Jiaqi Gu
# @date   Jun 2018
# @brief  Implement 2d dct, 2d idct, idxst(idct(x)), idct(idxst(x)) based on 2d fft
#

import numpy as np
import torch
from torch.autograd import Function
from torch import nn

from dreamplace.ops.dct.discrete_spectral_transform import get_exact_expk as precompute_expk

try:
    import dreamplace.ops.dct.dct2_fft2_cuda as dct2_fft2
except:
    pass


class DCT2Function(Function):
    @staticmethod
    def forward(ctx, x, expkM, expkN, out, buf):
        if x.is_cuda:
            dct2_fft2.dct2_fft2(x, expkM, expkN, out, buf)
            return out
        else:
            assert 0, "No CPU Implementation"


class DCT2(nn.Module):
    def __init__(self, M, N, dtype=torch.float64, device=torch.cuda, expkM=None, expkN=None):
        super(DCT2, self).__init__()

        if expkM is None or expkM.size(-1) != M or expkM.dtype != dtype:
            self.expkM = precompute_expk(M, dtype=dtype, device=device)
        else:
            self.expkM = expkM.to(device)

        if expkN is None or expkN.size(-1) != N or expkN.dtype != dtype:
            self.expkN = precompute_expk(N, dtype=dtype, device=device)
        else:
            self.expkN = expkN.to(device)

        self.out = torch.empty(M, N, dtype=dtype, device=device)
        self.buf = torch.empty(M, N / 2 + 1, 2, dtype=dtype, device=device)

    def forward(self, x):
        return DCT2Function.apply(x, self.expkM, self.expkN, self.out, self.buf)


class IDCT2Function(Function):
    @staticmethod
    def forward(ctx, x, expkM, expkN, out, buf):
        if x.is_cuda:
            dct2_fft2.idct2_fft2(x, expkM, expkN, out, buf)
            return out
        else:
            assert 0, "No CPU Implementation"


class IDCT2(nn.Module):
    def __init__(self, M, N, dtype=torch.float64, device=torch.cuda, expkM=None, expkN=None):
        super(IDCT2, self).__init__()

        if expkM is None or expkM.size(-1) != M or expkM.dtype != dtype:
            self.expkM = precompute_expk(M, dtype=dtype, device=device)
        else:
            self.expkM = expkM.to(device)

        if expkN is None or expkN.size(-1) != N or expkN.dtype != dtype:
            self.expkN = precompute_expk(N, dtype=dtype, device=device)
        else:
            self.expkN = expkN.to(device)

        self.out = torch.empty(M, N, dtype=dtype, device=device)
        self.buf = torch.empty(M, N / 2 + 1, 2, dtype=dtype, device=device)

    def forward(self, x):
        return IDCT2Function.apply(x, self.expkM, self.expkN, self.out, self.buf)


class IDCT_IDXSTFunction(Function):
    @staticmethod
    def forward(ctx, x, expkM, expkN, out, buf):
        if x.is_cuda:
            dct2_fft2.idct_idxst(x, expkM, expkN, out, buf)
            return out
        else:
            assert 0, "No CPU Implementation"


class IDCT_IDXST(nn.Module):
    def __init__(self, M, N, dtype=torch.float64, device=torch.cuda, expkM=None, expkN=None):
        super(IDCT_IDXST, self).__init__()

        if expkM is None or expkM.size(-1) != M or expkM.dtype != dtype:
            self.expkM = precompute_expk(M, dtype=dtype, device=device)
        else:
            self.expkM = expkM.to(device)

        if expkN is None or expkN.size(-1) != N or expkN.dtype != dtype:
            self.expkN = precompute_expk(N, dtype=dtype, device=device)
        else:
            self.expkN = expkN.to(device)

        self.out = torch.empty(M, N, dtype=dtype, device=device)
        self.buf = torch.empty(M, N / 2 + 1, 2, dtype=dtype, device=device)

    def forward(self, x):
        return IDCT_IDXSTFunction.apply(x, self.expkM, self.expkN, self.out, self.buf)


class IDXST_IDCTFunction(Function):
    @staticmethod
    def forward(ctx, x, expkM, expkN, out, buf):
        if x.is_cuda:
            dct2_fft2.idxst_idct(x, expkM, expkN, out, buf)
            return out
        else:
            assert 0, "No CPU Implementation"


class IDXST_IDCT(nn.Module):
    def __init__(self, M, N, dtype=torch.float64, device=torch.cuda, expkM=None, expkN=None):
        super(IDXST_IDCT, self).__init__()

        if expkM is None or expkM.size(-1) != M or expkM.dtype != dtype:
            self.expkM = precompute_expk(M, dtype=dtype, device=device)
        else:
            self.expkM = expkM.to(device)

        if expkN is None or expkN.size(-1) != N or expkN.dtype != dtype:
            self.expkN = precompute_expk(N, dtype=dtype, device=device)
        else:
            self.expkN = expkN.to(device)

        self.out = torch.empty(M, N, dtype=dtype, device=device)
        self.buf = torch.empty(M, N / 2 + 1, 2, dtype=dtype, device=device)

    def forward(self, x):
        return IDXST_IDCTFunction.apply(x, self.expkM, self.expkN, self.out, self.buf)
