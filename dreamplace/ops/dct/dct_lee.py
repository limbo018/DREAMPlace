##
# @file   dct_lee.py
# @author Yibo Lin
# @date   Oct 2018
#

import numpy as np
import torch
from torch.autograd import Function
from torch import nn
import pdb

import dreamplace.ops.dct.dct_lee_cpp as dct_cpp
import dreamplace.configure as configure
if configure.compile_configurations["CUDA_FOUND"] == "TRUE": 
    import dreamplace.ops.dct.dct_lee_cuda as dct_cuda

def dct(x, expk, buf, out):
    """compute discrete cosine transformation, DCT II 
    yk = \sum_{n=0}^{N-1} x_n cos(pi/N*n*(k+1/2))
    """
    if x.is_cuda:
        dct_cuda.dct(x.view([-1, x.size(-1)]), expk, buf, out)
    else:
        dct_cpp.dct(x.view([-1, x.size(-1)]), expk, buf, out, torch.get_num_threads())
    return out.view(x.size())  

class DCTFunction(Function):
    @staticmethod
    def forward(ctx, x, expk, buf, out):
        return dct(x, expk, buf, out)

class DCT(nn.Module):
    def __init__(self, expk=None):
        super(DCT, self).__init__()
        self.expk = expk
        self.buf = None
        self.out = None
    def forward(self, x): 
        if self.expk is None or self.expk.size(-1) != x.size(-1):
            self.expk = torch.empty(x.size(-1), dtype=x.dtype, device=x.device)
            if x.is_cuda: 
                dct_cuda.precompute_dct_cos(x.size(-1), self.expk)
            else:
                dct_cpp.precompute_dct_cos(x.size(-1), self.expk)
        if self.out is None or self.out.size() != x.size():
            self.buf = torch.empty_like(x)
            self.out = torch.empty_like(x)
        return DCTFunction.apply(x, self.expk, self.buf, self.out)

def idct(x, expk, buf, out):
    """Compute inverse discrete cosine transformation, which is also the DCT III
    yk = Re { 1/2*x0 + \sum_{n=1}^{N-1} xn exp(j*pi/(2N)*n*(2k+1)) }
    The actual yk will be scaled by 2 to match other python implementation
    """
    if x.is_cuda:
        dct_cuda.idct(x.view([-1, x.size(-1)]), expk, buf, out)
    else:
        dct_cpp.idct(x.view([-1, x.size(-1)]), expk, buf, out, torch.get_num_threads())
    return out.view(x.size())  

class IDCTFunction(Function):
    @staticmethod
    def forward(ctx, x, expk, buf, out):
        return idct(x, expk, buf, out)

class IDCT(nn.Module):
    def __init__(self, expk=None):
        super(IDCT, self).__init__()
        self.expk = expk
        self.buf = None
        self.out = None
    def forward(self, x): 
        if self.expk is None or self.expk.size(-1) != x.size(-1):
            self.expk = torch.empty(x.size(-1), dtype=x.dtype, device=x.device)
            if x.is_cuda: 
                dct_cuda.precompute_idct_cos(x.size(-1), self.expk)
            else:
                dct_cpp.precompute_idct_cos(x.size(-1), self.expk)
        if self.out is None or self.out.size() != x.size():
            self.buf = torch.empty_like(x)
            self.out = torch.empty_like(x)
        return IDCTFunction.apply(x, self.expk, self.buf, self.out)

def dct2(x, expk0, expk1, buf, out):
    """compute 2D discrete cosine transformation
    """
    if x.is_cuda:
        dct_cuda.dct2(x, expk0, expk1, buf, out)
    else:
        dct_cpp.dct2(x, expk0, expk1, buf, out, torch.get_num_threads())
    return out

class DCT2Function(Function):
    @staticmethod
    def forward(ctx, x, expk0, expk1, buf, out):
        return dct2(x, expk0, expk1, buf, out)

class DCT2(nn.Module):
    def __init__(self, expk0=None, expk1=None):
        super(DCT2, self).__init__()
        self.expk0 = expk0
        self.expk1 = expk1
        self.buf = None
        self.out = None
    def forward(self, x): 
        if self.expk0 is None or self.expk0.size(-1) != x.size(-2):
            self.expk0 = torch.empty(x.size(-2), dtype=x.dtype, device=x.device)
            if x.is_cuda: 
                dct_cuda.precompute_dct_cos(x.size(-2), self.expk0)
            else:
                dct_cpp.precompute_dct_cos(x.size(-2), self.expk0)
        if self.expk1 is None or self.expk1.size(-1) != x.size(-1):
            self.expk1 = torch.empty(x.size(-1), dtype=x.dtype, device=x.device)
            if x.is_cuda: 
                dct_cuda.precompute_dct_cos(x.size(-1), self.expk1)
            else:
                dct_cpp.precompute_dct_cos(x.size(-1), self.expk1)
        if self.out is None or self.out.size() != x.size():
            self.buf = torch.empty_like(x)
            self.out = torch.empty_like(x)
        return DCT2Function.apply(x, self.expk0, self.expk1, self.buf, self.out)

def idct2(x, expk0, expk1, buf, out):
    """compute 2D inverse discrete cosine transformation
    """
    if x.is_cuda:
        dct_cuda.idct2(x, expk0, expk1, buf, out)
    else:
        dct_cpp.idct2(x, expk0, expk1, buf, out, torch.get_num_threads())
    return out

class IDCT2Function(Function):
    @staticmethod
    def forward(ctx, x, expk0, expk1, buf, out):
        return idct2(x, expk0, expk1, buf, out)

class IDCT2(nn.Module):
    def __init__(self, expk0=None, expk1=None):
        super(IDCT2, self).__init__()
        self.expk0 = expk0
        self.expk1 = expk1
        self.buf = None
        self.out = None
    def forward(self, x): 
        if self.expk0 is None or self.expk0.size(-1) != x.size(-2):
            self.expk0 = torch.empty(x.size(-2), dtype=x.dtype, device=x.device)
            if x.is_cuda: 
                dct_cuda.precompute_idct_cos(x.size(-2), self.expk0)
            else:
                dct_cpp.precompute_idct_cos(x.size(-2), self.expk0)
        if self.expk1 is None or self.expk1.size(-1) != x.size(-1):
            self.expk1 = torch.empty(x.size(-2), dtype=x.dtype, device=x.device)
            if x.is_cuda: 
                dct_cuda.precompute_idct_cos(x.size(-1), self.expk1)
            else:
                dct_cpp.precompute_idct_cos(x.size(-1), self.expk1)
        if self.out is None or self.out.size() != x.size():
            self.buf = torch.empty_like(x)
            self.out = torch.empty_like(x)
        return IDCT2Function.apply(x, self.expk0, self.expk1, self.buf, self.out)

def dst(x, expk, buf, out):
    """compute discrete sine transformation
    yk = \sum_{n=0}^{N-1} x_n cos(pi/N*(n+1/2)*(k+1))
    """
    if x.is_cuda:
        dct_cuda.dst(x.view([-1, x.size(-1)]), expk, buf, out)
    else:
        dct_cpp.dst(x.view([-1, x.size(-1)]), expk, buf, out, torch.get_num_threads())
    return out.view(x.size())  

class DSTFunction(Function):
    @staticmethod
    def forward(ctx, x, expk, buf, out):
        return dst(x, expk, buf, out)

class DST(nn.Module):
    def __init__(self, expk=None):
        super(DST, self).__init__()
        self.expk = expk
        self.buf = None 
        self.out = None
    def forward(self, x): 
        if self.expk is None or self.expk.size(-1) != x.size(-1):
            self.expk = torch.empty(x.size(-1), dtype=x.dtype, device=x.device)
            if x.is_cuda: 
                dct_cuda.precompute_dct_cos(x.size(-1), self.expk)
            else:
                dct_cpp.precompute_dct_cos(x.size(-1), self.expk)
        if self.out is None or self.out.size() != x.size():
            self.buf = torch.empty_like(x)
            self.out = torch.empty_like(x)
        return DSTFunction.apply(x, self.expk, self.buf, self.out)

def idst(x, expk, buf, out):
    """Compute inverse discrete sine transformation, which is also the DST III
    yk = Im { (-1)^k*x_{N-1}/2 + \sum_{n=0}^{N-2} xn exp(j*pi/(2N)*(n+1)*(2k+1)) }
    The actual yk will be scaled by 2 to match other python implementation
    """
    if x.is_cuda:
        dct_cuda.idst(x.view([-1, x.size(-1)]), expk, buf, out)
    else:
        dct_cpp.idst(x.view([-1, x.size(-1)]), expk, buf, out, torch.get_num_threads())
    return out.view(x.size())  

class IDSTFunction(Function):
    @staticmethod
    def forward(ctx, x, expk, buf, out):
        return idst(x, expk, buf, out)

class IDST(nn.Module):
    def __init__(self, expk=None):
        super(IDST, self).__init__()
        self.expk = expk
        self.buf = None
        self.out = None
    def forward(self, x): 
        if self.expk is None or self.expk.size(-1) != x.size(-1):
            self.expk = torch.empty(x.size(-1), dtype=x.dtype, device=x.device)
            if x.is_cuda: 
                dct_cuda.precompute_idct_cos(x.size(-1), self.expk)
            else:
                dct_cpp.precompute_idct_cos(x.size(-1), self.expk)
        if self.out is None or self.out.size() != x.size():
            self.buf = torch.empty_like(x)
            self.out = torch.empty_like(x)
        return IDSTFunction.apply(x, self.expk, self.buf, self.out)

def idxct(x, expk, buf, out):
    """compute inverse discrete cosine transformation
    This is different from ordinary formulation for IDCT III
    yk = Re { \sum_{n=0}^{N-1} xn exp(j*pi/(2N)*n*(2k+1)) }
    """
    if x.is_cuda:
        dct_cuda.idxct(x.view([-1, x.size(-1)]), expk, buf, out)
    else:
        dct_cpp.idxct(x.view([-1, x.size(-1)]), expk, buf, out, torch.get_num_threads())
    #output = IDCTFunction.forward(ctx, x, expk)
    #output.add_(x[..., 0].unsqueeze(-1)).mul_(0.5)
    ##output.mul_(0.5).add_(x[..., 0].unsqueeze(-1).mul(0.5))
    return out.view(x.size()) 

class IDXCTFunction(Function):
    @staticmethod
    def forward(ctx, x, expk, buf, out):
        return idxct(x, expk, buf, out)

class IDXCT(nn.Module):
    def __init__(self, expk=None):
        super(IDXCT, self).__init__()
        self.expk = expk
        self.buf = None
        self.out = None
    def forward(self, x): 
        if self.expk is None or self.expk.size(-1) != x.size(-1):
            self.expk = torch.empty(x.size(-1), dtype=x.dtype, device=x.device)
            if x.is_cuda: 
                dct_cuda.precompute_idct_cos(x.size(-1), self.expk)
            else:
                dct_cpp.precompute_idct_cos(x.size(-1), self.expk)
        if self.out is None or self.out.size() != x.size():
            self.buf = torch.empty_like(x)
            self.out = torch.empty_like(x)
        return IDXCTFunction.apply(x, self.expk, self.buf, self.out)

def idxst(x, expk, buf, out):
    """compute inverse discrete sine transformation
    This is different from ordinary formulation for IDCT III
    yk = Im { \sum_{n=0}^{N-1} xn exp(j*pi/(2N)*n*(2k+1)) }
    """
    if x.is_cuda:
        dct_cuda.idxst(x.view([-1, x.size(-1)]), expk, buf, out)
    else:
        dct_cpp.idxst(x.view([-1, x.size(-1)]), expk, buf, out, torch.get_num_threads())
    return out.view(x.size())  

class IDXSTFunction(Function):
    @staticmethod
    def forward(ctx, x, expk, buf, out):
        return idxst(x, expk, buf, out)

class IDXST(nn.Module):
    def __init__(self, expk=None):
        super(IDXST, self).__init__()
        self.expk = expk
        self.buf = None
        self.out = None
    def forward(self, x): 
        if self.expk is None or self.expk.size(-1) != x.size(-1):
            self.expk = torch.empty(x.size(-1), dtype=x.dtype, device=x.device)
            if x.is_cuda: 
                dct_cuda.precompute_idct_cos(x.size(-1), self.expk)
            else:
                dct_cpp.precompute_idct_cos(x.size(-1), self.expk)
        if self.out is None or self.out.size() != x.size():
            self.buf = torch.empty_like(x)
            self.out = torch.empty_like(x)
        return IDXSTFunction.apply(x, self.expk, self.buf, self.out)

def idcct2(x, expk0, expk1, buf0, buf1, out):
    """compute inverse discrete cosine-sine transformation
    This is equivalent to idcct(idcct(x)^T)^T
    """
    if x.is_cuda:
        dct_cuda.idcct2(x.view([-1, x.size(-1)]), expk0, expk1, buf0, buf1, out)
    else:
        dct_cpp.idcct2(x.view([-1, x.size(-1)]), expk0, expk1, buf0, buf1, out, torch.get_num_threads())
    return out.view(x.size())  

class IDCCT2Function(Function):
    @staticmethod
    def forward(ctx, x, expk0, expk1, buf0, buf1, out):
        return idcct2(x, expk0, expk1, buf0, buf1, out)

class IDCCT2(nn.Module):
    def __init__(self, expk0=None, expk1=None):
        super(IDCCT2, self).__init__()
        self.expk0 = expk0
        self.expk1 = expk1
        self.buf0 = None
        self.buf1 = None
        self.out = None
    def forward(self, x): 
        if self.expk0 is None or self.expk0.size(-1) != x.size(-2):
            self.expk0 = torch.empty(x.size(-2), dtype=x.dtype, device=x.device)
            if x.is_cuda: 
                dct_cuda.precompute_idct_cos(x.size(-2), self.expk0)
            else:
                dct_cpp.precompute_idct_cos(x.size(-2), self.expk0)
        if self.expk1 is None or self.expk1.size(-1) != x.size(-1):
            self.expk1 = torch.empty(x.size(-2), dtype=x.dtype, device=x.device)
            if x.is_cuda: 
                dct_cuda.precompute_idct_cos(x.size(-1), self.expk1)
            else:
                dct_cpp.precompute_idct_cos(x.size(-1), self.expk1)
        if self.out is None or self.out.size() != x.size():
            self.buf0 = torch.empty_like(x)
            self.buf1 = torch.empty_like(x)
            self.out = torch.empty_like(x)
        return IDCCT2Function.apply(x, self.expk0, self.expk1, self.buf0, self.buf1, self.out)

def idcst2(x, expk0, expk1, buf0, buf1, out):
    """compute inverse discrete cosine-sine transformation
    This is equivalent to idxct(idxst(x)^T)^T
    """
    if x.is_cuda:
        dct_cuda.idcst2(x.view([-1, x.size(-1)]), expk0, expk1, buf0, buf1, out)
    else:
        dct_cpp.idcst2(x.view([-1, x.size(-1)]), expk0, expk1, buf0, buf1, out, torch.get_num_threads())
    return out.view(x.size())  

class IDCST2Function(Function):
    @staticmethod
    def forward(ctx, x, expk0, expk1, buf0, buf1, out):
        return idcst2(x, expk0, expk1, buf0, buf1, out)

class IDCST2(nn.Module):
    def __init__(self, expk0=None, expk1=None):
        super(IDCST2, self).__init__()
        self.expk0 = expk0
        self.expk1 = expk1
        self.buf0 = None
        self.buf1 = None
        self.out = None
    def forward(self, x): 
        if self.expk0 is None or self.expk0.size(-1) != x.size(-2):
            self.expk0 = torch.empty(x.size(-2), dtype=x.dtype, device=x.device)
            if x.is_cuda: 
                dct_cuda.precompute_idct_cos(x.size(-2), self.expk0)
            else:
                dct_cpp.precompute_idct_cos(x.size(-2), self.expk0)
        if self.expk1 is None or self.expk1.size(-1) != x.size(-1):
            self.expk1 = torch.empty(x.size(-2), dtype=x.dtype, device=x.device)
            if x.is_cuda: 
                dct_cuda.precompute_idct_cos(x.size(-1), self.expk1)
            else:
                dct_cpp.precompute_idct_cos(x.size(-1), self.expk1)
        if self.out is None or self.out.size() != x.size():
            self.buf0 = torch.empty_like(x)
            self.buf1 = torch.empty_like(x)
            self.out = torch.empty_like(x)
        return IDCST2Function.apply(x, self.expk0, self.expk1, self.buf0, self.buf1, self.out)

def idsct2(x, expk0, expk1, buf0, buf1, out):
    """compute inverse discrete cosine-sine transformation
    This is equivalent to idxst(idxct(x)^T)^T
    """
    if x.is_cuda:
        dct_cuda.idsct2(x.view([-1, x.size(-1)]), expk0, expk1, buf0, buf1, out)
    else:
        dct_cpp.idsct2(x.view([-1, x.size(-1)]), expk0, expk1, buf0, buf1, out, torch.get_num_threads())
    return out.view(x.size())  

class IDSCT2Function(Function):
    @staticmethod
    def forward(ctx, x, expk0, expk1, buf0, buf1, out):
        return idsct2(x, expk0, expk1, buf0, buf1, out)

class IDSCT2(nn.Module):
    def __init__(self, expk0=None, expk1=None):
        super(IDSCT2, self).__init__()
        self.expk0 = expk0
        self.expk1 = expk1
        self.buf0 = None
        self.buf1 = None
        self.out = None
    def forward(self, x): 
        if self.expk0 is None or self.expk0.size(-1) != x.size(-2):
            self.expk0 = torch.empty(x.size(-2), dtype=x.dtype, device=x.device)
            if x.is_cuda: 
                dct_cuda.precompute_idct_cos(x.size(-2), self.expk0)
            else:
                dct_cpp.precompute_idct_cos(x.size(-2), self.expk0)
        if self.expk1 is None or self.expk1.size(-1) != x.size(-1):
            self.expk1 = torch.empty(x.size(-2), dtype=x.dtype, device=x.device)
            if x.is_cuda: 
                dct_cuda.precompute_idct_cos(x.size(-1), self.expk1)
            else:
                dct_cpp.precompute_idct_cos(x.size(-1), self.expk1)
        if self.out is None or self.out.size() != x.size():
            self.buf0 = torch.empty_like(x)
            self.buf1 = torch.empty_like(x)
            self.out = torch.empty_like(x)
        return IDSCT2Function.apply(x, self.expk0, self.expk1, self.buf0, self.buf1, self.out)

