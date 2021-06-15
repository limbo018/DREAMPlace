##
# @file   discrete_spectral_transform.py
# @author Yibo Lin
# @date   Jun 2018
#

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import pdb
import dreamplace.ops.dct.torch_fft_api as torch_fft_api

""" Discrete spectral transformation leveraging fast fourier transform engine.
The math here mainly uses Prosthaphaeresis properties.
The trigonometric identities exploited by prosthaphaeresis relate products of trigonometric functions to sums.
sin(a) sin(b) = 1/2 * (cos(a-b) - cos(a+b))
cos(a) cos(b) = 1/2 * (cos(a-b) + cos(a+b))
sin(a) cos(b) = 1/2 * (sin(a+b) + sin(a-b))
cos(a) sin(b) = 1/2 * (sin(a-b) - sin(a+b))

A 2D FFT performs
y_{u, v} = \sum_i \sum_j x_{i, j} exp(-j*2*pi*u*i/M) exp(-j*2*pi*v*j/N)
         = \sum_i \sum_j x_{i, j} exp(-j*2*pi*(u*i/M + v*j/N))
         = \sum_i \sum_j x_{i, j} (cos(-2*pi*(u*i/M + v*j/N)) + j sin(-2*pi*(u*i/M + v*j/N))).

By mapping the original image from (i, j) to (i, N-j), we can have (u*i/M - v*j/N) inside exp.
This will enable us to derive various cos/sin transformation by computing FFT twice.
"""

def get_expk(N, dtype, device):
    """ Compute 2*exp(-1j*pi*u/(2N)), but not exactly the same.
    The actual return is 2*cos(pi*u/(2N)), 2*sin(pi*u/(2N)).
    This will make later multiplication easier.
    """
    pik_by_2N = torch.arange(N, dtype=dtype, device=device)
    pik_by_2N.mul_(np.pi/(2*N))
    # cos, sin
    # I use sin because the real part requires subtraction
    # this will be easier for multiplication
    expk = torch.stack([pik_by_2N.cos(), pik_by_2N.sin()], dim=-1)
    expk.mul_(2)

    return expk.contiguous()


def get_expkp1(N, dtype, device):
    """ Compute 2*exp(-1j*pi*(u+1)/(2N)), but not exactly the same.
    The actual return is 2*cos(pi*(u+1)/(2N)), 2*sin(pi*(u+1)/(2N))
    """
    neg_pik_by_2N = torch.arange(1, N+1, dtype=dtype, device=device)
    neg_pik_by_2N.mul_(np.pi/(2*N))
    # sin, -cos
    # I swap -cos and sin because we need the imag part
    # this will be easier for multiplication
    expk = torch.stack([neg_pik_by_2N.cos(), neg_pik_by_2N.sin()], dim=-1)
    expk.mul_(2)

    return expk.contiguous()


def get_exact_expk(N, dtype, device):
    # Compute exp(-j*pi*u/(2N)) = cos(pi*u/(2N)) - j * sin(pi*u/(2N))
    pik_by_2N = torch.arange(N, dtype=dtype, device=device)
    pik_by_2N.mul_(np.pi/(2*N))
    # cos, -sin
    expk = torch.stack([pik_by_2N.cos(), -pik_by_2N.sin()], dim=-1)
    return expk.contiguous()


def get_perm(N, dtype, device):
    """ Compute permutation to generate following array
    0, 2, 4, ..., 2*(N//2)-2, 2*(N//2)-1, 2*(N//2)-3, ..., 3, 1
    """
    perm = torch.zeros(N, dtype=dtype, device=device)
    perm[0:(N-1)//2+1] = torch.arange(0, N, 2, dtype=dtype, device=device)
    perm[(N-1)//2+1:] = torch.arange(2*(N//2)-1, 0, -2, dtype=dtype, device=device)

    return perm

def dct_2N(x, expk=None):
    """ Batch Discrete Cosine Transformation without normalization to coefficients.
    Compute y_u = \sum_i  x_i cos(pi*(2i+1)*u/(2N)),
    Impelements the 2N padding trick to solve DCT with FFT in the following link,
    https://dsp.stackexchange.com/questions/2807/fast-cosine-transform-via-fft

    1. Pad x by zeros
    2. Perform FFT
    3. Multiply by 2*exp(-1j*pi*u/(2N))
    4. Extract the real part
    """
    # last dimension
    N = x.size(-1)
    # pad last dimension
    x_pad = F.pad(x, (0, N), 'constant', 0)

    # the last dimension here becomes -2 because complex numbers introduce a new dimension
    y = torch_fft_api.rfft(x_pad, signal_ndim=1, normalized=False, onesided=True)[..., 0:N, :]
    y.mul_(1.0/N)

    if expk is None:
        expk = get_expk(N, dtype=x.dtype, device=x.device)

    # get real part
    y.mul_(expk)

    # I found add is much faster than sum
    #y = y.sum(dim=-1)
    return y[..., 0]+y[..., 1]


def dct_N(x, perm=None, expk=None):
    """ Batch Discrete Cosine Transformation without normalization to coefficients.
    Compute y_u = \sum_i  x_i cos(pi*(2i+1)*u/(2N)),
    Impelements the N permuting trick to solve DCT with FFT in the following link,
    https://dsp.stackexchange.com/questions/2807/fast-cosine-transform-via-fft

    1. permute x such that [a, b, c, d, e, f] becomes [a, c, e, f, d, b]
    2. Perform FFT
    3. Multiply by 2*exp(-1j*pi*u/(2N))
    4. Extract the real part
    """
    # last dimension
    N = x.size(-1)

    if perm is None:
        perm = get_perm(N, dtype=torch.int64, device=x.device)
    if x.ndimension() <= 1:
        x_reorder = x.view([1, N])
    else:
        x_reorder = x.clone()
    # switch from row-major to column-major for speedup
    x_reorder.transpose_(dim0=-2, dim1=-1)
    #x_reorder = x_reorder[..., perm, :]
    x_reorder = x_reorder.index_select(dim=-2, index=perm)
    # switch back
    x_reorder.transpose_(dim0=-2, dim1=-1)

    y = torch_fft_api.rfft(x_reorder, signal_ndim=1, normalized=False, onesided=False)[..., 0:N, :]
    y.mul_(1.0/N)

    if expk is None:
        expk = get_expk(N, dtype=x.dtype, device=x.device)

    # get real part
    y.mul_(expk)
    # I found add is much faster than sum
    #y = y.sum(dim=-1)
    return y[..., 0]+y[..., 1]


def idct_2N(x, expk=None):
    """ Batch Inverse Discrete Cosine Transformation without normalization to coefficients.
    Compute y_u = \sum_i  x_i cos(pi*(2u+1)*i/(2N)),
    Impelements the 2N padding trick to solve IDCT with IFFT in the following link,
    https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/ops/spectral_ops.py

    1. Multiply by 2*exp(1j*pi*u/(2N))
    2. Pad x by zeros
    3. Perform IFFT
    4. Extract the real part
    """
    # last dimension
    N = x.size(-1)

    if expk is None:
        expk = get_expk(N, dtype=x.dtype, device=x.device)

    # multiply by 2*exp(1j*pi*u/(2N))
    x_pad = x.unsqueeze(-1).mul(expk)
    # pad second last dimension, excluding the complex number dimension
    x_pad = F.pad(x_pad, (0, 0, 0, N), 'constant', 0)

    if len(x.size()) == 1:
        x_pad.unsqueeze_(0)

    # the last dimension here becomes -2 because complex numbers introduce a new dimension
    y = torch_fft_api.irfft(x_pad, signal_ndim=1, normalized=False, onesided=False, signal_sizes=[2*N])[..., 0:N]
    y.mul_(N)

    if len(x.size()) == 1:
        y.squeeze_(0)

    return y


def idct_N(x, expk=None):
    N = x.size(-1)

    if expk is None:
        expk = get_expk(N, dtype=x.dtype, device=x.device)

    size = list(x.size())
    size.append(2)
    x_reorder = torch.zeros(size, dtype=x.dtype, device=x.device)
    x_reorder[..., 0] = x
    x_reorder[..., 1:, 1] = x.flip([x.ndimension()-1])[..., :N-1].mul_(-1)

    x_reorder[..., 0] = x.mul(expk[..., 0]).sub_(x_reorder[..., 1].mul(expk[..., 1]))
    x_reorder[..., 1].mul_(expk[..., 0])
    x_reorder[..., 1].add_(x.mul(expk[..., 1]))
    # this is to match idct_2N
    # normal way should multiply 0.25
    x_reorder.mul_(0.5)

    y = torch_fft_api.ifft(x_reorder, signal_ndim=1, normalized=False)
    y.mul_(N)

    z = torch.empty_like(x)
    z[..., 0:N:2] = y[..., :(N+1)//2, 0]
    z[..., 1:N:2] = y[..., (N+1)//2:, 0].flip([x.ndimension()-1])

    return z


def dst(x, expkp1=None):
    """ Batch Discrete Sine Transformation without normalization to coefficients.
    Compute y_u = \sum_i  x_i sin(pi*(2i+1)*(u+1)/(2N)),
    Impelements the 2N padding trick to solve DCT with FFT in the following link,
    https://dsp.stackexchange.com/questions/2807/fast-cosine-transform-via-fft

    1. Pad x by zeros
    2. Perform FFT
    3. Multiply by 2*exp(-1j*pi*u/(2N))
    4. Extract the real part
    """
    # last dimension
    N = x.size(-1)
    # pad last dimension
    x_pad = F.pad(x, (0, N), 'constant', 0)

    # the last dimension here becomes -2 because complex numbers introduce a new dimension
    y = torch_fft_api.rfft(x_pad, signal_ndim=1, normalized=False, onesided=True)[..., 1:N+1, :]

    if expkp1 is None:
        expkp1 = get_expkp1(N, dtype=x.dtype, device=x.device)

    # get imag part
    y = y[..., 1].mul(expkp1[:, 0]) - y[..., 0].mul(expkp1[:, 1])

    return y


def idst(x, expkp1=None):
    """ Batch Inverse Discrete Sine Transformation without normalization to coefficients.
    Compute y_u = \sum_i  x_i cos(pi*(2u+1)*i/(2N)),
    Impelements the 2N padding trick to solve IDCT with IFFT in the following link,
    https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/ops/spectral_ops.py

    1. Multiply by 2*exp(1j*pi*u/(2N))
    2. Pad x by zeros
    3. Perform IFFT
    4. Extract the real part
    """
    # last dimension
    N = x.size(-1)

    if expkp1 is None:
        expkp1 = get_expkp1(N, dtype=x.dtype, device=x.device)

    # multiply by 2*exp(1j*pi*u/(2N))
    x_pad = x.unsqueeze(-1).mul(expkp1)
    # pad second last dimension, excluding the complex number dimension
    x_pad = F.pad(x_pad, (0, 0, 0, N), 'constant', 0)

    if len(x.size()) == 1:
        x_pad.unsqueeze_(0)

    # the last dimension here becomes -2 because complex numbers introduce a new dimension
    y = torch_fft_api.irfft(x_pad, signal_ndim=1, normalized=False, onesided=False, signal_sizes=[2*N])[..., 1:N+1]
    y.mul_(N)

    if len(x.size()) == 1:
        y.squeeze_(0)

    return y


def idxt(x, cos_or_sin_flag, expk=None):
    """ Batch Inverse Discrete Cosine Transformation without normalization to coefficients.
    Compute y_u = \sum_i  x_i cos(pi*(2u+1)*i/(2N)),
    Impelements the 2N padding trick to solve IDCT with IFFT in the following link,
    https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/ops/spectral_ops.py

    1. Multiply by 2*exp(1j*pi*u/(2N))
    2. Pad x by zeros
    3. Perform IFFT
    4. Extract the real part

    @param x batch 1D tensor for conversion
    @param cos_or_sin_flag 0 for cosine tranformation and 1 or sine transformation
    @param expk 2*exp(j*pi*k/(2N))
    """
    # last dimension
    N = x.size(-1)

    if expk is None:
        expk = get_expk(N, dtype=x.dtype, device=x.device)

    # multiply by 2*exp(1j*pi*u/(2N))
    x_pad = x.unsqueeze(-1).mul(expk)
    # pad second last dimension, excluding the complex number dimension
    x_pad = F.pad(x_pad, (0, 0, 0, N), 'constant', 0)

    if len(x.size()) == 1:
        x_pad.unsqueeze_(0)

    # the last dimension here becomes -2 because complex numbers introduce a new dimension
    # Must use IFFT here
    y = torch_fft_api.ifft(x_pad, signal_ndim=1, normalized=False)[..., 0:N, cos_or_sin_flag]
    y.mul_(N)

    if len(x.size()) == 1:
        y.squeeze_(0)

    return y


def dct2_2N(x, expk0=None, expk1=None):
    """ Batch 2D Discrete Cosine Transformation without normalization to coefficients.
    Compute 1D DCT twice.
    @param x batch tensor, the 2D part is MxN
    @param expk0 with length M
    @param expk1 with length N
    """
    return dct_2N(dct_2N(x.transpose(dim0=-2, dim1=-1), expk0).transpose_(dim0=-2, dim1=-1), expk1)


def dct2_N(x, perm0=None, expk0=None, perm1=None, expk1=None):
    """ Batch 2D Discrete Cosine Transformation without normalization to coefficients.
    Compute 1D DCT twice.
    @param x batch tensor, the 2D part is MxN
    @param perm0 with length M
    @param expk0 with length M
    @param perm1 with length N
    @param expk1 with length N
    """
    return dct_N(dct_N(x.transpose(dim0=-2, dim1=-1), perm=perm0, expk=expk0).transpose_(dim0=-2, dim1=-1), perm=perm1, expk=expk1)


def idct2_2N(x, expk0=None, expk1=None):
    """ Batch 2D Discrete Cosine Transformation without normalization to coefficients.
    Compute 1D DCT twice.
    @param x batch tensor, the 2D part is MxN
    @param expk0 with length M
    @param expk1 with length N
    """
    return idct_2N(idct_2N(x.transpose(dim0=-2, dim1=-1), expk0).transpose_(dim0=-2, dim1=-1), expk1)


def idct2_N(x, expk0=None, expk1=None):
    """ Batch 2D Discrete Cosine Transformation without normalization to coefficients.
    Compute 1D DCT twice.
    @param x batch tensor, the 2D part is MxN
    @param expk0 with length M
    @param expk1 with length N
    """
    return idct_N(idct_N(x.transpose(dim0=-2, dim1=-1), expk0).transpose_(dim0=-2, dim1=-1), expk1)


def dst2(x, expkp1_0=None, expkp1_1=None):
    """ Batch 2D Discrete Sine Transformation without normalization to coefficients.
    Compute 1D DST twice.
    @param x batch tensor, the 2D part is MxN
    @param expkp1_0 with length M
    @param expkp1_1 with length N
    """
    return dst(dst(x.transpose(dim0=-2, dim1=-1), expkp1_0).transpose_(dim0=-2, dim1=-1), expkp1_1)


def idcct2(x, expk_0=None, expk_1=None):
    """ Batch 2D Inverse Discrete Cosine-Cosine Transformation without normalization to coefficients.
    It computes following equation, which is slightly different from standard DCT formulation.
    y_{u, v} = \sum_p \sum_q x_{p, q} cos(pi/M*p*(u+0.5)) cos(pi/N*q*(v+0.5))
    Compute 1D DCT twice.
    @param x batch tensor, the 2D part is MxN
    @param expk_0 with length M, 2*exp(-1j*pi*k/(2M))
    @param expk_1 with length N, 2*exp(-1j*pi*k/(2N))
    """
    return idxt(idxt(x, 0, expk_1).transpose_(dim0=-2, dim1=-1), 0, expk_0).transpose(dim0=-2, dim1=-1)
    # return idxt(idxt(x.transpose(dim0=-2, dim1=-1), 0, expk_0).transpose_(dim0=-2, dim1=-1), 0, expk_1)


def idsct2(x, expk_0=None, expk_1=None):
    """ Batch 2D Inverse Discrete Sine-Cosine Transformation without normalization to coefficients.
    It computes following equation, which is slightly different from standard DCT formulation.
    y_{u, v} = \sum_p \sum_q x_{p, q} sin(pi/M*p*(u+0.5)) cos(pi/N*q*(v+0.5))
    Compute 1D DST and then 1D DCT.
    @param x batch tensor, the 2D part is MxN
    @param expk_0 with length M, 2*exp(-1j*pi*k/(2M))
    @param expk_1 with length N, 2*exp(-1j*pi*k/(2N))
    """
    return idxt(idxt(x, 0, expk_1).transpose_(dim0=-2, dim1=-1), 1, expk_0).transpose_(dim0=-2, dim1=-1)
    # return idxt(idxt(x.transpose(dim0=-2, dim1=-1), 1, expk_0).transpose_(dim0=-2, dim1=-1), 0, expk_1)


def idcst2(x, expk_0=None, expk_1=None):
    """ Batch 2D Inverse Discrete Cosine-Sine Transformation without normalization to coefficients.
    It computes following equation, which is slightly different from standard DCT formulation.
    y_{u, v} = \sum_p \sum_q x_{p, q} cos(pi/M*p*(u+0.5)) sin(pi/N*q*(v+0.5))
    Compute 1D DCT and then 1D DST.
    @param x batch tensor, the 2D part is MxN
    @param expk_0 with length M, 2*exp(-1j*pi*k/(2M))
    @param expk_1 with length N, 2*exp(-1j*pi*k/(2N))
    """
    return idxt(idxt(x, 1, expk_1).transpose_(dim0=-2, dim1=-1), 0, expk_0).transpose_(dim0=-2, dim1=-1)
    # return idxt(idxt(x.transpose(dim0=-2, dim1=-1), 0, expk_0).transpose_(dim0=-2, dim1=-1), 1, expk_1)


def idxst_idct(x, expk_0=None, expk_1=None):
    '''
    Batch 2D Inverse Discrete Sine-Cosine Transformation without normalization to coefficients.
    Compute idxst(idct(x))
    @param x batch tensor, the 2D part is MxN
    @param expk_0 with length M, 2*exp(-1j*pi*k/(2M))
    @param expk_1 with length N, 2*exp(-1j*pi*k/(2N))
    '''
    return idxt(idct_N(x, expk_1).transpose_(dim0=-2, dim1=-1), 1, expk_0).transpose_(dim0=-2, dim1=-1)


def idct_idxst(x, expk_0=None, expk_1=None):
    '''
    Batch 2D Inverse Discrete Cosine-Sine Transformation without normalization to coefficients.
    Compute idct(idxst(x)).
    @param x batch tensor, the 2D part is MxN
    @param expk_0 with length M, 2*exp(-1j*pi*k/(2M))
    @param expk_1 with length N, 2*exp(-1j*pi*k/(2N))
    '''
    return idct_N(idxt(x, 1, expk_1).transpose_(dim0=-2, dim1=-1), expk_0).transpose_(dim0=-2, dim1=-1)
