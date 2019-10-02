##
# @file   dct_unitest.py
# @author Yibo Lin
# @date   Mar 2019
#

import pdb
import os
import sys
import numpy as np
import unittest

import torch
from torch.autograd import Function, Variable
import time
import scipy
from scipy import fftpack

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dreamplace.ops.dct import dct2_fft2
from dreamplace.ops.dct import discrete_spectral_transform
from dreamplace.ops.dct import dct_lee
from dreamplace.ops.dct import dct
sys.path.pop()

dtype = torch.float32


class DCTOpTest(unittest.TestCase):
    def test_dctRandom(self):
        N = 4
        x = torch.empty(N, N, dtype=dtype).uniform_(0, 10.0)
        #x = Variable(torch.tensor([[1, 2, 7, 9, 20, 31], [4, 5, 9, 2, 1, 6]], dtype=dtype))

        golden_value = discrete_spectral_transform.dct_2N(x).data.numpy()
        print("golden_value")
        print(golden_value)

        # test cpu using N-FFT
        # pdb.set_trace()
        custom = dct.DCT(algorithm='N')
        dct_value = custom.forward(x)
        print("dct_value")
        print(dct_value.data.numpy())

        np.testing.assert_allclose(dct_value.data.numpy(), golden_value, rtol=1e-6, atol=1e-5)

        # test cpu using 2N-FFT
        # pdb.set_trace()
        custom = dct.DCT(algorithm='2N')
        dct_value = custom.forward(x)
        print("dct_value")
        print(dct_value.data.numpy())

        np.testing.assert_allclose(dct_value.data.numpy(), golden_value, rtol=1e-6, atol=1e-5)

        # test cpu using dct_lee
        # pdb.set_trace()
        custom = dct_lee.DCT()
        dct_value = custom.forward(x)
        print("dct_value")
        print(dct_value.data.numpy())

        np.testing.assert_allclose(dct_value.data.numpy(), golden_value, rtol=1e-6, atol=1e-5)

        if torch.cuda.device_count():
            # test gpu
            custom = dct.DCT(algorithm='N')
            dct_value = custom.forward(x.cuda()).cpu()
            print("dct_value cuda")
            print(dct_value.data.numpy())

            np.testing.assert_allclose(dct_value.data.numpy(), golden_value, rtol=1e-6, atol=1e-5)

            # test gpu
            custom = dct.DCT(algorithm='2N')
            dct_value = custom.forward(x.cuda()).cpu()
            print("dct_value cuda")
            print(dct_value.data.numpy())

            np.testing.assert_allclose(dct_value.data.numpy(), golden_value, rtol=1e-6, atol=1e-5)

            # test gpu
            custom = dct_lee.DCT()
            dct_value = custom.forward(x.cuda()).cpu()
            print("dct_value cuda")
            print(dct_value.data.numpy())

            np.testing.assert_allclose(dct_value.data.numpy(), golden_value, rtol=1e-6, atol=1e-5)

            #golden_value = discrete_spectral_transform.dct2_2N(x).data.numpy()
            #print("2D golden_value")
            # print(golden_value)

            #custom = dct.DCT()
            #dct2_value = custom.forward(dct_value.cuda().t().contiguous()).cpu()
            #dct2_value = dct2_value.t().contiguous()
            #print("dct2_value cuda")
            # print(dct2_value.data.numpy())

            #np.testing.assert_allclose(dct2_value.data.numpy(), golden_value)

    def test_idctRandom(self):
        N = 4
        x = torch.empty(N, N, dtype=dtype).uniform_(0, 10.0)
        #x = Variable(torch.tensor([[1, 2, 7, 9, 20, 31], [4, 5, 9, 2, 1, 6]], dtype=dtype))
        print("x")
        print(x)

        y = discrete_spectral_transform.dct_N(x)
        print("y")
        print(y.data.numpy())

        golden_value = discrete_spectral_transform.idct_2N(y).data.numpy()
        print("2D golden_value")
        print(golden_value)

        # test cpu use N-FFT
        # pdb.set_trace()
        custom = dct.IDCT(algorithm='N')
        dct_value = custom.forward(y)
        print("idct_value")
        print(dct_value.data.numpy())

        np.testing.assert_allclose(dct_value.data.numpy(), golden_value, rtol=1e-5)

        # test cpu use 2N-FFT
        # pdb.set_trace()
        custom = dct.IDCT(algorithm='2N')
        dct_value = custom.forward(y)
        print("idct_value")
        print(dct_value.data.numpy())

        np.testing.assert_allclose(dct_value.data.numpy(), golden_value, rtol=1e-5)

        # test cpu use dct_lee
        # pdb.set_trace()
        custom = dct_lee.IDCT()
        dct_value = custom.forward(y)
        print("idct_value")
        print(dct_value.data.numpy())

        np.testing.assert_allclose(dct_value.data.numpy(), golden_value, rtol=1e-5)

        if torch.cuda.device_count():
            # test gpu
            custom = dct.IDCT(algorithm='N')
            dct_value = custom.forward(y.cuda()).cpu()
            print("idct_value cuda")
            print(dct_value.data.numpy())

            np.testing.assert_allclose(dct_value.data.numpy(), golden_value, rtol=1e-5)

            # test gpu
            custom = dct.IDCT(algorithm='2N')
            dct_value = custom.forward(y.cuda()).cpu()
            print("idct_value cuda")
            print(dct_value.data.numpy())

            np.testing.assert_allclose(dct_value.data.numpy(), golden_value, rtol=1e-5)

            # test gpu
            custom = dct_lee.IDCT()
            dct_value = custom.forward(y.cuda()).cpu()
            print("idct_value cuda")
            print(dct_value.data.numpy())

            np.testing.assert_allclose(dct_value.data.numpy(), golden_value, rtol=1e-5)

    def test_dct2Random(self):
        torch.manual_seed(10)
        M = 4
        N = 8
        x = torch.empty(M, N, dtype=dtype).uniform_(0, 10.0)
        expkM = discrete_spectral_transform.get_exact_expk(M, dtype=x.dtype, device=x.device)
        expkN = discrete_spectral_transform.get_exact_expk(N, dtype=x.dtype, device=x.device)

        golden_value = discrete_spectral_transform.dct2_N(x).data.numpy()
        print("2D DCT golden_value")
        print(golden_value)

        # test cpu using N-FFT
        # pdb.set_trace()
        custom = dct.DCT2(algorithm='N')
        dct_value = custom.forward(x)
        print("2D dct_value")
        print(dct_value.data.numpy())

        np.testing.assert_allclose(dct_value.data.numpy(), golden_value, rtol=1e-6, atol=1e-5)

        # test cpu using 2N-FFT
        # pdb.set_trace()
        custom = dct.DCT2(algorithm='2N')
        dct_value = custom.forward(x)
        print("2D dct_value")
        print(dct_value.data.numpy())

        np.testing.assert_allclose(dct_value.data.numpy(), golden_value, rtol=1e-6, atol=1e-5)

        # test cpu using dct_lee
        # pdb.set_trace()
        custom = dct_lee.DCT2()
        dct_value = custom.forward(x)
        print("2D dct_value")
        print(dct_value.data.numpy())

        np.testing.assert_allclose(dct_value.data.numpy(), golden_value, rtol=1e-6, atol=1e-5)

        # test cpu using fft2
        custom = dct2_fft2.DCT2(expkM, expkN)
        dct_value = custom.forward(x)
        print("2D dct_value")
        print(dct_value.data.numpy())

        np.testing.assert_allclose(dct_value.data.numpy(), golden_value, rtol=1e-6, atol=1e-5)

        if torch.cuda.device_count():
            # test gpu
            custom = dct.DCT2(algorithm='N')
            dct_value = custom.forward(x.cuda()).cpu()
            print("2D dct_value cuda")
            print(dct_value.data.numpy())

            np.testing.assert_allclose(dct_value.data.numpy(), golden_value, rtol=1e-6, atol=1e-5)

            # test gpu
            custom = dct.DCT2(algorithm='2N')
            dct_value = custom.forward(x.cuda()).cpu()
            print("2D dct_value cuda")
            print(dct_value.data.numpy())

            np.testing.assert_allclose(dct_value.data.numpy(), golden_value, rtol=1e-6, atol=1e-5)

            # test gpu
            custom = dct_lee.DCT2()
            dct_value = custom.forward(x.cuda()).cpu()
            print("2D dct_value cuda")
            print(dct_value.data.numpy())

            np.testing.assert_allclose(dct_value.data.numpy(), golden_value, rtol=1e-6, atol=1e-5)

            # test gpu using fft2
            custom = dct2_fft2.DCT2(expkM.cuda(), expkN.cuda())
            dct_value = custom.forward(x.cuda()).cpu()
            print("2D dct_value cuda")
            print(dct_value.data.numpy())

            np.testing.assert_allclose(dct_value.data.numpy(), golden_value, rtol=1e-6, atol=1e-5)

    def test_idct2Random(self):
        torch.manual_seed(10)
        M = 4
        N = 8
        x = torch.empty(M, N, dtype=torch.int32).random_(0, 10).double()
        print("2D x")
        print(x)

        expkM = discrete_spectral_transform.get_exact_expk(M, dtype=x.dtype, device=x.device)
        expkN = discrete_spectral_transform.get_exact_expk(N, dtype=x.dtype, device=x.device)

        y = discrete_spectral_transform.dct2_2N(x)

        golden_value = discrete_spectral_transform.idct2_2N(y).data.numpy()
        print("2D idct golden_value")
        print(golden_value)

        # test cpu using N-FFT
        # pdb.set_trace()
        custom = dct.IDCT2(algorithm='N')
        dct_value = custom.forward(y)
        print("2D idct_value")
        print(dct_value.data.numpy())

        np.testing.assert_allclose(dct_value.data.numpy(), golden_value, rtol=1e-6, atol=1e-5)

        # test cpu using 2N-FFT
        # pdb.set_trace()
        custom = dct.IDCT2(algorithm='2N')
        dct_value = custom.forward(y)
        print("2D idct_value")
        print(dct_value.data.numpy())

        np.testing.assert_allclose(dct_value.data.numpy(), golden_value, rtol=1e-6, atol=1e-5)

        # test cpu using dct_lee
        # pdb.set_trace()
        custom = dct_lee.IDCT2()
        dct_value = custom.forward(y)
        print("2D idct_value")
        print(dct_value.data.numpy())

        np.testing.assert_allclose(dct_value.data.numpy(), golden_value, rtol=1e-6, atol=1e-5)

        # test cpu using fft2
        custom = dct2_fft2.IDCT2(expkM, expkN)
        dct_value = custom.forward(y)
        print("2D idct_value cuda")
        print(dct_value.data.numpy())

        np.testing.assert_allclose(dct_value.data.numpy(), golden_value, rtol=1e-6, atol=1e-5)

        if torch.cuda.device_count():
            # test gpu
            custom = dct.IDCT2(algorithm='N')
            dct_value = custom.forward(y.cuda()).cpu()
            print("2D idct_value cuda")
            print(dct_value.data.numpy())

            np.testing.assert_allclose(dct_value.data.numpy(), golden_value, rtol=1e-6, atol=1e-5)

            # test gpu
            custom = dct.IDCT2(algorithm='2N')
            dct_value = custom.forward(y.cuda()).cpu()
            print("2D idct_value cuda")
            print(dct_value.data.numpy())

            np.testing.assert_allclose(dct_value.data.numpy(), golden_value, rtol=1e-6, atol=1e-5)

            # test gpu
            custom = dct_lee.IDCT2()
            dct_value = custom.forward(y.cuda()).cpu()
            print("2D idct_value cuda")
            print(dct_value.data.numpy())

            np.testing.assert_allclose(dct_value.data.numpy(), golden_value, rtol=1e-6, atol=1e-5)

            # test gpu using ifft2
            custom = dct2_fft2.IDCT2(expkM.cuda(), expkN.cuda())
            dct_value = custom.forward(y.cuda()).cpu()
            print("2D idct_value cuda")
            print(dct_value.data.numpy())

            np.testing.assert_allclose(dct_value.data.numpy(), golden_value, rtol=1e-6, atol=1e-5)

    def test_idxct2Random(self):
        torch.manual_seed(10)
        M = 4
        N = 8
        x = torch.empty(M, N, dtype=torch.int32).random_(0, 10).double()
        print("2D x")
        print(x)

        golden_value = discrete_spectral_transform.idxt(x, 0).data.numpy()
        print("2D golden_value")
        print(golden_value)

        # test cpu
        # pdb.set_trace()
        custom = dct.IDXCT()
        dct_value = custom.forward(x)
        print("dxt_value")
        print(dct_value.data.numpy())

        np.testing.assert_allclose(dct_value.data.numpy(), golden_value, atol=1e-14)

        # test cpu
        # pdb.set_trace()
        custom = dct_lee.IDXCT()
        dct_value = custom.forward(x)
        print("dxt_value")
        print(dct_value.data.numpy())

        np.testing.assert_allclose(dct_value.data.numpy(), golden_value, atol=1e-14)

        if torch.cuda.device_count():
            # test gpu
            custom = dct.IDXCT()
            dct_value = custom.forward(x.cuda()).cpu()
            print("dxt_value cuda")
            print(dct_value.data.numpy())

            np.testing.assert_allclose(dct_value.data.numpy(), golden_value, atol=1e-14)

            # test gpu
            custom = dct_lee.IDXCT()
            dct_value = custom.forward(x.cuda()).cpu()
            print("dxt_value cuda")
            print(dct_value.data.numpy())

            np.testing.assert_allclose(dct_value.data.numpy(), golden_value, atol=1e-14)


class DSTOpTest(unittest.TestCase):
    def test_dstRandom(self):
        N = 4
        x = torch.empty(N, N, dtype=dtype).uniform_(0, 10.0)
        #x = Variable(torch.tensor([[1, 2, 7, 9, 20, 31], [4, 5, 9, 2, 1, 6]], dtype=dtype))
        import scipy
        from scipy import fftpack

        #golden_value = discrete_spectral_transform.dst(x).data.numpy()
        golden_value = torch.from_numpy(fftpack.dst(x.data.numpy())).data.numpy() / N
        print("golden_value")
        print(golden_value)

        # test cpu
        # pdb.set_trace()
        custom = dct.DST()
        dst_value = custom.forward(x)
        print("dst_value")
        print(dst_value.data.numpy())

        np.testing.assert_allclose(dst_value.data.numpy(), golden_value, rtol=1e-5)

        # test cpu
        # pdb.set_trace()
        custom = dct_lee.DST()
        dst_value = custom.forward(x)
        print("dst_value")
        print(dst_value.data.numpy())

        np.testing.assert_allclose(dst_value.data.numpy(), golden_value, rtol=1e-5)

        if torch.cuda.device_count():
            # test gpu
            custom = dct.DST()
            dst_value = custom.forward(x.cuda()).cpu()
            print("dst_value cuda")
            print(dst_value.data.numpy())

            np.testing.assert_allclose(dst_value.data.numpy(), golden_value, rtol=1e-5)

            # test gpu
            custom = dct_lee.DST()
            dst_value = custom.forward(x.cuda()).cpu()
            print("dst_value cuda")
            print(dst_value.data.numpy())

            np.testing.assert_allclose(dst_value.data.numpy(), golden_value, rtol=1e-5)

    def test_idstRandom(self):
        N = 4
        x = torch.empty(N, N, dtype=dtype).uniform_(0, 10.0)
        #x = Variable(torch.tensor([[1, 2, 7, 9, 20, 31], [4, 5, 9, 2, 1, 6]], dtype=dtype))
        print("x")
        print(x)
        import scipy
        from scipy import fftpack

        #y = discrete_spectral_transform.dst(x)
        y = torch.from_numpy(fftpack.dst(x.data.numpy()))
        print("y")
        print(y.data.numpy())

        #golden_value = discrete_spectral_transform.idst(y).data.numpy()
        golden_value = torch.from_numpy(fftpack.idst(y.data.numpy())).data.numpy()
        print("2D golden_value")
        print(golden_value)

        # test cpu
        # pdb.set_trace()
        custom = dct.IDST()
        dst_value = custom.forward(y)
        print("idst_value")
        print(dst_value.data.numpy())

        np.testing.assert_allclose(dst_value.data.numpy(), golden_value, rtol=1e-5)

        # test cpu
        # pdb.set_trace()
        custom = dct_lee.IDST()
        dst_value = custom.forward(y)
        print("idst_value")
        print(dst_value.data.numpy())

        np.testing.assert_allclose(dst_value.data.numpy(), golden_value, rtol=1e-5)

        if torch.cuda.device_count():
            # test gpu
            custom = dct.IDST()
            dst_value = custom.forward(y.cuda()).cpu()
            print("idst_value cuda")
            print(dst_value.data.numpy())

            np.testing.assert_allclose(dst_value.data.numpy(), golden_value, rtol=1e-5)

            # test gpu
            custom = dct_lee.IDST()
            dst_value = custom.forward(y.cuda()).cpu()
            print("idst_value cuda")
            print(dst_value.data.numpy())

            np.testing.assert_allclose(dst_value.data.numpy(), golden_value, rtol=1e-5)

    def test_idxst2Random(self):
        torch.manual_seed(10)
        M = 4
        N = 8
        x = torch.empty(M, N, dtype=torch.int32).random_(0, 10).double()
        print("2D x")
        print(x)

        golden_value = discrete_spectral_transform.idxt(x, 1).data.numpy()
        print("2D golden_value")
        print(golden_value)

        # test cpu
        # pdb.set_trace()
        custom = dct.IDXST()
        dst_value = custom.forward(x)
        print("dxt_value")
        print(dst_value.data.numpy())

        np.testing.assert_allclose(dst_value.data.numpy(), golden_value, atol=1e-14)

        # test cpu
        # pdb.set_trace()
        custom = dct_lee.IDXST()
        dst_value = custom.forward(x)
        print("dxt_value")
        print(dst_value.data.numpy())

        np.testing.assert_allclose(dst_value.data.numpy(), golden_value, atol=1e-14)

        if torch.cuda.device_count():
            # test gpu
            custom = dct.IDXST()
            dst_value = custom.forward(x.cuda()).cpu()
            print("dxt_value cuda")
            print(dst_value.data.numpy())

            np.testing.assert_allclose(dst_value.data.numpy(), golden_value, atol=1e-14)

            # test gpu
            custom = dct_lee.IDXST()
            dst_value = custom.forward(x.cuda()).cpu()
            print("dxt_value cuda")
            print(dst_value.data.numpy())

            np.testing.assert_allclose(dst_value.data.numpy(), golden_value, atol=1e-14)


class DXTOpTest(unittest.TestCase):
    def test_idcct2Random(self):
        torch.manual_seed(10)
        M = 4
        N = 8
        x = torch.empty(M, N, dtype=torch.int32).random_(0, 10).double()
        print("2D x")
        print(x)

        golden_value = discrete_spectral_transform.idcct2(x).data.numpy()
        print("2D golden_value")
        print(golden_value)

        # test cpu
        # pdb.set_trace()
        custom = dct.IDCCT2()
        dst_value = custom.forward(x)
        print("dxt_value")
        print(dst_value.data.numpy())

        np.testing.assert_allclose(dst_value.data.numpy(), golden_value, atol=1e-14)

        # test cpu
        # pdb.set_trace()
        custom = dct_lee.IDCCT2()
        dst_value = custom.forward(x)
        print("dxt_value")
        print(dst_value.data.numpy())

        np.testing.assert_allclose(dst_value.data.numpy(), golden_value, atol=1e-14)

        if torch.cuda.device_count():
            # test gpu
            custom = dct.IDCCT2()
            dst_value = custom.forward(x.cuda()).cpu()
            print("dxt_value cuda")
            print(dst_value.data.numpy())

            np.testing.assert_allclose(dst_value.data.numpy(), golden_value, atol=1e-14)

            # test gpu
            custom = dct_lee.IDCCT2()
            dst_value = custom.forward(x.cuda()).cpu()
            print("dxt_value cuda")
            print(dst_value.data.numpy())

            np.testing.assert_allclose(dst_value.data.numpy(), golden_value, atol=1e-14)

    def test_idcst2Random(self):
        torch.manual_seed(10)
        M = 4
        N = 8
        x = torch.empty(M, N, dtype=torch.int32).random_(0, 10).double()
        print("2D x")
        print(x)

        golden_value = discrete_spectral_transform.idcst2(x).data.numpy()
        print("2D golden_value")
        print(golden_value)

        # test cpu
        # pdb.set_trace()
        custom = dct.IDCST2()
        dst_value = custom.forward(x)
        print("dxt_value")
        print(dst_value.data.numpy())

        np.testing.assert_allclose(dst_value.data.numpy(), golden_value, atol=1e-14)

        # test cpu
        # pdb.set_trace()
        custom = dct_lee.IDCST2()
        dst_value = custom.forward(x)
        print("dxt_value")
        print(dst_value.data.numpy())

        np.testing.assert_allclose(dst_value.data.numpy(), golden_value, atol=1e-14)

        if torch.cuda.device_count():
            # test gpu
            custom = dct.IDCST2()
            dst_value = custom.forward(x.cuda()).cpu()
            print("dxt_value cuda")
            print(dst_value.data.numpy())

            np.testing.assert_allclose(dst_value.data.numpy(), golden_value, atol=1e-14)

            # test gpu
            custom = dct_lee.IDCST2()
            dst_value = custom.forward(x.cuda()).cpu()
            print("dxt_value cuda")
            print(dst_value.data.numpy())

            np.testing.assert_allclose(dst_value.data.numpy(), golden_value, atol=1e-14)

    def test_idsct2Random(self):
        torch.manual_seed(10)
        M = 4
        N = 8
        x = torch.empty(M, N, dtype=torch.int32).random_(0, 10).double()
        print("2D x")
        print(x)

        golden_value = discrete_spectral_transform.idsct2(x).data.numpy()
        print("2D golden_value")
        print(golden_value)

        # test cpu
        # pdb.set_trace()
        custom = dct.IDSCT2()
        dst_value = custom.forward(x)
        print("dxt_value")
        print(dst_value.data.numpy())

        np.testing.assert_allclose(dst_value.data.numpy(), golden_value, atol=1e-14)

        # test cpu
        # pdb.set_trace()
        custom = dct_lee.IDSCT2()
        dst_value = custom.forward(x)
        print("dxt_value")
        print(dst_value.data.numpy())

        np.testing.assert_allclose(dst_value.data.numpy(), golden_value, atol=1e-14)

        if torch.cuda.device_count():
            # test gpu
            custom = dct.IDSCT2()
            dst_value = custom.forward(x.cuda()).cpu()
            print("dxt_value cuda")
            print(dst_value.data.numpy())

            np.testing.assert_allclose(dst_value.data.numpy(), golden_value, atol=1e-14)

            # test gpu
            custom = dct_lee.IDSCT2()
            dst_value = custom.forward(x.cuda()).cpu()
            print("dxt_value cuda")
            print(dst_value.data.numpy())

            np.testing.assert_allclose(dst_value.data.numpy(), golden_value, atol=1e-14)

    def test_idct_idxstRandom(self):
        torch.manual_seed(10)
        M = 4
        N = 8
        x = torch.empty(M, N, dtype=torch.int32).random_(0, 10).double()
        print("2D x")
        print(x)

        expkM = discrete_spectral_transform.get_exact_expk(M, dtype=x.dtype, device=x.device)
        expkN = discrete_spectral_transform.get_exact_expk(N, dtype=x.dtype, device=x.device)

        golden_value = discrete_spectral_transform.idct_idxst(x).data.numpy()
        print("2D golden_value")
        print(golden_value)

        # test cpu
        custom = dct.IDCT_IDXST()
        idct_idxst_value = custom.forward(x)
        print("2D dct.idct_idxst")
        print(idct_idxst_value.data.numpy())

        np.testing.assert_allclose(idct_idxst_value.data.numpy(), golden_value * 2, atol=1e-14)

        # test gpu
        custom = dct2_fft2.IDCT_IDXST(expkM, expkN)
        idct_idxst_value = custom.forward(x)
        print("2D dct2_fft2.idct_idxst cuda")
        print(idct_idxst_value.data.numpy())

        # note the scale factor
        np.testing.assert_allclose(idct_idxst_value.data.numpy(), golden_value * 2, atol=1e-14)

        if torch.cuda.device_count():
            # test gpu
            custom = dct.IDCT_IDXST()
            idct_idxst_value = custom.forward(x.cuda()).cpu()
            print("2D dct.idct_idxst cuda")
            print(idct_idxst_value.data.numpy())

            np.testing.assert_allclose(idct_idxst_value.data.numpy(), golden_value * 2, atol=1e-14)

            # test gpu
            custom = dct2_fft2.IDCT_IDXST(expkM.cuda(), expkN.cuda())
            idct_idxst_value = custom.forward(x.cuda()).cpu()
            print("2D dct2_fft2.idct_idxst cuda")
            print(idct_idxst_value.data.numpy())

            # note the scale factor
            np.testing.assert_allclose(idct_idxst_value.data.numpy(), golden_value * 2, atol=1e-14)

    def test_idxst_idctRandom(self):
        torch.manual_seed(10)
        M = 4
        N = 8
        x = torch.empty(M, N, dtype=torch.int32).random_(0, 10).double()
        print("2D x")
        print(x)

        expkM = discrete_spectral_transform.get_exact_expk(M, dtype=x.dtype, device=x.device)
        expkN = discrete_spectral_transform.get_exact_expk(N, dtype=x.dtype, device=x.device)

        golden_value = discrete_spectral_transform.idxst_idct(x).data.numpy()
        print("2D golden_value")
        print(golden_value)

        # test cpu
        custom = dct.IDXST_IDCT()
        idxst_idct_value = custom.forward(x)
        print("2D dct.idxst_idct")
        print(idxst_idct_value.data.numpy())

        np.testing.assert_allclose(idxst_idct_value.data.numpy(), golden_value * 2, atol=1e-14)

        # test cpu
        custom = dct2_fft2.IDXST_IDCT(expkM, expkN)
        idxst_idct_value = custom.forward(x)
        print("2D dct2_fft2.idxst_idct cuda")
        print(idxst_idct_value.data.numpy())

        # note the scale factor
        np.testing.assert_allclose(idxst_idct_value.data.numpy(), golden_value* 2, atol=1e-14)

        if torch.cuda.device_count():
            # test gpu
            custom = dct.IDXST_IDCT()
            idxst_idct_value = custom.forward(x.cuda()).cpu()
            print("2D dct.idxst_idct cuda")
            print(idxst_idct_value.data.numpy())

            np.testing.assert_allclose(idxst_idct_value.data.numpy(), golden_value * 2, atol=1e-14)

            # test gpu
            custom = dct2_fft2.IDXST_IDCT(expkM.cuda(), expkN.cuda())
            idxst_idct_value = custom.forward(x.cuda()).cpu()
            print("2D dct2_fft2.idxst_idct cuda")
            print(idxst_idct_value.data.numpy())

            # note the scale factor
            np.testing.assert_allclose(idxst_idct_value.data.numpy(), golden_value* 2, atol=1e-14)

def eval_torch_rfft1d(x, runs):
    for i in range(100):
        a = torch.rfft(x, signal_ndim=1, onesided=True)
    torch.cuda.synchronize()
    tt = time.time()
    for i in range(runs):
        a = torch.rfft(x, signal_ndim=1, onesided=True)
    torch.cuda.synchronize()
    print("torch.rfft1d takes %.7f ms" % ((time.time()-tt)/runs*1000))

    b = torch.irfft(a, signal_ndim=1, onesided=True, signal_sizes=x.shape[1:])
    torch.cuda.synchronize()
    tt = time.time()
    for i in range(runs):
        b = torch.irfft(a, signal_ndim=1, onesided=True, signal_sizes=x.shape[1:])
    torch.cuda.synchronize()
    print("torch.irfft1d takes %.7f ms" % ((time.time()-tt)/runs*1000))

    print("")


def eval_torch_rfft2d(x, runs):
    for i in range(100):
        a = torch.rfft(x, signal_ndim=2, onesided=True)
    torch.cuda.synchronize()
    tt = time.time()
    for i in range(runs):
        a = torch.rfft(x, signal_ndim=2, onesided=True)
    torch.cuda.synchronize()
    print("torch.rfft2d takes %.7f ms" % ((time.time()-tt)/runs*1000))

    b = torch.irfft(a, signal_ndim=2, onesided=True, signal_sizes=x.shape)
    torch.cuda.synchronize()
    tt = time.time()
    for i in range(runs):
        b = torch.irfft(a, signal_ndim=2, onesided=True, signal_sizes=x.shape)
    torch.cuda.synchronize()
    print("torch.irfft2d takes %.7f ms" % ((time.time()-tt)/runs*1000))

    print("")


def eval_dct2d(x, expk0, expk1, expkM, expkN, runs):
    x_numpy = x.data.cpu().numpy()
    torch.cuda.synchronize()
    tt = time.time()
    y = fftpack.dct(fftpack.dct(x_numpy.T, norm=None).T/x.size(1), norm=None)/x.size(0)
    torch.cuda.synchronize()
    print("CPU scipy.fftpack.dct2d takes %.7f ms" % ((time.time()-tt)*1000))

    # 9s for 200 iterations 1024x1024 on GTX 1080
    torch.cuda.synchronize()
    tt = time.time()
    # with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for i in range(runs):
        y_2N = discrete_spectral_transform.dct2_2N(x, expk0=expk0, expk1=expk1)
    torch.cuda.synchronize()
    # print(prof)
    print("PyTorch: dct2d_2N takes %.7f ms" % ((time.time()-tt)/runs*1000))

    # 11s for 200 iterations 1024x1024 on GTX 1080
    perm0 = discrete_spectral_transform.get_perm(x.size(-2), dtype=torch.int64, device=x.device)
    perm1 = discrete_spectral_transform.get_perm(x.size(-1), dtype=torch.int64, device=x.device)
    torch.cuda.synchronize()
    tt = time.time()
    # with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for i in range(runs):
        y_N = discrete_spectral_transform.dct2_N(x, perm0=perm0, expk0=expk0, perm1=perm1, expk1=expk1)
    torch.cuda.synchronize()
    # print(prof)
    print("PyTorch: dct2d_N takes %.7f ms" % ((time.time()-tt)/runs*1000))

    dct2func = dct.DCT2(expk0, expk1, algorithm='2N')
    torch.cuda.synchronize()
    tt = time.time()
    # with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for i in range(runs):
        y_2N = dct2func.forward(x)
    torch.cuda.synchronize()
    # print(prof)
    print("DCT2d_2N Function takes %.7f ms" % ((time.time()-tt)/runs*1000))

    dct2func = dct.DCT2(expk0, expk1, algorithm='N')
    y_N = dct2func.forward(x)
    torch.cuda.synchronize()
    # with torch.autograd.profiler.profile(use_cuda=True) as prof:
    tt = time.time()
    for i in range(runs):
        y_N = dct2func.forward(x)
    torch.cuda.synchronize()
    # print(prof)
    print("DCT2d_N Function takes %.7f ms" % ((time.time()-tt)/runs*1000))

    # The implementation below only supports float64 by now
    dct2func = dct_lee.DCT2(expk0, expk1)
    torch.cuda.synchronize()
    tt = time.time()
    # with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for i in range(runs):
        y_N = dct2func.forward(x)
    torch.cuda.synchronize()
    # print(prof)
    print("DCT2d_Lee Function takes %.7f ms" % ((time.time()-tt)/runs*1000))

    dct2func = dct2_fft2.DCT2(expkM, expkN)
    y = dct2func.forward(x)
    torch.cuda.synchronize()
    tt = time.time()
    for i in range(runs):
        y_test = dct2func.forward(x)
    torch.cuda.synchronize()
    print("DCT2_FFT2 Function takes %.7f ms" % ((time.time()-tt)/runs*1000))

    print("")


def eval_idct2d(x, expk0, expk1, expkM, expkN, runs):
    y_N = discrete_spectral_transform.idct2_N(x, expk0=expk0, expk1=expk1)
    torch.cuda.synchronize()
    tt = time.time()
    for i in range(runs):
        y_N = discrete_spectral_transform.idct2_N(x, expk0=expk0, expk1=expk1)
    torch.cuda.synchronize()
    print("PyTorch idct2_N takes %.7f ms" % ((time.time()-tt)/runs*1000))

    idct2func = dct.IDCT2(expk0, expk1, algorithm='2N')
    y_N = idct2func.forward(x)
    torch.cuda.synchronize()
    tt = time.time()
    for i in range(runs):
        y_N = idct2func.forward(x)
    torch.cuda.synchronize()
    print("IDCT2_2N Function takes %.7f ms" % ((time.time()-tt)/runs*1000))

    idct2func = dct.IDCT2(expk0, expk1, algorithm='N')
    y_N = idct2func.forward(x)
    torch.cuda.synchronize()
    tt = time.time()
    # with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for i in range(runs):
        y_N = idct2func.forward(x)/x.size(0)/x.size(1)/4
    torch.cuda.synchronize()
    # print(prof)
    print("IDCT2_N Function takes %.7f ms" % ((time.time()-tt)/runs*1000))

    dct2func = dct2_fft2.IDCT2(expkM, expkN)
    y = dct2func.forward(x)
    torch.cuda.synchronize()
    tt = time.time()
    for i in range(runs):
        y_test = dct2func.forward(x)
    torch.cuda.synchronize()
    print("IDCT2_FFT2 Function takes %.7f ms" % ((time.time()-tt)/runs*1000))

    print("")


def eval_idxt2d(x, expk0, expk1, expkM, expkN, runs):
    dct2func = dct.IDXST_IDCT(expk0, expk1)
    y = dct2func.forward(x)
    torch.cuda.synchronize()
    tt = time.time()
    for i in range(runs):
        y_test = dct2func.forward(x)
    torch.cuda.synchronize()
    print("dct.IDXST_IDCT Function takes %.7f ms" % ((time.time()-tt)/runs*1000))

    y_N = discrete_spectral_transform.idxst_idct(x, expk_0=expk0, expk_1=expk1)
    torch.cuda.synchronize()
    tt = time.time()
    for i in range(runs):
        y_N = discrete_spectral_transform.idxst_idct(x, expk_0=expk0, expk_1=expk1)
    torch.cuda.synchronize()
    print("PyTorch: idxst_idct takes %.7f ms" % ((time.time()-tt)/runs*1000))

    dct2func = dct2_fft2.IDXST_IDCT(expkM, expkN)
    y = dct2func.forward(x)
    torch.cuda.synchronize()
    tt = time.time()
    for i in range(runs):
        y_test = dct2func.forward(x)
    torch.cuda.synchronize()
    print("dct2_fft2.IDXST_IDCT takes %.7f ms" % ((time.time()-tt)/runs*1000))

    dct2func = dct.IDCT_IDXST(expk0, expk1)
    y = dct2func.forward(x)
    torch.cuda.synchronize()
    tt = time.time()
    for i in range(runs):
        y_test = dct2func.forward(x)
    torch.cuda.synchronize()
    print("dct.IDCT_IDXST takes %.7f ms" % ((time.time()-tt)/runs*1000))

    y_N = discrete_spectral_transform.idct_idxst(x, expk_0=expk0, expk_1=expk1)
    torch.cuda.synchronize()
    tt = time.time()
    for i in range(runs):
        y_N = discrete_spectral_transform.idct_idxst(x, expk_0=expk0, expk_1=expk1)
    torch.cuda.synchronize()
    print("PyTorch: idct_idxst takes %.7f ms" % ((time.time()-tt)/runs*1000))

    dct2func = dct2_fft2.IDCT_IDXST(expkM, expkN)
    y = dct2func.forward(x)
    torch.cuda.synchronize()
    tt = time.time()
    for i in range(runs):
        y_test = dct2func.forward(x)
    torch.cuda.synchronize()
    print("dct2_fft2.IDCT_IDXST takes %.7f ms" % ((time.time()-tt)/runs*1000))

    print("")


def eval_others(x, expk0, expk1, expkM, expkN, runs):
    y_N = discrete_spectral_transform.idcct2(x, expk_0=expk0, expk_1=expk1)
    torch.cuda.synchronize()
    tt = time.time()
    # with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for i in range(runs):
        y_N = discrete_spectral_transform.idcct2(x, expk_0=expk0, expk_1=expk1)
    torch.cuda.synchronize()
    # print(prof)
    print("idcct2 takes %.7f ms" % ((time.time()-tt)/runs*1000))

    func = dct.IDCCT2(expk0, expk1)
    y_N = func.forward(x)
    torch.cuda.synchronize()
    tt = time.time()
    # with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for i in range(runs):
        y_N = func.forward(x)
    torch.cuda.synchronize()
    # print(prof)
    print("IDCCT2 Function takes %.7f ms" % ((time.time()-tt)/runs*1000))

    y_N = discrete_spectral_transform.idcst2(x, expk_0=expk0, expk_1=expk1)
    torch.cuda.synchronize()
    tt = time.time()
    # with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for i in range(runs):
        y_N = discrete_spectral_transform.idcst2(x, expk_0=expk0, expk_1=expk1)
    torch.cuda.synchronize()
    # print(prof)
    print("idcst2 takes %.7f ms" % ((time.time()-tt)/runs*1000))

    func = dct.IDCST2(expk0, expk1)
    y_N = func.forward(x)
    torch.cuda.synchronize()
    tt = time.time()
    # with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for i in range(runs):
        y_N = func.forward(x)
    torch.cuda.synchronize()
    # print(prof)
    print("IDCST2 Function takes %.7f ms" % ((time.time()-tt)/runs*1000))

    y_N = discrete_spectral_transform.idsct2(x, expk_0=expk0, expk_1=expk1)
    torch.cuda.synchronize()
    tt = time.time()
    # with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for i in range(runs):
        y_N = discrete_spectral_transform.idsct2(x, expk_0=expk0, expk_1=expk1)
    torch.cuda.synchronize()
    # print(prof)
    print("idsct2 takes %.7f ms" % ((time.time()-tt)/runs*1000))

    func = dct.IDSCT2(expk0, expk1)
    y_N = func.forward(x)
    torch.cuda.synchronize()
    tt = time.time()
    # with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for i in range(runs):
        y_N = func.forward(x)
    torch.cuda.synchronize()
    # print(prof)
    print("IDSCT2 Function takes %.7f ms" % ((time.time()-tt)/runs*1000))

    print("")


def eval_runtime():
    runs = 100

    M = 1024
    N = 1024
    dtype = torch.float64
    x = torch.empty(M, N, dtype=dtype).uniform_(0, 10.0).cuda()

    print("M = {}, N = {}".format(M, N))

    # 2cos(), 2sin()
    expk0 = discrete_spectral_transform.get_expk(M, dtype=x.dtype, device=x.device)
    expk1 = discrete_spectral_transform.get_expk(N, dtype=x.dtype, device=x.device)
    # cos(), -sin()
    expkM = discrete_spectral_transform.get_exact_expk(M, dtype=x.dtype, device=x.device)
    expkN = discrete_spectral_transform.get_exact_expk(N, dtype=x.dtype, device=x.device)

    eval_torch_rfft1d(x, runs)
    eval_torch_rfft2d(x, runs)
    eval_dct2d(x, expk0, expk1, expkM, expkN, runs)
    eval_idct2d(x, expk0, expk1, expkM, expkN, runs)
    eval_idxt2d(x, expk0, expk1, expkM, expkN, runs)
    eval_others(x, expk0, expk1, expkM, expkN, runs)


if __name__ == '__main__':
    torch.manual_seed(10)
    np.random.seed(10)

    print("usage: python dct_unitest.py test|eval")

    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        eval_runtime()
    else:
        unittest.main()
