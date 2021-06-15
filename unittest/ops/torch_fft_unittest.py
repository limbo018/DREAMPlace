
import os
import sys
import numpy as np 
import torch
#import torch_fft_api

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dreamplace.ops.dct import torch_fft_api
sys.path.pop()

def test_rfft_1D(N, dtype): 
    x = torch.empty(2, N, dtype=dtype).uniform_(0, 10.0)
    y1 = torch_fft_api.rfft(x, 1, False, onesided=True)
    y2 = torch_fft_api.rfft(x, 1, True, onesided=True)
    y3 = torch_fft_api.rfft(x, 1, False, onesided=False)
    y4 = torch_fft_api.rfft(x, 1, True, onesided=False)
    x1_hat = torch_fft_api.irfft(y1, 1, False, True, [N])
    x2_hat = torch_fft_api.irfft(y2, 1, True, True, [N])
    x3_hat = torch_fft_api.irfft(y3, 1, False, False, [N])
    x4_hat = torch_fft_api.irfft(y4, 1, True, False, [N])

    print("x")
    print(x)
    print("y1")
    print(y1)
    print("y2")
    print(y2)
    print("y3")
    print(y3)
    print("y4")
    print(y4)
    print("x1_hat")
    print(x1_hat)
    print("x2_hat")
    print(x2_hat)
    print("x3_hat")
    print(x3_hat)
    print("x4_hat")
    print(x4_hat)

def test_rfft_2D(N, dtype):
    x = torch.empty(2, N, N, dtype=dtype).uniform_(0, 10.0)
    y1 = torch_fft_api.rfft(x, 2, False, onesided=True)
    y2 = torch_fft_api.rfft(x, 2, True, onesided=True)
    y3 = torch_fft_api.rfft(x, 2, False, onesided=False)
    y4 = torch_fft_api.rfft(x, 2, True, onesided=False)
    x1_hat = torch_fft_api.irfft(y1, 2, False, True, [N, N])
    x2_hat = torch_fft_api.irfft(y2, 2, True, True, [N, N])
    x3_hat = torch_fft_api.irfft(y3, 2, False, False, [N, N])
    x4_hat = torch_fft_api.irfft(y4, 2, True, False, [N, N])

    print("x")
    print(x)
    print("y1")
    print(y1)
    print("y2")
    print(y2)
    print("y3")
    print(y3)
    print("y4")
    print(y4)
    print("x1_hat")
    print(x1_hat)
    print("x2_hat")
    print(x2_hat)
    print("x3_hat")
    print(x3_hat)
    print("x4_hat")
    print(x4_hat)

def test_rfft_3D(N, dtype): 
    x = torch.empty(2, N, N, N, dtype=dtype).uniform_(0, 10.0)
    y1 = torch_fft_api.rfft(x, 3, False, onesided=True)
    y2 = torch_fft_api.rfft(x, 3, True, onesided=True)
    y3 = torch_fft_api.rfft(x, 3, False, onesided=False)
    y4 = torch_fft_api.rfft(x, 3, True, onesided=False)
    x1_hat = torch_fft_api.irfft(y1, 3, False, True, [N, N, N])
    x2_hat = torch_fft_api.irfft(y2, 3, True, True, [N, N, N])
    x3_hat = torch_fft_api.irfft(y3, 3, False, False, [N, N, N])
    x4_hat = torch_fft_api.irfft(y4, 3, True, False, [N, N, N])

    print("x")
    print(x)
    print("y1")
    print(y1)
    print("y2")
    print(y2)
    print("y3")
    print(y3)
    print("y4")
    print(y4)
    print("x1_hat")
    print(x1_hat)
    print("x2_hat")
    print(x2_hat)
    print("x3_hat")
    print(x3_hat)
    print("x4_hat")
    print(x4_hat)

def test_fft_1D(N, dtype):
    x = torch.empty(2, N, 2, dtype=dtype).uniform_(0, 10.0)
    y1 = torch_fft_api.fft(x, 1, False)
    y2 = torch_fft_api.fft(x, 1, True)
    x1_hat = torch_fft_api.ifft(y1, 1, False)
    x2_hat = torch_fft_api.ifft(y2, 1, True)

    print("x")
    print(x)
    print("y1")
    print(y1)
    print("y2")
    print(y2)
    print("x1_hat")
    print(x1_hat)
    print("x2_hat")
    print(x2_hat)

def test_fft_2D(N, dtype):
    x = torch.empty(2, N, N, 2, dtype=dtype).uniform_(0, 10.0)
    y1 = torch_fft_api.fft(x, 2, False)
    y2 = torch_fft_api.fft(x, 2, True)
    x1_hat = torch_fft_api.ifft(y1, 2, False)
    x2_hat = torch_fft_api.ifft(y2, 2, True)

    print("x")
    print(x)
    print("y1")
    print(y1)
    print("y2")
    print(y2)
    print("x1_hat")
    print(x1_hat)
    print("x2_hat")
    print(x2_hat)

def test_fft_3D(N, dtype):
    x = torch.empty(2, N, N, N, 2, dtype=dtype).uniform_(0, 10.0)
    y1 = torch_fft_api.fft(x, 3, False)
    y2 = torch_fft_api.fft(x, 3, True)
    x1_hat = torch_fft_api.ifft(y1, 3, False)
    x2_hat = torch_fft_api.ifft(y2, 3, True)

    print("x")
    print(x)
    print("y1")
    print(y1)
    print("y2")
    print(y2)
    print("x1_hat")
    print(x1_hat)
    print("x2_hat")
    print(x2_hat)


if __name__ == '__main__':
    torch.manual_seed(10)
    np.random.seed(10)
    dtype = torch.float32
    N = 4

    # test 1D rfft/irfft
    #test_rfft_1D(N, dtype)

    # test 2D rfft/irfft
    #test_rfft_2D(N, dtype)

    # test 3D rfft/irfft
    #test_rfft_3D(N, dtype)

    # test 1D fft/ifft
    #test_fft_1D(N, dtype)

    # test 2D fft/ifft
    #test_fft_2D(N, dtype)

    # test 3D fft/ifft
    test_fft_3D(N, dtype)
