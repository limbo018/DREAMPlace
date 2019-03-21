##
# @file   tmp.py
# @author Yibo Lin
# @date   Sep 2018
#

import numpy as np 
import scipy 
from scipy import fftpack 
import torch
import pdb 

def myidst(x):
    N = x.shape[-1]
    z = np.zeros_like(x).astype(np.complex128)
    for k in range(len(z)):
        for i in range(len(x)):
            #content = str(x[i]) + " " + str(i)
            #print("[%d] %s" % (k, content))
            z[k] += x[i]*np.exp(1j*np.pi/N*((k+0.5)*(i+0)))
    #for k in range(len(z)):
    #    for i in range(len(x)):
    #        if i == N-1: 
    #            z[k] -= x[i]*np.exp(1j*np.pi/N*((k+0.5)*(i+0)))/2.0
    return z

def myidst_ext(x):
    N = x.shape[-1]
    z = np.zeros_like(x).astype(np.complex128)
    for k in range(len(z)):
        for i in range(len(x)):
            #content = str(x[i]) + " " + str(i+1)
            #print("[%d] %s" % (k, content))
            z[k] += x[i]*np.exp(1j*np.pi/N*((k+0.5)*(i+1)))
    #for k in range(len(z)):
    #    for i in range(len(x)):
    #        if i == N-1: 
    #            z[k] -= x[i]*np.exp(1j*np.pi/N*((k+0.5)*(i+0)))/2.0
    return z

def dst_type2(x):
    N = x.shape[-1]
    z = np.zeros_like(x).astype(np.complex128)
    for k in range(len(z)):
        for i in range(len(x)):
            z[k] += x[i]*np.sin(np.pi/N*((k+1)*(i+0.5)))
            #z[k] += x[i]*np.exp(1j*np.pi/N*((k+1)*(i+0.5)))
    return z

def dst_type3(x):
    N = x.shape[-1]
    z = np.zeros_like(x).astype(np.complex128)
    for k in range(len(z)):
        for i in range(len(x)):
            if i == N-1:
                #z[k] += ((-1)**k)/2.0*x[i]
                z[k] += x[i]*np.exp(1j*np.pi/N*((k+0.5)*(i+1)))/2.0
            else:
                #z[k] += x[i]*np.sin(np.pi/N*((k+0.5)*(i+1)))
                z[k] += x[i]*np.exp(1j*np.pi/N*((k+0.5)*(i+1)))
    return z*2

if __name__ == "__main__":
    x = np.array([1, 23, 5, 6, 7, 4]).astype(np.float64)
    xflip = np.flip(x, 0)
    N = len(x)
    print("scipy dst")
    print(fftpack.dst(x, 2)/2)

    ydst2 = dst_type2(x)
    print("dst_type2")
    print(ydst2)

    pdb.set_trace()

    print("scipy idst")
    print(fftpack.idst(ydst2, 2)/len(x))

    print("scipy dst III")
    print(fftpack.dst(ydst2, 3)/len(x))

    zidst2 = dst_type3(ydst2)/len(x)
    print("idst_type2")
    print(zidst2)

    zmyidst = myidst(ydst2)
    print("myidst")
    print(zmyidst)

    ydst2_ext = np.concatenate([ydst2[1:], [0]])
    zmyidst_ext = myidst_ext(ydst2_ext)
    print("myidst_ext")
    print(zmyidst_ext)

    print(fftpack.idct(np.flip(ydst2_ext, 0), 2)/2)

    #expk = 0.5*np.exp(np.arange(N)*1j*np.pi*2/(4*N))
    #v = np.zeros_like(expk)
    #for k in range(N):
    #    if k == 0: 
    #        v[k] = expk[k] * (-0 + 1j*ydst2_ext[k])
    #    else:
    #        v[k] = expk[k] * (-ydst2_ext[N-k] + 1j*ydst2_ext[k])
    #print(np.fft.ifft(v))

    pdb.set_trace()
