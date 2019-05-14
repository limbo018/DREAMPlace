##
# @file   electric_potential_unitest.py
# @author Yibo Lin
# @date   Mar 2019
#

import time
import numpy as np
import unittest

import torch
from torch.autograd import Function, Variable
import os
import imp
import sys
import gzip

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dreamplace.ops.dct import dct
from dreamplace.ops.dct import discrete_spectral_transform
from dreamplace.ops.electric_potential import electric_potential, electric_overflow
sys.path.pop()

if sys.version_info[0] < 3:
    import cPickle as pickle
else:
    import _pickle as pickle
import inspect
import pdb
from scipy import fftpack

import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def spectral(x, transform_type, normalize=False):
    M = x.shape[0]
    N = x.shape[1]
    y = np.zeros([M, N])
    for u in range(M):
        for v in range(N):
            for i in range(M):
                for j in range(N):
                    if transform_type == 'coscos':
                        y[u, v] += x[i, j]*np.cos(2*np.pi/M*i*u)*np.cos(2*np.pi/N*j*v)
                    elif transform_type == 'cossin':
                        y[u, v] += x[i, j]*np.cos(2*np.pi/M*i*u)*np.sin(2*np.pi/N*j*v)
                    elif transform_type == 'sincos':
                        y[u, v] += x[i, j]*np.sin(2*np.pi/M*i*u)*np.cos(2*np.pi/N*j*v)
                    elif transform_type == 'sinsin':
                        y[u, v] += x[i, j]*np.sin(2*np.pi/M*i*u)*np.sin(2*np.pi/N*j*v)
            if normalize:
                y[u, v] *= 1.0/(M*N)
    return y

def naive_idct(x):
    N = len(x)
    y = np.zeros(N)
    y += 0.5*x[0]
    for u in range(N):
        for k in range(1, N):
            y[u] += x[k]*np.cos(np.pi/(2*N)*k*(2*u+1))

    return y

def naive_idct2(x):
    M = x.shape[0]
    N = x.shape[1]

    y = np.zeros_like(x)

    #for u in range(M):
    #    y[u] = naive_idct(x[u])
    #y = np.transpose(y)
    #for v in range(N):
    #    y[v] = naive_idct(y[v])
    #y = np.transpose(y)

    for u in range(M):
        for v in range(N):
            for p in range(M):
                for q in range(N):
                    a = 1 #if p == 0 else 2
                    b = 1 #if q == 0 else 2
                    y[u, v] += a*b*x[p, q]*np.cos(np.pi/M*p*(u+0.5))*np.cos(np.pi/N*q*(v+0.5))
    return y

def naive_idct2_grad_x(x):
    M = x.shape[0]
    N = x.shape[1]

    y = np.zeros_like(x)
    for u in range(M):
        for v in range(N):
            for p in range(M):
                for q in range(N):
                    a = 1 #if p == 0 else 2
                    b = 1 #if q == 0 else 2
                    y[u, v] += -a*b*x[p, q]*(np.pi/M*p)*np.sin(np.pi/M*p*(u+0.5))*np.cos(np.pi/N*q*(v+0.5))
    return y

def naive_idct2_grad_y(x):
    M = x.shape[0]
    N = x.shape[1]

    y = np.zeros_like(x)
    for u in range(M):
        for v in range(N):
            for p in range(M):
                for q in range(N):
                    a = 1 #if p == 0 else 2
                    b = 1 #if q == 0 else 2
                    y[u, v] += -a*b*x[p, q]*(np.pi/N*q)*np.cos(np.pi/M*p*(u+0.5))*np.sin(np.pi/N*q*(v+0.5))
    return y

def naive_idsct2(x):
    M = x.shape[0]
    N = x.shape[1]

    y = np.zeros_like(x)
    for u in range(M):
        for v in range(N):
            for p in range(M):
                for q in range(N):
                    a = 1 #if p == 0 else 2
                    b = 1 #if q == 0 else 2
                    y[u, v] += -a*b*x[p, q]*np.sin(np.pi/M*p*(u+0.5))*np.cos(np.pi/N*q*(v+0.5))
    return y

def naive_idsct2_2(x):
    M = x.shape[0]
    N = x.shape[1]

    y1 = np.zeros_like(x)

    #for u in range(M):
    #    y1[u] = naive_idct(x[u])
    for v in range(N):
        for p in range(M):
            for q in range(N):
                y1[p, v] += x[p, q]*np.cos(np.pi/N*q*(v+0.5))
    #print("y1 = ", y1)
    #for u in range(M):
    #    for v in range(N):
    #        for p in range(M):
    #            for q in range(N):
    #                y1[u, p] += x[p, q]*np.cos(np.pi/N*q*(v+0.5))

    #y1 = np.transpose(y1)
    y2 = np.zeros_like(x)
    for u in range(M):
        for v in range(N):
            for p in range(M):
                y2[u, v] += -y1[p, v]*np.sin(np.pi/M*p*(u+0.5))
    #for u in range(M):
    #    for v in range(N):
    #        for p in range(M):
    #            for q in range(N):
    #                y2[u, v] += -y1[u, p]*np.sin(np.pi/M*p*(u+0.5))
    #y = np.transpose(y2)

    return y1, y2

"""
class DctOpTest(unittest.TestCase):
    def test_dct(self):
        xx = np.array([[1, 2], [3, 4]]).astype(np.float64).ravel()
        x = Variable(torch.from_numpy(xx), requires_grad=False)

        golden_value = fftpack.dct(xx)
        print("dct 1d even golden_value = ", golden_value)

        result = discrete_spectral_transform.dct(x)
        print("dct 1d even result = ", result)

        np.testing.assert_allclose(result.numpy(), golden_value, atol=1e-12)

        xx = xx.reshape([2, 2])
        x = x.view([2, 2])
        golden_value = fftpack.dct(xx)
        print("dct batch golden_value = ", golden_value)

        result = discrete_spectral_transform.dct(x)
        print("dct batch result = ", result)

        np.testing.assert_allclose(result.numpy(), golden_value, atol=1e-12)

        xx = np.array([[1, 2], [3, 4], [5, 6]]).astype(np.float64).ravel()
        x = Variable(torch.from_numpy(xx), requires_grad=False)

        golden_value = fftpack.dct(xx)
        print("dct 1d odd golden_value = ", golden_value)

        result = discrete_spectral_transform.dct(x)
        print("dct 1d odd result = ", result)

        np.testing.assert_allclose(result.numpy(), golden_value, atol=1e-12)

    def test_dct2(self):
        xx = np.array([[1, 2], [3, 4]]).astype(np.float64)
        x = Variable(torch.from_numpy(xx), requires_grad=False)

        golden_value = fftpack.dct(fftpack.dct(xx.T, norm=None).T, norm=None)
        print("dct 2d even golden_value = ", golden_value)

        result = discrete_spectral_transform.dct2(x)
        print("dct 2d even result = ", result)

        np.testing.assert_allclose(result.numpy(), golden_value, atol=1e-12)

    def test_idct(self):
        xx = np.array([[1, 2], [3, 4]]).astype(np.float64).ravel()
        yy = fftpack.dct(xx)
        y = Variable(torch.from_numpy(yy), requires_grad=False)

        print("original signal = ", xx)

        golden_value = fftpack.idct(yy)
        print("idct 1d even golden_value = ", golden_value)

        print("idct 1d even native_value = ", naive_idct(yy))

        result = discrete_spectral_transform.idct(y)*xx.size
        print("idct 1d even result = ", result)

        np.testing.assert_allclose(result.numpy(), golden_value, atol=1e-12)

        xx = np.array([[1, 2], [3, 4], [5, 6]]).astype(np.float64).ravel()
        yy = fftpack.dct(xx)
        y = Variable(torch.from_numpy(yy), requires_grad=False)

        print("original signal = ", xx)

        golden_value = fftpack.idct(yy)
        print("idct 1d even golden_value = ", golden_value)

        print("idct 1d even native_value = ", naive_idct(yy))

        result = discrete_spectral_transform.idct(y)*xx.size
        print("idct 1d even result = ", result)

        np.testing.assert_allclose(result.numpy(), golden_value, atol=1e-12)

    def test_idct2(self):
        xx = np.array([[1, 2], [3, 4]]).astype(np.float64)
        yy = fftpack.dct(fftpack.dct(xx.T, norm=None).T, norm=None)
        y = Variable(torch.from_numpy(yy), requires_grad=False)

        print("original signal = ", xx)

        golden_value = fftpack.idct(fftpack.idct(yy.T, norm=None).T, norm=None)
        print("idct 2d even golden_value = ", golden_value)

        result = discrete_spectral_transform.idct2(y)*xx.size
        print("idct 2d even result = ", result)

        np.testing.assert_allclose(result.numpy(), golden_value, atol=1e-12)

        xx = np.array([[1, 2], [3, 4], [5, 6]]).astype(np.float64)
        yy = fftpack.dct(fftpack.dct(xx.T, norm=None).T, norm=None)
        y = Variable(torch.from_numpy(yy), requires_grad=False)

        print("original signal = ", xx)

        golden_value = fftpack.idct(fftpack.idct(yy.T, norm=None).T, norm=None)
        print("idct 2d even golden_value = ", golden_value)

        result = discrete_spectral_transform.idct2(y)*xx.size
        print("idct 2d even result = ", result)

        np.testing.assert_allclose(result.numpy(), golden_value, atol=1e-12)

    #def test_idct2(self):
    #    xx = np.array([[1, 2], [3, 4]]).astype(np.float64)
    #    x = Variable(torch.from_numpy(xx), requires_grad=False)

    #    golden_value = fftpack.idct(fftpack.idct(xx.T, norm=None).T, norm=None)
    #    print("idct 2d even golden_value = ", golden_value)

    #    result = discrete_spectral_transform.idct2(x)
    #    print("idct 2d even result = ", result)

    #    np.testing.assert_allclose(result.numpy(), golden_value, atol=1e-12)

class DstOpTest(unittest.TestCase):
    def test_dst(self):
        xx = np.array([[1, 2], [3, 4]]).astype(np.float64).ravel()
        x = Variable(torch.from_numpy(xx), requires_grad=False)

        golden_value = fftpack.dst(xx)
        print("dst 1d even golden_value = ", golden_value)

        result = discrete_spectral_transform.dst(x)
        print("dst 1d even result = ", result)

        np.testing.assert_allclose(result.numpy(), golden_value, atol=1e-12)

        xx = xx.reshape([2, 2])
        x = x.view([2, 2])
        golden_value = fftpack.dst(xx)
        print("dst batch golden_value = ", golden_value)

        result = discrete_spectral_transform.dst(x)
        print("dst batch result = ", result)

        np.testing.assert_allclose(result.numpy(), golden_value, atol=1e-12)

        xx = np.array([[1, 2], [3, 4], [5, 6]]).astype(np.float64).ravel()
        x = Variable(torch.from_numpy(xx), requires_grad=False)

        golden_value = fftpack.dst(xx)
        print("dst 1d odd golden_value = ", golden_value)

        result = discrete_spectral_transform.dst(x)
        print("dst 1d odd result = ", result)

        np.testing.assert_allclose(result.numpy(), golden_value, atol=1e-12)

    def test_dst2(self):
        xx = np.array([[1, 2], [3, 4]]).astype(np.float64)
        x = Variable(torch.from_numpy(xx), requires_grad=False)

        golden_value = fftpack.dst(fftpack.dst(xx.T, norm=None).T, norm=None)
        print("dst 2d even golden_value = ", golden_value)

        result = discrete_spectral_transform.dst2(x)
        print("dst 2d even result = ", result)

        np.testing.assert_allclose(result.numpy(), golden_value, atol=1e-12)
"""

#class SpectralOpTest(unittest.TestCase):
#    def test_spectral_even(self):
#
#        xx = np.array([[1, 2], [3, 4]]).astype(np.float64)
#        x = Variable(torch.from_numpy(xx), requires_grad=False)
#
#        # coscos
#        golden_value = spectral(xx, 'coscos')
#        print("coscos golden_value = ", golden_value)
#
#        result = discrete_spectral_transform.DiscreteCosCosTransform(x)
#        print("coscos custom = ", result)
#
#        np.testing.assert_allclose(result.numpy(), golden_value, atol=1e-12)
#
#        # cossin
#        golden_value = spectral(xx, 'cossin')
#        print("cossin golden_value = ", golden_value)
#
#        result = discrete_spectral_transform.DiscreteCosSinTransform(x)
#        print("cossin custom = ", result)
#
#        np.testing.assert_allclose(result.numpy(), golden_value, atol=1e-12)
#
#        # sincos
#        golden_value = spectral(xx, 'sincos')
#        print("sincos golden_value = ", golden_value)
#
#        result = discrete_spectral_transform.DiscreteSinCosTransform(x)
#        print("sincos custom = ", result)
#
#        np.testing.assert_allclose(result.numpy(), golden_value, atol=1e-12)
#
#        # sinsin
#        golden_value = spectral(xx, 'sinsin')
#        print("sinsin golden_value = ", golden_value)
#
#        result = discrete_spectral_transform.DiscreteSinSinTransform(x)
#        print("sinsin custom = ", result)
#
#        np.testing.assert_allclose(result.numpy(), golden_value, atol=1e-12)
#
#    def test_spectral_odd(self):
#
#        xx = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.float64)
#        x = Variable(torch.from_numpy(xx), requires_grad=False)
#
#        # coscos
#        golden_value = spectral(xx, 'coscos')
#        print("coscos golden_value = ", golden_value)
#
#        result = discrete_spectral_transform.DiscreteCosCosTransform(x)
#        print("coscos custom = ", result)
#
#        np.testing.assert_allclose(result.numpy(), golden_value, atol=1e-12)
#
#        # cossin
#        golden_value = spectral(xx, 'cossin')
#        print("cossin golden_value = ", golden_value)
#
#        result = discrete_spectral_transform.DiscreteCosSinTransform(x)
#        print("cossin custom = ", result)
#
#        np.testing.assert_allclose(result.numpy(), golden_value, atol=1e-12)
#
#        # sincos
#        golden_value = spectral(xx, 'sincos')
#        print("sincos golden_value = ", golden_value)
#
#        result = discrete_spectral_transform.DiscreteSinCosTransform(x)
#        print("sincos custom = ", result)
#
#        np.testing.assert_allclose(result.numpy(), golden_value, atol=1e-12)
#
#        # sinsin
#        golden_value = spectral(xx, 'sinsin')
#        print("sinsin golden_value = ", golden_value)
#
#        result = discrete_spectral_transform.DiscreteSinSinTransform(x)
#        print("sinsin custom = ", result)
#
#        np.testing.assert_allclose(result.numpy(), golden_value, atol=1e-12)
#
#
class ElectricPotentialOpTest(unittest.TestCase):
    def test_densityOverflowRandom(self):
        dtype = np.float64
        xx = np.array([1.0, 2.0]).astype(dtype)
        yy = np.array([2.0, 1.5]).astype(dtype)
        node_size_x = np.array([0.5, 1.0]).astype(dtype)
        node_size_y = np.array([1.0, 1.0]).astype(dtype)
        #xx = np.array([2.0]).astype(dtype)
        #yy = np.array([1.5]).astype(dtype)
        #node_size_x = np.array([1.0]).astype(dtype)
        #node_size_y = np.array([1.0]).astype(dtype)
        num_nodes = len(xx)

        scale_factor = 1.0

        xl = 0.0
        yl = 0.0
        xh = 5.0
        yh = 5.0
        bin_size_x = 1.0
        bin_size_y = 1.0
        target_density = 0.1
        num_bins_x = int(np.ceil((xh-xl)/bin_size_x))
        num_bins_y = int(np.ceil((yh-yl)/bin_size_y))

        """
        return bin xl
        """
        def bin_xl(id_x):
            return xl+id_x*bin_size_x

        """
        return bin xh
        """
        def bin_xh(id_x):
            return min(bin_xl(id_x)+bin_size_x, xh)

        """
        return bin yl
        """
        def bin_yl(id_y):
            return yl+id_y*bin_size_y

        """
        return bin yh
        """
        def bin_yh(id_y):
            return min(bin_yl(id_y)+bin_size_y, yh)

        bin_center_x = np.zeros(num_bins_x, dtype=dtype)
        for id_x in range(num_bins_x):
            bin_center_x[id_x] = (bin_xl(id_x)+bin_xh(id_x))/2*scale_factor

        bin_center_y = np.zeros(num_bins_y, dtype=dtype)
        for id_y in range(num_bins_y):
            bin_center_y[id_y] = (bin_yl(id_y)+bin_yh(id_y))/2*scale_factor

        print("target_area = ", target_density*bin_size_x*bin_size_y)

        if dtype == np.float64:
            dtype = torch.float64
        elif dtype == np.float32:
            dtype = torch.float32
        # test cpu
        custom = electric_potential.ElectricPotential(
                    torch.tensor(node_size_x, requires_grad=False, dtype=dtype), torch.tensor(node_size_y, requires_grad=False, dtype=dtype),
                    torch.tensor(bin_center_x, requires_grad=False, dtype=dtype), torch.tensor(bin_center_y, requires_grad=False, dtype=dtype),
                    target_density=torch.tensor(target_density, requires_grad=False, dtype=dtype),
                    xl=xl, yl=yl, xh=xh, yh=yh,
                    bin_size_x=bin_size_x, bin_size_y=bin_size_y,
                    num_movable_nodes=num_nodes,
                    num_terminals=0,
                    num_filler_nodes=0,
                    padding=0
                    )

        pos = Variable(torch.from_numpy(np.concatenate([xx, yy])), requires_grad=True)
        result = custom.forward(pos)
        print("custom_result = ", result)
        print(result.type())
        result.backward()
        grad = pos.grad.clone()
        print("custom_grad = ", grad)

        # test cuda
        custom_cuda = electric_potential.ElectricPotential(
                    torch.tensor(node_size_x, requires_grad=False, dtype=dtype).cuda(), torch.tensor(node_size_y, requires_grad=False, dtype=dtype).cuda(),
                    torch.tensor(bin_center_x, requires_grad=False, dtype=dtype).cuda(), torch.tensor(bin_center_y, requires_grad=False, dtype=dtype).cuda(),
                    target_density=torch.tensor(target_density, requires_grad=False, dtype=dtype).cuda(),
                    xl=xl, yl=yl, xh=xh, yh=yh,
                    bin_size_x=bin_size_x, bin_size_y=bin_size_y,
                    num_movable_nodes=num_nodes,
                    num_terminals=0,
                    num_filler_nodes=0,
                    padding=0
                    )

        pos = Variable(torch.from_numpy(np.concatenate([xx, yy])).cuda(), requires_grad=True)
        #pos.grad.zero_()
        result_cuda = custom_cuda.forward(pos)
        print("custom_result_cuda = ", result_cuda.data.cpu())
        print(result_cuda.type())
        result_cuda.backward()
        grad_cuda = pos.grad.clone()
        print("custom_grad_cuda = ", grad_cuda.data.cpu())

        #np.testing.assert_allclose(result.detach().numpy(), result_cuda.data.cpu().detach().numpy())

def plot(plot_count, density_map, padding, name):
    """
    density map contour and heat map
    """
    density_map = density_map[padding:density_map.shape[0]-padding, padding:density_map.shape[1]-padding]
    print("max density = %g" % (np.amax(density_map)))
    print("mean density = %g" % (np.mean(density_map)))

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    x = np.arange(density_map.shape[0])
    y = np.arange(density_map.shape[1])

    x, y = np.meshgrid(x, y)
    # looks like x and y should be swapped
    ax.plot_surface(y, x, density_map, alpha=0.8)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('density')

    #plt.tight_layout()
    plt.savefig(name+".3d.png")
    plt.close()

    #plt.clf()

    #fig, ax = plt.subplots()

    #ax.pcolor(density_map)

    ## Loop over data dimensions and create text annotations.
    ##for i in range(density_map.shape[0]):
    ##    for j in range(density_map.shape[1]):
    ##        text = ax.text(j, i, density_map[i, j],
    ##                ha="center", va="center", color="w")
    #fig.tight_layout()
    #plt.savefig(name+".2d.%d.png" % (plot_count))
    #plt.close()

def eval_runtime(design):
    with gzip.open("../../benchmarks/ispd2005/density/%s_density.pklz" % (design), "rb") as f:
        node_size_x, node_size_y, bin_center_x, bin_center_y, target_density, xl, yl, xh, yh, bin_size_x, bin_size_y, num_movable_nodes, num_terminals, num_filler_nodes = pickle.load(f)

    dtype = torch.float64
    pos_var = Variable(torch.empty(len(node_size_x)*2, dtype=dtype).uniform_(xl, xh), requires_grad=True)
    custom = electric_potential.ElectricPotential(
                torch.tensor(node_size_x, requires_grad=False, dtype=dtype), torch.tensor(node_size_y, requires_grad=False, dtype=dtype),
                torch.tensor(bin_center_x, requires_grad=False, dtype=dtype), torch.tensor(bin_center_y, requires_grad=False, dtype=dtype),
                target_density=torch.tensor(target_density, requires_grad=False, dtype=dtype),
                xl=xl, yl=yl, xh=xh, yh=yh,
                bin_size_x=bin_size_x, bin_size_y=bin_size_y,
                num_movable_nodes=num_movable_nodes,
                num_terminals=num_terminals,
                num_filler_nodes=num_filler_nodes,
                padding=0
                )
    custom_cuda = electric_potential.ElectricPotential(
                torch.tensor(node_size_x, requires_grad=False, dtype=dtype).cuda(), torch.tensor(node_size_y, requires_grad=False, dtype=dtype).cuda(),
                torch.tensor(bin_center_x, requires_grad=False, dtype=dtype).cuda(), torch.tensor(bin_center_y, requires_grad=False, dtype=dtype).cuda(),
                target_density=torch.tensor(target_density, requires_grad=False, dtype=dtype).cuda(),
                xl=xl, yl=yl, xh=xh, yh=yh,
                bin_size_x=bin_size_x, bin_size_y=bin_size_y,
                num_movable_nodes=num_movable_nodes,
                num_terminals=num_terminals,
                num_filler_nodes=num_filler_nodes,
                padding=0
                )

    torch.cuda.synchronize()
    iters = 10
    tt = time.time()
    for i in range(iters):
        result = custom.forward(pos_var)
        result.backward()
    torch.cuda.synchronize()
    print("custom takes %.3f ms" % ((time.time()-tt)/iters*1000))

    pos_var = pos_var.cuda()
    tt = time.time()
    for i in range(iters):
        result = custom_cuda.forward(pos_var)
        result.backward()
    torch.cuda.synchronize()
    print("custom_cuda takes %.3f ms" % ((time.time()-tt)/iters*1000))

if __name__ == '__main__':
    unittest.main()
    exit()

    design = sys.argv[1]
    eval_runtime(design)
    exit()

    np.set_printoptions(linewidth=1000, edgeitems=5)

    x = []
    y = []
    density = []
    auv = []
    phi = []
    ex = []
    ey = []
    term_density = []
    cell_density = []
    fill_density = []
    filename = "/home/polaris/yibolin/Libraries/RePlAce/output/ispd/adaptec1.eplace/density_map0.csv"
    print(filename)
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                tokens = line.split(",")
                x.append(int(tokens[0].strip()))
                y.append(int(tokens[1].strip()))
                density.append(float(tokens[4].strip()))
                auv.append(float(tokens[5].strip()))
                phi.append(float(tokens[6].strip()))
                ex.append(float(tokens[7].strip()))
                ey.append(float(tokens[8].strip()))
                term_density.append(float(tokens[9].strip()))
                cell_density.append(float(tokens[10].strip()))
                fill_density.append(float(tokens[11].strip()))

        M = np.amax(x)+1
        N = np.amax(y)+1

        density_map = np.zeros([M, N])
        auv_map = np.zeros_like(density_map)
        phi_map = np.zeros_like(density_map)
        ex_map = np.zeros_like(density_map)
        ey_map = np.zeros_like(density_map)
        term_density_map = np.zeros_like(density_map)
        cell_density_map = np.zeros_like(density_map)
        fill_density_map = np.zeros_like(density_map)
        for i in range(len(x)):
            density_map[x[i], y[i]] = density[i]
            auv_map[x[i], y[i]] = auv[i]
            phi_map[x[i], y[i]] = phi[i]
            ex_map[x[i], y[i]] = ex[i]
            ey_map[x[i], y[i]] = ey[i]
            term_density_map[x[i], y[i]] = term_density[i]
            cell_density_map[x[i], y[i]] = cell_density[i]
            fill_density_map[x[i], y[i]] = fill_density[i]

    x = []
    y = []
    phi_in = []
    ex_in = []
    ey_in = []
    filename = "/home/polaris/yibolin/Libraries/RePlAce/output/ispd/adaptec1.eplace/phi_2d_st2_in0.csv"
    #filename = "./phi_2d_st2_in_4x4.csv"
    print(filename)
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                tokens = line.split(",")
                x.append(int(tokens[0].strip()))
                y.append(int(tokens[1].strip()))
                phi_in.append(float(tokens[2].strip()))
                ex_in.append(float(tokens[3].strip()))
                ey_in.append(float(tokens[4].strip()))

        M = np.amax(x)+1
        N = np.amax(y)+1

        phi_in_map = np.zeros([M, N])
        ex_in_map = np.zeros([M, N])
        ey_in_map = np.zeros([M, N])
        for i in range(len(x)):
            phi_in_map[x[i], y[i]] = phi_in[i]
            ex_in_map[x[i], y[i]] = ex_in[i]
            ey_in_map[x[i], y[i]] = ey_in[i]

    x = []
    y = []
    phi_out = []
    ex_out = []
    ey_out = []
    filename = "/home/polaris/yibolin/Libraries/RePlAce/output/ispd/adaptec1.eplace/phi_2d_st2_out0.csv"
    #filename = "./phi_2d_st2_out_4x4.csv"
    print(filename)
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                tokens = line.split(",")
                x.append(int(tokens[0].strip()))
                y.append(int(tokens[1].strip()))
                phi_out.append(float(tokens[2].strip()))
                ex_out.append(float(tokens[3].strip()))
                ey_out.append(float(tokens[4].strip()))

        M = np.amax(x)+1
        N = np.amax(y)+1

        phi_out_map = np.zeros([M, N])
        ex_out_map = np.zeros([M, N])
        ey_out_map = np.zeros([M, N])
        for i in range(len(x)):
            phi_out_map[x[i], y[i]] = phi_out[i]
            ex_out_map[x[i], y[i]] = ex_out[i]
            ey_out_map[x[i], y[i]] = ey_out[i]

    #np.savetxt("density_map.csv", density_map, delimiter=",")
    #np.savetxt("auv_map.csv", auv_map, delimiter=",")
    #np.savetxt("phi_map.csv", phi_map, delimiter=",")
    #np.savetxt("ex.csv", ex_map, delimiter=",")
    #np.savetxt("ey.csv", ey_map, delimiter=",")
    #np.savetxt("term_density_map.csv", term_density_map, delimiter=",")
    #np.savetxt("cell_density_map.csv", cell_density_map, delimiter=",")
    #np.savetxt("fill_density_map.csv", fill_density_map, delimiter=",")
    #np.savetxt("phi_in_map.csv", phi_in_map, delimiter=",")
    #np.savetxt("ex_in.csv", ex_in_map, delimiter=",")
    #np.savetxt("ey_in.csv", ey_in_map, delimiter=",")
    #np.savetxt("phi_out_map.csv", phi_out_map, delimiter=",")
    #np.savetxt("ex_out.csv", ex_out_map, delimiter=",")
    #np.savetxt("ey_out.csv", ey_out_map, delimiter=",")

    """
    #naive_auv = spectral(density_map, transform_type='coscos', normalize=True)
    #np.testing.assert_allclose(naive_auv, auv_map)

    # matlab implementation
    # ap = sqrt(1/M) if p == 0 else sqrt(2/M)
    # aq = sqrt(1/N) if q == 0 else sqrt(2/N)
    # e-place implementation
    # ap = 1/M if p == 0 else 2/M
    # aq = 1/N if q == 0 else 2/N
    scipy_auv = fftpack.dct(fftpack.dct(density_map.T, norm='ortho').T, norm='ortho')
    scipy_auv[1:, 1:] *= 2.0 / np.sqrt(M*N)
    scipy_auv[0, 1:] *= np.sqrt(2.0) / np.sqrt(M*N)
    scipy_auv[1:, 0] *= np.sqrt(2.0) / np.sqrt(M*N)
    scipy_auv[0, 0] *= 1.0 / np.sqrt(M*N)
    ratio = scipy_auv/auv_map
    print("scipy_auv/auv_map")
    print(ratio.min())
    print(ratio.max())
    print(ratio.mean())
    np.testing.assert_allclose(scipy_auv, auv_map, rtol=3e-1)
    """

    density_map = torch.from_numpy(density_map)
    # for DCT
    M = density_map.shape[0]
    N = density_map.shape[1]
    expk_M = discrete_spectral_transform.get_expk(M, dtype=density_map.dtype, device=density_map.device)
    expk_N = discrete_spectral_transform.get_expk(N, dtype=density_map.dtype, device=density_map.device)
    # wu and wv
    wu = torch.arange(M, dtype=density_map.dtype, device=density_map.device).mul(2*np.pi/M).view([M, 1])
    wv = torch.arange(N, dtype=density_map.dtype, device=density_map.device).mul(2*np.pi/N).view([1, N])
    wu2_plus_wv2 = wu.pow(2) + wv.pow(2)
    wu2_plus_wv2[0, 0] = 1.0 # avoid zero-division, it will be zeroed out
    inv_wu2_plus_wv2_2X = 2.0 / wu2_plus_wv2
    inv_wu2_plus_wv2_2X[0, 0] = 0.0
    wu_by_wu2_plus_wv2 = wu.mul(inv_wu2_plus_wv2_2X)
    wv_by_wu2_plus_wv2 = wv.mul(inv_wu2_plus_wv2_2X)

    # compute auv
    auv = discrete_spectral_transform.dct2(density_map, expk_M, expk_N)
    auv[0, :].mul_(0.5)
    auv[:, 0].mul_(0.5)
    ratio = auv.numpy() / auv_map
    print("auv/auv_map")
    print(ratio)

    # compute potential phi
    auv_by_wu2_plus_wv2 = auv.mul(inv_wu2_plus_wv2_2X).mul(2)
    ratio = auv_by_wu2_plus_wv2.numpy() / phi_in_map
    print("auv_by_wu2_plus_wv2/phi_in_map")
    print(ratio)

    # auv / (wu**2 + wv**2)
    potential_map = discrete_spectral_transform.idcct2(auv_by_wu2_plus_wv2, expk_M, expk_N)
    ratio = potential_map.numpy() / phi_out_map
    #plot(0, potential_map.numpy(), 0, "%d.potential_map" % (0))
    print("potential_map/phi_out_map")
    print(ratio)
    # compute field xi
    auv_by_wu2_plus_wv2_wu = auv.mul(wu_by_wu2_plus_wv2)
    ratio = auv_by_wu2_plus_wv2_wu.numpy() / ex_in_map
    print("auv_by_wu2_plus_wv2_wu / ex_in_map")
    print(ratio)
    field_map_x = discrete_spectral_transform.idsct2(auv_by_wu2_plus_wv2_wu, expk_M, expk_N)
    ratio = field_map_x.numpy() / ex_out_map
    print("field_map_x/ex_out_map")
    print(ratio)
    pdb.set_trace()
    auv_by_wu2_plus_wv2_wv = auv.mul(wv_by_wu2_plus_wv2)
    ratio = auv_by_wu2_plus_wv2_wv.numpy() / ey_in_map
    print("auv_by_wu2_plus_wv2_wv / ey_in_map")
    print(ratio)
    field_map_y = discrete_spectral_transform.idcst2(auv_by_wu2_plus_wv2_wv, expk_M, expk_N)
    ratio = field_map_y.numpy() / ey_out_map
    print("field_map_y/ey_out_map")
    print(ratio)
    pdb.set_trace()

    exit()

    """
    my_auv_map = discrete_spectral_transform.dct2(torch.from_numpy(density_map)) * (1.0/density_map.size)
    my_auv_map[0, :].mul_(0.5)
    my_auv_map[:, 0].mul_(0.5)
    ratio = my_auv_map.numpy() / auv_map
    print("my_auv_map/auv_map")
    print(ratio.min())
    print(ratio.max())
    print(ratio.mean())
    np.testing.assert_allclose(my_auv_map.numpy(), auv_map, rtol=3e-1)

    #scipy_phi_in_map = fftpack.dct(fftpack.dct(phi_out_map.T, norm=None).T, norm=None) * (1.0/phi_in_map.size)
    #scipy_phi_in_map[0, :] *= 0.5
    #scipy_phi_in_map[:, 0] *= 0.5
    #delta = np.absolute(scipy_phi_in_map - phi_in_map)
    #ratio = scipy_phi_in_map / (phi_in_map+1.0e-6)
    #print(scipy_phi_in_map)
    #print(phi_in_map)

    scipy_phi_in_map = phi_in_map
    scipy_phi_in_map[0, :] *= 2
    scipy_phi_in_map[:, 0] *= 2
    scipy_phi_in_map *= 0.25
    #scipy_phi_map = fftpack.idct(fftpack.idct(scipy_phi_in_map.T, norm=None).T, norm=None) #* (1.0/phi_in_map.size)
    #delta = np.absolute(scipy_phi_map - phi_out_map)
    #ratio = scipy_phi_map / (phi_out_map+1.0e-6)
    my_phi_map = discrete_spectral_transform.idct2(torch.from_numpy(scipy_phi_in_map)) * (phi_in_map.size)
    ratio = my_phi_map.numpy() / phi_out_map
    print(np.argmax(ratio))
    pdb.set_trace()

    exit()

    # compute auv
    density_map = torch.from_numpy(density_map)
    auv = discrete_spectral_transform.dct2(density_map)
    auv.mul_(1.0/density_map.numel())
    auv[0, :].mul_(0.5)
    auv[:, 0].mul_(0.5)
    ratio = auv.numpy() / auv_map
    print("auv/auv_map")
    print(ratio.min())
    print(ratio.max())
    print(ratio.mean())
    np.testing.assert_allclose(auv.numpy(), auv_map, rtol=3e-1)

    # compute potential phi
    wu = torch.arange(density_map.size(0), dtype=density_map.dtype, device=density_map.device).mul(2*np.pi/density_map.size(0)).view([density_map.size(0), 1])
    wv = torch.arange(density_map.size(1), dtype=density_map.dtype, device=density_map.device).mul(2*np.pi/density_map.size(1)).view([1, density_map.size(1)])
    wu2_plus_wv2 = wu.pow(2) + wv.pow(2)
    auv_by_wu2_plus_wv2 = torch.zeros_like(auv).view(-1)
    auv_by_wu2_plus_wv2[1:] = auv.view(-1)[1:] / wu2_plus_wv2.view(-1)[1:]
    auv_by_wu2_plus_wv2 = auv_by_wu2_plus_wv2.view_as(auv)
    # auv / (wu**2 + wv**2)
    potential_map = discrete_spectral_transform.dct2(auv_by_wu2_plus_wv2) * (1.0/density_map.numel())
    ratio = potential_map.numpy() / phi_map
    print("potential_map/phi_map")
    print(ratio)
    print(ratio.min())
    print(ratio.max())
    print(ratio.mean())
    np.testing.assert_allclose(potential_map.numpy(), phi_map, rtol=3e-1)
    exit()
    """

    #tmp1 = naive_idct2(phi_in_map)
    tmp1 = discrete_spectral_transform.idcct2(torch.from_numpy(phi_in_map)).numpy()#*phi_in_map.size
    print(tmp1)
    tmp3 = fftpack.idct(fftpack.idct(phi_in_map).T).T
    print(tmp3)
    tmp4 = naive_idct2_grad_y(phi_in_map)
    print(tmp4)
    #tmp5 = naive_idsct2(ex_in_map)
    #print(tmp5)
    tmp12 = discrete_spectral_transform.idsct2(torch.from_numpy(ex_in_map)).numpy()#*ex_in_map.size
    print(tmp12)
    tmp13 = discrete_spectral_transform.idcst2(torch.from_numpy(ey_in_map)).numpy()#*ey_in_map.size
    print(tmp13)
    pdb.set_trace()

    # compute field xi
    scipy_ex_in_map = ex_in_map
    scipy_ex_in_map[0, :] *= 2
    scipy_ex_in_map[:, 0] *= 2
    scipy_ex_in_map *= 0.25
    scipy_ex_map = fftpack.idct(fftpack.idst(scipy_ex_in_map.T).T, norm='ortho')
    ratio = scipy_ex_map / ex_out_map
    print("scipy_ex_map/ex_out_map")
    print(ratio)
    exit()


    field_map_x = discrete_spectral_transform.dsct2(auv_by_wu2_plus_wv2.mul(wu))
    field_map_y = discrete_spectral_transform.dcst2(auv_by_wu2_plus_wv2.mul(wv))

    # test dsct
    ratio = field_map_x.numpy() / ex_map
    print("field_map_x/ex_map")
    print(field_map_x)
    print(ex_map)
    print(ratio.min())
    print(ratio.max())
    print(ratio.mean())
    np.testing.assert_allclose(field_map_x.numpy(), ex_map, rtol=3e-1)

