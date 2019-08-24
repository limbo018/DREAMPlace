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
from dreamplace.ops.electric_potential import electric_potential
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
                        y[u, v] += x[i, j] * np.cos(2 * np.pi / M * i * u) * np.cos(2 * np.pi / N * j * v)
                    elif transform_type == 'cossin':
                        y[u, v] += x[i, j] * np.cos(2 * np.pi / M * i * u) * np.sin(2 * np.pi / N * j * v)
                    elif transform_type == 'sincos':
                        y[u, v] += x[i, j] * np.sin(2 * np.pi / M * i * u) * np.cos(2 * np.pi / N * j * v)
                    elif transform_type == 'sinsin':
                        y[u, v] += x[i, j] * np.sin(2 * np.pi / M * i * u) * np.sin(2 * np.pi / N * j * v)
            if normalize:
                y[u, v] *= 1.0 / (M * N)
    return y


def naive_idct(x):
    N = len(x)
    y = np.zeros(N)
    y += 0.5 * x[0]
    for u in range(N):
        for k in range(1, N):
            y[u] += x[k] * np.cos(np.pi / (2 * N) * k * (2 * u + 1))

    return y


def naive_idct2(x):
    M = x.shape[0]
    N = x.shape[1]

    y = np.zeros_like(x)

    # for u in range(M):
    #    y[u] = naive_idct(x[u])
    #y = np.transpose(y)
    # for v in range(N):
    #    y[v] = naive_idct(y[v])
    #y = np.transpose(y)

    for u in range(M):
        for v in range(N):
            for p in range(M):
                for q in range(N):
                    a = 1  # if p == 0 else 2
                    b = 1  # if q == 0 else 2
                    y[u, v] += a * b * x[p, q] * np.cos(np.pi / M * p * (u + 0.5)) * np.cos(np.pi / N * q * (v + 0.5))
    return y


def naive_idct2_grad_x(x):
    M = x.shape[0]
    N = x.shape[1]

    y = np.zeros_like(x)
    for u in range(M):
        for v in range(N):
            for p in range(M):
                for q in range(N):
                    a = 1  # if p == 0 else 2
                    b = 1  # if q == 0 else 2
                    y[u, v] += -a * b * x[p, q] * (np.pi / M * p) * np.sin(np.pi / M *
                                                                           p * (u + 0.5)) * np.cos(np.pi / N * q * (v + 0.5))
    return y


def naive_idct2_grad_y(x):
    M = x.shape[0]
    N = x.shape[1]

    y = np.zeros_like(x)
    for u in range(M):
        for v in range(N):
            for p in range(M):
                for q in range(N):
                    a = 1  # if p == 0 else 2
                    b = 1  # if q == 0 else 2
                    y[u, v] += -a * b * x[p, q] * (np.pi / N * q) * np.cos(np.pi / M *
                                                                           p * (u + 0.5)) * np.sin(np.pi / N * q * (v + 0.5))
    return y


def naive_idsct2(x):
    M = x.shape[0]
    N = x.shape[1]

    y = np.zeros_like(x)
    for u in range(M):
        for v in range(N):
            for p in range(M):
                for q in range(N):
                    a = 1  # if p == 0 else 2
                    b = 1  # if q == 0 else 2
                    y[u, v] += -a * b * x[p, q] * np.sin(np.pi / M * p * (u + 0.5)) * np.cos(np.pi / N * q * (v + 0.5))
    return y


def naive_idsct2_2(x):
    M = x.shape[0]
    N = x.shape[1]

    y1 = np.zeros_like(x)

    # for u in range(M):
    #    y1[u] = naive_idct(x[u])
    for v in range(N):
        for p in range(M):
            for q in range(N):
                y1[p, v] += x[p, q] * np.cos(np.pi / N * q * (v + 0.5))
    #print("y1 = ", y1)
    # for u in range(M):
    #    for v in range(N):
    #        for p in range(M):
    #            for q in range(N):
    #                y1[u, p] += x[p, q]*np.cos(np.pi/N*q*(v+0.5))

    #y1 = np.transpose(y1)
    y2 = np.zeros_like(x)
    for u in range(M):
        for v in range(N):
            for p in range(M):
                y2[u, v] += -y1[p, v] * np.sin(np.pi / M * p * (u + 0.5))
    # for u in range(M):
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

# class SpectralOpTest(unittest.TestCase):
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
        xh = 6.0
        yh = 6.0
        bin_size_x = 1.0
        bin_size_y = 1.0
        target_density = 0.1
        num_bins_x = int(np.ceil((xh - xl) / bin_size_x))
        num_bins_y = int(np.ceil((yh - yl) / bin_size_y))

        """
        return bin xl
        """
        def bin_xl(id_x):
            return xl + id_x * bin_size_x

        """
        return bin xh
        """
        def bin_xh(id_x):
            return min(bin_xl(id_x) + bin_size_x, xh)

        """
        return bin yl
        """
        def bin_yl(id_y):
            return yl + id_y * bin_size_y

        """
        return bin yh
        """
        def bin_yh(id_y):
            return min(bin_yl(id_y) + bin_size_y, yh)

        bin_center_x = np.zeros(num_bins_x, dtype=dtype)
        for id_x in range(num_bins_x):
            bin_center_x[id_x] = (bin_xl(id_x) + bin_xh(id_x)) / 2 * scale_factor

        bin_center_y = np.zeros(num_bins_y, dtype=dtype)
        for id_y in range(num_bins_y):
            bin_center_y[id_y] = (bin_yl(id_y) + bin_yh(id_y)) / 2 * scale_factor

        print("target_area = ", target_density * bin_size_x * bin_size_y)

        if dtype == np.float64:
            dtype = torch.float64
        elif dtype == np.float32:
            dtype = torch.float32
        movable_size_x = node_size_x[:num_nodes]
        _, sorted_node_map = torch.sort(torch.tensor(movable_size_x,requires_grad=False, dtype=dtype))
        sorted_node_map = sorted_node_map.to(torch.int32).contiguous()
        # test cpu
        custom = electric_potential.ElectricPotential(
            torch.tensor(node_size_x, requires_grad=False, dtype=dtype), torch.tensor(
                node_size_y, requires_grad=False, dtype=dtype),
            torch.tensor(bin_center_x, requires_grad=False, dtype=dtype), torch.tensor(
                bin_center_y, requires_grad=False, dtype=dtype),
            target_density=torch.tensor(target_density, requires_grad=False, dtype=dtype),
            xl=xl, yl=yl, xh=xh, yh=yh,
            bin_size_x=bin_size_x, bin_size_y=bin_size_y,
            num_movable_nodes=num_nodes,
            num_terminals=0,
            num_filler_nodes=0,
            padding=0,
            sorted_node_map=sorted_node_map
        )

        pos = Variable(torch.from_numpy(np.concatenate([xx, yy])), requires_grad=True)
        result = custom.forward(pos)
        print("custom_result = ", result)
        print(result.type())
        result.backward()
        grad = pos.grad.clone()
        print("custom_grad = ", grad)

        # test cuda
        if torch.cuda.device_count():
            custom_cuda = electric_potential.ElectricPotential(
                        torch.tensor(node_size_x, requires_grad=False, dtype=dtype).cuda(), torch.tensor(node_size_y, requires_grad=False, dtype=dtype).cuda(),
                        torch.tensor(bin_center_x, requires_grad=False, dtype=dtype).cuda(), torch.tensor(bin_center_y, requires_grad=False, dtype=dtype).cuda(),
                        target_density=torch.tensor(target_density, requires_grad=False, dtype=dtype).cuda(),
                        xl=xl, yl=yl, xh=xh, yh=yh,
                        bin_size_x=bin_size_x, bin_size_y=bin_size_y,
                        num_movable_nodes=num_nodes,
                        num_terminals=0,
                        num_filler_nodes=0,
                        padding=0,
                        sorted_node_map=sorted_node_map.cuda()
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
    density_map = density_map[padding:density_map.shape[0] - padding, padding:density_map.shape[1] - padding]
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

    # plt.tight_layout()
    plt.savefig(name + ".3d.png")
    plt.close()

    # plt.clf()

    #fig, ax = plt.subplots()

    # ax.pcolor(density_map)

    # Loop over data dimensions and create text annotations.
    # for i in range(density_map.shape[0]):
    # for j in range(density_map.shape[1]):
    # text = ax.text(j, i, density_map[i, j],
    # ha="center", va="center", color="w")
    # fig.tight_layout()
    #plt.savefig(name+".2d.%d.png" % (plot_count))
    # plt.close()


def eval_runtime(design):
    # e.g., adaptec1_density.pklz
    with gzip.open(design, "rb") as f:
        node_size_x, node_size_y, bin_center_x, bin_center_y, target_density, xl, yl, xh, yh, bin_size_x, bin_size_y, num_movable_nodes, num_terminals, num_filler_nodes = pickle.load(
            f)

    dtype = torch.float64
    movable_size_x = node_size_x[:num_movable_nodes]
    _, sorted_node_map = torch.sort(torch.tensor(movable_size_x,requires_grad=False, dtype=dtype).cuda())
    sorted_node_map = sorted_node_map.to(torch.int32).contiguous()

    pos_var = Variable(torch.empty(len(node_size_x) * 2, dtype=dtype).uniform_(xl, xh), requires_grad=True)
    custom = electric_potential.ElectricPotential(
        torch.tensor(node_size_x, requires_grad=False, dtype=dtype).cpu(),
        torch.tensor(node_size_y, requires_grad=False, dtype=dtype).cpu(),
        torch.tensor(bin_center_x, requires_grad=False, dtype=dtype).cpu(),
        torch.tensor(bin_center_y, requires_grad=False, dtype=dtype).cpu(),
        target_density=torch.tensor(target_density, requires_grad=False, dtype=dtype).cpu(),
        xl=xl, yl=yl, xh=xh, yh=yh,
        bin_size_x=bin_size_x, bin_size_y=bin_size_y,
        num_movable_nodes=num_movable_nodes,
        num_terminals=num_terminals,
        num_filler_nodes=num_filler_nodes,
        padding=0,
        sorted_node_map=sorted_node_map.cpu(), 
        num_threads=1
        )

    custom_cuda = electric_potential.ElectricPotential(
        torch.tensor(node_size_x, requires_grad=False, dtype=dtype).cuda(),
        torch.tensor(node_size_y, requires_grad=False, dtype=dtype).cuda(),
        torch.tensor(bin_center_x, requires_grad=False, dtype=dtype).cuda(),
        torch.tensor(bin_center_y, requires_grad=False, dtype=dtype).cuda(),
        target_density=torch.tensor(target_density, requires_grad=False, dtype=dtype).cuda(),
        xl=xl, yl=yl, xh=xh, yh=yh,
        bin_size_x=bin_size_x, bin_size_y=bin_size_y,
        num_movable_nodes=num_movable_nodes,
        num_terminals=num_terminals,
        num_filler_nodes=num_filler_nodes,
        padding=0,
        sorted_node_map=sorted_node_map)

    torch.cuda.synchronize()
    iters = 100
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
    print("custom_cuda takes %.3f ms" % ((time.time() - tt) / iters * 1000))


if __name__ == '__main__':
    #unittest.main()

    design = sys.argv[1]
    eval_runtime(design)
