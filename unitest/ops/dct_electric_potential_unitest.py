##
# @file   dct_electrical_potential_unitest.py
# @author Zixuan Jiang, Jiaqi Gu
# @date   Mar 2019
# @brief  compare two different transforms to calculate the electric potential
#       The fitst apporach is used in electric_potential_backup.py
#       The second approach is used in electric_potential.py

import torch
import os
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dreamplace.ops.dct import dct, discrete_spectral_transform, dct2_fft2
sys.path.pop()


def compare_different_methods(cuda_flag, M=1024, N=1024, dtype=torch.float64):
    density_map = torch.empty(M, N, dtype=dtype).uniform_(0, 10.0)
    if cuda_flag:
        density_map = density_map.cuda()
    expkM = discrete_spectral_transform.get_expk(M, dtype, density_map.device)
    expkN = discrete_spectral_transform.get_expk(N, dtype, density_map.device)
    exact_expkM = discrete_spectral_transform.get_exact_expk(M, dtype, density_map.device)
    exact_expkN = discrete_spectral_transform.get_exact_expk(N, dtype, density_map.device)
    print("M = {}, N = {}".format(M, N))

    wu = torch.arange(M, dtype=density_map.dtype, device=density_map.device).mul(2 * np.pi / M).view([M, 1])
    wv = torch.arange(N, dtype=density_map.dtype, device=density_map.device).mul(2 * np.pi / N).view([1, N])
    wu2_plus_wv2 = wu.pow(2) + wv.pow(2)
    wu2_plus_wv2[0, 0] = 1.0  # avoid zero-division, it will be zeroed out

    inv_wu2_plus_wv2_2X = 2.0 / wu2_plus_wv2
    inv_wu2_plus_wv2_2X[0, 0] = 0.0
    wu_by_wu2_plus_wv2_2X = wu.mul(inv_wu2_plus_wv2_2X)
    wv_by_wu2_plus_wv2_2X = wv.mul(inv_wu2_plus_wv2_2X)

    # the first approach is used as the ground truth
    auv_golden = dct.dct2(density_map, expk0=expkM, expk1=expkN)
    auv = auv_golden.clone()
    auv[0, :].mul_(0.5)
    auv[:, 0].mul_(0.5)
    auv_by_wu2_plus_wv2_wu = auv.mul(wu_by_wu2_plus_wv2_2X)
    auv_by_wu2_plus_wv2_wv = auv.mul(wv_by_wu2_plus_wv2_2X)
    field_map_x_golden = dct.idsct2(auv_by_wu2_plus_wv2_wu, expkM, expkN)
    field_map_y_golden = dct.idcst2(auv_by_wu2_plus_wv2_wv, expkM, expkN)
    # compute potential phi
    # auv / (wu**2 + wv**2)
    auv_by_wu2_plus_wv2 = auv.mul(inv_wu2_plus_wv2_2X).mul_(2)
    #potential_map = discrete_spectral_transform.idcct2(auv_by_wu2_plus_wv2, expkM, expkN)
    potential_map_golden = dct.idcct2(auv_by_wu2_plus_wv2, expkM, expkN)
    # compute energy
    energy_golden = potential_map_golden.mul(density_map).sum()

    if density_map.is_cuda:
        torch.cuda.synchronize()

    # the second approach uses the idxst_idct and idct_idxst
    dct2 = dct2_fft2.DCT2(exact_expkM, exact_expkN)
    idct2 = dct2_fft2.IDCT2(exact_expkM, exact_expkN)
    idct_idxst = dct2_fft2.IDCT_IDXST(exact_expkM, exact_expkN)
    idxst_idct = dct2_fft2.IDXST_IDCT(exact_expkM, exact_expkN)

    inv_wu2_plus_wv2 = 1.0 / wu2_plus_wv2
    inv_wu2_plus_wv2[0, 0] = 0.0
    wu_by_wu2_plus_wv2_half = wu.mul(inv_wu2_plus_wv2).mul_(0.5)
    wv_by_wu2_plus_wv2_half = wv.mul(inv_wu2_plus_wv2).mul_(0.5)

    buv = dct2.forward(density_map)

    buv_by_wu2_plus_wv2_wu = buv.mul(wu_by_wu2_plus_wv2_half)
    buv_by_wu2_plus_wv2_wv = buv.mul(wv_by_wu2_plus_wv2_half)
    field_map_x = idxst_idct.forward(buv_by_wu2_plus_wv2_wu)
    field_map_y = idct_idxst.forward(buv_by_wu2_plus_wv2_wv)
    buv_by_wu2_plus_wv2 = buv.mul(inv_wu2_plus_wv2)
    potential_map = idct2.forward(buv_by_wu2_plus_wv2)
    energy = potential_map.mul(density_map).sum()

    if density_map.is_cuda:
        torch.cuda.synchronize()

    # compare results
    np.testing.assert_allclose(buv.data.cpu().numpy(), auv_golden.data.cpu().numpy(), rtol=1e-6, atol=1e-5)
    np.testing.assert_allclose(field_map_x.data.cpu().numpy(), field_map_x_golden.data.cpu().numpy(), rtol=1e-6, atol=1e-5)
    np.testing.assert_allclose(field_map_y.data.cpu().numpy(), field_map_y_golden.data.cpu().numpy(), rtol=1e-6, atol=1e-5)
    np.testing.assert_allclose(potential_map.data.cpu().numpy(), potential_map_golden.data.cpu().numpy(), rtol=1e-6, atol=1e-5)
    np.testing.assert_allclose(energy.data.cpu().numpy(), energy_golden.data.cpu().numpy(), rtol=1e-6, atol=1e-5)

    # the third approach uses the dct.idxst_idct and dct.idxst_idct
    dct2 = dct.DCT2(expkM, expkN)
    idct2 = dct.IDCT2(expkM, expkN)
    idct_idxst = dct.IDCT_IDXST(expkM, expkN)
    idxst_idct = dct.IDXST_IDCT(expkM, expkN)

    cuv = dct2.forward(density_map)

    cuv_by_wu2_plus_wv2_wu = cuv.mul(wu_by_wu2_plus_wv2_half)
    cuv_by_wu2_plus_wv2_wv = cuv.mul(wv_by_wu2_plus_wv2_half)
    field_map_x = idxst_idct.forward(cuv_by_wu2_plus_wv2_wu)
    field_map_y = idct_idxst.forward(cuv_by_wu2_plus_wv2_wv)
    cuv_by_wu2_plus_wv2 = cuv.mul(inv_wu2_plus_wv2)
    potential_map = idct2.forward(cuv_by_wu2_plus_wv2)
    energy = potential_map.mul(density_map).sum()

    if density_map.is_cuda:
        torch.cuda.synchronize()

    # compare results
    np.testing.assert_allclose(cuv.data.cpu().numpy(), auv_golden.data.cpu().numpy(), rtol=1e-6, atol=1e-5)
    np.testing.assert_allclose(field_map_x.data.cpu().numpy(), field_map_x_golden.data.cpu().numpy(), rtol=1e-6, atol=1e-5)
    np.testing.assert_allclose(field_map_y.data.cpu().numpy(), field_map_y_golden.data.cpu().numpy(), rtol=1e-6, atol=1e-5)
    np.testing.assert_allclose(potential_map.data.cpu().numpy(), potential_map_golden.data.cpu().numpy(), rtol=1e-6, atol=1e-5)
    np.testing.assert_allclose(energy.data.cpu().numpy(), energy_golden.data.cpu().numpy(), rtol=1e-6, atol=1e-5)


if __name__ == "__main__":
    compare_different_methods(cuda_flag=False, M=1024, N=1024, dtype=torch.float64)
    if torch.cuda.device_count(): 
        compare_different_methods(cuda_flag=True, M=1024, N=1024, dtype=torch.float64)
