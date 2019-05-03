# compare two different methods to calculate the electric potential
# The fitst apporach is used in the electric_potential_backup.py
# The second approach is used in the electric_potential.py

import torch
import os
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dreamplace.ops.dct import dct, discrete_spectral_transform, dct2_fft2
from dreamplace.ops.electric_potential import electric_potential, electric_overflow
sys.path.pop()


def compare_two_different_methods(M=1024, N=1024, dtype=torch.float64):
    density_map = torch.empty(M, N, dtype=dtype).uniform_(0, 10.0).cuda()
    expk_M = discrete_spectral_transform.get_expk(M, dtype, density_map.device)
    expk_N = discrete_spectral_transform.get_expk(N, dtype, density_map.device)
    expkM = discrete_spectral_transform.get_exact_expk(M, dtype, density_map.device)
    expkN = discrete_spectral_transform.get_exact_expk(N, dtype, density_map.device)
    print("M = {}, N = {}".format(M, N))

    wu = torch.arange(M, dtype=density_map.dtype, device=density_map.device).mul(2 * np.pi / M).view([M, 1])
    wv = torch.arange(N, dtype=density_map.dtype, device=density_map.device).mul(2 * np.pi / N).view([1, N])
    wu2_plus_wv2 = wu.pow(2) + wv.pow(2)
    wu2_plus_wv2[0, 0] = 1.0  # avoid zero-division, it will be zeroed out

    inv_wu2_plus_wv2_2X = 2.0 / wu2_plus_wv2
    inv_wu2_plus_wv2_2X[0, 0] = 0.0
    wu_by_wu2_plus_wv2_2X = wu.mul(inv_wu2_plus_wv2_2X)
    wv_by_wu2_plus_wv2_2X = wv.mul(inv_wu2_plus_wv2_2X)

    auv_golden = dct.dct2(density_map, expk0=expk_M, expk1=expk_N)
    auv = auv_golden.clone()
    auv[0, :].mul_(0.5)
    auv[:, 0].mul_(0.5)
    auv_by_wu2_plus_wv2_wu = auv.mul(wu_by_wu2_plus_wv2_2X)
    auv_by_wu2_plus_wv2_wv = auv.mul(wv_by_wu2_plus_wv2_2X)
    field_map_x_golden = dct.idsct2(auv_by_wu2_plus_wv2_wu, expk_M, expk_N)
    field_map_y_golden = dct.idcst2(auv_by_wu2_plus_wv2_wv, expk_M, expk_N)
    # compute potential phi
    # auv / (wu**2 + wv**2)
    auv_by_wu2_plus_wv2 = auv.mul(inv_wu2_plus_wv2_2X).mul_(2)
    #potential_map = discrete_spectral_transform.idcct2(auv_by_wu2_plus_wv2, expk_M, expk_N)
    potential_map_golden = dct.idcct2(auv_by_wu2_plus_wv2, expk_M, expk_N)
    # compute energy
    energy_golden = potential_map_golden.mul_(density_map).sum()

    if density_map.is_cuda:
        torch.cuda.synchronize()

    dct2 = dct2_fft2.DCT2(M, N, density_map.dtype, density_map.device, expkM, expkN)
    idct2 = dct2_fft2.IDCT2(M, N, density_map.dtype, density_map.device, expkM, expkN)
    idct_idxst = dct2_fft2.IDCT_IDXST(M, N, density_map.dtype, density_map.device, expkM, expkN)
    idxst_idct = dct2_fft2.IDXST_IDCT(M, N, density_map.dtype, density_map.device, expkM, expkN)

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
    energy = potential_map.mul_(density_map).sum()

    if density_map.is_cuda:
        torch.cuda.synchronize()

    np.testing.assert_allclose(buv.data.cpu().numpy(), auv_golden.data.cpu().numpy(), rtol=1e-6, atol=1e-5)
    np.testing.assert_allclose(field_map_x.data.cpu().numpy(), field_map_x_golden.data.cpu().numpy(), rtol=1e-6, atol=1e-5)
    np.testing.assert_allclose(field_map_y.data.cpu().numpy(), field_map_y_golden.data.cpu().numpy(), rtol=1e-6, atol=1e-5)
    np.testing.assert_allclose(potential_map.data.cpu().numpy(), potential_map_golden.data.cpu().numpy(), rtol=1e-6, atol=1e-5)
    np.testing.assert_allclose(energy.data.cpu().numpy(), energy_golden.data.cpu().numpy(), rtol=1e-6, atol=1e-5)


if __name__ == "__main__":
    compare_two_different_methods(M=1024, N=1024, dtype=torch.float64)
