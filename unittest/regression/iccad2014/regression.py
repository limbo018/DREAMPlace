#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : regression.py
# Author            : Yibo Lin <yibolin@pku.edu.cn>
# Date              : 04.30.2020
# Last Modified Date: 04.30.2020
# Last Modified By  : Yibo Lin <yibolin@pku.edu.cn>

import os
import sys
import numpy as np
import unittest
import logging
import time
import pdb

import torch
from torch.autograd import Function, Variable
sys.path.append(
    os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
        "dreamplace"))
import Placer
import Params
import configure
sys.path.pop()

designs = [
    'vga_lcd',
    'b19',
    'mgc_edit_dist',
    'mgc_matrix_mult',
    'leon3mp',
    'leon2',
    'netcard',
]
# gpu flag
devices = ['cpu']
if configure.compile_configurations[
        "CUDA_FOUND"] == "TRUE" and torch.cuda.device_count():
    devices.append('gpu')
# deterministic flags
deterministics = ['deterministic', 'indeterministic']

# Yibo: I assume the results of different modes should be less than 0.5%
golden = {
    ('vga_lcd', 'gpu', 'deterministic'): {
        "GP": 2.575512E+07,
        "LG": 2.742744E+07,
        "DP": 2.701463E+07
    },
    ('vga_lcd', 'gpu', 'indeterministic'): {
        "GP": 2.575512E+07,
        "LG": 2.742744E+07,
        "DP": 2.701463E+07
    },
    ('vga_lcd', 'cpu', 'deterministic'): {
        "GP": 2.575512E+07,
        "LG": 2.742744E+07,
        "DP": 2.701463E+07
    },
    ('vga_lcd', 'cpu', 'indeterministic'): {
        "GP": 2.575512E+07,
        "LG": 2.742744E+07,
        "DP": 2.701463E+07
    },
    ('b19', 'gpu', 'deterministic'): {
        "GP": 2.437919E+07,
        "LG": 2.709975E+07,
        "DP": 2.636430E+07
    },
    ('b19', 'gpu', 'indeterministic'): {
        "GP": 2.437919E+07,
        "LG": 2.709975E+07,
        "DP": 2.636430E+07
    },
    ('b19', 'cpu', 'deterministic'): {
        "GP": 2.437919E+07,
        "LG": 2.709975E+07,
        "DP": 2.636430E+07
    },
    ('b19', 'cpu', 'indeterministic'): {
        "GP": 2.437919E+07,
        "LG": 2.709975E+07,
        "DP": 2.636430E+07
    },
    ('mgc_edit_dist', 'gpu', 'deterministic'): {
        "GP": 3.899874E+07,
        "LG": 4.038984E+07,
        "DP": 3.977641E+07
    },
    ('mgc_edit_dist', 'gpu', 'indeterministic'): {
        "GP": 3.899874E+07,
        "LG": 4.038984E+07,
        "DP": 3.977641E+07
    },
    ('mgc_edit_dist', 'cpu', 'deterministic'): {
        "GP": 3.899874E+07,
        "LG": 4.038984E+07,
        "DP": 3.977641E+07
    },
    ('mgc_edit_dist', 'cpu', 'indeterministic'): {
        "GP": 3.899874E+07,
        "LG": 4.038984E+07,
        "DP": 3.977641E+07
    },
    ('mgc_matrix_mult', 'gpu', 'deterministic'): {
        "GP": 2.207437E+07,
        "LG": 2.454287E+07,
        "DP": 2.380662E+07
    },
    ('mgc_matrix_mult', 'gpu', 'indeterministic'): {
        "GP": 2.207437E+07,
        "LG": 2.454287E+07,
        "DP": 2.380662E+07
    },
    ('mgc_matrix_mult', 'cpu', 'deterministic'): {
        "GP": 2.207437E+07,
        "LG": 2.454287E+07,
        "DP": 2.380662E+07
    },
    ('mgc_matrix_mult', 'cpu', 'indeterministic'): {
        "GP": 2.207437E+07,
        "LG": 2.454287E+07,
        "DP": 2.380662E+07
    },
    ('leon3mp', 'gpu', 'deterministic'): {
        "GP": 1.003386E+08,
        "LG": 1.087540E+08,
        "DP": 1.071010E+08
    },
    ('leon3mp', 'gpu', 'indeterministic'): {
        "GP": 1.003386E+08,
        "LG": 1.087540E+08,
        "DP": 1.071010E+08
    },
    ('leon3mp', 'cpu', 'deterministic'): {
        "GP": 1.003386E+08,
        "LG": 1.087540E+08,
        "DP": 1.071010E+08
    },
    ('leon3mp', 'cpu', 'indeterministic'): {
        "GP": 1.003386E+08,
        "LG": 1.087540E+08,
        "DP": 1.071010E+08
    },
    ('leon2', 'gpu', 'deterministic'): {
        "GP": 2.232737E+08,
        "LG": 2.343749E+08,
        "DP": 2.321112E+08
    },
    ('leon2', 'gpu', 'indeterministic'): {
        "GP": 2.232737E+08,
        "LG": 2.343749E+08,
        "DP": 2.321112E+08
    },
    ('leon2', 'cpu', 'deterministic'): {
        "GP": 2.232737E+08,
        "LG": 2.343749E+08,
        "DP": 2.321112E+08
    },
    ('leon2', 'cpu', 'indeterministic'): {
        "GP": 2.232737E+08,
        "LG": 2.343749E+08,
        "DP": 2.321112E+08
    },
    ('netcard', 'gpu', 'deterministic'): {
        "GP": 2.787224E+08,
        "LG": 2.879165E+08,
        "DP": 2.852960E+08
    },
    ('netcard', 'gpu', 'indeterministic'): {
        "GP": 2.787224E+08,
        "LG": 2.879165E+08,
        "DP": 2.852960E+08
    },
    ('netcard', 'cpu', 'deterministic'): {
        "GP": 2.787224E+08,
        "LG": 2.879165E+08,
        "DP": 2.852960E+08
    },
    ('netcard', 'cpu', 'indeterministic'): {
        "GP": 2.787224E+08,
        "LG": 2.879165E+08,
        "DP": 2.852960E+08
    },
}


class ICCAD2014Test(unittest.TestCase):
    def testAll(self):
        for design in designs:
            json_file = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "%s.json" % (design))
            params = Params.Params()
            params.load(json_file)
            # control numpy multithreading
            os.environ["OMP_NUM_THREADS"] = "%d" % (params.num_threads)
            for device_name in devices:
                for deterministic_name in deterministics:
                    params.gpu = 0 if device_name == 'cpu' else 1
                    params.deterministic_flag = 0 if deterministic_name == 'indeterministic' else 1
                    params.global_place_flag = 1
                    params.legalize_flag = 1
                    params.detaield_place_flag = 1
                    params.detailed_place_engine = ""
                    logging.info("%s, %s, %s" %
                                 (design, device_name, deterministic_name))
                    logging.info("parameters = %s" % (params))
                    # run placement
                    tt = time.time()
                    metrics = Placer.place(params)
                    logging.info("placement takes %.3f seconds" %
                                 (time.time() - tt))
                    # verify global placement results
                    np.testing.assert_allclose(
                        golden[(design, device_name,
                                deterministic_name)]["GP"],
                        metrics[-3][-1][-1].hpwl.cpu().numpy(),
                        rtol=0.010)
                    np.testing.assert_allclose(
                        golden[(design, device_name,
                                deterministic_name)]["LG"],
                        metrics[-2].hpwl.cpu().numpy(),
                        rtol=0.010)
                    np.testing.assert_allclose(
                        golden[(design, device_name,
                                deterministic_name)]["DP"],
                        metrics[-1].hpwl.cpu().numpy(),
                        rtol=0.005)


if __name__ == "__main__":
    """
    @brief main function to invoke the entire placement flow. 
    """
    logging.root.name = 'DREAMPlace'
    logging.basicConfig(level=logging.INFO,
                        format='[%(levelname)-7s] %(name)s - %(message)s',
                        filename="regression_iccad2014.log")

    unittest.main()
