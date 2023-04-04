#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : regression.py
# Author            : Yibo Lin <yibolin@pku.edu.cn>
# Date              : 04.03.2023
# Last Modified Date: 04.03.2023
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
    'superblue1',
    'superblue3',
    'superblue4',
    'superblue5',
    'superblue7',
    'superblue10',
    'superblue16',
    'superblue18',
]
# gpu flag
#devices = ['cpu']
devices = []
if configure.compile_configurations[
        "CUDA_FOUND"] == "TRUE" and torch.cuda.device_count():
    devices.append('gpu')
# deterministic flags
deterministics = ['deterministic', 'indeterministic']

# Yibo: I assume the results of different modes should be less than 0.5%
golden = {
}


class ICCAD2015Test(unittest.TestCase):
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
                    params.detaield_place_flag = 0
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
                    logging.info("verify %s, %s, %s" %
                                 (design, device_name, deterministic_name))
                    logging.info("LG TNS %.6f, WNS %.6f" % 
                            (metrics[-1].tns, metrics[-1].wns))
                    #np.testing.assert_allclose(
                    #    golden[(design, device_name,
                    #            deterministic_name)]["LG"]["TNS"],
                    #    metrics[-2].tns,
                    #    rtol=0.010)
                    #np.testing.assert_allclose(
                    #    golden[(design, device_name,
                    #            deterministic_name)]["LG"]["WNS"],
                    #    metrics[-2].wns,
                    #    rtol=0.010)


if __name__ == "__main__":
    """
    @brief main function to invoke the entire placement flow. 
    """
    logging.root.name = 'DREAMPlace'
    logging.basicConfig(level=logging.INFO,
                        format='[%(levelname)-7s] %(name)s - %(message)s',
                        filename="regression_iccad2015.ot.log")

    unittest.main()
