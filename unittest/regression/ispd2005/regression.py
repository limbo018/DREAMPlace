##
# @file   regression.py
# @author Yibo Lin
# @date   Apr 2020
#

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
    'adaptec1',
    'adaptec2',
    'adaptec3',
    'adaptec4',
    'bigblue1',
    'bigblue2',
    'bigblue3',
    'bigblue4',
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
    ('adaptec1', 'gpu', 'deterministic'): {
        "GP": 7.023626E+07,
        "LG": 7.381382E+07,
        "DP": 7.275838E+07
    },
    ('adaptec1', 'gpu', 'indeterministic'): {
        "GP": 7.023626E+07,
        "LG": 7.381382E+07,
        "DP": 7.275838E+07
    },
    ('adaptec1', 'cpu', 'deterministic'): {
        "GP": 7.023626E+07,
        "LG": 7.381382E+07,
        "DP": 7.275838E+07
    },
    ('adaptec1', 'cpu', 'indeterministic'): {
        "GP": 7.023626E+07,
        "LG": 7.381382E+07,
        "DP": 7.275838E+07
    },
    ('adaptec2', 'gpu', 'deterministic'): {
        "GP": 7.921833E+07,
        "LG": 8.297632E+07,
        "DP": 8.187338E+07
    },
    ('adaptec2', 'gpu', 'indeterministic'): {
        "GP": 7.921833E+07,
        "LG": 8.297632E+07,
        "DP": 8.187338E+07
    },
    ('adaptec2', 'cpu', 'deterministic'): {
        "GP": 7.921833E+07,
        "LG": 8.297632E+07,
        "DP": 8.187338E+07
    },
    ('adaptec2', 'cpu', 'indeterministic'): {
        "GP": 7.921833E+07,
        "LG": 8.297632E+07,
        "DP": 8.187338E+07
    },
    ('adaptec3', 'gpu', 'deterministic'): {
        "GP": 1.856957E+08,
        "LG": 1.977917E+08,
        "DP": 1.928894E+08
    },
    ('adaptec3', 'gpu', 'indeterministic'): {
        "GP": 1.856957E+08,
        "LG": 1.977917E+08,
        "DP": 1.928894E+08
    },
    ('adaptec3', 'cpu', 'deterministic'): {
        "GP": 1.856957E+08,
        "LG": 1.977917E+08,
        "DP": 1.928894E+08
    },
    ('adaptec3', 'cpu', 'indeterministic'): {
        "GP": 1.856957E+08,
        "LG": 1.977917E+08,
        "DP": 1.928894E+08
    },
    ('adaptec4', 'gpu', 'deterministic'): {
        "GP": 1.688209E+08,
        "LG": 1.774909E+08,
        "DP": 1.737706E+08
    },
    ('adaptec4', 'gpu', 'indeterministic'): {
        "GP": 1.688209E+08,
        "LG": 1.774909E+08,
        "DP": 1.737706E+08
    },
    ('adaptec4', 'cpu', 'deterministic'): {
        "GP": 1.688209E+08,
        "LG": 1.774909E+08,
        "DP": 1.737706E+08
    },
    ('adaptec4', 'cpu', 'indeterministic'): {
        "GP": 1.688209E+08,
        "LG": 1.774909E+08,
        "DP": 1.737706E+08
    },
    ('bigblue1', 'gpu', 'deterministic'): {
        "GP": 8.733162E+07,
        "LG": 8.978584E+07,
        "DP": 8.922830E+07
    },
    ('bigblue1', 'gpu', 'indeterministic'): {
        "GP": 8.733162E+07,
        "LG": 8.978584E+07,
        "DP": 8.922830E+07
    },
    ('bigblue1', 'cpu', 'deterministic'): {
        "GP": 8.733162E+07,
        "LG": 8.978584E+07,
        "DP": 8.922830E+07
    },
    ('bigblue1', 'cpu', 'indeterministic'): {
        "GP": 8.733162E+07,
        "LG": 8.978584E+07,
        "DP": 8.922830E+07
    },
    ('bigblue2', 'gpu', 'deterministic'): {
        "GP": 1.311660E+08,
        "LG": 1.392559E+08,
        "DP": 1.369483E+08
    },
    ('bigblue2', 'gpu', 'indeterministic'): {
        "GP": 1.311660E+08,
        "LG": 1.392559E+08,
        "DP": 1.369483E+08
    },
    ('bigblue2', 'cpu', 'deterministic'): {
        "GP": 1.311660E+08,
        "LG": 1.392559E+08,
        "DP": 1.369483E+08
    },
    ('bigblue2', 'cpu', 'indeterministic'): {
        "GP": 1.311660E+08,
        "LG": 1.392559E+08,
        "DP": 1.369483E+08
    },
    ('bigblue3', 'gpu', 'deterministic'): {
        "GP": 2.916528E+08,
        "LG": 3.102106E+08,
        "DP": 3.039599E+08
    },
    ('bigblue3', 'gpu', 'indeterministic'): {
        "GP": 2.916528E+08,
        "LG": 3.102106E+08,
        "DP": 3.039599E+08
    },
    ('bigblue3', 'cpu', 'deterministic'): {
        "GP": 2.916528E+08,
        "LG": 3.102106E+08,
        "DP": 3.039599E+08
    },
    ('bigblue3', 'cpu', 'indeterministic'): {
        "GP": 2.916528E+08,
        "LG": 3.102106E+08,
        "DP": 3.039599E+08
    },
    ('bigblue4', 'gpu', 'deterministic'): {
        "GP": 7.255144E+08,
        "LG": 7.485191E+08,
        "DP": 7.426981E+08
    },
    ('bigblue4', 'gpu', 'indeterministic'): {
        "GP": 7.255144E+08,
        "LG": 7.485191E+08,
        "DP": 7.426981E+08
    },
    ('bigblue4', 'cpu', 'deterministic'): {
        "GP": 7.255144E+08,
        "LG": 7.485191E+08,
        "DP": 7.426981E+08
    },
    ('bigblue4', 'cpu', 'indeterministic'): {
        "GP": 7.255144E+08,
        "LG": 7.485191E+08,
        "DP": 7.426981E+08
    },
}


class ISPD2005Test(unittest.TestCase):
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
                        filename="regression_ispd2005.log")

    unittest.main()
