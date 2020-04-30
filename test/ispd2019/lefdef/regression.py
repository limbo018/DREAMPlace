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

# Yibo: I observe that this set of benchmarks result in quite different HPWLs on CPU from GPU.
# So the tolerance of comparison is set to large, e.g., 1%.
# Probably the difference comes from DP.
designs = [
    #'ispd19_test1', # not converged
    'ispd19_test2',
    'ispd19_test3',
    'ispd19_test4',
    #'ispd19_test5', # fence region, cannot handle yet
    'ispd19_test6',
    'ispd19_test7',
    'ispd19_test8',
    'ispd19_test9',
    'ispd19_test10',
]
# gpu flag
devices = ['cpu']
if configure.compile_configurations[
        "CUDA_FOUND"] == "TRUE" and torch.cuda.device_count():
    devices.append('gpu')
# deterministic flags
deterministics = ['deterministic', 'indeterministic']

# Yibo: I assume the results of different modes should be small
golden = {
    ('ispd19_test1', 'gpu', 'deterministic'): {
        "GP": 4.304241E+05,
        "LG": 4.481719E+05,
        "DP": 4.163562E+05
    },
    ('ispd19_test1', 'gpu', 'indeterministic'): {
        "GP": 4.304241E+05,
        "LG": 4.481719E+05,
        "DP": 4.163562E+05
    },
    ('ispd19_test1', 'cpu', 'deterministic'): {
        "GP": 4.304241E+05,
        "LG": 4.481719E+05,
        "DP": 4.163562E+05
    },
    ('ispd19_test1', 'cpu', 'indeterministic'): {
        "GP": 4.304241E+05,
        "LG": 4.481719E+05,
        "DP": 4.163562E+05
    },
    ('ispd19_test2', 'gpu', 'deterministic'): {
        "GP": 1.718549E+07,
        "LG": 1.808001E+07,
        "DP": 1.791062E+07
    },
    ('ispd19_test2', 'gpu', 'indeterministic'): {
        "GP": 1.718549E+07,
        "LG": 1.808001E+07,
        "DP": 1.791062E+07
    },
    ('ispd19_test2', 'cpu', 'deterministic'): {
        "GP": 1.718549E+07,
        "LG": 1.808001E+07,
        "DP": 1.791062E+07
    },
    ('ispd19_test2', 'cpu', 'indeterministic'): {
        "GP": 1.718549E+07,
        "LG": 1.808001E+07,
        "DP": 1.791062E+07
    },
    ('ispd19_test3', 'gpu', 'deterministic'): {
        "GP": 6.294804E+05,
        "LG": 7.462954E+05,
        "DP": 7.086963E+05
    },
    ('ispd19_test3', 'gpu', 'indeterministic'): {
        "GP": 6.294804E+05,
        "LG": 7.462954E+05,
        "DP": 7.086963E+05
    },
    ('ispd19_test3', 'cpu', 'deterministic'): {
        "GP": 6.294804E+05,
        "LG": 7.462954E+05,
        "DP": 7.086963E+05
    },
    ('ispd19_test3', 'cpu', 'indeterministic'): {
        "GP": 6.294804E+05,
        "LG": 7.462954E+05,
        "DP": 7.086963E+05
    },
    ('ispd19_test4', 'gpu', 'deterministic'): {
        "GP": 1.660677E+07,
        "LG": 1.802024E+07,
        "DP": 1.721041E+07
    },
    ('ispd19_test4', 'gpu', 'indeterministic'): {
        "GP": 1.660677E+07,
        "LG": 1.802024E+07,
        "DP": 1.721041E+07
    },
    ('ispd19_test4', 'cpu', 'deterministic'): {
        "GP": 1.660677E+07,
        "LG": 1.802024E+07,
        "DP": 1.721041E+07
    },
    ('ispd19_test4', 'cpu', 'indeterministic'): {
        "GP": 1.660677E+07,
        "LG": 1.802024E+07,
        "DP": 1.721041E+07
    },
    #('ispd19_test5', 'gpu', 'deterministic'): {
    #    "GP": ,
    #    "LG": ,
    #    "DP":
    #},
    #('ispd19_test5', 'gpu', 'indeterministic'): {
    #    "GP": ,
    #    "LG": ,
    #    "DP":
    #},
    #('ispd19_test5', 'cpu', 'deterministic'): {
    #    "GP": ,
    #    "LG": ,
    #    "DP":
    #},
    #('ispd19_test5', 'cpu', 'indeterministic'): {
    #    "GP": ,
    #    "LG": ,
    #    "DP":
    #},
    ('ispd19_test6', 'gpu', 'deterministic'): {
        "GP": 4.299130E+07,
        "LG": 4.524393E+07,
        "DP": 4.480661E+07
    },
    ('ispd19_test6', 'gpu', 'indeterministic'): {
        "GP": 4.299130E+07,
        "LG": 4.524393E+07,
        "DP": 4.480661E+07
    },
    ('ispd19_test6', 'cpu', 'deterministic'): {
        "GP": 4.299130E+07,
        "LG": 4.524393E+07,
        "DP": 4.480661E+07
    },
    ('ispd19_test6', 'cpu', 'indeterministic'): {
        "GP": 4.299130E+07,
        "LG": 4.524393E+07,
        "DP": 4.480661E+07
    },
    ('ispd19_test7', 'gpu', 'deterministic'): {
        "GP": 8.458533E+07,
        "LG": 9.081408E+07,
        "DP": 8.950081E+07
    },
    ('ispd19_test7', 'gpu', 'indeterministic'): {
        "GP": 8.458533E+07,
        "LG": 9.081408E+07,
        "DP": 8.950081E+07
    },
    ('ispd19_test7', 'cpu', 'deterministic'): {
        "GP": 8.458533E+07,
        "LG": 9.081408E+07,
        "DP": 8.950081E+07
    },
    ('ispd19_test7', 'cpu', 'indeterministic'): {
        "GP": 8.458533E+07,
        "LG": 9.081408E+07,
        "DP": 8.950081E+07
    },
    ('ispd19_test8', 'gpu', 'deterministic'): {
        "GP": 1.284594E+08,
        "LG": 1.353531E+08,
        "DP": 1.340507E+08
    },
    ('ispd19_test8', 'gpu', 'indeterministic'): {
        "GP": 1.284594E+08,
        "LG": 1.353531E+08,
        "DP": 1.340507E+08
    },
    ('ispd19_test8', 'cpu', 'deterministic'): {
        "GP": 1.284594E+08,
        "LG": 1.353531E+08,
        "DP": 1.340507E+08
    },
    ('ispd19_test8', 'cpu', 'indeterministic'): {
        "GP": 1.284594E+08,
        "LG": 1.353531E+08,
        "DP": 1.340507E+08
    },
    ('ispd19_test9', 'gpu', 'deterministic'): {
        "GP": 1.983457E+08,
        "LG": 2.074886E+08,
        "DP": 2.059705E+08
    },
    ('ispd19_test9', 'gpu', 'indeterministic'): {
        "GP": 1.983457E+08,
        "LG": 2.074886E+08,
        "DP": 2.059705E+08
    },
    ('ispd19_test9', 'cpu', 'deterministic'): {
        "GP": 1.983457E+08,
        "LG": 2.074886E+08,
        "DP": 2.059705E+08
    },
    ('ispd19_test9', 'cpu', 'indeterministic'): {
        "GP": 1.983457E+08,
        "LG": 2.074886E+08,
        "DP": 2.059705E+08
    },
    ('ispd19_test10', 'gpu', 'deterministic'): {
        "GP": 2.017845E+08,
        "LG": 2.116758E+08,
        "DP": 2.098444E+08
    },
    ('ispd19_test10', 'gpu', 'indeterministic'): {
        "GP": 2.017845E+08,
        "LG": 2.116758E+08,
        "DP": 2.098444E+08
    },
    ('ispd19_test10', 'cpu', 'deterministic'): {
        "GP": 2.017845E+08,
        "LG": 2.116758E+08,
        "DP": 2.098444E+08
    },
    ('ispd19_test10', 'cpu', 'indeterministic'): {
        "GP": 2.017845E+08,
        "LG": 2.116758E+08,
        "DP": 2.098444E+08
    },
}


class ISPD2019Test(unittest.TestCase):
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
                        rtol=0.010)


if __name__ == "__main__":
    """
    @brief main function to invoke the entire placement flow. 
    """
    logging.root.name = 'DREAMPlace'
    logging.basicConfig(level=logging.INFO,
                        format='[%(levelname)-7s] %(name)s - %(message)s',
                        filename="regression_ispd2019.log")

    unittest.main()
