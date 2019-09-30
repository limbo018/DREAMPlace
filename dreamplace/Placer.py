##
# @file   Placer.py
# @author Yibo Lin
# @date   Apr 2018
# @brief  Main file to run the entire placement flow. 
#

import matplotlib 
matplotlib.use('Agg')
import os
import sys 
import time 
import numpy as np 
import logging
# for consistency between python2 and python3
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
	sys.path.append(root_dir)
import Params 
import PlaceDB
import NonLinearPlace 
import pdb 

def place(params):
    """
    @brief Top API to run the entire placement flow. 
    @param params parameters 
    """

    np.random.seed(params.random_seed)
    # read database 
    tt = time.time()
    placedb = PlaceDB.PlaceDB()
    placedb(params)
    logging.info("reading database takes %.2f seconds" % (time.time()-tt))

    # solve placement 
    tt = time.time()
    placer = NonLinearPlace.NonLinearPlace(params, placedb)
    logging.info("non-linear placement initialization takes %.2f seconds" % (time.time()-tt))
    metrics = placer(params, placedb)
    logging.info("non-linear placement takes %.2f seconds" % (time.time()-tt))

    # write placement solution 
    path = "%s/%s" % (params.result_dir, params.design_name())
    if not os.path.exists(path):
        os.system("mkdir -p %s" % (path))
    gp_out_file = os.path.join(path, "%s.gp.%s" % (params.design_name(), params.solution_file_suffix()))
    placedb.write(params, gp_out_file)

    # call external detailed placement
    # TODO: support more external placers, currently only NTUplace3 with Bookshelf format 
    if params.detailed_place_engine and os.path.exists(params.detailed_place_engine) and params.solution_file_suffix() == "pl": 
        logging.info("Use external detailed placement engine %s" % (params.detailed_place_engine))
        dp_out_file = gp_out_file.replace(".gp.pl", "")
        # add target density constraint if provided 
        target_density_cmd = ""
        if params.target_density < 1.0:
            target_density_cmd = " -util %f" % (params.target_density)
        cmd = "%s -aux %s -loadpl %s %s -out %s -noglobal %s" % (params.detailed_place_engine, params.aux_input, gp_out_file, target_density_cmd, dp_out_file, params.detailed_place_command)
        logging.info("%s" % (cmd))
        tt = time.time()
        os.system(cmd)
        logging.info("External detailed placement takes %.2f seconds" % (time.time()-tt))

        if params.plot_flag: 
            # read solution and evaluate 
            placedb.read_pl(params, dp_out_file+".ntup.pl")
            iteration = len(metrics)
            pos = placer.init_pos
            pos[0:placedb.num_physical_nodes] = placedb.node_x
            pos[placedb.num_nodes:placedb.num_nodes+placedb.num_physical_nodes] = placedb.node_y
            hpwl, density_overflow, max_density = placer.validate(placedb, pos, iteration)
            logging.info("iteration %4d, HPWL %.3E, overflow %.3E, max density %.3E" % (iteration, hpwl, density_overflow, max_density))
            placer.plot(params, placedb, iteration, pos)
    elif params.detailed_place_engine:
        logging.warning("External detailed placement engine %s or aux file NOT found" % (params.detailed_place_engine))

if __name__ == "__main__":
    """
    @brief main function to invoke the entire placement flow. 
    """
    logging.root.name = 'DREAMPlace'
    logging.basicConfig(level=logging.INFO, format='[%(levelname)-7s] %(name)s - %(message)s')
    params = Params.Params()
    params.printWelcome()
    if len(sys.argv) == 1 or '-h' in sys.argv[1:] or '--help' in sys.argv[1:]:
        params.printHelp()
        exit()
    elif len(sys.argv) != 2:
        logging.error("One input parameters in json format in required")
        params.printHelp()
        exit()

    # load parameters 
    params.load(sys.argv[1])
    logging.info("parameters = %s" % (params))

    # run placement 
    tt = time.time()
    place(params)
    logging.info("placement takes %.3f seconds" % (time.time()-tt))
