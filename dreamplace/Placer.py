##
# @file   Placer.py
# @author Yibo Lin
# @date   Apr 2018
#

import matplotlib 
matplotlib.use('Agg')
import os
import sys 
import time 
import numpy as np 
import Params 
import PlaceDB
import NonLinearPlace 
import pdb 

# print all contents of numpy array 
#np.set_printoptions(threshold=np.nan)

def place(params):

    enable_gp_flag = True
    enable_dp_flag = True

    np.random.seed(params.random_seed)
    # read database 
    tt = time.time()
    placedb = PlaceDB.PlaceDB()
    placedb(params)
    print("[I] reading database takes %.2f seconds" % (time.time()-tt))
    #placedb.write_nets(params, "tmp.nets")

    # solve placement 
    tt = time.time()
    placer = NonLinearPlace.NonLinearPlace(params, placedb)
    print("[I] non-linear placement initialization takes %.2f seconds" % (time.time()-tt))
    metrics = placer(params, placedb, enable_gp_flag)
    print("[I] non-linear placement takes %.2f seconds" % (time.time()-tt))

    # write placement solution 
    path = "summary/%s" % (os.path.splitext(os.path.basename(params.aux_file))[0])
    if not os.path.exists(path):
        os.system("mkdir -p %s" % (path))
    gp_out_file = os.path.join(path, os.path.basename(params.aux_file).replace(".aux", ".gp.pl"))
    if enable_gp_flag:
        placedb.write_pl(params, gp_out_file)

    # call detailed placement
    dp_out_file = gp_out_file.replace(".gp.pl", "")
    # add target density constraint if provided 
    target_density_cmd = ""
    if params.target_density < 1.0:
        target_density_cmd = " -util %f" % (params.target_density)
    if params.legalize_flag:
        legalize = "-nolegal"
    else:
        legalize = ""
    if params.detailed_place_flag:
        detailed_place = "-nodetail"
    else:
        detailed_place = ""
    cmd = "./thirdparty/ntuplace3 -aux %s -loadpl %s %s -out %s -noglobal %s %s" % (params.aux_file, gp_out_file, target_density_cmd, dp_out_file, legalize, detailed_place)
    print("[I] %s" % (cmd))
    tt = time.time()
    if enable_dp_flag:
        os.system(cmd)
    print("[I] detailed placement takes %.2f seconds" % (time.time()-tt))

    # read solution and evaluate 
    placedb.read_pl(dp_out_file+".ntup.pl")
    placedb.scale_pl(params.scale_factor)
    iteration = len(metrics)
    pos = placer.init_pos
    pos[0:placedb.num_physical_nodes] = placedb.node_x
    pos[placedb.num_nodes:placedb.num_nodes+placedb.num_physical_nodes] = placedb.node_y
    hpwl, density_overflow, max_density = placer.validate(placedb, pos, iteration)
    print("[I] iteration %4d, HPWL %.3E, overflow %.3E, max density %.3E" % (iteration, hpwl, density_overflow, max_density))
    placer.plot(params, placedb, iteration, pos)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("[E] input parameters in json format in required")
    paramsArray = []
    for i in range(1, len(sys.argv)):
        params = Params.Params()
        params.load(sys.argv[i])
        paramsArray.append(params)
    print("[I] parameters[%d] = %s" % (len(paramsArray), paramsArray))

    tt = time.time()
    for params in paramsArray: 
        place(params)
    print("[I] placement takes %.3f seconds" % (time.time()-tt))
