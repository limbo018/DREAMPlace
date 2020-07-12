##
# @file   nctugr_binary.py
# @author Yibo Lin
# @date   Jan 2020
#

import os
import stat
import sys
import logging
import torch
from torch.autograd import Function
from torch import nn
import pdb

import dreamplace.ops.place_io.place_io as place_io

logger = logging.getLogger(__name__)


class NCTUgr(object):
    def __init__(self, aux_input_file, param_setting_file, tmp_pl_file,
                 tmp_output_file, horizontal_routing_capacities,
                 vertical_routing_capacities, params, placedb):
        self.aux_input_file = os.path.realpath(aux_input_file)
        self.param_setting_file = os.path.realpath(param_setting_file)
        self.tmp_pl_file = os.path.realpath(tmp_pl_file)
        self.tmp_output_file = os.path.realpath(tmp_output_file)
        self.routing_capacities = (horizontal_routing_capacities +
                                   vertical_routing_capacities).view([
                                       1, 1,
                                       len(horizontal_routing_capacities)
                                   ])
        self.params = params
        self.placedb = placedb
        self.nctugr_dir = "%s/../../../thirdparty/NCTUgr.ICCAD2012" % (
            os.path.dirname(os.path.realpath(__file__)))

        nctugr_bin = "%s/NCTUgr" % (self.nctugr_dir)
        st = os.stat(nctugr_bin)
        os.chmod(nctugr_bin, st.st_mode | stat.S_IEXEC)

    def __call__(self, pos):
        return self.forward(pos)

    def forward(self, pos):
        if pos.is_cuda:
            pos_cpu = pos.cpu().data.numpy()
        else:
            pos_cpu = pos.data.numpy()

        num_nodes = pos.numel() // 2
        if not os.path.exists(os.path.dirname(self.tmp_pl_file)):
            os.system("mkdir -p %s" % (os.path.dirname(self.tmp_pl_file)))
        place_io.PlaceIOFunction.write(self.placedb.rawdb, self.tmp_pl_file,
                                       place_io.SolutionFileFormat.BOOKSHELF,
                                       pos_cpu[:num_nodes],
                                       pos_cpu[num_nodes:])
        #self.placedb.write_pl(self.params, self.tmp_pl_file, pos_cpu[:pos_cpu.numel()//2], pos_cpu[pos_cpu.numel()//2:])
        cmd = "ln -s %s/PORT9.dat .; \
               ln -s %s/POST9.dat .; \
               ln -s %s/POWV9.dat .; \
                %s/NCTUgr ICCAD %s %s %s %s ;\
                rm PORT9.dat POST9.dat POWV9.dat ; \
                " % (self.nctugr_dir, self.nctugr_dir, self.nctugr_dir,
                     self.nctugr_dir, self.aux_input_file, self.tmp_pl_file,
                     self.param_setting_file, self.tmp_output_file)
        logger.info(cmd)
        os.system(cmd)

        congestion_map = torch.zeros((self.placedb.num_routing_grids_x,
                                      self.placedb.num_routing_grids_y,
                                      self.placedb.num_routing_layers),
                                     dtype=pos.dtype)
        with open(self.tmp_output_file + ".ofinfo", "r") as f:
            status = 0
            for line in f:
                line = line.strip()
                if line.startswith("Overflowed grid edges :"):
                    status = 1
                elif line.startswith("end") and status:
                    status = 0
                    break
                elif line.startswith("(") and status:
                    tokens = line.split()
                    start = (int(tokens[0][1:-1]), int(tokens[1][:-1]),
                             int(tokens[2][:-1]))
                    end = (int(tokens[4][1:-1]), int(tokens[5][:-1]),
                           int(tokens[6][:-1]))
                    overflow = int(tokens[7])
                    assert start[2] == end[2]
                    congestion_map[start[0], start[1], start[2]] = overflow

        cmd = "rm %s.*" % (self.tmp_output_file)
        logger.info(cmd)
        os.system(cmd)

        if self.routing_capacities.device != pos.device:
            self.routing_capacities = self.routing_capacities.to(pos.device)
        overflow_map = congestion_map.to(
            pos.device) / (self.routing_capacities + 1e-6) + 1
        #horizontal_overflow_map = overflow_map[:, :, 0:self.placedb.num_routing_layers:2].mean(dim=2)
        #vertical_overflow_map = overflow_map[:, :, 1:self.placedb.num_routing_layers:2].mean(dim=2)
        #ret = torch.max(horizontal_overflow_map, vertical_overflow_map)
        ret = overflow_map.max(dim=2)[0]

        return ret
