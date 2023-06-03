##
# @file   place_io.py
# @author Yibo Lin
# @date   Aug 2018
#

from torch.autograd import Function

import dreamplace.ops.place_io.place_io_cpp as place_io_cpp
from dreamplace.ops.place_io.place_io_cpp import SolutionFileFormat, Direction1DType, Direction2DType, OrientEnum, PlaceStatusEnum, MultiRowAttrEnum, SignalDirectEnum, PlanarDirectEnum, RegionTypeEnum


class PlaceIOFunction(Function):
    @staticmethod
    def read(params):
        """
        @brief read design and store in placement database
        """
        args = "DREAMPlace"
        if "aux_input" in params.__dict__ and params.aux_input:
            args += " --bookshelf_aux_input %s" % (params.aux_input)
        if "lef_input" in params.__dict__ and params.lef_input:
            if isinstance(params.lef_input, list):
                for lef in params.lef_input:
                    args += " --lef_input %s" % (lef)
            else:
                args += " --lef_input %s" % (params.lef_input)
        if "def_input" in params.__dict__ and params.def_input:
            args += " --def_input %s" % (params.def_input)
        if "verilog_input" in params.__dict__ and params.verilog_input:
            args += " --verilog_input %s" % (params.verilog_input)
        if "sort_nets_by_degree" in params.__dict__:
            args += " --sort_nets_by_degree %s" % (params.sort_nets_by_degree)

        return place_io_cpp.forward(args.split(' '))

    @staticmethod
    def pydb(raw_db):
        """
        @brief convert to python database 
        @param raw_db original placement database 
        """
        return place_io_cpp.pydb(raw_db)

    @staticmethod
    def write(raw_db, filename, sol_file_format, node_x, node_y):
        """
        @brief write solution in specific format 
        @param raw_db original placement database 
        @param filename output file 
        @param sol_file_format solution file format, DEF|DEFSIMPLE|BOOKSHELF|BOOKSHELFALL
        @param node_x x coordinates of cells, only need movable cells; if none, use original position 
        @param node_y y coordinates of cells, only need movable cells; if none, use original position
        """
        return place_io_cpp.write(raw_db, filename, sol_file_format, node_x,
                                  node_y)

    @staticmethod
    def apply(raw_db, node_x, node_y):
        """
        @brief apply solution 
        @param raw_db original placement database 
        @param node_x x coordinates of cells, only need movable cells
        @param node_y y coordinates of cells, only need movable cells
        """
        return place_io_cpp.apply(raw_db, node_x, node_y)
