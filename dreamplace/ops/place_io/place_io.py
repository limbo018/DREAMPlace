##
# @file   place_io.py
# @author Yibo Lin
# @date   Aug 2018
#

from torch.autograd import Function

import dreamplace.ops.place_io.place_io_cpp as place_io_cpp

class PlaceIOFunction(Function):
    @staticmethod
    def forward(params):
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

        return place_io_cpp.forward(args.split(' '))
