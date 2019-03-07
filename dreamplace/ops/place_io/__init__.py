##
# @file   __init__.py
# @author Yibo Lin
# @date   Aug 2018
#

import os 
import sys
if sys.version_info[0] < 3: 
    import src.place_io as place_io
else:
    from .src import place_io

class Params (object):
    def __init__(self):
        self.aux_file = None

if __name__ == "__main__":
    params = Params()
    params.aux_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../benchmarks/simple/simple.aux"))

    db = place_io.PlaceIOFunction.forward(params)

    print("num_nodes = ", db.num_nodes)
    print("num_terminals = ", db.num_terminals)
    print("node_name2id_map = ", db.node_name2id_map)
    print("node_names = ", db.node_names)
    print("node_x = ", db.node_x)
    print("node_y = ", db.node_y)
    print("node_orient = ", db.node_orient)
    print("node_size_x = ", db.node_size_x)
    print("node_size_y = ", db.node_size_y)
    print("pin_direct = ", db.pin_direct)
    print("pin_offset_x = ", db.pin_offset_x)
    print("pin_offset_y = ", db.pin_offset_y)
    print("net_name2id_map = ", db.net_name2id_map)
    print("net_names = ", db.net_names)
    print("net2pin_map = ", db.net2pin_map)
    print("flat_net2pin_map = ", db.flat_net2pin_map)
    print("flat_net2pin_start_map = ", db.flat_net2pin_start_map)
    print("node2pin_map = ", db.node2pin_map)
    print("flat_node2pin_map = ", db.flat_node2pin_map)
    print("flat_node2pin_start_map = ", db.flat_node2pin_start_map)
    print("pin2node_map = ", db.pin2node_map)
    print("pin2net_map = ", db.pin2net_map)
    print("rows = ", db.rows)
    print("xl = ", db.xl)
    print("yl = ", db.yl)
    print("xh = ", db.xh)
    print("yh = ", db.yh)
    print("row_height = ", db.row_height)
    print("site_width = ", db.site_width)
    print("num_movable_pins = ", db.num_movable_pins)
