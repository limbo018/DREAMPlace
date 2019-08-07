##
# @file   place_io_unitest.py
# @author Yibo Lin
# @date   Mar 2019
#

import os 
import sys
import numpy as np 
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from dreamplace.ops.place_io import place_io
sys.path.pop()

class Params (object):
    def __init__(self):
        self.aux_input = None

def name2id_map2str(m):
    id2name_map = [None]*len(m)
    for k in m.keys():
        id2name_map[m[k]] = k
    content = ""
    for i in range(len(m)):
        if i:
            content += ", "
        content += "%s : %d" % (id2name_map[i], i)
    return "{%s}" % (content)

def array2str(a):
    content = ""
    for v in a:
        if content:
            content += ", "
        content += "%s" % (v)
    return "[%s]" % (content)

class PlaceIOOpTest(unittest.TestCase):
    def test_simple(self):
        params = Params()
        design = os.path.dirname(os.path.realpath(__file__))
        params.aux_input = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.join(design, "simple/simple.aux")))

        db = place_io.PlaceIOFunction.read(params)
        pydb = place_io.PlaceIOFunction.pydb(db)

        content = ""
        content += "num_nodes = %s\n" % (pydb.num_nodes)
        content += "num_terminals = %s\n" % (pydb.num_terminals)
        content += "node_name2id_map = %s\n" % (name2id_map2str(pydb.node_name2id_map))
        content += "node_names = %s\n" % (array2str(pydb.node_names))
        content += "node_x = %s\n" % (pydb.node_x)
        content += "node_y = %s\n" % (pydb.node_y)
        content += "node_orient = %s\n" % (array2str(pydb.node_orient))
        content += "node_size_x = %s\n" % (pydb.node_size_x)
        content += "node_size_y = %s\n" % (pydb.node_size_y)
        content += "pin_direct = %s\n" % (array2str(pydb.pin_direct))
        content += "pin_offset_x = %s\n" % (pydb.pin_offset_x)
        content += "pin_offset_y = %s\n" % (pydb.pin_offset_y)
        content += "net_name2id_map = %s\n" % (name2id_map2str(pydb.net_name2id_map))
        content += "net_names = %s\n" % (array2str(pydb.net_names))
        content += "net2pin_map = %s\n" % (pydb.net2pin_map)
        content += "flat_net2pin_map = %s\n" % (pydb.flat_net2pin_map)
        content += "flat_net2pin_start_map = %s\n" % (pydb.flat_net2pin_start_map)
        content += "node2pin_map = %s\n" % (pydb.node2pin_map)
        content += "flat_node2pin_map = %s\n" % (pydb.flat_node2pin_map)
        content += "flat_node2pin_start_map = %s\n" % (pydb.flat_node2pin_start_map)
        content += "pin2node_map = %s\n" % (pydb.pin2node_map)
        content += "pin2net_map = %s\n" % (pydb.pin2net_map)
        content += "rows = %s\n" % (pydb.rows)
        content += "xl = %s\n" % (pydb.xl)
        content += "yl = %s\n" % (pydb.yl)
        content += "xh = %s\n" % (pydb.xh)
        content += "yh = %s\n" % (pydb.yh)
        content += "row_height = %s\n" % (pydb.row_height)
        content += "site_width = %s\n" % (pydb.site_width)
        content += "num_movable_pins = %s\n" % (pydb.num_movable_pins)
        print(content)

        with open(os.path.join(design, "simple.golden"), "r") as f:
            golden = f.read()

        np.testing.assert_array_equal(content.strip(), golden.strip())

if __name__ == "__main__":
    unittest.main()
