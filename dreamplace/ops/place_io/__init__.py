##
# @file   __init__.py
# @author Yibo Lin
# @date   Aug 2018
#

import os 
import sys
import numpy as np 
import unittest

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import place_io
sys.path.pop()

class Params (object):
    def __init__(self):
        self.aux_file = None

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
        design = os.path.join(os.path.dirname(os.path.realpath(__file__)), "unitest")
        params.aux_file = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.join(design, "simple/simple.aux")))

        db = place_io.PlaceIOFunction.forward(params)

        content = ""
        content += "num_nodes = %s\n" % (db.num_nodes)
        content += "num_terminals = %s\n" % (db.num_terminals)
        content += "node_name2id_map = %s\n" % (name2id_map2str(db.node_name2id_map))
        content += "node_names = %s\n" % (array2str(db.node_names))
        content += "node_x = %s\n" % (db.node_x)
        content += "node_y = %s\n" % (db.node_y)
        content += "node_orient = %s\n" % (array2str(db.node_orient))
        content += "node_size_x = %s\n" % (db.node_size_x)
        content += "node_size_y = %s\n" % (db.node_size_y)
        content += "pin_direct = %s\n" % (array2str(db.pin_direct))
        content += "pin_offset_x = %s\n" % (db.pin_offset_x)
        content += "pin_offset_y = %s\n" % (db.pin_offset_y)
        content += "net_name2id_map = %s\n" % (name2id_map2str(db.net_name2id_map))
        content += "net_names = %s\n" % (array2str(db.net_names))
        content += "net2pin_map = %s\n" % (db.net2pin_map)
        content += "flat_net2pin_map = %s\n" % (db.flat_net2pin_map)
        content += "flat_net2pin_start_map = %s\n" % (db.flat_net2pin_start_map)
        content += "node2pin_map = %s\n" % (db.node2pin_map)
        content += "flat_node2pin_map = %s\n" % (db.flat_node2pin_map)
        content += "flat_node2pin_start_map = %s\n" % (db.flat_node2pin_start_map)
        content += "pin2node_map = %s\n" % (db.pin2node_map)
        content += "pin2net_map = %s\n" % (db.pin2net_map)
        content += "rows = %s\n" % (db.rows)
        content += "xl = %s\n" % (db.xl)
        content += "yl = %s\n" % (db.yl)
        content += "xh = %s\n" % (db.xh)
        content += "yh = %s\n" % (db.yh)
        content += "row_height = %s\n" % (db.row_height)
        content += "site_width = %s\n" % (db.site_width)
        content += "num_movable_pins = %s\n" % (db.num_movable_pins)
        print(content)

        with open(os.path.join(design, "simple.golden"), "r") as f:
            golden = f.read()

        np.testing.assert_array_equal(content.strip(), golden.strip())

if __name__ == "__main__":
    unittest.main()
