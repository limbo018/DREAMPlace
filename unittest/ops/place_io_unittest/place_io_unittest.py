##
# @file   place_io_unitest.py
# @author Yibo Lin
# @date   Mar 2019
#

import os 
import sys
import numpy as np 
import unittest
import tempfile

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
    def test_def_row_continuations(self):
        lef = """VERSION 5.8 ;
BUSBITCHARS "[]" ;
DIVIDERCHAR "/" ;
UNITS DATABASE MICRONS 1000 ; END UNITS
MANUFACTURINGGRID 0.001 ;
LAYER M1
  TYPE ROUTING ; DIRECTION HORIZONTAL ;
  PITCH 0.2 ; WIDTH 0.1 ; SPACING 0.1 ;
END M1
SITE core
  CLASS CORE ; SYMMETRY Y ; SIZE 1 BY 2 ;
END core
MACRO CELL
  CLASS CORE ; ORIGIN 0 0 ; SIZE 2 BY 2 ; SYMMETRY X Y ; SITE core ;
  PIN A
    DIRECTION INPUT ; USE SIGNAL ;
    PORT LAYER M1 ; RECT 0.4 0.4 0.6 0.6 ; END
  END A
END CELL
END LIBRARY
"""
        definition = """VERSION 5.8 ;
DIVIDERCHAR "/" ;
BUSBITCHARS "[]" ;
DESIGN tiny ;
UNITS DISTANCE MICRONS 1000 ;
DIEAREA ( 0 0 ) ( 10000 4000 ) ;
%s
ROW r1 core 0 2000 FS DO 10 BY 1 STEP 1000 0 ;
COMPONENTS 3 ;
- c0 CELL + PLACED ( 0 0 ) N ;
- c1 CELL + PLACED ( 2000 0 ) N ;
- fixed CELL + FIXED ( 6000 0 ) N ;
END COMPONENTS
NETS 1 ;
- n0 ( c0 A ) ( c1 A ) ( fixed A ) ;
END NETS
END DESIGN
"""
        row = "ROW r0 core 0 0 N DO 10 BY 1 STEP 1000 0 ;"
        rows = [row, row.replace("N DO", "N\n  DO"),
                row.replace("BY 1 STEP", "BY 1\n  STEP"),
                "ROW r0 core\n  0 0 N\n\n  DO 10\n  BY 1\n  STEP 1000 0\n  ;"]
        expected = ["core 0 0 N DO 10 BY 1 STEP 1000 0".split(),
                    "core 0 2000 FS DO 10 BY 1 STEP 1000 0".split()]
        with tempfile.TemporaryDirectory() as design:
            params = Params()
            params.lef_input = os.path.join(design, "tiny.lef")
            params.def_input = os.path.join(design, "tiny.def")
            params.sort_nets_by_degree = False
            output = os.path.join(design, "out.def")
            with open(params.lef_input, "w") as f:
                f.write(lef)
            for row_text in rows:
                with self.subTest(row=row_text):
                    with open(params.def_input, "w") as f:
                        f.write(definition % row_text)
                    db = place_io.PlaceIOFunction.read(params)
                    pydb = place_io.PlaceIOFunction.pydb(db)
                    self.assertEqual(list(pydb.rows),
                                     [(0, 0, 10000, 2000), (0, 2000, 10000, 4000)])
                    self.assertTrue(place_io.PlaceIOFunction.write(
                        db, output, place_io.SolutionFileFormat.DEF,
                        np.asarray(pydb.node_x, dtype=float),
                        np.asarray(pydb.node_y, dtype=float)))
                    with open(output, "r") as f:
                        content = f.read()
                    for name, tokens in zip(["r0", "r1"], expected):
                        actual = content.split("ROW " + name, 1)[1].split(";", 1)[0].split()
                        self.assertEqual(actual, tokens)

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
