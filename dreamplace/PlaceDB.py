##
# @file   PlaceDB.py
# @author Yibo Lin
# @date   Apr 2018
# @brief  placement database 
#

import sys
import os
import re
import time 
import numpy as np 
import cairocffi as cairo 
import Params
from pytorch.ops import place_io
import pdb 

datatypes = {
        'float32' : np.float32, 
        'float64' : np.float64
        }

class PlaceDB (object):
    """
    initialization
    To avoid the usage of list, I flatten everything.  
    """
    def __init__(self):
        self.num_physical_nodes = 0 # number of real nodes, including movable nodes, terminals
        self.num_terminals = 0 # number of terminals 
        self.node_name2id_map = {} # node name to id map, cell name 
        self.node_names = None # 1D array, cell name 
        self.node_x = None # 1D array, cell position x 
        self.node_y = None # 1D array, cell position y 
        self.node_orient = None # 1D array, cell orientation 
        self.node_size_x = None # 1D array, cell width  
        self.node_size_y = None # 1D array, cell height

        self.pin_direct = None # 1D array, pin direction IO 
        self.pin_offset_x = None # 1D array, pin offset x to its node 
        self.pin_offset_y = None # 1D array, pin offset y to its node 

        self.net_name2id_map = {} # net name to id map
        self.net_names = None # net name 
        self.net2pin_map = None # array of 1D array, each row stores pin id
        self.flat_net2pin_map = None # flatten version of net2pin_map 
        self.flat_net2pin_start_map = None # starting index of each net in flat_net2pin_map

        self.node2pin_map = None # array of 1D array, contains pin id of each node 
        self.pin2node_map = None # 1D array, contain parent node id of each pin 
        self.pin2net_map = None # 1D array, contain parent net id of each pin 

        self.rows = None # NumRows x 4 array, stores xl, yl, xh, yh of each row 

        self.xl = None 
        self.yl = None 
        self.xh = None 
        self.yh = None 

        self.row_height = None
        self.site_width = None

        self.bin_size_x = None 
        self.bin_size_y = None
        self.num_bins_x = None
        self.num_bins_y = None
        self.bin_center_x = None 
        self.bin_center_y = None

        self.num_movable_pins = None 

        self.total_movable_node_area = None
        self.total_fixed_node_area = None

        # enable filler cells 
        # the Idea from e-place and RePlace 
        self.total_filler_node_area = None 
        self.num_filler_nodes = None

        self.dtype = None 

    """
    scale placement solution only
    """
    def scale_pl(self, scale_factor):
        self.node_x *= scale_factor
        self.node_y *= scale_factor 

    """
    scale distances
    """
    def scale(self, scale_factor):
        print("[I] scale coordinate system by %g" % (scale_factor))
        self.scale_pl(scale_factor)
        self.node_size_x *= scale_factor
        self.node_size_y *= scale_factor
        self.pin_offset_x *= scale_factor
        self.pin_offset_y *= scale_factor
        self.xl *= scale_factor 
        self.yl *= scale_factor
        self.xh *= scale_factor
        self.yh *= scale_factor
        self.row_height *= scale_factor
        self.site_width *= scale_factor

    """
    sort net by degree 
    sort pin array such that pins belonging to the same net is abutting each other
    """
    def sort(self):
        print("\t[I] sort nets by degree and pins by net")

        # sort nets by degree 
        net_degrees = np.array([len(pins) for pins in self.net2pin_map])
        net_order = net_degrees.argsort() # indexed by new net_id, content is old net_id
        self.net_names = self.net_names[net_order]
        self.net2pin_map = self.net2pin_map[net_order]
        for net_id, net_name in enumerate(self.net_names):
            self.net_name2id_map[net_name] = net_id
        for new_net_id in range(len(net_order)):
            for pin_id in self.net2pin_map[new_net_id]:
                self.pin2net_map[pin_id] = new_net_id
        ## check 
        #for net_id in range(len(self.net2pin_map)):
        #    for j in range(len(self.net2pin_map[net_id])):
        #        assert self.pin2net_map[self.net2pin_map[net_id][j]] == net_id

        # sort pins such that pins belonging to the same net is abutting each other
        pin_order = self.pin2net_map.argsort() # indexed new pin_id, content is old pin_id 
        self.pin2net_map = self.pin2net_map[pin_order]
        self.pin2node_map = self.pin2node_map[pin_order]
        self.pin_direct = self.pin_direct[pin_order]
        self.pin_offset_x = self.pin_offset_x[pin_order]
        self.pin_offset_y = self.pin_offset_y[pin_order]
        old2new_pin_id_map = np.zeros(len(pin_order), dtype=np.int32)
        for new_pin_id in range(len(pin_order)):
            old2new_pin_id_map[pin_order[new_pin_id]] = new_pin_id
        for i in range(len(self.net2pin_map)):
            for j in range(len(self.net2pin_map[i])):
                self.net2pin_map[i][j] = old2new_pin_id_map[self.net2pin_map[i][j]]
        for i in range(len(self.node2pin_map)):
            for j in range(len(self.node2pin_map[i])):
                self.node2pin_map[i][j] = old2new_pin_id_map[self.node2pin_map[i][j]]
        ## check 
        #for net_id in range(len(self.net2pin_map)):
        #    for j in range(len(self.net2pin_map[net_id])):
        #        assert self.pin2net_map[self.net2pin_map[net_id][j]] == net_id
        #for node_id in range(len(self.node2pin_map)):
        #    for j in range(len(self.node2pin_map[node_id])):
        #        assert self.pin2node_map[self.node2pin_map[node_id][j]] == node_id

    """
    return number of movable nodes 
    """
    @property
    def num_movable_nodes(self):
        return self.num_physical_nodes - self.num_terminals
    """
    return number of movable nodes, terminals and fillers
    """
    @property 
    def num_nodes(self):
        return self.num_physical_nodes + self.num_filler_nodes
    """
    return number of nets
    """
    @property
    def num_nets(self):
        return len(self.net2pin_map)
    """
    return number of pins
    """
    @property
    def num_pins(self):
        return len(self.pin2net_map)
    """
    return width of layout 
    """
    @property
    def width(self):
        return self.xh-self.xl
    """
    return height of layout 
    """
    @property
    def height(self):
        return self.yh-self.yl
    """
    return area of layout 
    """
    @property
    def area(self):
        return self.width*self.height

    """
    return bin index in x direction 
    """
    def bin_index_x(self, x): 
        if x < self.xl:
            return 0 
        elif x > self.xh:
            return int(np.floor((self.xh-self.xl)/self.bin_size_x))
        else:
            return int(np.floor((x-self.xl)/self.bin_size_x))

    """
    return bin index in y direction 
    """
    def bin_index_y(self, y): 
        if y < self.yl:
            return 0 
        elif y > self.yh:
            return int(np.floor((self.yh-self.yl)/self.bin_size_y))
        else:
            return int(np.floor((y-self.yl)/self.bin_size_y))

    """
    return bin xl
    """
    def bin_xl(self, id_x):
        return self.xl+id_x*self.bin_size_x

    """
    return bin xh
    """
    def bin_xh(self, id_x):
        return min(self.bin_xl(id_x)+self.bin_size_x, self.xh)

    """
    return bin yl
    """
    def bin_yl(self, id_y):
        return self.yl+id_y*self.bin_size_y

    """
    return bin yh
    """
    def bin_yh(self, id_y):
        return min(self.bin_yl(id_y)+self.bin_size_y, self.yh)

    """
    @param l lower bound 
    @param h upper bound 
    @param bin_size bin size 
    @return number of bins 
    """
    def num_bins(self, l, h, bin_size):
        return int(np.ceil((h-l)/bin_size))

    """
    @param l lower bound 
    @param h upper bound 
    @param bin_size bin size 
    @return array of bin centers 
    """
    def bin_centers(self, l, h, bin_size):
        num_bins = self.num_bins(l, h, bin_size)
        centers = np.zeros(num_bins, dtype=self.dtype)
        for id_x in range(num_bins): 
            bin_l = l+id_x*bin_size
            bin_h = min(bin_l+bin_size, h)
            centers[id_x] = (bin_l+bin_h)/2
        return centers 

    """
    return hpwl of a net 
    """
    def net_hpwl(self, x, y, net_id): 
        pins = self.net2pin_map[net_id]
        nodes = self.pin2node_map[pins]
        hpwl_x = np.amax(x[nodes]+self.pin_offset_x[pins]) - np.amin(x[nodes]+self.pin_offset_x[pins])
        hpwl_y = np.amax(y[nodes]+self.pin_offset_y[pins]) - np.amin(y[nodes]+self.pin_offset_y[pins])

        return hpwl_x+hpwl_y

    """
    return hpwl of all nets
    """
    def hpwl(self, x, y):
        wl = 0
        for net_id in range(len(self.net2pin_map)):
            wl += self.net_hpwl(x, y, net_id)
        return wl 

    """
    return overlap area between two rectangles
    """
    def overlap(self, xl1, yl1, xh1, yh1, xl2, yl2, xh2, yh2):
        return max(min(xh1, xh2)-max(xl1, xl2), 0.0) * max(min(yh1, yh2)-max(yl1, yl2), 0.0)

    """
    return density map 
    this density map evaluates the overlap between cell and bins 
    """
    def density_map(self, x, y):
        bin_index_xl = np.maximum(np.floor(x/self.bin_size_x).astype(np.int32), 0)
        bin_index_xh = np.minimum(np.ceil((x+self.node_size_x)/self.bin_size_x).astype(np.int32), self.num_bins_x-1)
        bin_index_yl = np.maximum(np.floor(y/self.bin_size_y).astype(np.int32), 0)
        bin_index_yh = np.minimum(np.ceil((y+self.node_size_y)/self.bin_size_y).astype(np.int32), self.num_bins_y-1)

        density_map = np.zeros([self.num_bins_x, self.num_bins_y])

        for node_id in range(self.num_physical_nodes):
            for ix in range(bin_index_xl[node_id], bin_index_xh[node_id]+1):
                for iy in range(bin_index_yl[node_id], bin_index_yh[node_id]+1):
                    density_map[ix, iy] += self.overlap(
                            self.bin_xl(ix), self.bin_yl(iy), self.bin_xh(ix), self.bin_yh(iy), 
                            x[node_id], y[node_id], x[node_id]+self.node_size_x[node_id], y[node_id]+self.node_size_y[node_id]
                            )

        for ix in range(self.num_bins_x):
            for iy in range(self.num_bins_y):
                density_map[ix, iy] /= (self.bin_xh(ix)-self.bin_xl(ix))*(self.bin_yh(iy)-self.bin_yl(iy))

        return density_map

    """
    return density overflow cost 
    if density of a bin is larger than target_density, consider as overflow bin 
    """
    def density_overflow(self, x, y, target_density):
        density_map = self.density_map(x, y)
        return np.sum(np.square(np.maximum(density_map-target_density, 0.0)))

    """
    print node information 
    """
    def print_node(self, node_id): 
        print("node %s(%d), size (%g, %g), pos (%g, %g)" % (self.node_names[node_id], node_id, self.node_size_x[node_id], self.node_size_y[node_id], self.node_x[node_id], self.node_y[node_id]))
        pins = "pins "
        for pin_id in self.node2pin_map[node_id]:
            pins += "%s(%s, %d) " % (self.node_names[self.pin2node_map[pin_id]], self.net_names[self.pin2net_map[pin_id]], pin_id)
        print(pins)

    """
    print net information
    """
    def print_net(self, net_id):
        print("net %s(%d)" % (self.net_names[net_id], net_id))
        pins = "pins "
        for pin_id in self.net2pin_map[net_id]:
            pins += "%s(%s, %d) " % (self.node_names[self.pin2node_map[pin_id]], self.net_names[self.pin2net_map[pin_id]], pin_id)
        print(pins)

    """
    print row information 
    """
    def print_row(self, row_id):
        print("row %d %s" % (row_id, self.rows[row_id]))

    """
    read using python 
    """
    def read_bookshelf(self, params): 
        self.dtype = datatypes[params.dtype]
        node_file = None 
        net_file = None
        pl_file = None 
        scl_file = None

        # read aux file 
        aux_dir = os.path.dirname(params.aux_file)
        with open(params.aux_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    files = line.split(":")[1].strip()
                    files = files.split()
                    for file in files:
                        file = file.strip()
                        if file.endswith(".nodes"):
                            node_file = os.path.join(aux_dir, file)
                        elif file.endswith(".nets"):
                            net_file = os.path.join(aux_dir, file)
                        elif file.endswith(".pl"):
                            pl_file = os.path.join(aux_dir, file)
                        elif file.endswith(".scl"):
                            scl_file = os.path.join(aux_dir, file)
                        else:
                            print("[W] ignore files not used: %s" % (file))

        if node_file:
            self.read_nodes(node_file)
        if net_file:
            self.read_nets(net_file)
        if pl_file:
            self.read_pl(pl_file)
        if scl_file:
            self.read_scl(scl_file)

        # convert node2pin_map to array of array 
        for i in range(len(self.node2pin_map)):
            self.node2pin_map[i] = np.array(self.node2pin_map[i], dtype=np.int32)
        self.node2pin_map = np.array(self.node2pin_map)

        # convert net2pin_map to array of array 
        for i in range(len(self.net2pin_map)):
            self.net2pin_map[i] = np.array(self.net2pin_map[i], dtype=np.int32)
        self.net2pin_map = np.array(self.net2pin_map)

        # sort nets and pins
        #self.sort()

        # construct flat_net2pin_map and flat_net2pin_start_map
        # flat netpin map, length of #pins
        # starting index in netpin map for each net, length of #nets+1, the last entry is #pins  
        self.flat_net2pin_map, self.flat_net2pin_start_map = self.flatten_nested_map(self.net2pin_map)
        # construct flat_node2pin_map and flat_node2pin_start_map
        # flat nodepin map, length of #pins
        # starting index in nodepin map for each node, length of #nodes+1, the last entry is #pins  
        self.flat_node2pin_map, self.flat_node2pin_start_map = self.flatten_nested_map(self.node2pin_map)

    def flatten_nested_map(self, net2pin_map): 
        # flat netpin map, length of #pins
        flat_net2pin_map = np.zeros(len(pin2net_map), dtype=np.int32)
        # starting index in netpin map for each net, length of #nets+1, the last entry is #pins  
        flat_net2pin_start_map = np.zeros(len(net2pin_map)+1, dtype=np.int32)
        count = 0
        for i in range(len(net2pin_map)):
            flat_net2pin_map[count:count+len(net2pin_map[i])] = net2pin_map[i]
            flat_net2pin_start_map[i] = count 
            count += len(net2pin_map[i])
        assert flat_net2pin_map[-1] != 0
        flat_net2pin_start_map[len(net2pin_map)] = len(pin2net_map)

        return flat_net2pin_map, flat_net2pin_start_map

    """
    read using c++ 
    """
    def read(self, params): 
        self.dtype = datatypes[params.dtype]
        db = place_io.PlaceIOFunction.forward(params)
        self.num_physical_nodes = db.num_nodes
        self.num_terminals = db.num_terminals
        self.node_name2id_map = db.node_name2id_map
        self.node_names = np.array(db.node_names, dtype=np.string_)
        self.node_x = np.array(db.node_x, dtype=self.dtype)
        self.node_y = np.array(db.node_y, dtype=self.dtype)
        self.node_orient = np.array(db.node_orient, dtype=np.string_)
        self.node_size_x = np.array(db.node_size_x, dtype=self.dtype)
        self.node_size_y = np.array(db.node_size_y, dtype=self.dtype)
        self.pin_direct = np.array(db.pin_direct, dtype=np.string_)
        self.pin_offset_x = np.array(db.pin_offset_x, dtype=self.dtype)
        self.pin_offset_y = np.array(db.pin_offset_y, dtype=self.dtype)
        self.net_name2id_map = db.net_name2id_map
        self.net_names = np.array(db.net_names, dtype=np.string_)
        self.net2pin_map = db.net2pin_map
        self.flat_net2pin_map = np.array(db.flat_net2pin_map, dtype=np.int32)
        self.flat_net2pin_start_map = np.array(db.flat_net2pin_start_map, dtype=np.int32)
        self.node2pin_map = db.node2pin_map
        self.flat_node2pin_map = np.array(db.flat_node2pin_map, dtype=np.int32)
        self.flat_node2pin_start_map = np.array(db.flat_node2pin_start_map, dtype=np.int32)
        self.pin2node_map = np.array(db.pin2node_map, dtype=np.int32)
        self.pin2net_map = np.array(db.pin2net_map, dtype=np.int32)
        self.rows = np.array(db.rows, dtype=self.dtype)
        self.xl = float(db.xl)
        self.yl = float(db.yl)
        self.xh = float(db.xh)
        self.yh = float(db.yh)
        self.row_height = float(db.row_height)
        self.site_width = float(db.site_width)
        self.num_movable_pins = db.num_movable_pins

        # convert node2pin_map to array of array 
        for i in range(len(self.node2pin_map)):
            self.node2pin_map[i] = np.array(self.node2pin_map[i], dtype=np.int32)
        self.node2pin_map = np.array(self.node2pin_map)

        # convert net2pin_map to array of array 
        for i in range(len(self.net2pin_map)):
            self.net2pin_map[i] = np.array(self.net2pin_map[i], dtype=np.int32)
        self.net2pin_map = np.array(self.net2pin_map)

    """
    top API to read placement files 
    """
    def __call__(self, params):
        tt = time.time()

        #self.read_bookshelf(params)
        self.read(params)

        # scale 
        self.scale(params.scale_factor)

        print("=============== Benchmark Statistics ===============")
        print("\t#nodes = %d, #terminals = %d, #movable = %d, #nets = %d" % (self.num_physical_nodes, self.num_terminals, self.num_movable_nodes, len(self.net_names)))
        print("\tdie area = (%g, %g, %g, %g) %g" % (self.xl, self.yl, self.xh, self.yh, self.area))
        print("\trow height = %g, site width = %g" % (self.row_height, self.site_width))

        # set number of bins 
        self.num_bins_x = params.num_bins_x #self.num_bins(self.xl, self.xh, self.bin_size_x)
        self.num_bins_y = params.num_bins_y #self.num_bins(self.yl, self.yh, self.bin_size_y)
        # set bin size 
        self.bin_size_x = (self.xh-self.xl)/params.num_bins_x 
        self.bin_size_y = (self.yh-self.yl)/params.num_bins_y 

        # bin center array 
        self.bin_center_x = self.bin_centers(self.xl, self.xh, self.bin_size_x)
        self.bin_center_y = self.bin_centers(self.yl, self.yh, self.bin_size_y)

        print("\tnum_bins = %dx%d, bin sizes = %gx%g" % (self.num_bins_x, self.num_bins_y, self.bin_size_x/self.row_height, self.bin_size_y/self.row_height))

        # set num_movable_pins 
        if self.num_movable_pins is None:
            self.num_movable_pins = 0 
            for node_id in self.pin2node_map:
                if node_id < self.num_movable_nodes:
                    self.num_movable_pins += 1
        print("\t#pins = %d, #movable_pins = %d" % (self.num_pins, self.num_movable_pins))
        # set total cell area 
        self.total_movable_node_area = float(np.sum(self.node_size_x[:self.num_movable_nodes]*self.node_size_y[:self.num_movable_nodes]))
        # total fixed node area should exclude the area outside the layout 
        self.total_fixed_node_area = float(np.sum(
                np.maximum(
                    np.minimum(self.node_x[self.num_movable_nodes:self.num_physical_nodes]+self.node_size_x[self.num_movable_nodes:self.num_physical_nodes], self.xh)
                    - np.maximum(self.node_x[self.num_movable_nodes:self.num_physical_nodes], self.xl), 
                    0.0) * np.maximum(
                        np.minimum(self.node_y[self.num_movable_nodes:self.num_physical_nodes]+self.node_size_y[self.num_movable_nodes:self.num_physical_nodes], self.yh)
                        - np.maximum(self.node_y[self.num_movable_nodes:self.num_physical_nodes], self.yl), 
                        0.0)
                ))
        #self.total_fixed_node_area = float(np.sum(self.node_size_x[self.num_movable_nodes:]*self.node_size_y[self.num_movable_nodes:]))
        print("\ttotal_movable_node_area = %g, total_fixed_node_area = %g" % (self.total_movable_node_area, self.total_fixed_node_area))

        # insert filler nodes 
        if params.enable_fillers: 
            self.total_filler_node_area = max((self.area-self.total_fixed_node_area)*params.target_density-self.total_movable_node_area, 0.0)
            node_size_order = np.argsort(self.node_size_x[:self.num_movable_nodes])
            filler_size_x = np.mean(self.node_size_x[node_size_order[int(self.num_movable_nodes*0.05):int(self.num_movable_nodes*0.95)]])
            filler_size_y = self.row_height
            self.num_filler_nodes = int(round(self.total_filler_node_area/(filler_size_x*filler_size_y)))
            self.node_size_x = np.concatenate([self.node_size_x, np.full(self.num_filler_nodes, fill_value=filler_size_x, dtype=self.node_size_x.dtype)])
            self.node_size_y = np.concatenate([self.node_size_y, np.full(self.num_filler_nodes, fill_value=filler_size_y, dtype=self.node_size_y.dtype)])
        else:
            self.total_filler_node_area = 0 
            self.num_filler_nodes = 0
        print("\ttotal_filler_node_area = %g, #fillers = %g, filler sizes = %gx%g" % (self.total_filler_node_area, self.num_filler_nodes, filler_size_x, filler_size_y))
        print("====================================================")

        print("[I] reading benchmark takes %g seconds" % (time.time()-tt))

    """
    read .node file 
    """
    def read_nodes(self, node_file): 
        print("[I] reading %s" % (node_file))
        count = 0 
        with open(node_file, "r") as f: 
            for line in f:
                line = line.strip()
                if line.startswith("UCLA") or line.startswith("#"):
                    continue
                # NumNodes
                num_physical_nodes = re.search(r"NumNodes\s*:\s*(\d+)", line)
                if num_physical_nodes: 
                    self.num_physical_nodes = int(num_physical_nodes.group(1))
                    self.node_names = np.chararray(self.num_physical_nodes, itemsize=64)
                    self.node_x = np.zeros(self.num_physical_nodes)
                    self.node_y = np.zeros(self.num_physical_nodes)
                    self.node_orient = np.chararray(self.num_physical_nodes, itemsize=2)
                    self.node_size_x = np.zeros(self.num_physical_nodes)
                    self.node_size_y = np.zeros(self.num_physical_nodes)
                    self.node2pin_map = [None]*self.num_physical_nodes
                # NumTerminals
                num_terminals = re.search(r"NumTerminals\s*:\s*(\d+)", line)
                if num_terminals:
                    self.num_terminals = int(num_terminals.group(1))
                # nodes and terminals 
                node = re.search(r"(\w+)\s+([+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)\s+([+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)", line)
                if node: 
                    self.node_name2id_map[node.group(1)] = count 
                    self.node_names[count] = node.group(1)
                    self.node_size_x[count] = float(node.group(2))
                    self.node_size_y[count] = float(node.group(6))
                    count += 1
                    # I assume the terminals append to the list, so I will not store extra information about it 
    """
    read .net file 
    """
    def read_nets(self, net_file): 
        print("[I] reading %s" % (net_file))
        net_count = 0 
        pin_count = 0
        degree_count = 0
        with open(net_file, "r") as f: 
            for line in f:
                line = line.strip()
                if line.startswith("UCLA"):
                    pass 
                # NumNets 
                elif line.startswith("NumNets"): 
                    num_nets = re.search(r"NumNets\s*:\s*(\d+)", line)
                    if num_nets:
                        num_nets = int(num_nets.group(1))
                        self.net_names = np.chararray(num_nets, itemsize=32)
                        self.net2pin_map = [None]*num_nets 
                # NumPins 
                elif line.startswith("NumPins"): 
                    num_pins = re.search(r"NumPins\s*:\s*(\d+)", line)
                    if num_pins:
                        num_pins = int(num_pins.group(1))
                        self.pin_direct = np.chararray(num_pins, itemsize=1)
                        self.pin_offset_x = np.zeros(num_pins)
                        self.pin_offset_y = np.zeros(num_pins)
                        self.pin2node_map = np.zeros(num_pins).astype(np.int32)
                        self.pin2net_map = np.zeros(num_pins).astype(np.int32)
                # NetDegree 
                elif line.startswith("NetDegree"): 
                    net_degree = re.search(r"NetDegree\s*:\s*(\d+)\s*(\w+)", line)
                    if net_degree:
                        self.net_name2id_map[net_degree.group(2)] = net_count 
                        self.net_names[net_count] = net_degree.group(2)
                        self.net2pin_map[net_count] = np.zeros(int(net_degree.group(1))).astype(np.int32)
                        net_count += 1
                        degree_count = 0
                # net pin 
                else: 
                    net_pin = re.search(r"(\w+)\s*([IO])\s*:\s*([+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)\s+([+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)", line)
                    if net_pin:
                        node_id = self.node_name2id_map[net_pin.group(1)]
                        self.pin_direct[pin_count] = net_pin.group(2)
                        self.pin_offset_x[pin_count] = float(net_pin.group(3))
                        self.pin_offset_y[pin_count] = float(net_pin.group(7))
                        self.pin2node_map[pin_count] = node_id
                        self.pin2net_map[pin_count] = net_count-1
                        self.net2pin_map[net_count-1][degree_count] = pin_count
                        if self.node2pin_map[node_id] is None: 
                            self.node2pin_map[node_id] = []
                        self.node2pin_map[node_id].append(pin_count)
                        pin_count += 1
                        degree_count += 1
    """
    read .scl file 
    """
    def read_scl(self, scl_file):
        print("[I] reading %s" % (scl_file))
        count = 0 
        with open(scl_file, "r") as f:
            for line in f:
                line = line.strip()
                ## CoreRow 
                #core_row = re.search(r"CoreRow\s*(\w+)", line)
                # End 
                if line.startswith("UCLA"):
                    pass
                # NumRows 
                elif line.startswith("NumRows"): 
                    num_rows = re.search(r"NumRows\s*:\s*(\d+)", line)
                    if num_rows:
                        self.rows = np.zeros((int(num_rows.group(1)), 4))
                elif line == "End":
                    count += 1
                # Coordinate 
                elif line.startswith("Coordinate"): 
                    coordinate = re.search(r"Coordinate\s*:\s*([+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)", line)
                    if coordinate:
                        self.rows[count][1] = float(coordinate.group(1))
                # Height 
                elif line.startswith("Height"): 
                    height = re.search(r"Height\s*:\s*([+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)", line)
                    if height:
                        self.rows[count][3] = self.rows[count][1] + float(height.group(1))
                # Sitewidth 
                elif line.startswith("Sitewidth"):
                    sitewidth = re.search(r"Sitewidth\s*:\s*([+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)", line)
                    if sitewidth: 
                        self.rows[count][2] = float(sitewidth.group(1)) # temporily store to row.yh 
                        if self.site_width is None:
                            self.site_width = self.rows[count][2]
                        else:
                            assert self.site_width == self.rows[count][2]
                # Sitespacing 
                elif line.startswith("Sitespacing"):
                    sitespacing = re.search(r"Sitespacing\s*:\s*([+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)", line)
                    if sitespacing: 
                        sitespacing = float(sitespacing.group(1))
                # SubrowOrigin
                elif line.startswith("SubrowOrigin"):
                    subrow_origin = re.search(r"SubrowOrigin\s*:\s*([+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)\s+NumSites\s*:\s*(\d+)", line)
                    if subrow_origin:
                        self.rows[count][0] = float(subrow_origin.group(1))
                        self.rows[count][2] = self.rows[count][0] + float(subrow_origin.group(5))*self.rows[count][2]

            # set xl, yl, xh, yh 
            self.xl = np.finfo(self.dtype).max
            self.yl = np.finfo(self.dtype).max 
            self.xh = np.finfo(self.dtype).min
            self.yh = np.finfo(self.dtype).min
            for row in self.rows:
                self.xl = min(self.xl, row[0])
                self.yl = min(self.yl, row[1])
                self.xh = max(self.xh, row[2])
                self.yh = max(self.yh, row[3])

            # set row height 
            self.row_height = self.rows[0][3]-self.rows[0][1]
    """
    read .pl file 
    """
    def read_pl(self, pl_file):
        print("[I] reading %s" % (pl_file))
        count = 0 
        with open(pl_file, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("UCLA"):
                    continue
                # node positions 
                pos = re.search(r"(\w+)\s+([+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)\s+([+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)\s*:\s*(\w+)", line)
                if pos: 
                    node_id = self.node_name2id_map[pos.group(1)]
                    self.node_x[node_id] = float(pos.group(2))
                    self.node_y[node_id] = float(pos.group(6))
                    #print("%g, %g" % (self.node_x[node_id], self.node_y[node_id]))
                    self.node_orient[node_id] = pos.group(10)
                    orient = pos.group(4)
    """
    write .pl file
    """
    def write_pl(self, params, pl_file):
        tt = time.time()
        print("[I] writing to %s" % (pl_file))
        content = "UCLA pl 1.0\n"
        str_node_names = np.array(self.node_names).astype(np.str)
        str_node_orient = np.array(self.node_orient).astype(np.str)
        for i in range(self.num_physical_nodes):
            content += "\n%s %g %g : %s" % (
                    str_node_names[i],
                    self.node_x[i]/params.scale_factor, 
                    self.node_y[i]/params.scale_factor, 
                    str_node_orient[i]
                    )
            if i >= self.num_movable_nodes:
                content += " /FIXED"
        with open(pl_file, "w") as f:
            f.write(content)
        print("[I] write_pl takes %.3f seconds" % (time.time()-tt))
    """
    write .net file
    """
    def write_nets(self, params, net_file):
        tt = time.time()
        print("[I] writing to %s" % (net_file))
        content = "UCLA nets 1.0\n"
        content += "\nNumNets : %d" % (len(self.net2pin_map))
        content += "\nNumPins : %d" % (len(self.pin2net_map))
        content += "\n"

        for net_id in range(len(self.net2pin_map)):
            pins = self.net2pin_map[net_id]
            content += "\nNetDegree : %d %s" % (len(pins), self.net_names[net_id])
            for pin_id in pins: 
                content += "\n\t%s %s : %d %d" % (self.node_names[self.pin2node_map[pin_id]], self.pin_direct[pin_id], self.pin_offset_x[pin_id]/params.scale_factor, self.pin_offset_y[pin_id]/params.scale_factor)

        with open(net_file, "w") as f:
            f.write(content)
        print("[I] write_nets takes %.3f seconds" % (time.time()-tt))

    """
    plot layout
    """
    def plot(self, params, iteration, x, y): 
        try: 
            tt = time.time()
            ## dump intermediate solutions for debug 
            #if iteration in [1, 200, 250, 300, 350, 400, 450, 500]:
            #    with open("summary/train/xy%s.pkl" % ('{:04}'.format(iteration)), "wb") as f:
            #        pickle.dump([x, y], f)
            width = 800
            height = 800
            surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
            ctx = cairo.Context(surface)
            ctx.scale(width, height)  # Normalizing the canvas

            layout_xl = min(np.amin(self.node_x[self.num_movable_nodes:self.num_physical_nodes]), self.xl)
            layout_yl = min(np.amin(self.node_y[self.num_movable_nodes:self.num_physical_nodes]), self.yl)
            layout_xh = max(np.amax(self.node_x[self.num_movable_nodes:self.num_physical_nodes]+self.node_size_x[self.num_movable_nodes:self.num_physical_nodes]), self.xh)
            layout_yh = max(np.amax(self.node_y[self.num_movable_nodes:self.num_physical_nodes]+self.node_size_y[self.num_movable_nodes:self.num_physical_nodes]), self.yh)

            def normalize_x(xx):
                return (xx - (layout_xl-2*self.bin_size_x))/(layout_xh-layout_xl+4*self.bin_size_x)
            def normalize_y(xx):
                return (xx - (layout_yl-2*self.bin_size_y))/(layout_yh-layout_yl+4*self.bin_size_y)
            def draw_rect(x1, y1, x2, y2):
                ctx.move_to(x1, y1)
                ctx.line_to(x1, y2)
                ctx.line_to(x2, y2)
                ctx.line_to(x2, y1)
                ctx.close_path()
                ctx.stroke()

            # draw layout region 
            ctx.set_source_rgb(1, 1, 1)
            draw_layout_xl = normalize_x(layout_xl-2*self.bin_size_x)
            draw_layout_yl = normalize_y(layout_yl-2*self.bin_size_y)
            draw_layout_xh = normalize_x(layout_xh+2*self.bin_size_x)
            draw_layout_yh = normalize_y(layout_yh+2*self.bin_size_y)
            ctx.rectangle(draw_layout_xl, draw_layout_yl, draw_layout_xh, draw_layout_yh)
            ctx.fill()
            ctx.set_line_width(0.001)
            ctx.set_source_rgba(0.1, 0.1, 0.1, alpha=0.8)
            #ctx.move_to(normalize_x(self.xl), normalize_y(self.yl))
            #ctx.line_to(normalize_x(self.xl), normalize_y(self.yh))
            #ctx.line_to(normalize_x(self.xh), normalize_y(self.yh))
            #ctx.line_to(normalize_x(self.xh), normalize_y(self.yl))
            #ctx.close_path()
            #ctx.stroke()
            # draw bins 
            for i in range(1, self.num_bins_x):
                ctx.move_to(normalize_x(self.bin_xl(i)), normalize_y(self.yl))
                ctx.line_to(normalize_x(self.bin_xl(i)), normalize_y(self.yh))
                ctx.close_path()
                ctx.stroke()
            for i in range(1, self.num_bins_y):
                ctx.move_to(normalize_x(self.xl), normalize_y(self.bin_yl(i)))
                ctx.line_to(normalize_x(self.xh), normalize_y(self.bin_yl(i)))
                ctx.close_path()
                ctx.stroke()

            # draw cells
            xl = x
            yl = layout_yl+layout_yh-(y+self.node_size_y[0:len(y)]) # flip y 
            xh = xl+self.node_size_x[0:len(x)]
            yh = layout_yl+layout_yh-y # flip y 
            xl = normalize_x(xl)
            yl = normalize_y(yl)
            xh = normalize_x(xh)
            yh = normalize_y(yh)
            ctx.set_line_width(0.001)
            #print("plot layout")
            # draw fixed macros
            ctx.set_source_rgba(1, 0, 0, alpha=0.5)
            for i in range(self.num_movable_nodes, self.num_physical_nodes):
                ctx.rectangle(xl[i], yl[i], xh[i]-xl[i], yh[i]-yl[i])  # Rectangle(xl, yl, w, h)
                ctx.fill()
            ctx.set_source_rgba(0, 0, 0, alpha=1.0)  # Solid color
            for i in range(self.num_movable_nodes, self.num_physical_nodes):
                draw_rect(xl[i], yl[i], xh[i], yh[i])
            # draw fillers
            if len(xl) > self.num_physical_nodes: # filler is included 
                ctx.set_line_width(0.001)
                ctx.set_source_rgba(230/255.0, 230/255.0, 250/255.0, alpha=0.3)  # Solid color
                for i in range(self.num_physical_nodes, self.num_nodes):
                    draw_rect(xl[i], yl[i], xh[i], yh[i])
            # draw cells 
            ctx.set_line_width(0.002)
            ctx.set_source_rgba(0, 0, 1, alpha=0.8)  # Solid color
            for i in range(self.num_movable_nodes):
                draw_rect(xl[i], yl[i], xh[i], yh[i])

            # show iteration, not working  
            ctx.set_source_rgb(0, 0, 0)
            ctx.set_line_width(0.1)
            ctx.select_font_face("monospace", cairo.FONT_SLANT_NORMAL, 
                    cairo.FONT_WEIGHT_NORMAL)
            ctx.set_font_size(72)
            ctx.move_to(normalize_x((self.xl+self.xh)/2), normalize_y((self.yl+self.yh)/2))
            ctx.show_text('{:04}'.format(iteration))

            figname = "./summary/train/train%s.png" % ('{:04}'.format(iteration))
            surface.write_to_png(figname)  # Output to PNG
            print("[I] plotting to %s takes %.3f seconds" % (figname, time.time()-tt))
            #print(self.session.run(self.grads))
            #print(self.session.run(self.masked_grads))
        except Exception as e:
            print("[E] failed to plot")
            print(str(e))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("[E] input parameters in json format in required")
    paramsArray = []
    for i in range(1, len(sys.argv)):
        params = Params.Params()
        params.load(sys.argv[i])
        paramsArray.append(params)
    print("[I] parameters[%d] = %s" % (len(paramsArray), paramsArray))

    for params in paramsArray: 
        db = PlaceDB()
        db(params)

        db.print_node(1)
        db.print_net(1)
        db.print_row(1)
