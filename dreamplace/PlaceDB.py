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
import Params
import dreamplace 
import dreamplace.ops.place_io.place_io as place_io 
import pdb 

datatypes = {
        'float32' : np.float32, 
        'float64' : np.float64
        }

class PlaceDB (object):
    """
    @brief placement database 
    """
    def __init__(self):
        """
        initialization
        To avoid the usage of list, I flatten everything.  
        """
        self.rawdb = None # raw placement database, a C++ object 

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
        self.net_weights = None # weights for each net

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

    def scale_pl(self, scale_factor):
        """
        @brief scale placement solution only
        @param scale_factor scale factor 
        """
        self.node_x *= scale_factor
        self.node_y *= scale_factor 

    def scale(self, scale_factor):
        """
        @brief scale distances
        @param scale_factor scale factor 
        """
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

    def sort(self):
        """
        @brief Sort net by degree. 
        Sort pin array such that pins belonging to the same net is abutting each other
        """
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

    @property
    def num_movable_nodes(self):
        """
        @return number of movable nodes 
        """
        return self.num_physical_nodes - self.num_terminals

    @property 
    def num_nodes(self):
        """
        @return number of movable nodes, terminals and fillers
        """
        return self.num_physical_nodes + self.num_filler_nodes

    @property
    def num_nets(self):
        """
        @return number of nets
        """
        return len(self.net2pin_map)

    @property
    def num_pins(self):
        """
        @return number of pins
        """
        return len(self.pin2net_map)

    @property
    def width(self):
        """
        @return width of layout 
        """
        return self.xh-self.xl

    @property
    def height(self):
        """
        @return height of layout 
        """
        return self.yh-self.yl

    @property
    def area(self):
        """
        @return area of layout 
        """
        return self.width*self.height

    def bin_index_x(self, x): 
        """
        @param x horizontal location 
        @return bin index in x direction 
        """
        if x < self.xl:
            return 0 
        elif x > self.xh:
            return int(np.floor((self.xh-self.xl)/self.bin_size_x))
        else:
            return int(np.floor((x-self.xl)/self.bin_size_x))

    def bin_index_y(self, y): 
        """
        @param y vertical location 
        @return bin index in y direction 
        """
        if y < self.yl:
            return 0 
        elif y > self.yh:
            return int(np.floor((self.yh-self.yl)/self.bin_size_y))
        else:
            return int(np.floor((y-self.yl)/self.bin_size_y))

    def bin_xl(self, id_x):
        """
        @param id_x horizontal index 
        @return bin xl
        """
        return self.xl+id_x*self.bin_size_x

    def bin_xh(self, id_x):
        """
        @param id_x horizontal index 
        @return bin xh
        """
        return min(self.bin_xl(id_x)+self.bin_size_x, self.xh)

    def bin_yl(self, id_y):
        """
        @param id_y vertical index 
        @return bin yl
        """
        return self.yl+id_y*self.bin_size_y

    def bin_yh(self, id_y):
        """
        @param id_y vertical index 
        @return bin yh
        """
        return min(self.bin_yl(id_y)+self.bin_size_y, self.yh)

    def num_bins(self, l, h, bin_size):
        """
        @brief compute number of bins 
        @param l lower bound 
        @param h upper bound 
        @param bin_size bin size 
        @return number of bins 
        """
        return int(np.ceil((h-l)/bin_size))

    def bin_centers(self, l, h, bin_size):
        """
        @brief compute bin centers 
        @param l lower bound 
        @param h upper bound 
        @param bin_size bin size 
        @return array of bin centers 
        """
        num_bins = self.num_bins(l, h, bin_size)
        centers = np.zeros(num_bins, dtype=self.dtype)
        for id_x in range(num_bins): 
            bin_l = l+id_x*bin_size
            bin_h = min(bin_l+bin_size, h)
            centers[id_x] = (bin_l+bin_h)/2
        return centers 

    def net_hpwl(self, x, y, net_id): 
        """
        @brief compute HPWL of a net 
        @param x horizontal cell locations 
        @param y vertical cell locations
        @return hpwl of a net 
        """
        pins = self.net2pin_map[net_id]
        nodes = self.pin2node_map[pins]
        hpwl_x = np.amax(x[nodes]+self.pin_offset_x[pins]) - np.amin(x[nodes]+self.pin_offset_x[pins])
        hpwl_y = np.amax(y[nodes]+self.pin_offset_y[pins]) - np.amin(y[nodes]+self.pin_offset_y[pins])

        return (hpwl_x+hpwl_y)*self.net_weights[net_id]

    def hpwl(self, x, y):
        """
        @brief compute total HPWL 
        @param x horizontal cell locations 
        @param y vertical cell locations 
        @return hpwl of all nets
        """
        wl = 0
        for net_id in range(len(self.net2pin_map)):
            wl += self.net_hpwl(x, y, net_id)
        return wl 

    def overlap(self, xl1, yl1, xh1, yh1, xl2, yl2, xh2, yh2):
        """
        @brief compute overlap between two boxes 
        @return overlap area between two rectangles
        """
        return max(min(xh1, xh2)-max(xl1, xl2), 0.0) * max(min(yh1, yh2)-max(yl1, yl2), 0.0)

    def density_map(self, x, y):
        """
        @brief this density map evaluates the overlap between cell and bins 
        @param x horizontal cell locations 
        @param y vertical cell locations 
        @return density map 
        """
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

    def density_overflow(self, x, y, target_density):
        """
        @brief if density of a bin is larger than target_density, consider as overflow bin 
        @param x horizontal cell locations 
        @param y vertical cell locations 
        @param target_density target density 
        @return density overflow cost 
        """
        density_map = self.density_map(x, y)
        return np.sum(np.square(np.maximum(density_map-target_density, 0.0)))

    def print_node(self, node_id): 
        """
        @brief print node information 
        @param node_id cell index 
        """
        print("node %s(%d), size (%g, %g), pos (%g, %g)" % (self.node_names[node_id], node_id, self.node_size_x[node_id], self.node_size_y[node_id], self.node_x[node_id], self.node_y[node_id]))
        pins = "pins "
        for pin_id in self.node2pin_map[node_id]:
            pins += "%s(%s, %d) " % (self.node_names[self.pin2node_map[pin_id]], self.net_names[self.pin2net_map[pin_id]], pin_id)
        print(pins)

    def print_net(self, net_id):
        """
        @brief print net information
        @param net_id net index 
        """
        print("net %s(%d)" % (self.net_names[net_id], net_id))
        pins = "pins "
        for pin_id in self.net2pin_map[net_id]:
            pins += "%s(%s, %d) " % (self.node_names[self.pin2node_map[pin_id]], self.net_names[self.pin2net_map[pin_id]], pin_id)
        print(pins)

    def print_row(self, row_id):
        """
        @brief print row information 
        @param row_id row index 
        """
        print("row %d %s" % (row_id, self.rows[row_id]))

    def flatten_nested_map(self, net2pin_map): 
        """
        @brief flatten an array of array to two arrays like CSV format 
        @param net2pin_map array of array 
        @return a pair of (elements, cumulative column indices of the beginning element of each row)
        """
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

    def read(self, params): 
        """
        @brief read using c++ 
        @param params parameters 
        """
        self.dtype = datatypes[params.dtype]
        self.rawdb = place_io.PlaceIOFunction.read(params)
        pydb = place_io.PlaceIOFunction.pydb(self.rawdb)

        self.num_physical_nodes = pydb.num_nodes
        self.num_terminals = pydb.num_terminals
        self.node_name2id_map = pydb.node_name2id_map
        self.node_names = np.array(pydb.node_names, dtype=np.string_)
        self.node_x = np.array(pydb.node_x, dtype=self.dtype)
        self.node_y = np.array(pydb.node_y, dtype=self.dtype)
        self.node_orient = np.array(pydb.node_orient, dtype=np.string_)
        self.node_size_x = np.array(pydb.node_size_x, dtype=self.dtype)
        self.node_size_y = np.array(pydb.node_size_y, dtype=self.dtype)
        self.pin_direct = np.array(pydb.pin_direct, dtype=np.string_)
        self.pin_offset_x = np.array(pydb.pin_offset_x, dtype=self.dtype)
        self.pin_offset_y = np.array(pydb.pin_offset_y, dtype=self.dtype)
        self.net_name2id_map = pydb.net_name2id_map
        self.net_names = np.array(pydb.net_names, dtype=np.string_)
        self.net2pin_map = pydb.net2pin_map
        self.flat_net2pin_map = np.array(pydb.flat_net2pin_map, dtype=np.int32)
        self.flat_net2pin_start_map = np.array(pydb.flat_net2pin_start_map, dtype=np.int32)
        self.net_weights = np.array(pydb.net_weights, dtype=self.dtype)
        self.node2pin_map = pydb.node2pin_map
        self.flat_node2pin_map = np.array(pydb.flat_node2pin_map, dtype=np.int32)
        self.flat_node2pin_start_map = np.array(pydb.flat_node2pin_start_map, dtype=np.int32)
        self.pin2node_map = np.array(pydb.pin2node_map, dtype=np.int32)
        self.pin2net_map = np.array(pydb.pin2net_map, dtype=np.int32)
        self.rows = np.array(pydb.rows, dtype=self.dtype)
        self.xl = float(pydb.xl)
        self.yl = float(pydb.yl)
        self.xh = float(pydb.xh)
        self.yh = float(pydb.yh)
        self.row_height = float(pydb.row_height)
        self.site_width = float(pydb.site_width)
        self.num_movable_pins = pydb.num_movable_pins

        # convert node2pin_map to array of array 
        for i in range(len(self.node2pin_map)):
            self.node2pin_map[i] = np.array(self.node2pin_map[i], dtype=np.int32)
        self.node2pin_map = np.array(self.node2pin_map)

        # convert net2pin_map to array of array 
        for i in range(len(self.net2pin_map)):
            self.net2pin_map[i] = np.array(self.net2pin_map[i], dtype=np.int32)
        self.net2pin_map = np.array(self.net2pin_map)

    def __call__(self, params):
        """
        @brief top API to read placement files 
        @param params parameters 
        """
        tt = time.time()

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

    def write(self, params, filename, sol_file_format=None):
        """
        @brief write placement solution
        @param filename output file name 
        @param sol_file_format solution file format, DEF|DEFSIMPLE|BOOKSHELF|BOOKSHELFALL
        """
        tt = time.time()
        print("[I] writing to %s" % (filename))
        if sol_file_format is None: 
            if filename.endswith(".def"): 
                sol_file_format = place_io.SolutionFileFormat.DEF 
            else:
                sol_file_format = place_io.SolutionFileFormat.BOOKSHELF

        # unscale locations 
        unscale_factor = 1.0/params.scale_factor
        if unscale_factor == 1.0:
            node_x = self.node_x
            node_y = self.node_y
        else:
            node_x = self.node_x * unscale_factor
            node_y = self.node_y * unscale_factor

        place_io.PlaceIOFunction.write(self.rawdb, filename, sol_file_format, node_x, node_y)
        print("[I] write %s takes %.3f seconds" % (str(sol_file_format), time.time()-tt))

    def write_pl(self, params, pl_file):
        """
        @brief write .pl file
        @param pl_file .pl file 
        """
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

    def write_nets(self, params, net_file):
        """
        @brief write .net file
        @param params parameters 
        @param net_file .net file 
        """
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

    def apply(self, params, node_x, node_y):
        """
        @brief apply placement solution and update database 
        """
        # assign solution 
        self.node_x[:self.num_movable_nodes] = node_x[:self.num_movable_nodes]
        self.node_y[:self.num_movable_nodes] = node_y[:self.num_movable_nodes]

        # unscale locations 
        unscale_factor = 1.0/params.scale_factor
        if unscale_factor == 1.0:
            node_x = self.node_x
            node_y = self.node_y
        else:
            node_x = self.node_x * unscale_factor
            node_y = self.node_y * unscale_factor

        # update raw database 
        place_io.PlaceIOFunction.apply(self.rawdb, node_x, node_y)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("[E] One input parameters in json format in required")

    params = Params.Params()
    params.load(sys.argv[sys.argv[1]])
    print("[I] parameters = %s" % (params))

    db = PlaceDB()
    db(params)

    db.print_node(1)
    db.print_net(1)
    db.print_row(1)
