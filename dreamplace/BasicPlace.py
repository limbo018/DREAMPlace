##
# @file   BasicPlace.py
# @author Yibo Lin
# @date   Jun 2018
# @brief  Base placement class 
#

import os 
import sys
import time 
import gzip 
if sys.version_info[0] < 3: 
    import cPickle as pickle
else:
    import _pickle as pickle
import re 
import numpy as np 
import torch 
import torch.nn as nn
import dreamplace.ops.move_boundary.move_boundary as move_boundary 
import dreamplace.ops.hpwl.hpwl as hpwl 
import dreamplace.ops.density_overflow.density_overflow as density_overflow 
import dreamplace.ops.electric_potential.electric_overflow as electric_overflow 
import dreamplace.ops.rmst_wl.rmst_wl as rmst_wl 
import dreamplace.ops.greedy_legalize.greedy_legalize as greedy_legalize 
import dreamplace.ops.draw_place.draw_place as draw_place 
import pdb 

class PlaceDataCollection (object):
    """
    @brief A wraper for all data tensors on device for building ops 
    """
    def __init__(self, pos, params, placedb, device):
        """
        @brief initialization 
        @param pos locations of cells 
        @param params parameters 
        @param placedb placement database 
        @param device cpu or cuda 
        """
        torch.set_num_threads = params.num_threads
        # position should be parameter 
        self.pos = pos 
        # other tensors required to build ops 
        self.node_size_x = torch.from_numpy(placedb.node_size_x).to(device)
        self.node_size_y = torch.from_numpy(placedb.node_size_y).to(device)

        self.pin_offset_x = torch.tensor(placedb.pin_offset_x, dtype=self.pos[0].dtype, device=device)
        self.pin_offset_y = torch.tensor(placedb.pin_offset_y, dtype=self.pos[0].dtype, device=device)

        self.pin2node_map = torch.from_numpy(placedb.pin2node_map).to(device)
        self.flat_node2pin_map = torch.from_numpy(placedb.flat_node2pin_map).to(device)
        self.flat_node2pin_start_map = torch.from_numpy(placedb.flat_node2pin_start_map).to(device)

        self.pin2net_map = torch.from_numpy(placedb.pin2net_map).to(device)
        self.flat_net2pin_map = torch.from_numpy(placedb.flat_net2pin_map).to(device)
        self.flat_net2pin_start_map = torch.from_numpy(placedb.flat_net2pin_start_map).to(device)

        self.net_mask_all = torch.from_numpy(np.ones(placedb.num_nets, dtype=np.uint8)).to(device) # all nets included 
        net_degrees = np.array([len(net2pin) for net2pin in placedb.net2pin_map])
        net_mask = np.logical_and(2 <= net_degrees, net_degrees < params.ignore_net_degree).astype(np.uint8)
        self.net_mask_ignore_large_degrees = torch.from_numpy(net_mask).to(device) # nets with large degrees are ignored 

        # avoid computing gradient for fixed macros 
        # 1 is for fixed macros 
        self.pin_mask_ignore_fixed_macros = (self.pin2node_map >= placedb.num_movable_nodes)

        self.bin_center_x = torch.from_numpy(placedb.bin_center_x).to(device)
        self.bin_center_y = torch.from_numpy(placedb.bin_center_y).to(device)

    def bin_center_x_padded(self, placedb, padding): 
        """
        @brief compute array of bin center horizontal coordinates with padding 
        @param placedb placement database 
        @param padding number of bins padding to boundary of placement region 
        """
        if padding == 0: 
            return self.bin_center_x 
        else:
            local_num_bins_x = num_bins_x + 2*padding 
            xl = placedb.xl - padding*bin_size_x
            xh = placedb.xh + padding*bin_size_x
            self.bin_center_x_padded = torch.from_numpy(placedb.bin_centers(xl, xh, bin_size_x)).to(device)
            return self.bin_center_x_padded

    def bin_center_y_padded(self, placedb, padding): 
        """
        @brief compute array of bin center vertical coordinates with padding 
        @param placedb placement database 
        @param padding number of bins padding to boundary of placement region 
        """
        if padding == 0: 
            return self.bin_center_y 
        else:
            local_num_bins_y = num_bins_y + 2*padding 
            yl = placedb.yl - padding*bin_size_y
            yh = placedb.yh + padding*bin_size_y
            self.bin_center_y_padded = torch.from_numpy(placedb.bin_centers(yl, yh, bin_size_y)).to(device)
            return self.bin_center_y_padded

class PlaceOpCollection (object):
    """
    @brief A wrapper for all ops 
    """
    def __init__(self):
        """
        @brief initialization
        """
        self.pin_pos_op = None 
        self.move_boundary_op = None 
        self.hpwl_op = None
        self.rmst_wl_op = None 
        self.density_overflow_op = None 
        self.greedy_legalize_op = None 
        self.detailed_place_op = None
        self.wirelength_op = None 
        self.update_gamma_op = None 
        self.density_op = None 
        self.update_density_weight_op = None
        self.precondition_op = None 
        self.noise_op = None 
        self.draw_place_op = None

class BasicPlace (nn.Module):
    """
    @brief Base placement class. 
    All placement engines should be derived from this class. 
    """
    def __init__(self, params, placedb):
        """
        @brief initialization
        @param params parameter 
        @param placedb placement database 
        """
        torch.manual_seed(params.random_seed)
        super(BasicPlace, self).__init__()

        #tt = time.time()
        self.init_pos = np.zeros(placedb.num_nodes*2, dtype=placedb.dtype)
        # x position 
        self.init_pos[0:placedb.num_physical_nodes] = placedb.node_x
        if params.global_place_flag: # move to center of layout 
            print("[I] move cells to the center of layout with random noise")
            self.init_pos[0:placedb.num_movable_nodes] = np.random.normal(loc=(placedb.xl*1.0+placedb.xh*1.0)/2, scale=(placedb.xh-placedb.xl)*0.001, size=placedb.num_movable_nodes)
        #self.init_pos[0:placedb.num_movable_nodes] = init_x[0:placedb.num_movable_nodes]*0.01 + (placedb.xl+placedb.xh)/2
        # y position 
        self.init_pos[placedb.num_nodes:placedb.num_nodes+placedb.num_physical_nodes] = placedb.node_y
        if params.global_place_flag: # move to center of layout 
            self.init_pos[placedb.num_nodes:placedb.num_nodes+placedb.num_movable_nodes] = np.random.normal(loc=(placedb.yl*1.0+placedb.yh*1.0)/2, scale=(placedb.yh-placedb.yl)*0.001, size=placedb.num_movable_nodes)
        #init_y[0:placedb.num_movable_nodes] = init_y[0:placedb.num_movable_nodes]*0.01 + (placedb.yl+placedb.yh)/2

        if placedb.num_filler_nodes: # uniformly distribute filler cells in the layout 
            self.init_pos[placedb.num_physical_nodes:placedb.num_nodes] = np.random.uniform(low=placedb.xl, high=placedb.xh-placedb.node_size_x[-placedb.num_filler_nodes], size=placedb.num_filler_nodes)
            self.init_pos[placedb.num_nodes+placedb.num_physical_nodes:placedb.num_nodes*2] = np.random.uniform(low=placedb.yl, high=placedb.yh-placedb.node_size_y[-placedb.num_filler_nodes], size=placedb.num_filler_nodes)

        #print("prepare init_pos takes %.2f seconds" % (time.time()-tt))

        self.device = torch.device("cuda" if params.gpu else "cpu")

        # position should be parameter 
        # must be defined in BasicPlace 
        #tt = time.time()
        self.pos = nn.ParameterList([nn.Parameter(torch.from_numpy(self.init_pos).to(self.device))])
        #print("build pos takes %.2f seconds" % (time.time()-tt))
        # shared data on device for building ops  
        # I do not want to construct the data from placedb again and again for each op 
        #tt = time.time()
        self.data_collections = PlaceDataCollection(self.pos, params, placedb, self.device)
        #print("build data_collections takes %.2f seconds" % (time.time()-tt))
        # similarly I wrap all ops 
        #tt = time.time()
        self.op_collections = PlaceOpCollection()
        #print("build op_collections takes %.2f seconds" % (time.time()-tt))

        #tt = time.time()
        # position to pin position
        self.op_collections.pin_pos_op = self.build_pin_pos(params, placedb, self.data_collections, self.device)
        # bound nodes to layout region 
        self.op_collections.move_boundary_op = self.build_move_boundary(params, placedb, self.data_collections, self.device)
        # hpwl and density overflow ops for evaluation 
        self.op_collections.hpwl_op = self.build_hpwl(params, placedb, self.data_collections, self.op_collections.pin_pos_op, self.device)
        # rectilinear minimum steiner tree wirelength from flute 
        # can only be called once 
        #self.op_collections.rmst_wl_op = self.build_rmst_wl(params, placedb, self.op_collections.pin_pos_op, torch.device("cpu"))
        #self.op_collections.density_overflow_op = self.build_density_overflow(params, placedb, self.data_collections, self.device)
        self.op_collections.density_overflow_op = self.build_electric_overflow(params, placedb, self.data_collections, self.device)
        # legalization 
        self.op_collections.greedy_legalize_op = self.build_greedy_legalization(params, placedb, self.data_collections, self.device)
        # draw placement 
        self.op_collections.draw_place_op = self.build_draw_placement(params, placedb)

        # flag for rmst_wl_op
        # can only read once 
        self.read_lut_flag = True

        #print("build BasicPlace ops takes %.2f seconds" % (time.time()-tt))

    def __call__(self, params, placedb):
        """
        @brief Solve placement.  
        placeholder for derived classes. 
        @param params parameters 
        @param placedb placement database 
        """
        pass 

    def build_pin_pos(self, params, placedb, data_collections, device):
        """
        @brief sum up the pins for each cell 
        @param params parameters 
        @param placedb placement database 
        @param data_collections a collection of all data and variables required for constructing the ops 
        @param device cpu or cuda 
        """
        def build_pin_pos_op(pos): 
            pin_x = data_collections.pin_offset_x.add(torch.index_select(pos[0:placedb.num_physical_nodes], dim=0, index=data_collections.pin2node_map.long()))
            pin_y = data_collections.pin_offset_y.add(torch.index_select(pos[placedb.num_nodes:placedb.num_nodes+placedb.num_physical_nodes], dim=0, index=data_collections.pin2node_map.long()))
            pin_pos = torch.cat([pin_x, pin_y], dim=0)

            return pin_pos 
        return build_pin_pos_op

    def build_move_boundary(self, params, placedb, data_collections, device):
        """
        @brief bound nodes into layout region 
        @param params parameters 
        @param placedb placement database 
        @param data_collections a collection of all data and variables required for constructing the ops 
        @param device cpu or cuda 
        """
        return move_boundary.MoveBoundary(
                data_collections.node_size_x, data_collections.node_size_y, 
                xl=placedb.xl, yl=placedb.yl, xh=placedb.xh, yh=placedb.yh, 
                num_movable_nodes=placedb.num_movable_nodes, 
                num_filler_nodes=placedb.num_filler_nodes, 
                num_threads=params.num_threads
                )

    def build_hpwl(self, params, placedb, data_collections, pin_pos_op, device):
        """
        @brief compute half-perimeter wirelength 
        @param params parameters 
        @param placedb placement database 
        @param data_collections a collection of all data and variables required for constructing the ops 
        @param pin_pos_op the op to compute pin locations according to cell locations 
        @param device cpu or cuda 
        """

        wirelength_for_pin_op = hpwl.HPWL(
                flat_netpin=data_collections.flat_net2pin_map, 
                netpin_start=data_collections.flat_net2pin_start_map,
                pin2net_map=data_collections.pin2net_map, 
                net_mask=data_collections.net_mask_all, 
                algorithm='atomic', 
                num_threads=params.num_threads
                )

        # wirelength for position 
        def build_wirelength_op(pos): 
            return wirelength_for_pin_op(pin_pos_op(pos))

        return build_wirelength_op

    def build_rmst_wl(self, params, placedb, pin_pos_op, device):
        """
        @brief compute rectilinear minimum spanning tree wirelength with flute 
        @param params parameters 
        @param placedb placement database 
        @param pin_pos_op the op to compute pin locations according to cell locations 
        @param device cpu or cuda 
        """
        # wirelength cost 

        POWVFILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../thirdparty/flute-3.1/POWV9.dat"))
        POSTFILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../thirdparty/flute-3.1/POST9.dat"))
        print("POWVFILE = %s" % (POWVFILE))
        print("POSTFILE = %s" % (POSTFILE))
        wirelength_for_pin_op = rmst_wl.RMSTWL(
                flat_netpin=torch.from_numpy(placedb.flat_net2pin_map).to(device), 
                netpin_start=torch.from_numpy(placedb.flat_net2pin_start_map).to(device),
                ignore_net_degree=params.ignore_net_degree, 
                POWVFILE=POWVFILE, 
                POSTFILE=POSTFILE
                )

        # wirelength for position 
        def build_wirelength_op(pos): 
            pin_pos = pin_pos_op(pos)
            wls = wirelength_for_pin_op(pin_pos.clone().cpu(), self.read_lut_flag)
            self.read_lut_flag = False
            return wls 

        return build_wirelength_op

    def build_density_overflow(self, params, placedb, data_collections, device):
        """
        @brief compute density overflow 
        @param params parameters 
        @param placedb placement database 
        @param data_collections a collection of all data and variables required for constructing the ops 
        @param device cpu or cuda 
        """
        return density_overflow.DensityOverflow(
                data_collections.node_size_x, data_collections.node_size_x, 
                data_collections.bin_center_x, data_collections.bin_center_y, 
                target_density=params.target_density, 
                xl=placedb.xl, yl=placedb.yl, xh=placedb.xh, yh=placedb.yh, 
                bin_size_x=placedb.bin_size_x, bin_size_y=placedb.bin_size_y, 
                num_movable_nodes=placedb.num_movable_nodes, 
                num_terminals=placedb.num_terminals, 
                num_filler_nodes=0,
                algorithm='by-node', 
                num_threads=params.num_threads
                )

    def build_electric_overflow(self, params, placedb, data_collections, device):
        """
        @brief compute electric density overflow 
        @param params parameters 
        @param placedb placement database 
        @param data_collections a collection of all data and variables required for constructing the ops 
        @param device cpu or cuda 
        """
        return electric_overflow.ElectricOverflow(
                data_collections.node_size_x, data_collections.node_size_y, 
                data_collections.bin_center_x, data_collections.bin_center_y, 
                target_density=params.target_density, 
                xl=placedb.xl, yl=placedb.yl, xh=placedb.xh, yh=placedb.yh, 
                bin_size_x=placedb.bin_size_x, bin_size_y=placedb.bin_size_y, 
                num_movable_nodes=placedb.num_movable_nodes, 
                num_terminals=placedb.num_terminals, 
                num_filler_nodes=0,
                padding=0, 
                num_threads=params.num_threads
                )

    def build_greedy_legalization(self, params, placedb, data_collections, device):
        """
        @brief greedy legalization 
        @param params parameters 
        @param placedb placement database 
        @param data_collections a collection of all data and variables required for constructing the ops 
        @param device cpu or cuda 
        """
        return greedy_legalize.GreedyLegalize(
                node_size_x=data_collections.node_size_x, node_size_y=data_collections.node_size_y, 
                xl=placedb.xl, yl=placedb.yl, xh=placedb.xh, yh=placedb.yh, 
                site_width=placedb.site_width, row_height=placedb.row_height, 
                num_bins_x=1, num_bins_y=64, 
                #num_bins_x=64, num_bins_y=64, 
                num_movable_nodes=placedb.num_movable_nodes, 
                num_filler_nodes=placedb.num_filler_nodes
                )

    def build_draw_placement(self, params, placedb):
        """
        @brief plot placement  
        @param params parameters 
        @param placedb placement database 
        """
        return draw_place.DrawPlace(placedb)

    def validate(self, placedb, pos, iteration):
        """
        @brief validate placement 
        @param placedb placement database 
        @param pos locations of cells 
        @param iteration optimization step 
        """
        pos = torch.from_numpy(pos).to(self.device)
        hpwl = self.op_collections.hpwl_op(pos)
        #rmst_wls = self.rmst_wl_op(pos)
        #rmst_wl = rmst_wls.sum()
        overflow, max_density = self.op_collections.density_overflow_op(pos)

        #return hpwl, rmst_wl, overflow, max_density
        return hpwl, overflow, max_density

    def plot(self, params, placedb, iteration, pos): 
        """
        @brief plot layout
        @param params parameters 
        @param placedb placement database 
        @param iteration optimization step 
        @param pos locations of cells 
        """
        tt = time.time()
        figname = "%s/plot/iter%s.png" % (params.result_dir, '{:04}'.format(iteration))
        os.system("mkdir -p %s" % (os.path.dirname(figname)))
        if isinstance(pos, np.ndarray):
            pos = torch.from_numpy(pos)
        self.op_collections.draw_place_op(pos, figname)
        print("[I] plotting to %s takes %.3f seconds" % (figname, time.time()-tt))
