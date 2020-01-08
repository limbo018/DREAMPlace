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
import logging
import torch 
import torch.nn as nn
import dreamplace.ops.move_boundary.move_boundary as move_boundary 
import dreamplace.ops.hpwl.hpwl as hpwl 
import dreamplace.ops.density_overflow.density_overflow as density_overflow 
import dreamplace.ops.electric_potential.electric_overflow as electric_overflow 
import dreamplace.ops.rmst_wl.rmst_wl as rmst_wl 
import dreamplace.ops.macro_legalize.macro_legalize as macro_legalize 
import dreamplace.ops.greedy_legalize.greedy_legalize as greedy_legalize 
import dreamplace.ops.abacus_legalize.abacus_legalize as abacus_legalize 
import dreamplace.ops.legality_check.legality_check as legality_check 
import dreamplace.ops.draw_place.draw_place as draw_place 
import dreamplace.ops.pin_pos.pin_pos as pin_pos
import dreamplace.ops.global_swap.global_swap as global_swap 
import dreamplace.ops.k_reorder.k_reorder as k_reorder
import dreamplace.ops.independent_set_matching.independent_set_matching as independent_set_matching
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
        self.device = device
        torch.set_num_threads(params.num_threads)
        # position should be parameter 
        self.pos = pos 
        
        with torch.no_grad(): 
            # other tensors required to build ops 
            
            self.node_size_x = torch.from_numpy(placedb.node_size_x).to(device)
            self.node_size_y = torch.from_numpy(placedb.node_size_y).to(device)
            # original node size for legalization, since they will be adjusted in global placement
            if params.routability_opt_flag: 
                self.original_node_size_x = self.node_size_x.clone()
                self.original_node_size_y = self.node_size_y.clone()

            self.pin_offset_x = torch.tensor(placedb.pin_offset_x, dtype=self.pos[0].dtype, device=device)
            self.pin_offset_y = torch.tensor(placedb.pin_offset_y, dtype=self.pos[0].dtype, device=device)
            # original pin offset for legalization, since they will be adjusted in global placement
            if params.routability_opt_flag: 
                self.original_pin_offset_x = self.pin_offset_x.clone()
                self.original_pin_offset_y = self.pin_offset_y.clone()

            self.target_density = torch.empty(1, dtype=self.pos[0].dtype, device=device)
            self.target_density.data.fill_(params.target_density)

            # detect movable macros and scale down the density to avoid halos 
            # I use a heuristic that cells whose areas are 10x of the mean area will be regarded movable macros in global placement 
            node_areas = self.node_size_x * self.node_size_y
            if self.target_density < 1: 
                mean_area = node_areas[:placedb.num_movable_nodes].mean().mul_(10)
                row_height = self.node_size_y[:placedb.num_movable_nodes].min().mul_(2)
                self.movable_macro_mask = (node_areas[:placedb.num_movable_nodes] > mean_area) & (self.node_size_y[:placedb.num_movable_nodes] > row_height)
            else: # no movable macros 
                self.movable_macro_mask = None

            self.pin2node_map = torch.from_numpy(placedb.pin2node_map).to(device)
            self.flat_node2pin_map = torch.from_numpy(placedb.flat_node2pin_map).to(device)
            self.flat_node2pin_start_map = torch.from_numpy(placedb.flat_node2pin_start_map).to(device)
            # number of pins for each cell  
            self.pin_weights = (self.flat_node2pin_start_map[1:] - self.flat_node2pin_start_map[:-1]).to(self.node_size_x.dtype)

            self.unit_pin_capacity = torch.empty(1, dtype=self.pos[0].dtype, device=device)
            self.unit_pin_capacity.data.fill_(params.unit_pin_capacity)
            if params.routability_opt_flag:
                unit_pin_capacity = self.pin_weights[:placedb.num_movable_nodes] / node_areas[:placedb.num_movable_nodes]
                avg_pin_capacity = unit_pin_capacity.mean() * self.target_density
                # min(computed, params.unit_pin_capacity)
                self.unit_pin_capacity = avg_pin_capacity.clamp_(max=params.unit_pin_capacity)
                logging.info("unit_pin_capacity = %g" % (self.unit_pin_capacity))

            # routing information 
            # project initial routing utilization map to one layer 
            self.initial_horizontal_utilization_map = None 
            self.initial_vertical_utilization_map = None
            if params.routability_opt_flag and placedb.initial_horizontal_demand_map is not None: 
                self.initial_horizontal_utilization_map = torch.from_numpy(placedb.initial_horizontal_demand_map).to(device).div_(placedb.routing_grid_size_y * placedb.unit_horizontal_capacity)
                self.initial_vertical_utilization_map = torch.from_numpy(placedb.initial_vertical_demand_map).to(device).div_(placedb.routing_grid_size_x * placedb.unit_vertical_capacity)

            self.pin2net_map = torch.from_numpy(placedb.pin2net_map).to(device)
            self.flat_net2pin_map = torch.from_numpy(placedb.flat_net2pin_map).to(device)
            self.flat_net2pin_start_map = torch.from_numpy(placedb.flat_net2pin_start_map).to(device)
            if np.amin(placedb.net_weights) != np.amax(placedb.net_weights): # weights are meaningful 
                self.net_weights = torch.from_numpy(placedb.net_weights).to(device)
            else: # an empty tensor 
                logging.warning("net weights are all the same, ignored")
                self.net_weights = torch.Tensor().to(device)

            # regions 
            self.flat_region_boxes = torch.from_numpy(placedb.flat_region_boxes).to(device)
            self.flat_region_boxes_start = torch.from_numpy(placedb.flat_region_boxes_start).to(device)
            self.node2fence_region_map = torch.from_numpy(placedb.node2fence_region_map).to(device)
            
            self.net_mask_all = torch.from_numpy(np.ones(placedb.num_nets, dtype=np.uint8)).to(device) # all nets included 
            net_degrees = np.array([len(net2pin) for net2pin in placedb.net2pin_map])
            net_mask = np.logical_and(2 <= net_degrees, net_degrees < params.ignore_net_degree).astype(np.uint8)
            self.net_mask_ignore_large_degrees = torch.from_numpy(net_mask).to(device) # nets with large degrees are ignored 

            # avoid computing gradient for fixed macros 
            # 1 is for fixed macros 
            self.pin_mask_ignore_fixed_macros = (self.pin2node_map >= placedb.num_movable_nodes)

            self.bin_center_x = torch.from_numpy(placedb.bin_center_x).to(device)
            self.bin_center_y = torch.from_numpy(placedb.bin_center_y).to(device)

            # sort nodes by size, return their sorted indices, designed for memory coalesce in electrical force
            movable_size_x = self.node_size_x[:placedb.num_movable_nodes]
            _, self.sorted_node_map = torch.sort(movable_size_x)
            self.sorted_node_map = self.sorted_node_map.to(torch.int32) 
            # self.sorted_node_map = torch.arange(0, placedb.num_movable_nodes, dtype=torch.int32, device=device)

            # logging.debug(self.node_size_x[placedb.num_movable_nodes//2 :placedb.num_movable_nodes//2+20])
            # logging.debug(self.sorted_node_map[placedb.num_movable_nodes//2 :placedb.num_movable_nodes//2+20])
            # logging.debug(self.node_size_x[self.sorted_node_map[0: 10].long()])
            # logging.debug(self.node_size_x[self.sorted_node_map[-10:].long()])

        
    def bin_center_x_padded(self, placedb, padding): 
        """
        @brief compute array of bin center horizontal coordinates with padding 
        @param placedb placement database 
        @param padding number of bins padding to boundary of placement region 
        """
        if padding == 0: 
            return self.bin_center_x 
        else:
            xl = placedb.xl - padding * placedb.bin_size_x
            xh = placedb.xh + padding * placedb.bin_size_x
            self.bin_center_x_padded = torch.from_numpy(placedb.bin_centers(xl, xh, placedb.bin_size_x)).to(self.device)
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
            yl = placedb.yl - padding * placedb.bin_size_y
            yh = placedb.yh + padding * placedb.bin_size_y
            self.bin_center_y_padded = torch.from_numpy(placedb.bin_centers(yl, yh, placedb.bin_size_y)).to(self.device)
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
        self.legality_check_op = None 
        self.legalize_op = None 
        self.detailed_place_op = None
        self.wirelength_op = None 
        self.update_gamma_op = None 
        self.density_op = None 
        self.update_density_weight_op = None
        self.precondition_op = None 
        self.noise_op = None 
        self.draw_place_op = None
        self.route_utilization_map_op = None
        self.pin_utilization_map_op = None 
        self.nctugr_congestion_map_op = None 
        self.adjust_node_area_op = None

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

        tt = time.time()
        self.init_pos = np.zeros(placedb.num_nodes*2, dtype=placedb.dtype)
        # x position 
        self.init_pos[0:placedb.num_physical_nodes] = placedb.node_x
        if params.global_place_flag and params.random_center_init_flag: # move to center of layout 
            logging.info("move cells to the center of layout with random noise")
            self.init_pos[0:placedb.num_movable_nodes] = np.random.normal(loc=(placedb.xl*1.0+placedb.xh*1.0)/2, scale=(placedb.xh-placedb.xl)*0.001, size=placedb.num_movable_nodes)
        #self.init_pos[0:placedb.num_movable_nodes] = init_x[0:placedb.num_movable_nodes]*0.01 + (placedb.xl+placedb.xh)/2
        # y position 
        self.init_pos[placedb.num_nodes:placedb.num_nodes+placedb.num_physical_nodes] = placedb.node_y
        if params.global_place_flag and params.random_center_init_flag: # move to center of layout 
            self.init_pos[placedb.num_nodes:placedb.num_nodes+placedb.num_movable_nodes] = np.random.normal(loc=(placedb.yl*1.0+placedb.yh*1.0)/2, scale=(placedb.yh-placedb.yl)*0.001, size=placedb.num_movable_nodes)
        #init_y[0:placedb.num_movable_nodes] = init_y[0:placedb.num_movable_nodes]*0.01 + (placedb.yl+placedb.yh)/2

        if placedb.num_filler_nodes: # uniformly distribute filler cells in the layout 
            self.init_pos[placedb.num_physical_nodes:placedb.num_nodes] = np.random.uniform(low=placedb.xl, high=placedb.xh-placedb.node_size_x[-placedb.num_filler_nodes], size=placedb.num_filler_nodes)
            self.init_pos[placedb.num_nodes+placedb.num_physical_nodes:placedb.num_nodes*2] = np.random.uniform(low=placedb.yl, high=placedb.yh-placedb.node_size_y[-placedb.num_filler_nodes], size=placedb.num_filler_nodes)

        logging.debug("prepare init_pos takes %.2f seconds" % (time.time()-tt))

        self.device = torch.device("cuda" if params.gpu else "cpu")

        # position should be parameter 
        # must be defined in BasicPlace 
        tt = time.time()
        self.pos = nn.ParameterList([nn.Parameter(torch.from_numpy(self.init_pos).to(self.device))])
        logging.debug("build pos takes %.2f seconds" % (time.time()-tt))
        # shared data on device for building ops  
        # I do not want to construct the data from placedb again and again for each op 
        tt = time.time()
        self.data_collections = PlaceDataCollection(self.pos, params, placedb, self.device)
        logging.debug("build data_collections takes %.2f seconds" % (time.time()-tt))
        # similarly I wrap all ops 
        tt = time.time()
        self.op_collections = PlaceOpCollection()
        logging.debug("build op_collections takes %.2f seconds" % (time.time()-tt))

        tt = time.time()
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
        # legality check 
        self.op_collections.legality_check_op = self.build_legality_check(params, placedb, self.data_collections, self.device)
        # legalization 
        self.op_collections.legalize_op = self.build_legalization(params, placedb, self.data_collections, self.device)
        # detailed placement 
        self.op_collections.detailed_place_op = self.build_detailed_placement(params, placedb, self.data_collections, self.device)
        # draw placement 
        self.op_collections.draw_place_op = self.build_draw_placement(params, placedb)

        # flag for rmst_wl_op
        # can only read once 
        self.read_lut_flag = True

        logging.debug("build BasicPlace ops takes %.2f seconds" % (time.time()-tt))

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
        # Yibo: I found CPU version of this is super slow, more than 2s for ISPD2005 bigblue4 with 10 threads. 
        # So I implemented a custom CPU version, which is around 20ms
        #pin2node_map = data_collections.pin2node_map.long()
        #def build_pin_pos_op(pos): 
        #    pin_x = data_collections.pin_offset_x.add(torch.index_select(pos[0:placedb.num_physical_nodes], dim=0, index=pin2node_map))
        #    pin_y = data_collections.pin_offset_y.add(torch.index_select(pos[placedb.num_nodes:placedb.num_nodes+placedb.num_physical_nodes], dim=0, index=pin2node_map))
        #    pin_pos = torch.cat([pin_x, pin_y], dim=0)

        #    return pin_pos 
        #return build_pin_pos_op

        return pin_pos.PinPos(
                pin_offset_x=data_collections.pin_offset_x, 
                pin_offset_y=data_collections.pin_offset_y, 
                pin2node_map=data_collections.pin2node_map, 
                flat_node2pin_map=data_collections.flat_node2pin_map, 
                flat_node2pin_start_map=data_collections.flat_node2pin_start_map, 
                num_physical_nodes=placedb.num_physical_nodes, 
                num_threads=params.num_threads
                )

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
                net_weights=data_collections.net_weights, 
                net_mask=data_collections.net_mask_all, 
                algorithm='net-by-net', 
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

        POWVFILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../thirdparty/NCTUgr.ICCAD2012/POWV9.dat"))
        POSTFILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../thirdparty/NCTUgr.ICCAD2012/POST9.dat"))
        logging.info("POWVFILE = %s" % (POWVFILE))
        logging.info("POSTFILE = %s" % (POSTFILE))
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
                target_density=data_collections.target_density, 
                xl=placedb.xl, yl=placedb.yl, xh=placedb.xh, yh=placedb.yh, 
                bin_size_x=placedb.bin_size_x, bin_size_y=placedb.bin_size_y, 
                num_movable_nodes=placedb.num_movable_nodes, 
                num_terminals=placedb.num_terminals, 
                num_filler_nodes=0,
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
                node_size_x=data_collections.node_size_x, node_size_y=data_collections.node_size_y, 
                bin_center_x=data_collections.bin_center_x, bin_center_y=data_collections.bin_center_y, 
                target_density=data_collections.target_density, 
                xl=placedb.xl, yl=placedb.yl, xh=placedb.xh, yh=placedb.yh, 
                bin_size_x=placedb.bin_size_x, bin_size_y=placedb.bin_size_y, 
                num_movable_nodes=placedb.num_movable_nodes, 
                num_terminals=placedb.num_terminals, 
                num_filler_nodes=0,
                padding=0, 
                sorted_node_map=data_collections.sorted_node_map,
                movable_macro_mask=data_collections.movable_macro_mask, 
                num_threads=params.num_threads
                )

    def build_legality_check(self, params, placedb, data_collections, device):
        """
        @brief legality check 
        @param params parameters 
        @param placedb placement database 
        @param data_collections a collection of all data and variables required for constructing the ops 
        @param device cpu or cuda 
        """
        return legality_check.LegalityCheck(
                node_size_x=data_collections.node_size_x, node_size_y=data_collections.node_size_y, 
                flat_region_boxes=data_collections.flat_region_boxes, flat_region_boxes_start=data_collections.flat_region_boxes_start, node2fence_region_map=data_collections.node2fence_region_map, 
                xl=placedb.xl, yl=placedb.yl, xh=placedb.xh, yh=placedb.yh, 
                site_width=placedb.site_width, row_height=placedb.row_height, 
                num_terminals=placedb.num_terminals, 
                num_movable_nodes=placedb.num_movable_nodes
                )

    def build_legalization(self, params, placedb, data_collections, device):
        """
        @brief legalization 
        @param params parameters 
        @param placedb placement database 
        @param data_collections a collection of all data and variables required for constructing the ops 
        @param device cpu or cuda 
        """
        # for movable macro legalization 
        # the number of bins control the search granularity 
        ml = macro_legalize.MacroLegalize(
                node_size_x=data_collections.node_size_x, node_size_y=data_collections.node_size_y, 
                flat_region_boxes=data_collections.flat_region_boxes, flat_region_boxes_start=data_collections.flat_region_boxes_start, node2fence_region_map=data_collections.node2fence_region_map, 
                xl=placedb.xl, yl=placedb.yl, xh=placedb.xh, yh=placedb.yh, 
                site_width=placedb.site_width, row_height=placedb.row_height, 
                num_bins_x=params.num_bins_x, num_bins_y=params.num_bins_y, 
                num_movable_nodes=placedb.num_movable_nodes, 
                num_terminal_NIs=placedb.num_terminal_NIs, 
                num_filler_nodes=placedb.num_filler_nodes
                )
        # for standard cell legalization
        gl = greedy_legalize.GreedyLegalize(
                node_size_x=data_collections.node_size_x, node_size_y=data_collections.node_size_y, 
                flat_region_boxes=data_collections.flat_region_boxes, flat_region_boxes_start=data_collections.flat_region_boxes_start, node2fence_region_map=data_collections.node2fence_region_map, 
                xl=placedb.xl, yl=placedb.yl, xh=placedb.xh, yh=placedb.yh, 
                site_width=placedb.site_width, row_height=placedb.row_height, 
                num_bins_x=1, num_bins_y=64, 
                #num_bins_x=64, num_bins_y=64, 
                num_movable_nodes=placedb.num_movable_nodes, 
                num_terminal_NIs=placedb.num_terminal_NIs, 
                num_filler_nodes=placedb.num_filler_nodes
                )
        # for standard cell legalization
        al = abacus_legalize.AbacusLegalize(
                node_size_x=data_collections.node_size_x, node_size_y=data_collections.node_size_y, 
                flat_region_boxes=data_collections.flat_region_boxes, flat_region_boxes_start=data_collections.flat_region_boxes_start, node2fence_region_map=data_collections.node2fence_region_map, 
                xl=placedb.xl, yl=placedb.yl, xh=placedb.xh, yh=placedb.yh, 
                site_width=placedb.site_width, row_height=placedb.row_height, 
                num_bins_x=1, num_bins_y=64, 
                #num_bins_x=64, num_bins_y=64, 
                num_movable_nodes=placedb.num_movable_nodes, 
                num_terminal_NIs=placedb.num_terminal_NIs, 
                num_filler_nodes=placedb.num_filler_nodes
                )
        def build_legalization_op(pos): 
            logging.info("Start legalization")
            pos1 = ml(pos, pos)
            pos2 = gl(pos1, pos1)
            legal = self.op_collections.legality_check_op(pos2)
            if not legal:
                logging.error("legality check failed in greedy legalization")
                return pos2 
            return al(pos1, pos2)
        return build_legalization_op

    def build_detailed_placement(self, params, placedb, data_collections, device):
        """
        @brief detailed placement consisting of global swap and independent set matching 
        @param params parameters 
        @param placedb placement database 
        @param data_collections a collection of all data and variables required for constructing the ops 
        @param device cpu or cuda 
        """
        gs = global_swap.GlobalSwap(
                node_size_x=data_collections.node_size_x, node_size_y=data_collections.node_size_y, 
                flat_region_boxes=data_collections.flat_region_boxes, flat_region_boxes_start=data_collections.flat_region_boxes_start, node2fence_region_map=data_collections.node2fence_region_map, 
                flat_net2pin_map=data_collections.flat_net2pin_map, flat_net2pin_start_map=data_collections.flat_net2pin_start_map, pin2net_map=data_collections.pin2net_map, 
                flat_node2pin_map=data_collections.flat_node2pin_map, flat_node2pin_start_map=data_collections.flat_node2pin_start_map, pin2node_map=data_collections.pin2node_map, 
                pin_offset_x=data_collections.pin_offset_x, pin_offset_y=data_collections.pin_offset_y, 
                net_mask=data_collections.net_mask_ignore_large_degrees, 
                xl=placedb.xl, yl=placedb.yl, xh=placedb.xh, yh=placedb.yh, 
                site_width=placedb.site_width, row_height=placedb.row_height, 
                #num_bins_x=placedb.num_bins_x//16, num_bins_y=placedb.num_bins_y//16, 
                num_bins_x=placedb.num_bins_x//2, num_bins_y=placedb.num_bins_y//2, 
                num_movable_nodes=placedb.num_movable_nodes, 
                num_terminal_NIs=placedb.num_terminal_NIs, 
                num_filler_nodes=placedb.num_filler_nodes, 
                batch_size=256, 
                max_iters=2, 
                algorithm='concurrent', 
                num_threads=params.num_threads
                )
        kr = k_reorder.KReorder(
                node_size_x=data_collections.node_size_x, node_size_y=data_collections.node_size_y, 
                flat_region_boxes=data_collections.flat_region_boxes, flat_region_boxes_start=data_collections.flat_region_boxes_start, node2fence_region_map=data_collections.node2fence_region_map, 
                flat_net2pin_map=data_collections.flat_net2pin_map, flat_net2pin_start_map=data_collections.flat_net2pin_start_map, pin2net_map=data_collections.pin2net_map, 
                flat_node2pin_map=data_collections.flat_node2pin_map, flat_node2pin_start_map=data_collections.flat_node2pin_start_map, pin2node_map=data_collections.pin2node_map, 
                pin_offset_x=data_collections.pin_offset_x, pin_offset_y=data_collections.pin_offset_y, 
                net_mask=data_collections.net_mask_ignore_large_degrees, 
                xl=placedb.xl, yl=placedb.yl, xh=placedb.xh, yh=placedb.yh, 
                site_width=placedb.site_width, row_height=placedb.row_height, 
                num_bins_x=placedb.num_bins_x, num_bins_y=placedb.num_bins_y, 
                num_movable_nodes=placedb.num_movable_nodes, 
                num_terminal_NIs=placedb.num_terminal_NIs, 
                num_filler_nodes=placedb.num_filler_nodes, 
                K=4, 
                max_iters=2, 
                num_threads=params.num_threads
                )
        ism = independent_set_matching.IndependentSetMatching(
                node_size_x=data_collections.node_size_x, node_size_y=data_collections.node_size_y, 
                flat_region_boxes=data_collections.flat_region_boxes, flat_region_boxes_start=data_collections.flat_region_boxes_start, node2fence_region_map=data_collections.node2fence_region_map, 
                flat_net2pin_map=data_collections.flat_net2pin_map, flat_net2pin_start_map=data_collections.flat_net2pin_start_map, pin2net_map=data_collections.pin2net_map, 
                flat_node2pin_map=data_collections.flat_node2pin_map, flat_node2pin_start_map=data_collections.flat_node2pin_start_map, pin2node_map=data_collections.pin2node_map, 
                pin_offset_x=data_collections.pin_offset_x, pin_offset_y=data_collections.pin_offset_y, 
                net_mask=data_collections.net_mask_ignore_large_degrees, 
                xl=placedb.xl, yl=placedb.yl, xh=placedb.xh, yh=placedb.yh, 
                site_width=placedb.site_width, row_height=placedb.row_height, 
                num_bins_x=placedb.num_bins_x, num_bins_y=placedb.num_bins_y, 
                num_movable_nodes=placedb.num_movable_nodes, 
                num_terminal_NIs=placedb.num_terminal_NIs, 
                num_filler_nodes=placedb.num_filler_nodes, 
                batch_size=2048, 
                set_size=128, 
                max_iters=50, 
                algorithm='concurrent', 
                num_threads=params.num_threads
                )

        # wirelength for position 
        def build_detailed_placement_op(pos): 
            logging.info("Start ABCDPlace for refinement")
            pos1 = pos 
            for i in range(1): 
                pos1 = kr(pos1)
                legal = self.op_collections.legality_check_op(pos1)
                logging.info("K-Reorder legal flag = %d" % (legal))
                if not legal:
                    return pos1 
                pos1 = ism(pos1)
                legal = self.op_collections.legality_check_op(pos1)
                logging.info("Independent set matching legal flag = %d" % (legal))
                if not legal:
                    return pos1 
                pos1 = gs(pos1)
                legal = self.op_collections.legality_check_op(pos1)
                logging.info("Global swap legal flag = %d" % (legal))
                if not legal:
                    return pos1 
                pos1 = kr(pos1)
                legal = self.op_collections.legality_check_op(pos1)
                logging.info("K-Reorder legal flag = %d" % (legal))
                if not legal:
                    return pos1 
            return pos1 
        return build_detailed_placement_op

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
        path = "%s/%s" % (params.result_dir, params.design_name())
        figname = "%s/plot/iter%s.png" % (path, '{:04}'.format(iteration))
        os.system("mkdir -p %s" % (os.path.dirname(figname)))
        if isinstance(pos, np.ndarray):
            pos = torch.from_numpy(pos)
        self.op_collections.draw_place_op(pos, figname)
        logging.info("plotting to %s takes %.3f seconds" % (figname, time.time()-tt))

    def dump(self, params, placedb, pos, filename):
        """
        @brief dump intermediate solution as compressed pickle file (.pklz)
        @param params parameters 
        @param placedb placement database 
        @param iteration optimization step 
        @param pos locations of cells 
        @param filename output file name 
        """
        with gzip.open(filename, "wb") as f:
            pickle.dump((self.data_collections.node_size_x.cpu(), self.data_collections.node_size_y.cpu(), 
                self.data_collections.flat_net2pin_map.cpu(), self.data_collections.flat_net2pin_start_map.cpu(), self.data_collections.pin2net_map.cpu(), 
                self.data_collections.flat_node2pin_map.cpu(), self.data_collections.flat_node2pin_start_map.cpu(), self.data_collections.pin2node_map.cpu(), 
                self.data_collections.pin_offset_x.cpu(), self.data_collections.pin_offset_y.cpu(), 
                self.data_collections.net_mask_ignore_large_degrees.cpu(), 
                placedb.xl, placedb.yl, placedb.xh, placedb.yh, 
                placedb.site_width, placedb.row_height, 
                placedb.num_bins_x, placedb.num_bins_y, 
                placedb.num_movable_nodes, 
                placedb.num_terminal_NIs, 
                placedb.num_filler_nodes, 
                pos 
                ), f)
