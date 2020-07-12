##
# @file   independent_set_matching_unitest.py
# @author Yibo Lin
# @date   Mar 2019
#

import os
import sys
import math
import numpy as np
import unittest
import cairocffi as cairo
import time
import math

import torch
from torch.autograd import Function, Variable
from scipy.optimize import linear_sum_assignment
import gzip
if sys.version_info[0] < 3:
    import cPickle as pickle
else:
    import _pickle as pickle

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from dreamplace.ops.independent_set_matching import independent_set_matching as independent_set_matching
sys.path.pop()

import pdb


def plot(figname, node_x, node_y, node_size_x, node_size_y, layout_xl,
         layout_yl, layout_xh, layout_yh, num_bins_x, num_bins_y, num_nodes,
         num_movable_nodes, num_physical_nodes, num_filler_nodes):
    tt = time.time()
    width = 800
    height = 800
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    ctx = cairo.Context(surface)

    if num_movable_nodes < num_physical_nodes:
        layout_xl2 = min(np.amin(node_x[num_movable_nodes:num_physical_nodes]),
                         layout_xl)
        layout_yl2 = min(np.amin(node_y[num_movable_nodes:num_physical_nodes]),
                         layout_yl)
        layout_xh2 = max(
            np.amax(node_x[num_movable_nodes:num_physical_nodes] +
                    node_size_x[num_movable_nodes:num_physical_nodes]),
            layout_xh)
        layout_yh2 = max(
            np.amax(node_y[num_movable_nodes:num_physical_nodes] +
                    node_size_y[num_movable_nodes:num_physical_nodes]),
            layout_yh)
    else:
        layout_xl2 = layout_xl
        layout_yl2 = layout_yl
        layout_xh2 = layout_xh
        layout_yh2 = layout_yh

    bin_size_x = (layout_xh - layout_xl) / num_bins_x
    bin_size_y = (layout_yh - layout_yl) / num_bins_y

    def normalize_x(xx):
        return (xx - (layout_xl - bin_size_x)) / (layout_xh - layout_xl +
                                                  2 * bin_size_x) * width

    def normalize_y(xx):
        return (xx - (layout_yl - bin_size_y)) / (layout_yh - layout_yl +
                                                  2 * bin_size_y) * height

    def draw_rect(x1, y1, x2, y2, text=None):
        ctx.move_to(x1, y1)
        ctx.line_to(x1, y2)
        ctx.line_to(x2, y2)
        ctx.line_to(x2, y1)
        ctx.close_path()
        ctx.stroke()
        if text:
            empty_pixels = surface.get_data()[:]
            ctx.move_to(float((x1 + x2) / 2), float((y1 + y2) / 2))
            #ctx.set_source_rgb(0, 0, 0)
            ctx.show_text(text)
            text_pixels = surface.get_data()[:]
            assert empty_pixels != text_pixels

    def bin_xl(i):
        return layout_xl + i * bin_size_x

    def bin_yl(i):
        return layout_yl + i * bin_size_y

    # draw layout region
    ctx.set_source_rgb(1, 1, 1)
    draw_layout_xl = normalize_x(layout_xl2 - 1 * bin_size_x)
    draw_layout_yl = normalize_y(layout_yl2 - 1 * bin_size_y)
    draw_layout_xh = normalize_x(layout_xh2 + 1 * bin_size_x)
    draw_layout_yh = normalize_y(layout_yh2 + 1 * bin_size_y)
    ctx.rectangle(draw_layout_xl, draw_layout_yl, draw_layout_xh,
                  draw_layout_yh)
    ctx.fill()
    ctx.set_line_width(1)
    ctx.set_source_rgba(0.1, 0.1, 0.1, alpha=0.8)
    draw_rect(normalize_x(layout_xl), normalize_y(layout_yl),
              normalize_x(layout_xh), normalize_y(layout_yh))
    #ctx.move_to(normalize_x(xl), normalize_y(yl))
    #ctx.line_to(normalize_x(xl), normalize_y(yh))
    #ctx.line_to(normalize_x(xh), normalize_y(yh))
    #ctx.line_to(normalize_x(xh), normalize_y(yl))
    #ctx.close_path()
    #ctx.stroke()
    # draw bins
    for i in range(1, num_bins_x):
        ctx.move_to(normalize_x(bin_xl(i)), normalize_y(layout_yl))
        ctx.line_to(normalize_x(bin_xl(i)), normalize_y(layout_yh))
        ctx.close_path()
        ctx.stroke()
    for i in range(1, num_bins_y):
        ctx.move_to(normalize_x(layout_xl), normalize_y(bin_yl(i)))
        ctx.line_to(normalize_x(layout_xh), normalize_y(bin_yl(i)))
        ctx.close_path()
        ctx.stroke()

    # draw cells
    node_xl = node_x
    node_yl = layout_yl + layout_yh - (node_y + node_size_y[0:len(node_y)]
                                       )  # flip y
    node_xh = node_x + node_size_x[0:len(node_x)]
    node_yh = layout_yl + layout_yh - node_y  # flip y
    node_xl = normalize_x(node_xl)
    node_yl = normalize_y(node_yl)
    node_xh = normalize_x(node_xh)
    node_yh = normalize_y(node_yh)
    ctx.set_line_width(1)
    #print("plot layout")
    # draw fixed macros
    ctx.set_source_rgba(1, 0, 0, alpha=0.5)
    for i in range(num_movable_nodes, num_physical_nodes):
        ctx.rectangle(node_xl[i], node_yl[i], node_xh[i] - node_xl[i],
                      node_yh[i] - node_yl[i])  # Rectangle(xl, yl, w, h)
        ctx.fill()
    ctx.set_source_rgba(0, 0, 0, alpha=1.0)  # Solid color
    for i in range(num_movable_nodes, num_physical_nodes):
        draw_rect(node_xl[i], node_yl[i], node_xh[i], node_yh[i])
    # draw fillers
    if len(node_xl) > num_physical_nodes:  # filler is included
        ctx.set_line_width(1)
        ctx.set_source_rgba(230 / 255.0, 230 / 255.0, 250 / 255.0,
                            alpha=0.3)  # Solid color
        for i in range(num_physical_nodes, num_nodes):
            draw_rect(node_xl[i], node_yl[i], node_xh[i], node_yh[i])
    # draw cells
    ctx.set_line_width(2)
    ctx.set_source_rgba(0, 0, 1, alpha=0.8)  # Solid color
    #ctx.select_font_face("Purisa", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
    ctx.set_font_size(32)
    for i in range(num_movable_nodes):
        draw_rect(node_xl[i], node_yl[i], node_xh[i], node_yh[i], "%d" % i)

    #ctx.scale(width, height)  # Normalizing the canvas, this is not compatible with show_text
    surface.write_to_png(figname)  # Output to PNG
    print("[I] plotting to %s takes %.3f seconds" %
          (figname, time.time() - tt))
    #print(session.run(grads))
    #print(session.run(masked_grads))


def flatten_2D_map(net2pin_map):
    num_pins = 0
    for pins in net2pin_map:
        num_pins += len(pins)
    # pin2net_map
    pin2net_map = np.zeros(num_pins, dtype=np.int32)
    for net_id, pins in enumerate(net2pin_map):
        for pin in pins:
            pin2net_map[pin] = net_id
    # construct flat_net2pin_map and flat_net2pin_start_map
    # flat netpin map, length of #pins
    flat_net2pin_map = np.zeros(num_pins, dtype=np.int32)
    # starting index in netpin map for each net, length of #nets+1, the last entry is #pins
    flat_net2pin_start_map = np.zeros(len(net2pin_map) + 1, dtype=np.int32)
    count = 0
    for i in range(len(net2pin_map)):
        flat_net2pin_map[count:count + len(net2pin_map[i])] = net2pin_map[i]
        flat_net2pin_start_map[i] = count
        count += len(net2pin_map[i])
    flat_net2pin_start_map[len(net2pin_map)] = num_pins

    return pin2net_map, flat_net2pin_map, flat_net2pin_start_map


def test_ispd2005(design, algorithm, device_str):
    with gzip.open(design, "rb") as f:
        if sys.version_info[0] < 3:
            data_collections = pickle.load(f)
        else:
            data_collections = pickle.load(f, encoding='bytes')
        node_size_x = data_collections[0]
        node_size_y = data_collections[1]
        flat_net2pin_map = data_collections[2]
        flat_net2pin_start_map = data_collections[3]
        pin2net_map = data_collections[4]
        flat_node2pin_map = data_collections[5]
        flat_node2pin_start_map = data_collections[6]
        pin2node_map = data_collections[7]
        pin_offset_x = data_collections[8]
        pin_offset_y = data_collections[9]
        net_mask_ignore_large_degrees = data_collections[10]
        xl = data_collections[11]
        yl = data_collections[12]
        xh = data_collections[13]
        yh = data_collections[14]
        site_width = data_collections[15]
        row_height = data_collections[16]
        num_bins_x = data_collections[17]
        num_bins_y = data_collections[18]
        num_movable_nodes = data_collections[19]
        num_terminal_NIs = data_collections[20]
        num_filler_nodes = data_collections[21]
        pos = data_collections[22]

        #net_mask = net_mask_ignore_large_degrees
        net_mask = np.ones_like(net_mask_ignore_large_degrees)
        for i in range(1, len(flat_net2pin_start_map)):
            degree = flat_net2pin_start_map[i] - flat_net2pin_start_map[i - 1]
            if degree > 100:
                net_mask[i - 1] = 0
        net_mask = torch.from_numpy(net_mask)

        #max_node_degree = 0
        #for i in range(1, len(flat_node2pin_start_map)):
        #    if i <= num_movable_nodes:
        #        max_node_degree = max(max_node_degree, flat_node2pin_start_map[i]-flat_node2pin_start_map[i-1])
        #print("max node degree %d" % (max_node_degree))

        device = torch.device(device_str)

        print("bins %dx%d" % (num_bins_x, num_bins_y))
        print("num_movable_nodes %d, num_nodes %d" %
              (num_movable_nodes,
               node_size_x.numel() - num_filler_nodes - num_terminal_NIs))

        pos = pos.float().to(device)

        torch.set_num_threads(20)
        custom = independent_set_matching.IndependentSetMatching(
            node_size_x=node_size_x.float().to(device),
            node_size_y=node_size_y.float().to(device),
            flat_net2pin_map=flat_net2pin_map.to(device),
            flat_net2pin_start_map=flat_net2pin_start_map.to(device),
            pin2net_map=pin2net_map.to(device),
            flat_node2pin_map=flat_node2pin_map.to(device),
            flat_node2pin_start_map=flat_node2pin_start_map.to(device),
            pin2node_map=pin2node_map.to(device),
            pin_offset_x=pin_offset_x.float().to(device),
            pin_offset_y=pin_offset_y.float().to(device),
            net_mask=net_mask.to(device),
            xl=xl,
            yl=yl,
            xh=xh,
            yh=yh,
            site_width=site_width,
            row_height=row_height,
            num_bins_x=num_bins_x // 1,
            num_bins_y=num_bins_y // 1,
            num_movable_nodes=num_movable_nodes,
            num_terminal_NIs=num_terminal_NIs,
            num_filler_nodes=num_filler_nodes,
            batch_size=2048,
            set_size=128,
            max_iters=50,
            algorithm=algorithm)

        result = custom(pos)

        #num_bins_x = 512
        #num_bins_y = 512
        #with gzip.open("adaptec1.dp.ism.pklz", "wb") as f:
        #    pickle.dump((node_size_x.cpu(), node_size_y.cpu(),
        #        flat_net2pin_map.cpu(), flat_net2pin_start_map.cpu(), pin2net_map.cpu(),
        #        flat_node2pin_map.cpu(), flat_node2pin_start_map.cpu(), pin2node_map.cpu(),
        #        pin_offset_x.cpu(), pin_offset_y.cpu(),
        #        net_mask_ignore_large_degrees.cpu(),
        #        xl, yl, xh, yh,
        #        site_width, row_height,
        #        num_bins_x, num_bins_y,
        #        num_movable_nodes,
        #        num_terminal_NIs,
        #        num_filler_nodes,
        #        result.cpu()
        #        ), f)
        #    exit()


if __name__ == '__main__':
    #unittest.main()
    if len(sys.argv) < 4:
        print(
            "usage: python script.py design.pklz sequential|concurrent cpu|cuda"
        )
    else:
        design = sys.argv[1]
        algorithm = sys.argv[2]
        device_str = sys.argv[3]
        test_ispd2005(design, algorithm, device_str)
