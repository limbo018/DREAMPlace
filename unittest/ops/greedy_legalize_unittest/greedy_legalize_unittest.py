##
# @file   greedy_legalize_unitest.py
# @author Yibo Lin
# @date   Mar 2019
#

import os 
import sys
import numpy as np
import unittest
import cairocffi as cairo 
import time 

import torch
from torch.autograd import Function, Variable

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from dreamplace.ops.greedy_legalize import greedy_legalize
sys.path.pop()

def plot(figname, 
        node_x, node_y, 
        node_size_x, node_size_y, 
        layout_xl, layout_yl, layout_xh, layout_yh, 
        num_bins_x, num_bins_y, 
        num_nodes, num_movable_nodes, num_physical_nodes, num_filler_nodes 
        ): 
    tt = time.time()
    width = 800
    height = 800
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    ctx = cairo.Context(surface)
    ctx.scale(width, height)  # Normalizing the canvas

    if num_movable_nodes < num_physical_nodes: 
        layout_xl2 = min(np.amin(node_x[num_movable_nodes:num_physical_nodes]), layout_xl)
        layout_yl2 = min(np.amin(node_y[num_movable_nodes:num_physical_nodes]), layout_yl)
        layout_xh2 = max(np.amax(node_x[num_movable_nodes:num_physical_nodes]+node_size_x[num_movable_nodes:num_physical_nodes]), layout_xh)
        layout_yh2 = max(np.amax(node_y[num_movable_nodes:num_physical_nodes]+node_size_y[num_movable_nodes:num_physical_nodes]), layout_yh)
    else:
        layout_xl2 = layout_xl
        layout_yl2 = layout_yl
        layout_xh2 = layout_xh
        layout_yh2 = layout_yh

    bin_size_x = (layout_xh-layout_xl)/num_bins_x
    bin_size_y = (layout_yh-layout_yl)/num_bins_y

    def normalize_x(xx):
        return (xx - (layout_xl-bin_size_x))/(layout_xh-layout_xl+2*bin_size_x)
    def normalize_y(xx):
        return (xx - (layout_yl-bin_size_y))/(layout_yh-layout_yl+2*bin_size_y)
    def draw_rect(x1, y1, x2, y2):
        ctx.move_to(x1, y1)
        ctx.line_to(x1, y2)
        ctx.line_to(x2, y2)
        ctx.line_to(x2, y1)
        ctx.close_path()
        ctx.stroke()

    def bin_xl(i):
        return layout_xl+i*bin_size_x
    def bin_yl(i):
        return layout_yl+i*bin_size_y

    # draw layout region 
    ctx.set_source_rgb(1, 1, 1)
    draw_layout_xl = normalize_x(layout_xl2-1*bin_size_x)
    draw_layout_yl = normalize_y(layout_yl2-1*bin_size_y)
    draw_layout_xh = normalize_x(layout_xh2+1*bin_size_x)
    draw_layout_yh = normalize_y(layout_yh2+1*bin_size_y)
    ctx.rectangle(draw_layout_xl, draw_layout_yl, draw_layout_xh, draw_layout_yh)
    ctx.fill()
    ctx.set_line_width(0.001)
    ctx.set_source_rgba(0.1, 0.1, 0.1, alpha=0.8)
    draw_rect(normalize_x(layout_xl), normalize_y(layout_yl), normalize_x(layout_xh), normalize_y(layout_yh))
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
    node_yl = layout_yl+layout_yh-(node_y+node_size_y[0:len(node_y)]) # flip y 
    node_xh = node_x+node_size_x[0:len(node_x)]
    node_yh = layout_yl+layout_yh-node_y # flip y 
    node_xl = normalize_x(node_xl)
    node_yl = normalize_y(node_yl)
    node_xh = normalize_x(node_xh)
    node_yh = normalize_y(node_yh)
    ctx.set_line_width(0.001)
    #print("plot layout")
    # draw fixed macros
    ctx.set_source_rgba(1, 0, 0, alpha=0.5)
    for i in range(num_movable_nodes, num_physical_nodes):
        ctx.rectangle(node_xl[i], node_yl[i], node_xh[i]-node_xl[i], node_yh[i]-node_yl[i])  # Rectangle(xl, yl, w, h)
        ctx.fill()
    ctx.set_source_rgba(0, 0, 0, alpha=1.0)  # Solid color
    for i in range(num_movable_nodes, num_physical_nodes):
        draw_rect(node_xl[i], node_yl[i], node_xh[i], node_yh[i])
    # draw fillers
    if len(node_xl) > num_physical_nodes: # filler is included 
        ctx.set_line_width(0.001)
        ctx.set_source_rgba(230/255.0, 230/255.0, 250/255.0, alpha=0.3)  # Solid color
        for i in range(num_physical_nodes, num_nodes):
            draw_rect(node_xl[i], node_yl[i], node_xh[i], node_yh[i])
    # draw cells 
    ctx.set_line_width(0.002)
    ctx.set_source_rgba(0, 0, 1, alpha=0.8)  # Solid color
    for i in range(num_movable_nodes):
        draw_rect(node_xl[i], node_yl[i], node_xh[i], node_yh[i])

    surface.write_to_png(figname)  # Output to PNG
    print("[I] plotting to %s takes %.3f seconds" % (figname, time.time()-tt))
    #print(session.run(grads))
    #print(session.run(masked_grads))

class GreedyLegalizeOpTest(unittest.TestCase):
    def test_greedyLegalizeRandom(self):
        dtype = np.float64
        xx = np.array([1.0, 0.5, 3.0]).astype(dtype)
        yy = np.array([0.5, 0.8, 1.5]).astype(dtype)
        node_size_x = np.array([0.5, 1.5, 1.0]).astype(dtype)
        node_size_y = np.array([2.0, 2.0, 4.0]).astype(dtype)
        node_weights = np.ones_like(node_size_x) 
        num_nodes = len(xx)
        
        xl = 1.0 
        yl = 1.0 
        xh = 5.0
        yh = 5.0
        num_terminals = 0 
        num_terminal_NIs = 0 
        num_filler_nodes = 0
        num_movable_nodes = len(xx)-num_terminals-num_terminal_NIs-num_filler_nodes
        site_width = 1 
        row_height = 2 
        num_bins_x = 2
        num_bins_y = 2
        flat_region_boxes = np.zeros(0, dtype=dtype)
        flat_region_boxes_start = np.array([0], dtype=np.int32)
        node2fence_region_map = np.zeros(0, dtype=np.int32)

        plot("initial.png", 
                xx, yy, 
                node_size_x, node_size_y, 
                xl, yl, xh, yh, 
                num_bins_x, num_bins_y, 
                num_movable_nodes+num_terminals+num_terminal_NIs+num_filler_nodes, num_movable_nodes, num_movable_nodes+num_terminals+num_terminal_NIs, num_filler_nodes)

        # test cpu 
        custom = greedy_legalize.GreedyLegalize(
                    torch.from_numpy(node_size_x), torch.from_numpy(node_size_y), torch.from_numpy(node_weights), 
                    flat_region_boxes=torch.from_numpy(flat_region_boxes), flat_region_boxes_start=torch.from_numpy(flat_region_boxes_start), node2fence_region_map=torch.from_numpy(node2fence_region_map), 
                    xl=xl, yl=yl, xh=xh, yh=yh, 
                    site_width=site_width, row_height=row_height, 
                    num_bins_x=num_bins_x, num_bins_y=num_bins_y, 
                    num_movable_nodes=num_movable_nodes, 
                    num_terminal_NIs=num_terminal_NIs, 
                    num_filler_nodes=num_filler_nodes)

        pos = Variable(torch.from_numpy(np.concatenate([xx, yy])))
        result = custom(pos, pos)
        print("custom_result = ", result)

        print("average displacement = %g" % (np.sum(np.absolute(result.numpy() - np.concatenate([xx, yy])))/num_movable_nodes))

        plot("final.png", 
                result.numpy()[0:len(xx)], result.numpy()[len(xx):], 
                node_size_x, node_size_y, 
                xl, yl, xh, yh, 
                num_bins_x, num_bins_y, 
                num_movable_nodes+num_terminals+num_terminal_NIs+num_filler_nodes, num_movable_nodes, num_movable_nodes+num_terminals+num_terminal_NIs, num_filler_nodes)

        # test cuda 
        if torch.cuda.device_count(): 
            custom_cuda = greedy_legalize.GreedyLegalize(
                        torch.from_numpy(node_size_x).cuda(), torch.from_numpy(node_size_y).cuda(), torch.from_numpy(node_weights).cuda(), 
                        flat_region_boxes=torch.from_numpy(flat_region_boxes).cuda(), flat_region_boxes_start=torch.from_numpy(flat_region_boxes_start).cuda(), node2fence_region_map=torch.from_numpy(node2fence_region_map).cuda(), 
                        xl=xl, yl=yl, xh=xh, yh=yh, 
                        site_width=site_width, row_height=row_height, 
                        num_bins_x=num_bins_x, num_bins_y=num_bins_y, 
                        num_movable_nodes=num_movable_nodes, 
                        num_terminal_NIs=num_terminal_NIs, 
                        num_filler_nodes=num_filler_nodes)

            pos = Variable(torch.from_numpy(np.concatenate([xx, yy]))).cuda()
            result_cuda = custom_cuda(pos, pos)
            print("custom_result = ", result_cuda.data.cpu())

            #np.testing.assert_allclose(result, result_cuda.data.cpu())

if __name__ == '__main__':
    unittest.main()
