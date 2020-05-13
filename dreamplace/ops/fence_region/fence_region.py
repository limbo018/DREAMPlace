import time

import torch
from shapely.geometry import (GeometryCollection, LineString, MultiPolygon,
                              Polygon, box)
from shapely.ops import unary_union
import numpy as np

__all__ = ["gen_macros_for_non_fence_region", "gen_macros_for_fence_region", "slice_non_fence_region"]

def slice_non_fence_region(regions, xl, yl, xh, yh, merge=False, plot=False):
    if(type(regions) == list):
        if(isinstance(regions[0], np.ndarray)):
            regions = torch.from_numpy(np.concatenate(regions, 0))
        elif(isinstance(regions[0], torch.Tensor)):
            regions = torch.cat(regions, dim=0) # [n_box, 4]
    num_boxes = regions.size(0)
    regions = regions.view(num_boxes, 2, 2)
    fence_regions = MultiPolygon(
        [box(regions[i, 0, 0], regions[i, 0, 1], regions[i, 1, 0], regions[i, 1, 1]) for i in range(num_boxes)])
    fence_regions = unary_union(fence_regions)
    site = box(xl, yl, xh, yh)
    non_fence_region = unary_union(site.difference(fence_regions))

    slices = []
    xs = regions[:, :, 0].view(-1).sort()[0]
    for i in range(xs.size(0)+1):
        x_l = xl if i == 0 else xs[i-1]
        x_h = xh if i == xs.size(0) else xs[i]
        cvx_hull = box(x_l, yl, x_h, yh)
        if(not cvx_hull.is_valid):
            continue
        intersect = non_fence_region.intersection(cvx_hull)

        if(isinstance(intersect, Polygon)):
            slices.append(intersect.bounds)
        elif(isinstance(intersect, (GeometryCollection, MultiPolygon))):
            slices.extend(
                [j.bounds for j in intersect if(isinstance(j, Polygon))])

    if(merge):
        raw_bbox_list = sorted(slices, key=lambda x: (x[1], x[0]))

        cur_bbox = None
        bbox_list = []
        for i, p in enumerate(raw_bbox_list):
            minx, miny, maxx, maxy = p
            if(cur_bbox is None):
                cur_bbox = [minx, miny, maxx, maxy]
            elif(cur_bbox[1] == miny and cur_bbox[3] == maxy):
                cur_bbox[2:] = p[2:]
            else:
                bbox_list.append(cur_bbox)
                cur_bbox = [minx, miny, maxx, maxy]
        else:
            bbox_list.append(cur_bbox)
    else:
        bbox_list = slices


    if(plot):
        from matplotlib import pyplot as plt
        from descartes.patch import PolygonPatch
        from figures import BLUE, SIZE, set_limits, plot_coords, color_isvalid
        res = []
        for bbox in bbox_list:
            res.append(box(*bbox))
        res = MultiPolygon(res)
        fig = plt.figure(1, figsize=SIZE, dpi=90)
        ax = fig.add_subplot(121)
        for polygon in res:
            # plot_coords(ax, polygon.exterior)
            patch = PolygonPatch(polygon, facecolor=color_isvalid(
                non_fence_region), edgecolor=color_isvalid(non_fence_region, valid=BLUE), alpha=0.5, zorder=2)
            ax.add_patch(patch)

        set_limits(ax, -1, 20, -1, 20)
        ax = fig.add_subplot(122)
        patch = PolygonPatch(non_fence_region, facecolor=color_isvalid(
            non_fence_region), edgecolor=color_isvalid(non_fence_region, valid=BLUE), alpha=0.5, zorder=2)
        ax.add_patch(patch)
        set_limits(ax, -1, 20, -1, 20)
        plt.savefig('non_fence_region.png')

    bbox_list = torch.tensor(bbox_list, device=regions.device)
    return bbox_list


def gen_macros_for_non_fence_region(macro_pos_x, macro_pos_y, macro_size_x, macro_size_y, regions, yl, yh, merge=False, plot=False):
    # tt = time.time()
    macros = MultiPolygon([box(macro_pos_x[i], macro_pos_y[i], macro_pos_x[i]+macro_size_x[i],
                               macro_pos_y[i]+macro_size_y[i]) for i in range(macro_size_x.size(0))])
    # print("macro:", time.time()-tt)

    # tt = time.time()
    num_boxes = regions.size(0)
    regions = regions.view(num_boxes, 2, 2)
    fence_regions = MultiPolygon(
        [box(regions[i, 0, 0], regions[i, 0, 1], regions[i, 1, 0], regions[i, 1, 1]) for i in range(num_boxes)])
    # print("fence region:", time.time()-tt)

    merged_macros = macros.union(fence_regions)

    slices = []
    for p in merged_macros:
        boundary_x, _ = p.boundary.xy
        boundary_x = boundary_x.tolist()
        if(len(boundary_x) == 5):
            slices.append(p.bounds)
        else:
            xs = sorted(list(set(boundary_x)))
            for i, x_l in enumerate(xs[:-1]):
                x_h = xs[i+1]
                cvx_hull = box(x_l, yl, x_h, yh)
                intersect = p.intersection(cvx_hull)
                if(isinstance(intersect, Polygon)):
                    slices.append(intersect.bounds)
                elif(isinstance(intersect, (GeometryCollection, MultiPolygon))):
                    slices.extend(
                        [j.bounds for j in intersect if(isinstance(j, Polygon))])

    # tt = time.time()
    if(merge):
        raw_bbox_list = sorted(slices, key=lambda x: (x[1], x[0]))
        cur_bbox = None
        bbox_list = []
        for i, p in enumerate(raw_bbox_list):
            minx, miny, maxx, maxy = p
            if(cur_bbox is None):
                cur_bbox = [minx, miny, maxx, maxy]
            elif(cur_bbox[1] == miny and cur_bbox[3] == maxy):
                cur_bbox[2:] = p[2:]
            else:
                bbox_list.append(cur_bbox)
                cur_bbox = [minx, miny, maxx, maxy]
        else:
            bbox_list.append(cur_bbox)
    else:
        bbox_list = slices
    # print("merge:", time.time()-tt)

    bbox_list = torch.tensor(bbox_list).float()
    pos_x = bbox_list[:, 0]
    pos_y = bbox_list[:, 1]
    node_size_x = bbox_list[:, 2] - bbox_list[:, 0]
    node_size_y = bbox_list[:, 3] - bbox_list[:, 1]

    if(plot):
        from matplotlib import pyplot as plt
        from descartes.patch import PolygonPatch
        from figures import BLUE, SIZE, set_limits, plot_coords, color_isvalid
        res = []
        for bbox in bbox_list:
            res.append(box(*bbox))
        res = MultiPolygon(res)
        fig = plt.figure(1, figsize=SIZE, dpi=90)
        ax = fig.add_subplot(111)
        for polygon in res:
            # plot_coords(ax, polygon.exterior)
            patch = PolygonPatch(polygon, facecolor=color_isvalid(
                merged_macros), edgecolor=color_isvalid(merged_macros, valid=BLUE), alpha=0.5, zorder=2)
            ax.add_patch(patch)

        set_limits(ax, -1, 20, -1, 20)
        # ax = fig.add_subplot(122)
        # patch = PolygonPatch(reverse, facecolor=color_isvalid(reverse), edgecolor=color_isvalid(reverse, valid=BLUE), alpha=0.5, zorder=2)
        # ax.add_patch(patch)
        # set_limits(ax, -1, 20, -1, 20)
        plt.savefig('macro.png')

    return pos_x, pos_y, node_size_x, node_size_y


def gen_macros_for_fence_region(macro_pos_x, macro_pos_y, macro_size_x, macro_size_y, regions, xl, xh, yl, yh, merge=False, plot=False):
    # tt = time.time()
    macros = MultiPolygon([box(macro_pos_x[i], macro_pos_y[i], macro_pos_x[i]+macro_size_x[i],
                               macro_pos_y[i]+macro_size_y[i]) for i in range(macro_size_x.size(0))])
    # print("macro:", time.time()-tt)

    # tt = time.time()
    num_boxes = regions.size(0)
    regions = regions.view(num_boxes, 2, 2)
    fence_regions = MultiPolygon(
        [box(regions[i, 0, 0], regions[i, 0, 1], regions[i, 1, 0], regions[i, 1, 1]) for i in range(num_boxes)])

    site = box(xl, yl, xh, yh)
    reverse = site.difference(fence_regions).union(macros)
    # print("fence region:", time.time()-tt)

    # tt = time.time()
    slices = []
    xs = torch.cat([regions[:, :, 0].view(-1), macro_pos_x,
                    macro_pos_x + macro_size_x], dim=0).sort()[0]
    for i in range(xs.size(0)+1):
        x_l = xl if i == 0 else xs[i-1]
        x_h = xh if i == xs.size(0) else xs[i]

        # line1 = LineString([(x_l, yl), (x_l, yh)])
        # line2 = LineString([(x_h, yl), (x_h, yh)])

        # cvx_hull = MultiLineString([line1, line2]).convex_hull
        cvx_hull = box(x_l, yl, x_h, yh)
        intersect = reverse.intersection(cvx_hull)

        if(isinstance(intersect, Polygon)):
            slices.append(intersect.bounds)
        elif(isinstance(intersect, (GeometryCollection, MultiPolygon))):
            slices.extend(
                [j.bounds for j in intersect if(isinstance(j, Polygon))])

    # print("slicing:", time.time()-tt)

    # tt = time.time()
    if(merge):
        raw_bbox_list = sorted(slices, key=lambda x: (x[1], x[0]))

        cur_bbox = None
        bbox_list = []
        for i, p in enumerate(raw_bbox_list):
            minx, miny, maxx, maxy = p
            if(cur_bbox is None):
                cur_bbox = [minx, miny, maxx, maxy]
            elif(cur_bbox[1] == miny and cur_bbox[3] == maxy):
                cur_bbox[2:] = p[2:]
            else:
                bbox_list.append(cur_bbox)
                cur_bbox = [minx, miny, maxx, maxy]
        else:
            bbox_list.append(cur_bbox)
    else:
        bbox_list = slices
    # print("merge:", time.time()-tt)

    bbox_list = torch.tensor(bbox_list).float()
    pos_x = bbox_list[:, 0]
    pos_y = bbox_list[:, 1]
    node_size_x = bbox_list[:, 2] - bbox_list[:, 0]
    node_size_y = bbox_list[:, 3] - bbox_list[:, 1]

    if(plot):
        from matplotlib import pyplot as plt
        from descartes.patch import PolygonPatch
        from figures import BLUE, SIZE, set_limits, plot_coords, color_isvalid
        res = []
        for bbox in bbox_list:
            res.append(box(*bbox))
        res = MultiPolygon(res)
        fig = plt.figure(1, figsize=SIZE, dpi=90)
        ax = fig.add_subplot(121)
        for polygon in res:
            # plot_coords(ax, polygon.exterior)
            patch = PolygonPatch(polygon, facecolor=color_isvalid(
                fence_regions), edgecolor=color_isvalid(fence_regions, valid=BLUE), alpha=0.5, zorder=2)
            ax.add_patch(patch)

        set_limits(ax, -1, 20, -1, 20)
        ax = fig.add_subplot(122)
        patch = PolygonPatch(reverse, facecolor=color_isvalid(
            reverse), edgecolor=color_isvalid(reverse, valid=BLUE), alpha=0.5, zorder=2)
        ax.add_patch(patch)
        set_limits(ax, -1, 20, -1, 20)
        plt.savefig('polygon.png')

    return pos_x, pos_y, node_size_x, node_size_y


if __name__ == "__main__":
    xl, yl, xh, yh = 0, 0, 20, 20
    macro_pos_x = torch.tensor([8]).float()
    macro_pos_y = torch.tensor([5]).float()
    macro_size_x = torch.tensor([10]).float()
    macro_size_y = torch.tensor([10]).float()
    regions = torch.tensor(
        [[1, 1, 5, 5], (3, 6, 9, 9), (6, 3, 10, 5.5)]).float()

    slice_non_fence_region(regions, xl, yl, xh, yh, merge=True, plot=True)
    exit(1)
    gen_macros_for_non_fence_region(
        macro_pos_x, macro_pos_y, macro_size_x, macro_size_y, regions, yl, yh, merge=True)
    exit(1)
    import time
    t = time.time()
    for _ in range(10):
        gen_macros_for_fence_region(
            macro_pos_x, macro_pos_y, macro_size_x, macro_size_y, regions, xl, xh, yl, yh, merge=True)
    print((time.time()-t)/10)
