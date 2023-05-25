import time
from math import sqrt

import numpy as np
import torch
from shapely import affinity
from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Polygon, box
from shapely.ops import unary_union

__all__ = ["gen_macros_for_non_fence_region", "gen_macros_for_fence_region", "slice_non_fence_region"]

GM = (sqrt(5) - 1.0) / 2.0
W = 8.0
H = W * GM
SIZE = (W, H)

BLUE = "#6699cc"
GRAY = "#999999"
DARKGRAY = "#333333"
YELLOW = "#ffcc33"
GREEN = "#339933"
RED = "#ff3333"
BLACK = "#000000"

COLOR_ISVALID = {
    True: BLUE,
    False: RED,
}


def plot_line(ax, ob, color=GRAY, zorder=1, linewidth=3, alpha=1):
    x, y = ob.xy
    ax.plot(x, y, color=color, linewidth=linewidth, solid_capstyle="round", zorder=zorder, alpha=alpha)


def plot_coords(ax, ob, color=GRAY, zorder=1, alpha=1):
    x, y = ob.xy
    ax.plot(x, y, "o", color=color, zorder=zorder, alpha=alpha)


def color_isvalid(ob, valid=BLUE, invalid=RED):
    if ob.is_valid:
        return valid
    else:
        return invalid


def color_issimple(ob, simple=BLUE, complex=YELLOW):
    if ob.is_simple:
        return simple
    else:
        return complex


def plot_line_isvalid(ax, ob, **kwargs):
    kwargs["color"] = color_isvalid(ob)
    plot_line(ax, ob, **kwargs)


def plot_line_issimple(ax, ob, **kwargs):
    kwargs["color"] = color_issimple(ob)
    plot_line(ax, ob, **kwargs)


def plot_bounds(ax, ob, zorder=1, alpha=1):
    x, y = zip(*list((p.x, p.y) for p in ob.boundary))
    ax.plot(x, y, "o", color=BLACK, zorder=zorder, alpha=alpha)


def add_origin(ax, geom, origin):
    x, y = xy = affinity.interpret_origin(geom, origin, 2)
    ax.plot(x, y, "o", color=GRAY, zorder=1)
    ax.annotate(str(xy), xy=xy, ha="center", textcoords="offset points", xytext=(0, 8))


def set_limits(ax, x0, xN, y0, yN, dx=1, dy=1):
    ax.set_xlim(x0, xN)
    ax.set_xticks(range(x0, xN + 1, dx))
    ax.set_ylim(y0, yN)
    ax.set_yticks(range(y0, yN + 1, dy))
    ax.set_aspect("equal")


def calc_region_area(regions):
    if type(regions) == list:
        if isinstance(regions[0], np.ndarray):
            regions = torch.from_numpy(np.concatenate(regions, 0))
        elif isinstance(regions[0], torch.Tensor):
            regions = torch.cat(regions, dim=0)  # [n_box, 4]
    elif isinstance(regions, np.ndarray):
        regions = torch.from_numpy(regions)

    num_boxes = regions.size(0)
    regions = regions.view(num_boxes, 2, 2)
    fence_regions = MultiPolygon(
        [
            box(regions[i, 0, 0], regions[i, 0, 1], regions[i, 1, 0], regions[i, 1, 1])
            for i in range(num_boxes)
        ]
    )
    fence_regions = unary_union(fence_regions)
    return fence_regions.area


def slice_non_fence_region(
    regions,
    xl,
    yl,
    xh,
    yh,
    macro_pos_x=None,
    macro_pos_y=None,
    macro_size_x=None,
    macro_size_y=None,
    merge=False,
    plot=False,
    figname="non_fence_region.png",
    device=torch.device("cuda:0"),
):
    if type(regions) == list:
        if isinstance(regions[0], np.ndarray):
            regions = torch.from_numpy(np.concatenate(regions, 0)).to(device)
        elif isinstance(regions[0], torch.Tensor):
            regions = torch.cat(regions, dim=0).to(device)  # [n_box, 4]
    elif isinstance(regions, np.ndarray):
        regions = torch.from_numpy(regions).to(device)

    if macro_pos_x is not None:
        if isinstance(macro_pos_x, np.ndarray):
            macro_pos_x = torch.from_numpy(macro_pos_x).to(device).float()
            macro_pos_y = torch.from_numpy(macro_pos_y).to(device).float()
            macro_size_x = torch.from_numpy(macro_size_x).to(device).float()
            macro_size_y = torch.from_numpy(macro_size_y).to(device).float()

        regions = torch.cat(
            [
                regions,
                torch.stack(
                    [macro_pos_x, macro_pos_y, macro_pos_x + macro_size_x, macro_pos_y + macro_size_y], 0
                ).t(),
            ],
            0,
        )

    num_boxes = regions.size(0)
    regions = regions.view(num_boxes, 2, 2)
    fence_regions = MultiPolygon(
        [
            box(regions[i, 0, 0], regions[i, 0, 1], regions[i, 1, 0], regions[i, 1, 1])
            for i in range(num_boxes)
        ]
    )
    fence_regions = unary_union(fence_regions)
    site = box(xl, yl, xh, yh)

    non_fence_region = unary_union(site.difference(fence_regions))

    slices = []
    xs = regions[:, :, 0].view(-1).sort()[0]
    for i in range(xs.size(0) + 1):
        x_l = xl if i == 0 else xs[i - 1]
        x_h = xh if i == xs.size(0) else xs[i]
        cvx_hull = box(x_l, yl, x_h, yh)

        if x_l >= x_h or not cvx_hull.is_valid:
            continue
        intersect = non_fence_region.intersection(cvx_hull)

        if isinstance(intersect, Polygon) and len(intersect.bounds) == 4:
            slices.append(intersect.bounds)
        elif isinstance(intersect, (GeometryCollection, MultiPolygon)):
            slices.extend([j.bounds for j in intersect.geoms if (isinstance(j, Polygon) and len(j.bounds) == 4)])

    if merge:
        raw_bbox_list = sorted(slices, key=lambda x: (x[1], x[0]))

        cur_bbox = None
        bbox_list = []
        for i, p in enumerate(raw_bbox_list):
            minx, miny, maxx, maxy = p
            if cur_bbox is None:
                cur_bbox = [minx, miny, maxx, maxy]
            elif cur_bbox[1] == miny and cur_bbox[3] == maxy and cur_bbox[2] == minx:
                cur_bbox[2:] = p[2:]
            else:
                bbox_list.append(cur_bbox)
                cur_bbox = [minx, miny, maxx, maxy]
        else:
            bbox_list.append(cur_bbox)
    else:
        bbox_list = slices

    if plot:
        from descartes.patch import PolygonPatch
        from matplotlib import pyplot as plt

        # from figures import BLUE, SIZE, set_limits, plot_coords, color_isvalid
        res = []
        bbox_list_np = np.array(bbox_list)
        bbox_list_np *= 1000 / np.max(bbox_list_np)
        for bbox in bbox_list_np:
            res.append(box(*bbox.tolist()))
        res = MultiPolygon(res)
        fig = plt.figure(1, figsize=SIZE, dpi=90)
        ax = fig.add_subplot(121)
        for polygon in res:
            # plot_coords(ax, polygon.exterior)
            patch = PolygonPatch(
                polygon,
                facecolor=color_isvalid(non_fence_region),
                edgecolor=color_isvalid(non_fence_region, valid=BLUE),
                alpha=0.5,
                zorder=2,
            )
            ax.add_patch(patch)

        set_limits(ax, -1, 1000, -1, 1000, dx=100, dy=100)
        # ax = fig.add_subplot(122)
        # patch = PolygonPatch(non_fence_region, facecolor=color_isvalid(
        #     non_fence_region), edgecolor=color_isvalid(non_fence_region, valid=BLUE), alpha=0.5, zorder=2)
        # ax.add_patch(patch)
        # set_limits(ax, -1, 1000, -1, 1000, dx=100, dy=100)
        plt.savefig(figname)
        plt.close()

    bbox_list = torch.tensor(bbox_list, device=device)
    # print("non fence region area after slicing:", ((bbox_list[:,2]-bbox_list[:,0])*(bbox_list[:,3]-bbox_list[:,1])).sum().item())
    return bbox_list


def gen_macros_for_non_fence_region(
    macro_pos_x, macro_pos_y, macro_size_x, macro_size_y, regions, yl, yh, merge=False, plot=False
):
    # tt = time.time()
    macros = MultiPolygon(
        [
            box(
                macro_pos_x[i],
                macro_pos_y[i],
                macro_pos_x[i] + macro_size_x[i],
                macro_pos_y[i] + macro_size_y[i],
            )
            for i in range(macro_size_x.size(0))
        ]
    )
    # print("macro:", time.time()-tt)

    # tt = time.time()
    num_boxes = regions.size(0)
    regions = regions.view(num_boxes, 2, 2)
    fence_regions = MultiPolygon(
        [
            box(regions[i, 0, 0], regions[i, 0, 1], regions[i, 1, 0], regions[i, 1, 1])
            for i in range(num_boxes)
        ]
    )
    # print("fence region:", time.time()-tt)

    merged_macros = macros.union(fence_regions)

    slices = []
    for p in merged_macros:
        boundary_x, _ = p.boundary.xy
        boundary_x = boundary_x.tolist()
        if len(boundary_x) == 5:
            slices.append(p.bounds)
        else:
            xs = sorted(list(set(boundary_x)))
            for i, x_l in enumerate(xs[:-1]):
                x_h = xs[i + 1]
                cvx_hull = box(x_l, yl, x_h, yh)
                intersect = p.intersection(cvx_hull)
                if isinstance(intersect, Polygon):
                    slices.append(intersect.bounds)
                elif isinstance(intersect, (GeometryCollection, MultiPolygon)):
                    slices.extend([j.bounds for j in intersect if (isinstance(j, Polygon))])

    # tt = time.time()
    if merge:
        raw_bbox_list = sorted(slices, key=lambda x: (x[1], x[0]))
        cur_bbox = None
        bbox_list = []
        for i, p in enumerate(raw_bbox_list):
            minx, miny, maxx, maxy = p
            if cur_bbox is None:
                cur_bbox = [minx, miny, maxx, maxy]
            elif cur_bbox[1] == miny and cur_bbox[3] == maxy:
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

    if plot:
        from descartes.patch import PolygonPatch
        from matplotlib import pyplot as plt

        from figures import BLUE, SIZE, color_isvalid, plot_coords, set_limits

        res = []
        for bbox in bbox_list:
            res.append(box(*bbox))
        res = MultiPolygon(res)
        fig = plt.figure(1, figsize=SIZE, dpi=90)
        ax = fig.add_subplot(111)
        for polygon in res:
            # plot_coords(ax, polygon.exterior)
            patch = PolygonPatch(
                polygon,
                facecolor=color_isvalid(merged_macros),
                edgecolor=color_isvalid(merged_macros, valid=BLUE),
                alpha=0.5,
                zorder=2,
            )
            ax.add_patch(patch)

        set_limits(ax, -1, 20, -1, 20)
        # ax = fig.add_subplot(122)
        # patch = PolygonPatch(reverse, facecolor=color_isvalid(reverse), edgecolor=color_isvalid(reverse, valid=BLUE), alpha=0.5, zorder=2)
        # ax.add_patch(patch)
        # set_limits(ax, -1, 20, -1, 20)
        plt.savefig("macro.png")

    return pos_x, pos_y, node_size_x, node_size_y


def gen_macros_for_fence_region(
    macro_pos_x, macro_pos_y, macro_size_x, macro_size_y, regions, xl, xh, yl, yh, merge=False, plot=False
):
    # tt = time.time()
    macros = MultiPolygon(
        [
            box(
                macro_pos_x[i],
                macro_pos_y[i],
                macro_pos_x[i] + macro_size_x[i],
                macro_pos_y[i] + macro_size_y[i],
            )
            for i in range(macro_size_x.size(0))
        ]
    )
    # print("macro:", time.time()-tt)

    # tt = time.time()
    num_boxes = regions.size(0)
    regions = regions.view(num_boxes, 2, 2)
    fence_regions = MultiPolygon(
        [
            box(regions[i, 0, 0], regions[i, 0, 1], regions[i, 1, 0], regions[i, 1, 1])
            for i in range(num_boxes)
        ]
    )

    site = box(xl, yl, xh, yh)
    reverse = site.difference(fence_regions).union(macros)
    # print("fence region:", time.time()-tt)

    # tt = time.time()
    slices = []
    xs = torch.cat([regions[:, :, 0].view(-1), macro_pos_x, macro_pos_x + macro_size_x], dim=0).sort()[0]
    for i in range(xs.size(0) + 1):
        x_l = xl if i == 0 else xs[i - 1]
        x_h = xh if i == xs.size(0) else xs[i]

        cvx_hull = box(x_l, yl, x_h, yh)
        intersect = reverse.intersection(cvx_hull)

        if isinstance(intersect, Polygon):
            slices.append(intersect.bounds)
        elif isinstance(intersect, (GeometryCollection, MultiPolygon)):
            slices.extend([j.bounds for j in intersect if (isinstance(j, Polygon))])

    # print("slicing:", time.time()-tt)

    # tt = time.time()
    if merge:
        raw_bbox_list = sorted(slices, key=lambda x: (x[1], x[0]))

        cur_bbox = None
        bbox_list = []
        for i, p in enumerate(raw_bbox_list):
            minx, miny, maxx, maxy = p
            if cur_bbox is None:
                cur_bbox = [minx, miny, maxx, maxy]
            elif cur_bbox[1] == miny and cur_bbox[3] == maxy:
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

    if plot:
        from descartes.patch import PolygonPatch
        from matplotlib import pyplot as plt

        from figures import BLUE, SIZE, color_isvalid, plot_coords, set_limits

        res = []
        for bbox in bbox_list:
            res.append(box(*bbox))
        res = MultiPolygon(res)
        fig = plt.figure(1, figsize=SIZE, dpi=90)
        ax = fig.add_subplot(121)
        for polygon in res:
            # plot_coords(ax, polygon.exterior)
            patch = PolygonPatch(
                polygon,
                facecolor=color_isvalid(fence_regions),
                edgecolor=color_isvalid(fence_regions, valid=BLUE),
                alpha=0.5,
                zorder=2,
            )
            ax.add_patch(patch)

        set_limits(ax, -1, 20, -1, 20)
        ax = fig.add_subplot(122)
        patch = PolygonPatch(
            reverse,
            facecolor=color_isvalid(reverse),
            edgecolor=color_isvalid(reverse, valid=BLUE),
            alpha=0.5,
            zorder=2,
        )
        ax.add_patch(patch)
        set_limits(ax, -1, 20, -1, 20)
        plt.savefig("polygon.png")

    return pos_x, pos_y, node_size_x, node_size_y


def draw_ispd2015():
    regions = [
        np.array([(47200, 252000, 99200, 492000)], dtype=np.float32) / 1000,
        np.array(
            [
                (136000, 252000, 297800, 300000),
                (194800, 346000, 297800, 396000),
                (297800, 252000, 361200, 396000),
                (136000, 346000, 194800, 490000),
                (194800, 440000, 361200, 490000),
            ],
            dtype=np.float32,
        )
        / 1000,
        np.array(
            [
                (483200, 254000, 484400, 408000),
                (484400, 364000, 565400, 408000),
                (426000, 250000, 483200, 490000),
                (483200, 450000, 565400, 490000),
                (565400, 364000, 622600, 490000),
            ],
            dtype=np.float32,
        )
        / 1000,
        np.array(
            [
                (725000, 252000, 828000, 300000),
                (668200, 252000, 725000, 490000),
                (725000, 448000, 828000, 490000),
                (828000, 252000, 856200, 490000),
            ],
            dtype=np.float32,
        )
        / 1000,
    ]
    xl, yl, xh, yh = 0, 0, 1000, 1000

    # non_fence_regions_ex = slice_non_fence_region(
    #     regions, xl, yl, xh, yh, merge=True, plot=True, figname="nonfence_ex.png")
    # non_fence_regions = [slice_non_fence_region(
    #     region, xl, yl, xh, yh, merge=True, plot=True, figname=f"nonfence_{i}.png") for i, region in enumerate(regions)]
    macro_pos_x = torch.tensor([600.0])
    macro_pos_y = torch.tensor([600.0])
    macro_size_x = torch.tensor([200.0])
    macro_size_y = torch.tensor([200.0])
    inner_fence_region = slice_non_fence_region(
        regions,
        xl,
        yl,
        xh,
        yh,
        merge=True,
        plot=True,
        figname="inner_nonfence_ex.png",
        macro_pos_x=macro_pos_x,
        macro_pos_y=macro_pos_y,
        macro_size_x=macro_size_x,
        macro_size_y=macro_size_y,
        device="cpu",
    )


if __name__ == "__main__":
    draw_ispd2015()
