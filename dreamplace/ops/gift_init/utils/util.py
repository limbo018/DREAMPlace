import re
import os
import numpy as np
from pathlib import Path
from scipy.sparse import csc_matrix, lil_matrix, save_npz, load_npz, csgraph, linalg


def make_dir(path):
    if os.path.isdir(path):
        print(path, ' dir exists')
    else:
        os.makedirs(path)
        print(path, 'is created')


class Parser:
    def __init__(self):
        data = []

    def parser_net(self, netfilepath):
        """
        read XX.def file, extract cells, IO and nets
        """
        with open(netfilepath, 'r') as f:
            info = f.read()
            # read NET file
            netRegex = re.compile(r'\bnets\s+\d+\s*;',
                                  re.IGNORECASE)  # \\b matches the empty string, but only at the beginning or end of a word
            endNetRegex = re.compile(r'\bend\s+nets\s+', re.IGNORECASE)
            netInfo = info[info.find(re.search(netRegex, info).group()):info.find(re.search(endNetRegex, info).group())]

            # read cell info
            totalCellNumber = int(re.search(r'COMPONENTS\s(\d+)\s;', info).group(1))
            print('total cell num', totalCellNumber)
            cellInfo = info[info.find('COMPONENTS'):info.find('END COMPONENTS')]
            cellList = re.split(r';', cellInfo)
            cellList.pop(0)  # delete 'COMPONENTS xxxx'
            cellList.pop(-1)  # delete '\n'

            # # cellName2Index dict，按顺序存放cells and IOs, 先cells，然后IOs
            cellName2Index = dict()
            for i in range(totalCellNumber):
                CellName = re.split(r'\s', re.search(r'-\s+(.*)', cellList[i]).group(1))[0]
                cellName2Index.setdefault(CellName, i)

            # read PIN(IO Pad) info
            PINSRegex = re.compile(r'pins\s+(\d+)', re.IGNORECASE)
            totalPinNumber = int(re.search(PINSRegex, info).group(1)) - 1  # warning!!!!delete clk pin
            print('total io num', totalPinNumber)
            PINInfo = info[info.find('PINS'):info.find('END PINS')]
            PINList = re.split(r';', PINInfo)
            PINList.pop(0)
            PINList.pop(-1)
            for i in range(totalPinNumber):
                PINName = re.search(r'-\s+(\w+)\s+', PINList[i]).group(1)
                cellName2Index.setdefault(PINName, i + totalCellNumber)
        # ****************
        #  read net info
        # ****************
        subNetRegex = re.compile(r'-\s(.*?)\s')
        connectCellRegex = re.compile(r'\(\s+(.*?)\s+(.*?)\s+\)')

        # subnet_cellIndex: save cell indexes in the same net
        subNetName, subnet_cellIndex = [], []
        netInfoList = re.split(r';', netInfo)
        flag = 0
        for subNetInfo in netInfoList:
            flag += 1
            # print('net',flag)
            NetName = re.search(subNetRegex, subNetInfo)

            if NetName is not None and NetName.group(1) != 'ispd_clk':
                subNetName.append(NetName.group(1))  # save net name, delete ispd_clk
                connectCell = re.findall(connectCellRegex, subNetInfo)  # save cell names in the same net
                tmp_cell_index = []
                if len(connectCell) != 0:
                    for cell in connectCell:
                        if cell[0] == 'PIN':
                            tmp_cell_index.append(cellName2Index[cell[1]])
                        else:
                            tmp_cell_index.append(cellName2Index[cell[0]])

                subnet_cellIndex.append(tmp_cell_index)

        print('net num', len(subNetName))
        cell_num = totalCellNumber
        io_num = totalPinNumber
        cells_dict = cellName2Index
        nets_list = subnet_cellIndex

        return cell_num, io_num, cells_dict, nets_list


def cluster_level_connectivity_sparse(clusterResult, subnetlist):
    # create group-level connectivity matrix
    group_size = int(max(clusterResult)) + 1
    group_level_feature = lil_matrix((group_size, group_size))

    # Clique model
    for flag in range(len(subnetlist)):
        node_num_in_net = len(subnetlist[flag])
        for x in subnetlist[flag]:
            for y in subnetlist[flag]:
                xx = clusterResult[x]
                yy = clusterResult[y]
                if xx != yy:
                    group_level_feature[xx, yy] += 2 / node_num_in_net
                else:
                    pass

    # Cells in the same net are only connected to the first cell
    # for flag in range(len(subnetlist)):
    #     node_num_in_net = len(subnetlist[flag])
    #     xx = clusterResult[subnetlist[flag][0]]
    #     for y in subnetlist[flag][1:]:
    #         yy = clusterResult[y]
    #         if xx != yy:
    #             group_level_feature[xx, yy] += 2 / node_num_in_net
    #         else:
    #             pass

    group_level_feature = csc_matrix(group_level_feature)

    return group_level_feature


def cluster_connectivity_macro(cell_num, io_num, nets_list):
    # Construct the adjacency matrix and save it in the .npz file
    movable_node_num = cell_num
    fix_node_num = io_num
    cluster_num = movable_node_num + fix_node_num
    cluster = list(range(cluster_num))

    group_level_feature = cluster_level_connectivity_sparse(cluster, nets_list)
    saveFile = Path('forQua_conn.npz')
    save_npz(saveFile, group_level_feature)

    return group_level_feature, cluster, cluster_num


def find_fixed_point_def(file):
    io_id = []
    io_pos = []
    with open(file, 'r') as f:
        info = f.read()
        totalCellNumber = int(re.search(r'COMPONENTS\s(\d+)\s;', info).group(1))

        # read PIN(IO Pad) info
        PINSRegex = re.compile(r'pins\s+(\d+)', re.IGNORECASE)
        totalPinNumber = int(re.search(PINSRegex, info).group(1)) - 1  # remove clk pin

        PINInfo = info[info.find('PINS'):info.find('END PINS')]
        PINList = re.split(r';', PINInfo)
        PINList.pop(0)
        PINList.pop(-1)

        for i in range(totalPinNumber):
            io_id.append(i + totalCellNumber)
            pos_info = PINList[i].split('\n')[3]
            io_pos.append([int(pos_info.split()[3]), int(pos_info.split()[4])])
    io_pos = np.array(io_pos)

    return totalCellNumber, totalPinNumber, io_id, io_pos


def placement_region(fixed_pos):
    xf = fixed_pos[:, 0]
    yf = fixed_pos[:, 1]
    x_min = np.min(xf)
    x_max = np.max(xf)
    y_min = np.min(yf)
    y_max = np.max(yf)
    print('placement region: ', x_min, y_min, x_max, y_max)
    return x_min, y_min, x_max, y_max


def generate_initial_locations(fixed_cell_location, movable_num, scale):
    x_min, y_min, x_max, y_max = placement_region(fixed_cell_location)
    random_initial = np.random.rand(int(movable_num), 2)
    random_initial[:, 1] += (random_initial[:, 0] == random_initial[:, 1]) * 0.01  # avoid x_coor == y_coor
    xcenter = (x_max - x_min) / 2 + x_min
    ycenter = (y_max - y_min) / 2 + y_min
    random_initial[:, 0] = ((random_initial[:, 0] - 0.5) * (x_max - x_min) * scale) + xcenter
    random_initial[:, 1] = ((random_initial[:, 1] - 0.5) * (y_max - y_min) * scale) + ycenter
    return random_initial
