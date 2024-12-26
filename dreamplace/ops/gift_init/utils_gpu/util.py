import re
import time
import json
import csv
import time
import numpy as np
import scipy
import matplotlib.pyplot as plt
import os
import numpy as np
import statistics
from scipy import stats
from pathlib import Path
from scipy.sparse import csc_matrix, lil_matrix, save_npz, load_npz, csgraph, linalg

import logging
logger = logging.getLogger(__name__)

def make_dir(path):
    if os.path.isdir(path):
        print(path, ' dir exists')
    else:
        os.makedirs(path)
        print(path, 'is created')


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
    logger.info('placement region: (%g, %g, %g, %g)', x_min, y_min, x_max, y_max)
    return x_min, y_min, x_max, y_max


def generate_initial_locations(fixed_cell_location, movable_num, scale):
    x_min, y_min, x_max, y_max = placement_region(fixed_cell_location)
    random_initial = np.random.rand(int(movable_num), 2)
    xcenter = (x_max - x_min) / 2 + x_min
    ycenter = (y_max - y_min) / 2 + y_min
    random_initial[:, 0] = ((random_initial[:, 0] - 0.5) * (x_max - x_min) * scale) + xcenter
    random_initial[:, 1] = ((random_initial[:, 1] - 0.5) * (y_max - y_min) * scale) + ycenter
    return random_initial
