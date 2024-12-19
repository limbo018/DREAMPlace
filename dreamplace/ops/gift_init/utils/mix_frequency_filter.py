import torch
import time
from torch import nn
import scipy.sparse as sp
import numpy as np
from scipy.sparse import csc_matrix, lil_matrix, save_npz, load_npz, csgraph, linalg, diags, identity


class GiFt(object):
    def __init__(self, adj_mat):
        self.adj_mat = adj_mat
        # adjacency matrix

    def train(self, sigma):
        adj_mat = self.adj_mat
        adj_mat = csc_matrix(adj_mat)
        dim = adj_mat.shape[0]
        start = time.time()
        adj_mat = adj_mat + sigma * identity(dim)  # augmented adj
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)  # D^-0.5
        norm_adj = d_mat.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat)
        self.norm_adj = norm_adj  # (D^-0.5(A+sigmaI)D^-0.5)
        self.d_mat = d_mat
        end = time.time()
        print('training time', end - start)

    def get_cell_position(self, k, cell_pos):
        norm_adj = self.norm_adj
        print("non-zero number norm adj: ", norm_adj.shape)

        for _ in range(k):
            start = time.time()
            cell_pos = norm_adj @ cell_pos
            end = time.time()
        return cell_pos

    def ideal_low_pass_filter(self):
        laplacianmatrix = csgraph.laplacian(self.adj_mat, normed=True)
        eigenvalue, eigenvector = linalg.eigsh(laplacianmatrix, k=3, which='SM')
        vt = eigenvector[:, 1:]
        return vt
