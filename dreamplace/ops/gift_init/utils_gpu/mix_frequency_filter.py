import torch
import time
from torch import nn
import scipy.sparse as sp
import numpy as np
import torch
import pdb 
# from sparsesvd import sparsesvd
from scipy.sparse import csc_matrix, lil_matrix, save_npz, load_npz, csgraph, linalg, diags, identity


class GiFt_GPU(object):
    def __init__(self, adj_mat, device):
        self.adj_mat = adj_mat
        self.device = device 

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
        # self.norm_adj = norm_adj.tocsc()  # (D^-0.5(A+sigmaI)D^-0.5)
        # self.d_mat = d_mat.tocsc()
        end = time.time()
        # print('training time', end - start)

    def get_cell_position(self, k, cell_pos):
        norm_adj = self.norm_adj
        trainAdj = norm_adj.tocoo()
        edge_index = np.vstack((trainAdj.row, trainAdj.col)).transpose()
        edge_index = torch.from_numpy(edge_index).long()
        edge_index = edge_index.t().to(self.device)
        edge_weight = torch.from_numpy(trainAdj.data).float().to(self.device)

        norm_adj = torch.sparse.FloatTensor(edge_index,edge_weight).to(self.device)
        for _ in range(k):
            start = time.time()
            cell_pos = torch.sparse.mm(norm_adj, cell_pos)
            end = time.time()
        # torch.sparse.mm will create huge cache memory on GPU, which cannot be released automatically
        if cell_pos.is_cuda:
            torch.cuda.empty_cache()
        return cell_pos

