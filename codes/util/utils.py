# -*- coding: utf-8 -*-
import numpy as np
import scipy.sparse as sp
import time
import sys
import os
import shutil

def tprint(s):
    """print the time and string"""
    print("[" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "] " + s)

def get_sets(is_label=False):
    if is_label:
        sets = ["train", "valid", "test", "test2"]
    else:
        sets = ["train", "valid", "test"]
    return sets

def get_dataset_dir(options):
    dataset_dir = "%s/%s/"%(options.data_dir, options.dataset)
    return dataset_dir

def np_load(data_dir):
    if np.__version__ >= '1.16.2':
        data = np.load(data_dir, allow_pickle=True)
    else:
        data = np.load(data_dir)
    return data

def get_npz_arrays_name(npz_data):
    if sys.version_info.major > 2:
        arrays_name = npz_data.files
    else:
        arrays_name = npz_data.keys()
    return arrays_name

def exists_dir(path):
    file_path, file_name = os.path.split(path)
    return os.path.exists(file_path)

def exists_file(path):
    return os.path.exists(path)

def makedir(path):
    file_path, file_name = os.path.split(path)
    os.makedirs(file_path)
    return 0

def save_npz(path, *args, **kwds):
    if not exists_dir(path):
        makedir(path)
    np.savez_compressed(path, *args, **kwds)
    return 0

def clear_dir(path):
    if exists_dir(path):
        shutil.rmtree(path)

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(normalized_laplacian)


def calculate_random_walk_matrix(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1.0).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return sparse_to_tuple(random_walk_mx)

def get_dc_support(adj, self_conn=False):
    supports = []
    if self_conn:
        adj = adj + sp.eye(adj.shape[0])
    supports.append(calculate_random_walk_matrix(adj))
    supports.append(calculate_random_walk_matrix(adj.transpose()))
    return supports