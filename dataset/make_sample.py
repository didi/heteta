import os
import scipy.io as sio
import scipy.sparse as sp
import numpy as np
import random

dir_path = 'toy_sample/'

def print_shape():
    adj_files = ["adj.mat", "adj_gap_top5.mat"]
    for file in adj_files:
        data = sio.loadmat(dir_path+file)
        print(data.keys())
        for k in data.keys():
            if k in ['__header__', '__version__', '__globals__']:
                continue
            print("*"*50)
            print(data[k].shape, type(data[k]))
            print(sp.coo_matrix(data))

def make_a_random_adj(nodes_num, max_value=1):
    edges_rate = random.uniform(0, 0.005)
    edges_num = int(nodes_num*nodes_num*edges_rate)
    indices_left = [random.randint(0, nodes_num-1) for __ in range(edges_num)]
    indices_right = [random.randint(0, nodes_num-1) for __ in range(edges_num)]
    if max_value == 1:
        values = np.ones((edges_num,))
    else:
        values = [random.randint(1, max_value) for __ in range(edges_num)]
    sparseM = sp.csc_matrix((values,(indices_left, indices_right)), shape=(nodes_num, nodes_num))
    print("sparse matrix shape:", sparseM.get_shape())
    return sparseM

def make_adj_dict(nodes_num, relation_names):
    adj_dict = {}
    for r in relation_names:
        a_adj = make_a_random_adj(nodes_num)
        a_adj.sort_indices()
        adj_dict[r] = a_adj
    return adj_dict

def normalize_matrix(adj):
    adj.sort_indices()
    adj_mx = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv = np.power(d, -1.0).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    normalized = d_mat_inv.dot(adj_mx)
    if not sp.isspmatrix_csr(normalized):
        normalized = sp.csr_matrix(normalized)
    return normalized

def make_road_adj(nodes_num):
    relation_names = [str(i) for i in range(7)]
    road_adj_dict = make_adj_dict(nodes_num, relation_names)
    sio.savemat(dir_path+"adj.mat", road_adj_dict, do_compression=True)


def make_vehicle_adj(nodes_num):
    adj = make_a_random_adj(nodes_num, max_value=2000)
    car_adj_out = normalize_matrix(adj)
    car_adj_in = normalize_matrix(adj.transpose())
    vehicle_adj = {'3_0': car_adj_out, '3_1': car_adj_in}
    sio.savemat(dir_path+"adj_gap_top5.mat", vehicle_adj, do_compression=True)

def np_load(data_dir):
    if np.__version__ >= '1.16.2':
        data = np.load(data_dir, allow_pickle=True)
    else:
        data = np.load(data_dir)
    return data

def print_npz(the_npz):
    print(the_npz.keys())
    for k in the_npz.keys():
        if isinstance(the_npz[k], list) or isinstance(the_npz[k], tuple):
            print(k, 'list', len(the_npz[k]))
            ele_num = len(the_npz[k][0])
            for i in range(ele_num):
                print(k, i, type(the_npz[k][0][i]), the_npz[k][0][i].shape)
        else:
            print(k, 'array', the_npz[k].shape)

def make_dynamic_fes(nodes_num):
    periods = range(0, 8366)
    print("periods num:", len(periods))
    speed = np.random.uniform(low=0.0, high=40.0, size=(len(periods), nodes_num, 1))
    periods = np.array(periods)
    np.savez(dir_path+"dynamic_fes.npz", periods=periods, fes=speed)


def make_eta_label(nodes_num):
    train_periods = range(8065, 8165)
    valid_periods = range(8165, 8265)
    test_periods = range(8265, 8365)
    train_periods_len = len(train_periods)
    valid_periods_len = len(valid_periods)
    test_periods_len = len(test_periods)
    print("train period len", train_periods_len)
    print("valid period len", valid_periods_len)
    print("test period len", test_periods_len)
    the_len = [train_periods_len, valid_periods_len, test_periods_len]
    the_set = ["train", "valid", "test"]
    eta_label_list = []
    for d in range(3):
        the_eta_label = []
        for p in xrange(the_len[d]):
            # for each period
            o_num = random.randint(3, 100)
            o_list = []
            for o in xrange(o_num):
                # for each order
                # each line is a list, including list(list(link_idxs), list(link_moves), timespent)
                link_num = random.randint(1,100)
                link_idxs = [random.randint(0, nodes_num-1) for __ in range(link_num)]
                link_moves = [random.uniform(1, 3000) for __ in range(link_num)]
                timespent = random.uniform(30, 5000)
                o_list.append((link_idxs, link_moves, timespent))
            the_eta_label.append(o_list)
        eta_label_list.append(the_eta_label)
    np.savez(dir_path+"eta_label.npz", train=eta_label_list[0], train_periods=train_periods, \
                                       valid=eta_label_list[1], valid_periods=valid_periods, \
                                       test=eta_label_list[2], test_periods=test_periods)

def make_links_info(nodes_num):
    features_name = ['fes_'+str(i+1) for i in range(16)]
    features_type = {"fes_1": {"float": 1}, "fes_2": {"float": 1}, "fes_3": {"one_hot": 4, "start": 0},
                         "fes_4": {"one_hot": 5, "start": 1}, "fes_5": {"one_hot": 4, "start": 0},
                         "fes_6": {"binary": 1}, "fes_7": {"binary": 1}, "fes_8": {"binary": 1},
                         "fes_9": {"binary": 1}, "fes_10": {"binary": 1}, "fes_11": {"float": 1},
                         "fes_12": {"float": 1}, "fes_13": {"float": 1}, "fes_14": {"binary": 1},
                         "fes_15": {"one_hot": 3, "start": 0}, "fes_16": {"one_hot": 3, "start": 0}}
    link_info_list = []
    for n in xrange(nodes_num):
        n_fes_list = []
        for fes in range(len(features_name)):
            fes_name = features_name[fes]
            fes_type = features_type[fes_name]
            if "float" in fes_type:
                n_fes_list.append(random.uniform(-8, 8))
            elif "one_hot" in fes_type:
                one_hot_num = fes_type['one_hot']
                v = random.randint(0, one_hot_num-1)
                v += fes_type['start']
                n_fes_list.append(v)
            elif "binary" in fes_type:
                v = random.randint(0, 1)
                n_fes_list.append(v)
            else:
                raise TypeError("unkown feature type:", fes_type)
        link_info_list.append(n_fes_list)
    link_info = np.array(link_info_list)
    np.savez(dir_path+"link_info.npz", link_info=link_info)

def check():
    link_info = np_load(dir_path+"link_info.npz")
    print("*"*25,"link_info")
    print_npz(link_info)
    dynamic_fes = np_load(dir_path+"dynamic_fes.npz")
    print("*"*25,"dynamic_fes")
    print_npz(dynamic_fes)
    eta_label = np_load(dir_path+"eta_label.npz")
    print("*"*25,"eta_label")
    print_npz(eta_label)

if __name__ == "__main__":
    nodes_num = 300
    make_road_adj(nodes_num)
    make_vehicle_adj(nodes_num)
    print_shape()
    make_links_info(nodes_num)
    make_dynamic_fes(nodes_num)
    make_eta_label(nodes_num)
    check()
