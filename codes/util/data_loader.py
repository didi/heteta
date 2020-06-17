# -*- coding: utf-8 -*-
import random
import numpy as np
import os
import tensorflow as tf
from scaler import StandardScaler, StaticFeatureScaler
from utils import tprint, np_load

def get_one_sample(the_args):
    eta_idx, loader = the_args
    period = loader.eta_periods[eta_idx]
    if loader.seq_len == 1:  # baseline
        batch_fes = np.reshape(loader.dynamic_fes[loader.fe_period2idx[period - 1]],
                                newshape=loader.num_of_links)  # [link_num, 1]->link_num
    else:
        fe_idxs = map(lambda x: loader.fe_period2idx[x], range(period - loader.seq_len, period))
        batch_fes = np.concatenate([loader.dynamic_fes[fe_idxs], loader.static_fes],
                                    axis=2)  # [4, link_num, feature_dim]
    batch_link_move_sp, batch_sum_time = loader.period2batch[period]
    return (batch_fes, batch_link_move_sp, batch_sum_time)


class DataLoader(object):
    def __init__(self, dynamic_fes, static_fes, links_time_list, seq_len, days, weeks, fe_periods, eta_periods, shuffle=False, per_period=5):
        """
        :param dynamic_fes: [periods, link_num, dynamic_feature_dim]
        :param static_fes: [link_num, static_feature_dim]
        :param links_time_list: list idx by period_idx, element: list(list(link_idxs), list(link_moves), timespent)
        :param fe_periods: 表示dynamic_fes的第i行对应的period
        :param eta_periods: 表示links_time_list的第i行对应的period
        """
        self.fe_period2idx = dict(zip(fe_periods, range(len(fe_periods))))
        self.eta_periods = eta_periods
        eta_period2idx = dict(zip(eta_periods, range(len(eta_periods))))
        self.num_of_links = len(static_fes)
        self.seq_len = seq_len
        self.days = days
        self.weeks = weeks
        self.dynamic_fes = dynamic_fes
        self.static_fes = np.repeat(np.expand_dims(static_fes, axis=0), seq_len+days+weeks, axis=0)  # [4, link_num, fe_static_dim]
        self.order = range(len(eta_periods))
        self.shuffle = shuffle
        self.per_period = per_period
        # Preprocessing the batches: making the same period as a batch
        self.period2batch = dict()
        for period in eta_periods:
            eta_idx = eta_period2idx[period]
            period = self.eta_periods[eta_idx]
            # Sparse storage within a batch
            rowcol2val = dict()
            batch_sum_time = []
            for i, (link_idxs, link_moves, time_spent) in enumerate(links_time_list[eta_idx]):
                for link_idx, link_move in zip(link_idxs, link_moves):
                    rowcol2val[(i, link_idx)] = rowcol2val.get((i, link_idx), 0.0) + link_move
                batch_sum_time.append(time_spent)
            # Each line represents how many road segments have been traveled in an order
            # to sort sparse tensor!
            sorted_keys = rowcol2val.keys()
            sorted_keys.sort()
            batch_link_move_sp = tf.SparseTensorValue(indices=np.array(sorted_keys),
                                                      values=map(rowcol2val.get, sorted_keys),
                                                      dense_shape=[len(batch_sum_time), self.num_of_links])
            batch_sum_time = np.array(batch_sum_time)
            self.period2batch[period] = (batch_link_move_sp, batch_sum_time)
        print("Init dataloader, total num of batches=%d." % len(eta_periods))

    def get_static_fes(self):
        return self.static_fes

    def get_iterator(self, batch_num=1):  # Samples in the same period make up a batch
        def _wrapper():
            oneday_periods = int(24*60/self.per_period)
            all_order_num = len(self.order)
            sidx = 0  #smaple_index
            while sidx < all_order_num:
                eta_idx_list = []
                b_sidx = sidx
                for b in range(batch_num):
                    eta_idx = self.order[b_sidx]
                    period = self.eta_periods[eta_idx]
                    batch_link_move_sp, batch_sum_time = self.period2batch[period]
                    while batch_sum_time.shape[0] == 0 or batch_link_move_sp.indices.shape[0] == 0:
                        b_sidx = (b_sidx + 1)%all_order_num
                        sidx += 1
                        eta_idx = self.order[b_sidx]
                        period = self.eta_periods[eta_idx]
                        batch_link_move_sp, batch_sum_time = self.period2batch[period]
                    eta_idx_list.append(self.order[b_sidx])
                    b_sidx = (b_sidx + 1)%all_order_num
                    sidx += 1
                batch_samples = []
                for eta_idx in eta_idx_list:
                    period = self.eta_periods[eta_idx]
                    if self.seq_len == 1:  # baseline
                        batch_fes = np.reshape(self.dynamic_fes[self.fe_period2idx[period - 1]],
                                            newshape=self.num_of_links)  # [link_num, 1]->link_num
                    else:
                        # for having recent dynamic data
                        if self.seq_len:
                            if period - self.seq_len not in self.fe_period2idx:
                                continue
                            fe_idxs_recent = map(lambda x: self.fe_period2idx[x], range(period - self.seq_len, period))
                        if self.days:
                            # for having few days ago dynamic data: future information
                            fe_idxs_days = map(lambda x: self.fe_period2idx[x], range(period - oneday_periods*self.days, period, oneday_periods))
                        if self.weeks:
                            # for having few weeks ago dynamic data
                            fe_idxs_weeks = map(lambda x: self.fe_period2idx[x], range(period - oneday_periods*7*self.weeks, period, oneday_periods*7))
                        if self.weeks and self.days:
                            if self.seq_len:
                                fe_idxs = fe_idxs_weeks + fe_idxs_days + fe_idxs_recent
                            else:
                                fe_idxs = fe_idxs_weeks + fe_idxs_days
                        elif self.weeks:
                            fe_idxs = fe_idxs_weeks
                            if self.seq_len:
                                fe_idxs += fe_idxs_recent
                        elif self.days:
                            fe_idxs = fe_idxs_days
                            if self.seq_len:
                                fe_idxs += fe_idxs_recent
                        elif self.seq_len:
                            fe_idxs = fe_idxs_recent
                        else:
                            raise Exception("No historical information.")
                        batch_fes = np.concatenate([self.dynamic_fes[fe_idxs], self.static_fes],
                                                axis=2)  # [4+4+4, link_num, feature_dim]
                    batch_link_move_sp, batch_sum_time = self.period2batch[period]
                    batch_samples.append((batch_fes, batch_link_move_sp, batch_sum_time))
                yield batch_samples
        if self.shuffle:
            random.shuffle(self.order)
        return _wrapper()


def load_dataset(config, used_days=4, used_weeks=4, days=4, weeks=4):
    base_dir = config['data'].get('dataset_dir')
    dataset_dir = config['data'].get('dataset_dir')
    static_dim = config['data'].get('static_dim')
    dynamic_dim = config['data'].get('dynamic_dim')
    method = config['model'].get('method')
    per_period = config['data'].get('per_period', 5)
    seq_len = 1 if method == 'baseline' else config['model'].get('seq_len')
    data = {}  # to return
    tprint("Loading Dataset: " + dataset_dir)
    # loading node features
    info = np.load(os.path.join(base_dir, 'link_info.npz'))['link_info']
    data['link_length'] = info[:, 0] * 1000  # the length of road segments: km -> m
    # static feature normlize
    scaler0 = StaticFeatureScaler()
    static_fes = scaler0.transform(info)[:, 0:static_dim]
    print("static_fes.shape=", static_fes.shape)

    dynamic_fes = np_load(os.path.join(dataset_dir, 'dynamic_fes.npz'))
    eta_label = np_load(os.path.join(dataset_dir, 'eta_label.npz'))  # each row is a list, including list(link_idxs, link_move, timespent)
    # dynamic feature normlize
    fes, fe_periods = dynamic_fes['fes'], dynamic_fes['periods']
    scale_fes = fes[fe_periods < min(eta_label['valid_periods'])] # All samples before the valid periods can be used to scale
    scaler1 = StandardScaler(mean=scale_fes.mean(), std=scale_fes.std())
    scaler1.save(os.path.join(dataset_dir, 'scaler1.npz'))
    if method == 'baseline':
        dynamic_fes0 = fes
    else:
        dynamic_fes0 = scaler1.transform(fes)
    for prefix in ('train', 'valid', 'test'):
        tprint("(%s)dynamic_fes.shape=%s" % (prefix, str(dynamic_fes0.shape)))
        assert (dynamic_fes0.shape[-1] == dynamic_dim)
        data['%s_loader' % prefix] = DataLoader(dynamic_fes=dynamic_fes0,
                                                static_fes=static_fes,
                                                links_time_list=eta_label[prefix].tolist(),
                                                seq_len=seq_len,
                                                days=used_days,
                                                weeks=used_weeks,
                                                fe_periods=dynamic_fes['periods'],
                                                eta_periods=eta_label['%s_periods' % prefix],
                                                shuffle=('train' == prefix),
                                                per_period=per_period)
    tprint('Dataset loaded successfully.')
    return data
