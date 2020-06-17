from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import tensorflow as tf
import yaml
from scipy import sparse
from util.data_loader import load_dataset
from util.utils import tprint
from model.supervisor import Supervisor
import scipy.io as sio


def main(args):
    with open(args.config) as f:
        tprint("Loading conifg file.")
        config = yaml.load(f)
        if args.model_dir != "data/model/HetETA":
            config['model']['model_dir'] = args.model_dir
        if args.dataset_dir != "../dataset/6pk":
            config['data']['dataset_dir'] = args.dataset_dir
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    method = config['model']['method']
    days = config['model'].get('days', 0)
    weeks = config['model'].get('weeks', 0)
    tf.reset_default_graph()
    with tf.Session(config=tf_config) as sess:
        with tf.device('/cpu:0'):
            tprint("Loading dataset.")
            data = load_dataset(config, used_days=days, used_weeks=weeks)
            if method in ["HetETA"]:
                adj_files = ["adj.mat", "adj_gap_top5.mat"]
                adj_mx = {}
                for file in adj_files:
                    adj_path = os.path.join(config['data']['dataset_dir'], file)
                    tprint("Loading adj matrix: " + adj_path)
                    adj_mx_file = sio.loadmat(adj_path)
                    suffix = file[-7:-4]
                    for k in adj_mx_file:
                        if k in ['__header__', '__version__', '__globals__']:
                            continue
                        if k not in adj_mx:
                            adj_mx[k+suffix] = adj_mx_file[k]
                        else:
                            raise Exception("There is already a %s adj matrix"%k)
                supervisor = Supervisor(data, adj_mx, config)
                supervisor.train(sess=sess)
            else:
                raise Exception("unk method " + method)
    tprint('end')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="data/config.yaml", type=str,
                        help='Configuration filename.')
    parser.add_argument('--model_dir', default="data/model/HetETA", type=str,
                        help='Save model filename.')
    parser.add_argument('--dataset_dir', default="../dataset/6pk", type=str,
                        help='Dataset filename.')
    main(parser.parse_args())