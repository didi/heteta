# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from model.cell.cheb_layer import STConvBlock, STLastLayer, STPredictLayer

class HetETAModel(object):
    def __init__(self, config, supports_num, atten_supports_num, cpu_supports, optimizer):
        num_nodes = config['model']['num_nodes']
        seq_len = config['model']['seq_len']
        fe_dim = config['data']['dynamic_dim'] + config['data']['static_dim']
        bad_case_thre = config['train'].get('threshold', 5)
        output_dim = config['model']['rnn_units']
        heads_num = config['model']['heads_num']
        regular_rate = config['train'].get('regular_rate', 0.0005)
        spatial_kernel_layer = config['model']['max_diffusion_step']
        days = config['model'].get('days', 4)
        weeks = config['model'].get('weeks', 4)
        road_net_num = config['model'].get('road_net_num', 7)
        car_net_num = config['model'].get('car_net_num', 1)
        self.max_grad_norm = config['train']['max_grad_norm']
        assert (road_net_num + car_net_num)*2 == atten_supports_num
        if road_net_num and car_net_num:
            assert (spatial_kernel_layer+1)*2 == supports_num
        else:
            assert (spatial_kernel_layer+1) == supports_num
        blocks = [[fe_dim, 8, output_dim]]
        Ko, Ks, Kt = {}, {},{}
        Ko["recent"], Ks["recent"], Kt["recent"] = seq_len, spatial_kernel_layer, 2  # kernel output, kernel spatial, kernel temporal
        Ko["days"], Ks["days"], Kt["days"] = days, spatial_kernel_layer, 2
        Ko["weeks"], Ks["weeks"], Kt["weeks"] = weeks, spatial_kernel_layer, 2
        Istart = {"weeks":0, "days":weeks, "recent":weeks+days}
        Iend = {"weeks":weeks, "days":weeks+days, "recent":weeks+days+seq_len}
        input_len = seq_len+days+weeks
        types_list = self.get_type_list(recents=seq_len, weeks=weeks, days=days)

        self._ph = {
            'features': tf.placeholder(tf.float32, shape=(input_len, num_nodes, fe_dim), name='features'),
            'supports': [tf.sparse_placeholder(tf.float32, name='supports_%d' % i) for i in range(supports_num)],
            'atten_supports': [tf.sparse_placeholder(tf.float32, name='atten_supports_%d' % i) for i in range(atten_supports_num)],
            'batch_link_move_sp': tf.sparse_placeholder(tf.float32, shape=(None, num_nodes), name='batch_link_move_sp'),
            'batch_sum_time': tf.placeholder(tf.float32, shape=None, name='batch_sum_time'),
            'dropout': tf.placeholder_with_default(0., shape=None, name='dropout'),
        }

        all_inputs = tf.reshape(self._ph['features'], (1, input_len, num_nodes, fe_dim))
        all_outputs = []
        for time_type in types_list:
            x = all_inputs[:, Istart[time_type]:Iend[time_type], :, :]
            Ko_time_type = Ko[time_type]
            for i, channels in enumerate(blocks):
                st_conv_block = STConvBlock(Ks=Ks[time_type], Kt=Kt[time_type], num_nodes=num_nodes,
                                            supports=self._ph['supports'],
                                            atten_supports=self._ph['atten_supports'],
                                            road_num=road_net_num,
                                            car_num=car_net_num,
                                            scope=time_type+"_"+str(i), channels=channels,
                                            dropout=self._ph['dropout'],
                                            heads_num=heads_num)
                x = st_conv_block(x)  #[batch_size, seq_len-2(Kt-1), num_nodes, output_dim2]
                Ko_time_type -= 2*(Kt[time_type] - 1)  # temporal conv will reduce T dim
            if Ko_time_type > 1:
                st_output_layer = STLastLayer(Ko_time_type, num_nodes, blocks[-1][-1], scope='last_layer_'+time_type)
                x = st_output_layer(x) #[batch_size, 1, num_nodes, output_dim2]
            all_outputs.append(x)
        con_state = tf.concat(all_outputs, axis=-1) #[batch_size, 1, num_nodes, output_dim2*3]
        x = tf.reshape(con_state, [1, 1, num_nodes, blocks[-1][-1]*len(types_list)])  #[batch_size, 1, num_nodes, output_dim2*3]
        st_predict_layer = STPredictLayer(num_nodes, blocks[-1][-1]*len(types_list), blocks[-1][-1])
        last_state, x = st_predict_layer(x)
        output = tf.nn.sigmoid(tf.reshape(x, [num_nodes, 1]))
        pred_link_speed = tf.clip_by_value(output * 120/3.6, 0.1, 120/3.6)  # 120km/h --> m/s
        self._pred_link_speed = tf.transpose(pred_link_speed) # [1, num_nodes]
        pred_sum_time = tf.sparse_reduce_sum(self._ph['batch_link_move_sp'] / self._pred_link_speed, axis=1)
        err = pred_sum_time - self._ph['batch_sum_time']
        batch_diff = tf.abs(err)
        relative_err = batch_diff / self._ph['batch_sum_time']
        #######evaluation###########
        self.mape = tf.reduce_mean(relative_err)
        self.mae = tf.reduce_mean(batch_diff)
        self.mse = tf.reduce_mean(tf.square(err))
        rates = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.badcase_rate = []
        for r in rates:
            badcases = tf.where(relative_err > r)
            badcases_values = tf.gather(batch_diff, badcases)
            badcases_large5 = tf.where(badcases_values >= bad_case_thre*60)
            bad_rate = tf.divide(tf.shape(badcases_large5)[0], tf.shape(err)[0])
            self.badcase_rate.append(bad_rate)
        reg_ws1 = 0.
        var_lists1 = tf.get_collection(tf.GraphKeys.WEIGHTS)
        for var in var_lists1:
            reg_ws1 += tf.nn.l2_loss(var)
        var_lists2 = tf.get_collection("weight_decay")
        reg_ws2 = tf.add_n(var_lists2)
        self.batch_num = tf.shape(err)[0]
        self._loss = self.mape + regular_rate*reg_ws1 + regular_rate*reg_ws2
        #######construct embedding #######
        history_speed = tf.transpose(self._ph['features'][:, :, 0]) # [num_nodes，input_len]
        last_state = tf.reshape(last_state, [num_nodes, -1])
        con_state_reshape = tf.reshape(con_state, [num_nodes, -1])
        self._embedding = tf.concat([history_speed, con_state_reshape, last_state, pred_link_speed],
                                    axis=1, name="emb") # [num_nodes, ?]
        print("embedding shape=%s"+str(self._embedding.shape))
        print("part1=>history speed，shape=" + str(history_speed.shape))
        print("part2=>concat state，shape=" + str(con_state_reshape.shape))
        print("part3=>last state，shape=" + str(last_state.shape))
        print("part4=>final pred speed，shape=" + str(pred_link_speed.shape))

    @property
    def placeholders(self):
        return self._ph

    @property
    def pred_link_speed(self):
        return self._pred_link_speed

    @property
    def loss(self):
        return self._loss

    @property
    def embedding(self):
        return self._embedding

    # @property
    def grads(self, tvars):
        _grads = tf.gradients(self._loss, tvars)
        the_grads, _ = tf.clip_by_global_norm(_grads, self.max_grad_norm)
        return the_grads
    
    def get_type_list(self, recents, weeks, days):
        if recents:
            types = ["recent"]
        else:
            types = []
        if days:
            types += ["days"]
        if weeks:
            types += ["weeks"]
        assert len(types)>0 
        return types
    
    def get_het_list(self, road_net_num, car_net_num):
        types = []
        if road_net_num:
            types.append("road")
        if car_net_num:
            types.append("car")
        return types
