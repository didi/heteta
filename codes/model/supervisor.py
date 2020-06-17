# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import random

from util.AMSGrad import AMSGrad
from util.utils import tprint, clear_dir
from util.support import *
from model.HetETA_model import HetETAModel


class Supervisor(object):
    """
    Do experiments using Graph Random Walk RNN model.
    """
    def __init__(self, data, adj_mx, config):
        self._num_gpu = config['train'].get('gpu_num', 4)
        self._data = data
        self._dropout = config['train']['dropout']
        self._base_lr = config['train']['base_lr']
        self._lr_decay_ratio = config['train']['lr_decay_ratio']
        self._lr_decay_epoch = config['train']['lr_decay_epoch']
        self._epochs = config['train']['epochs']
        self._model_file = config['model'].get('model_file', '')
        self._model_dir = config['model']['model_dir']
        self._patience = config['train']['patience']
        self._method = config['model']['method']
        self._Skernel_layer = config['model']['max_diffusion_step']
        self._supports, self._atten_supports = self.load_adj(adj_mx, self._method)
        self._max_grad_norm = config['train']['max_grad_norm']
        clear_dir(self._model_dir+"/")
        # reduce model on cpu
        self.max_to_keep = 1
        # Learning rate.
        global_step = tf.train.get_or_create_global_step()
        self._lr = tf.train.exponential_decay(self._base_lr, global_step,
                                           int(288*self._lr_decay_epoch/self._num_gpu), 1-self._lr_decay_ratio, staircase=True)
        # Configure optimizer
        optimizer_name = config['train']['optimizer'].lower()
        epsilon = float(config['train']['epsilon'])
        self.optimizer = self.get_optimizer(optimizer_name, epsilon)
        
        # build models
        tprint("Building %s Model on GPU tower..." % self._method)
        self.models = []
        for gpu_id in range(self._num_gpu):
            with tf.device('/gpu:%d' % gpu_id):
                print('\t Initing tower: %d...'% gpu_id)
                with tf.name_scope('tower_%d' % gpu_id):
                    a_model = self.get_model(config, len(self._supports), reuse=tf.AUTO_REUSE)
                    self.models.append(a_model)
        tprint("%s Model GPUs inited." % self._method)


        tvars = tf.trainable_variables()
        tower_grads = []
        for m in self.models:
            grad = m.grads(tvars)
            tower_grads.append(grad)
        tower_batch_sizes = [m.batch_num for m in self.models]
        aver_grads = self.average_gradients(tower_grads, tower_batch_sizes)
        self.apply_gradients_ops = self.optimizer.apply_gradients(zip(aver_grads, tvars), global_step=global_step, name='train_op')
        self._saver = tf.train.Saver(tf.global_variables(), max_to_keep=self.max_to_keep)
        tprint("%s Model CPU inited." % self._method)
    
    def load_adj(self, adj, method="HetETA"):
        atten_supports = None
        if method in ["HetETA"]:
            supports, atten_supports = get_model_cheb_support(adj=adj, K=self._Skernel_layer+1)
            print("the num of supports", len(supports))
            print("the num of atten_supports", len(atten_supports))
        else:
            supports = []
        return supports, atten_supports
    
    def get_optimizer(self, optimizer_name, epsilon):
        if optimizer_name == 'adam':
            optimizer = tf.train.AdamOptimizer(self._lr, epsilon=epsilon)
        elif optimizer_name == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(self._lr)
        elif optimizer_name == 'amsgrad':
            optimizer = AMSGrad(self._lr, epsilon=epsilon)
        elif optimizer_name == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(self._lr, epsilon=epsilon)
        else:
            raise Exception("unknown opt type")
        return optimizer
    
    def average_gradients(self, tower_grads, tower_batch_sizes):
        average_grads = []
        batch_sizes = tf.stack(tower_batch_sizes, axis=0)
        batch_sizes = tf.cast(tf.reshape(tower_batch_sizes, (-1,)), dtype=tf.float32)
        all_sizes = tf.reduce_sum(batch_sizes)
        batch_sizes = batch_sizes / all_sizes
        for grads in zip(*tower_grads):
            # Average over the 'tower' dimension.
            weighted_grads = []
            for i in range(len(grads)):
                weighted_grads.append(grads[i]*batch_sizes[i])
            grad = tf.reduce_sum(tf.stack(weighted_grads, axis=0), axis=0)
            average_grads.append(grad)
        return average_grads

    def get_model(self, config, supports_num, reuse):
        cpu_supports = None
        with tf.variable_scope('MODEL', reuse=reuse):
            if self._method == 'HetETA':
                if self._atten_supports:
                    model = HetETAModel(config, supports_num, len(self._atten_supports), cpu_supports, self.optimizer)
                else:
                    model = HetETAModel(config, supports_num)
            else:
                raise Exception("Can not find model of method " + self._method)
        return model
    
    def feed_all_gpu(self, inp_dict, models, batch_inputs):
        gpus = len(models)
        assert gpus == len(batch_inputs)
        for i in range(gpus):
            batch_fes, batch_link_move_sp, batch_sum_time = batch_inputs[i]
            inp_dict.update({
                models[i].placeholders['features']: batch_fes,
                models[i].placeholders['batch_link_move_sp']: batch_link_move_sp,
                models[i].placeholders['batch_sum_time']: batch_sum_time
            })
        return inp_dict

    def run_epoch_generator(self, sess, models, data_generator, mode='train'):
        assert mode in ('train', 'test', 'valid')
        rates = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        rates_num = len(rates)
        gpus_num = len(models)
        # setting fetches
        fetches = {}
        for i in range(gpus_num):
            fetches['loss_%d'%i] = models[i].loss
            fetches['mape_%d'%i] = models[i].mape
            fetches['mae_%d'%i] = models[i].mae
            fetches['mse_%d'%i] = models[i].mse
            fetches['badcase_rate_%d'%i] = models[i].badcase_rate
        if mode == 'train':
            fetches['train_op'] = self.apply_gradients_ops
        # setting numpy values
        values_dict = {"loss":0., "mape":0., "mae":0., "mse":0., "badcase_rate": [0.]*rates_num, "order_num":0}
        sum_orders = 0.0
        #setting feed_dict
        inp_dict = {}
        for i in range(gpus_num):
            inp_dict.update({models[i].placeholders['dropout']: self._dropout})
            inp_dict.update({models[i].placeholders['supports'][j]: self._supports[j] for j in range(len(self._supports))})
            if self._atten_supports:
                inp_dict.update({models[i].placeholders['atten_supports'][j]: self._atten_supports[j] for j in range(len(self._atten_supports))})

        # training/testing
        for i, batch_samples in enumerate(data_generator):
            # Note that each samples in batch_samples is:
            # (batch_fes, batch_link_move_sp, batch_sum_time)
            batch_order_sizes = [s[2].shape[0] for s in batch_samples]
            # print("batch_order_sizes:", batch_order_sizes)
            inp_dict = self.feed_all_gpu(inp_dict=inp_dict, models=self.models, batch_inputs=batch_samples)
            values = sess.run(fetches, feed_dict=inp_dict)
            for gpu in range(gpus_num):
                for key in values_dict:
                    if key == "order_num":
                        continue
                    if key != "badcase_rate":
                        values_dict[key] += values[key+"_%d"%gpu].item() * batch_order_sizes[gpu]
                    else:
                        bad_case_rates = values[key+"_%d"%gpu]
                        for r in range(rates_num):
                            values_dict[key][r] += bad_case_rates[r] * batch_order_sizes[gpu]
                sum_orders += batch_order_sizes[gpu]
        for key in values_dict:
            if key == "order_num":
                values_dict[key] = sum_orders
            elif key != "badcase_rate":
                values_dict[key] /= sum_orders
        if "mse" in values_dict:
            values_dict["rmse"] = np.sqrt(values_dict["mse"])
        return values_dict

    def get_lr(self, sess):
        return np.asscalar(sess.run(self._lr))

    def set_lr(self, sess, lr):
        sess.run(self._lr_update, feed_dict={self._new_lr: lr})
    
    def get_support_weights(self, sess):
        fetches = {}
        for s in range(len(self._supports)):
            fetches["weights_supports_" + str(s)] = self._model.vars["weights_supports_" + str(s)]
        values = sess.run(fetches)
        return values

    def train(self, sess):
        min_val_loss = float('inf')
        last_eval_loss = 100.0
        best_model = ""
        wait = 0
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=self.max_to_keep)

        if self._model_file not in (None, ''):
            saver.restore(sess, self._model_file)
        else:
            sess.run(tf.global_variables_initializer())
        tprint('Start training ...')

        for epoch in range(self._epochs):
            cur_lr = sess.run(self._lr)
            start_time = time.time()
            train_values = self.run_epoch_generator(sess, self.models,
                                                  self._data['train_loader'].get_iterator(batch_num=self._num_gpu),
                                                  mode='train')
            if train_values["loss"] > 1e5:
                tprint('Gradient explosion detected. Ending...')
                break

            global_step = sess.run(tf.train.get_or_create_global_step())
            # Compute validation error.
            val_values = self.run_epoch_generator(sess, self.models,
                                                self._data['valid_loader'].get_iterator(batch_num=self._num_gpu),
                                                mode='valid')
            end_time = time.time()
            tprint('Epoch [{}/{}] ({}) train_loss: {:.4f}, valid_loss: {:.4f} lr:{:.6f} {:.1f}s'.format(
                epoch, self._epochs, global_step, train_values["loss"], val_values["loss"], cur_lr, (end_time - start_time)))
            tprint("train_mape: {:.2f}%, valid_mape: {:.2f}%, train_mae: {:.4f}, valid_mae: {:.4f}, train_rmse: {:.4f}, valid_rmse: {:.4f}, train_order: {:.1f}, valid_order: {:.1f}.".format(
                train_values["mape"]*100, val_values["mape"]*100, train_values["mae"], val_values["mae"], train_values["rmse"], val_values["rmse"], train_values["order_num"], val_values["order_num"]
            ))
            print("train_badcase: ", train_values["badcase_rate"])
            print("valid_badcase: ", val_values["badcase_rate"])
            if val_values["mape"] <= min_val_loss:
                if min_val_loss - val_values["mape"] < 0.0001 :
                    wait +=1
                    if wait > self._patience:
                        tprint('Early stopping.')
                        break
                else:
                    wait = 0
                tprint('Valid loss decrease from %.4f to %.4f' % (min_val_loss, val_values["mape"]))
                min_val_loss = val_values["mape"]
                best_model = self.save(sess, val_values["mape"])
                tprint('saving to %s' % best_model)
                if last_eval_loss - val_values["mape"] > 0.0005:
                    last_eval_loss = val_values["mape"]
                    self.evaluate(sess)
            else:
                wait += 1
                if wait > self._patience:
                    tprint('Early stopping.')
                    break
            sys.stdout.flush()
        tprint("Training END.")
        self.load(sess, best_model)
        self.evaluate(sess)
        tprint("Best model is: %s" % best_model)
        return min_val_loss

    def evaluate(self, sess):
        test_values = self.run_epoch_generator(sess, self.models,
                                             self._data['test_loader'].get_iterator(batch_num=self._num_gpu),
                                             mode='test')
        tprint('=> Test loss=%.4f' % test_values["loss"])
        tprint("test_mape: {:.2f}%, test_mae: {:.4f}, test_rmse: {:.4f}, test_order: {:.1f}.".format(
                test_values["mape"]*100, test_values["mae"],test_values["rmse"],test_values["order_num"]
            ))
        print("Test badcase: ", test_values["badcase_rate"])

    def load(self, sess, model_filename):
        """Restore from saved model."""
        self._saver.restore(sess, model_filename)

    def save(self, sess, val_loss):
        global_step = np.asscalar(sess.run(tf.train.get_or_create_global_step()))
        prefix = os.path.join(self._model_dir, '{}-models-{:.4f}'.format(self._method, val_loss))
        model_filename = self._saver.save(sess, prefix, global_step=global_step, write_meta_graph=True)
        return model_filename
