# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from model.util.layers import Layer, MultiAttentionCheb


class NormLayer(Layer):
    def __init__(self, n, dim, **kwargs):
        super(NormLayer, self).__init__(**kwargs)
        self.n = n
        self.dim = dim
        self.gamma = tf.get_variable('gamma', initializer=tf.ones([1, 1, n, dim]), dtype=tf.float32)
        self.beta = tf.get_variable('beta', initializer=tf.zeros([1, 1, n, dim]), dtype=tf.float32)

    def _call(self, inputs):
        """
        :param inputs [batch_size, seq_len, num_nodes, dim].
        :return outputs [batch_size, seq_len, num_nodes, dim]
        """
        _, _, n, dim = inputs.get_shape().as_list()
        assert (self.n == n)
        assert (self.dim == dim)
        mu, sigma = tf.nn.moments(inputs, axes=[2, 3], keep_dims=True)
        return (inputs - mu) / tf.sqrt(sigma + 1e-6) * self.gamma + self.beta


class TemporalConvLayer(Layer):
    def __init__(self, Kt, input_dim, output_dim, act_func='relu', **kwargs):
        """
        :param Kt: int, kernel size of temporal convolution.
        """
        super(TemporalConvLayer, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.Kt = Kt
        self.act_func = act_func

        # parameter for alignment
        if input_dim > output_dim:
            self.w_input = tf.get_variable('wt_input', shape=[1, 1, input_dim, output_dim], dtype=tf.float32) #[filter_height, filter_width, in_channels, out_channels]
            tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(self.w_input))
        if act_func == 'GLU':  # parameters for GLU
            self.wt = tf.get_variable(name='wt', shape=[Kt, 1, input_dim, 2 * output_dim], dtype=tf.float32)
            tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(self.wt))
            self.bt = tf.get_variable(name='bt', initializer=tf.zeros([2 * output_dim]), dtype=tf.float32)
        else:  # linear/relu/sigmoid
            self.wt = tf.get_variable(name='wt', shape=[Kt, 1, input_dim, output_dim], dtype=tf.float32)
            tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(self.wt))
            self.bt = tf.get_variable(name='bt', initializer=tf.zeros([output_dim]), dtype=tf.float32)

    def _call(self, inputs):
        """
        :param inputs [batch_size, seq_len, num_nodes, input_dim].--》[batch_size, in_height, in_width, n_channels]
        :return outputs [batch_size, seq_len-Kt+1, num_nodes, output_dim]
        """
        batch_size, seq_len, n, c_in = inputs.get_shape().as_list()
        assert (c_in == self.input_dim)
        # align to output_dim
        if self.input_dim > self.output_dim:
            x_input = tf.nn.conv2d(inputs, self.w_input, strides=[1, 1, 1, 1], padding='SAME')
        elif self.input_dim < self.output_dim:
            x_input = tf.concat([inputs, tf.zeros([batch_size, seq_len, n, self.output_dim - self.input_dim])], axis=3)
        else:
            x_input = inputs
        # keep the original input for residual connection.
        x_input = x_input[:, self.Kt - 1:seq_len, :, :]
        # x_conv -> [batch_size, seq_len-Kt+1, n, output_dim]
        x_conv = tf.nn.conv2d(inputs, self.wt, strides=[1, 1, 1, 1], padding='VALID') + self.bt
        if self.act_func == 'GLU':
            g = tf.nn.sigmoid(x_conv[:, :, :, -self.output_dim:])  # gate rate
            return (x_conv[:, :, :, 0:self.output_dim] + x_input) * g
        elif self.act_func == 'relu':
            return tf.nn.relu(x_conv + x_input)
        elif self.act_func == 'sigmoid':
            return tf.nn.sigmoid(x_conv)
        elif self.act_func == 'linear':
            return x_conv
        else:
            raise ValueError('ERROR: activation function "%s" is not defined.' % self.act_func)


class SpatioConvLayerCheb(Layer):
    def __init__(self, Ks, input_dim, output_dim, supports, atten_supports, heads_num, **kwargs):
        """
        :param Ks: int, kernel size of spatial convolution.
        :param kernel: Cheb Poly kernels, List of sparse [num_nodes, num_nodes], Ks elements
        """
        super(SpatioConvLayerCheb, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.Ks = Ks
        self.atten_conv = MultiAttentionCheb(input_dim, output_dim, supports, atten_supports, kernels=self.Ks, heads_num=heads_num, is_concat=False, activation=lambda x:x, is_residual=False, **kwargs)
        # parameter for alignment
        if input_dim > output_dim:  # bottleneck down-sampling
            self.w_input = tf.get_variable(self.name+'_ws_input', shape=[1, 1, input_dim, output_dim], dtype=tf.float32)
            tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(self.w_input))

    def _call(self, inputs):
        """
        :param inputs [batch_size, seq_len, num_nodes, input_dim].
        :return outputs [batch_size, seq_len, num_nodes, output_dim]
        """
        batch_size, seq_len, n, c_in = inputs.get_shape().as_list()
        assert (c_in == self.input_dim)
        # align to output_dim
        if self.input_dim > self.output_dim:
            x_input = tf.nn.conv2d(inputs, self.w_input, strides=[1, 1, 1, 1], padding='SAME')
        elif self.input_dim < self.output_dim:
            x_input = tf.concat([inputs, tf.zeros([batch_size, seq_len, n, self.output_dim - self.input_dim])], axis=3)
        else:
            x_input = inputs
        # Graph Convolution
        x = tf.reshape(inputs, [batch_size*seq_len, n, self.input_dim])  # -> [batch_size*seq_len, num_nodes, input_dim]
        batch_gconv = []
        for b in range(batch_size*seq_len):
            x_gconv = self.atten_conv(x[b,...]) #[num_nodes, output_dim]
            batch_gconv.append(x_gconv)
        x_gc = tf.stack(batch_gconv) # batch_size*seq_len, num_nodes, output_dim]
        x_gc = tf.reshape(x_gc, [batch_size, seq_len, n, self.output_dim])
        return tf.nn.relu(x_gc + x_input)

class STConvBlock(Layer):
    def __init__(self, Ks, Kt, num_nodes, supports, atten_supports, road_num, car_num, scope, channels, dropout, heads_num, act_func='GLU', **kwargs):
        """
        Spatio-temporal convolutional block, which contains two temporal gated convolution layers
        and one spatial graph convolution layer in the middle.
        :param Ks: int, kernel size of spatial convolution.
        :param Kt: int, kernel size of temporal convolution.
        :param supports: Cheb Poly kernels, List of sparse [num_nodes, num_nodes], Ks elements
        :param channels: list of 3 elemetents, dims in a single st_conv block.
        :param scope: str, variable scope.
        """
        super(STConvBlock, self).__init__(**kwargs)
        self.road_num = road_num
        self.car_num = car_num
        Het_supports = {"road":0, "car":Ks+1}
        if road_num == 0:
            Het_supports["car"] = 0
        self.hidden_num = self.get_hidden_num(road_num, car_num)
        with tf.variable_scope('stn_block_%s_in' % scope, reuse=tf.AUTO_REUSE):
            self.temporal_layer1 = TemporalConvLayer(Kt, channels[0], channels[1], act_func=act_func)
            if self.road_num:
                parms = {"name": "road"}
                self.road_spatio_layer = SpatioConvLayerCheb(Ks, channels[1], channels[1], supports[0:Ks+1], atten_supports[0:self.road_num*2], heads_num, **parms)
            if self.car_num:
                parms = {"name": "car"}
                self.car_spatio_layer = SpatioConvLayerCheb(Ks, channels[1], channels[1], supports[Het_supports["car"]:], atten_supports[road_num*2:], heads_num, **parms)
        with tf.variable_scope('stn_block_%s_out' % scope, reuse=tf.AUTO_REUSE):
            self.temporal_layer2 = TemporalConvLayer(Kt, self.hidden_num*channels[1], channels[2])
        with tf.variable_scope('layer_norm_%s' % scope, reuse=tf.AUTO_REUSE):
            self.norm_layer = NormLayer(num_nodes, channels[2])
        self.dropout = dropout
    
    def get_hidden_num(self, road_num, car_num):
        types = 0
        if road_num:
            types += 1
        if car_num:
            types += 1
        return types

    def _call(self, inputs):
        """
        :param inputs [batch_size, seq_len, num_nodes, input_dim].
        :return output [batch_size, seq_len-2(Kt-1), num_nodes, output_dim]
        """
        x = self.temporal_layer1(inputs) # [batch_size, seq_len-Kt+1, num_nodes, output_dim1]
        if self.road_num:
            x1 = self.road_spatio_layer(x) # [batch_size, seq_len-Kt+1, num_nodes, output_dim1]
        if self.car_num:
            x2 = self.car_spatio_layer(x) # [batch_size, seq_len-Kt+1, num_nodes, output_dim1]
        if self.road_num and self.car_num:
            x = tf.concat([x1, x2], axis=-1)
        elif self.road_num:
            x = x1
        elif self.car_num:
            x = x2
        else:
            raise ValueError('ERROR: neither road_net nor car_net.')
        x = self.temporal_layer2(x) # [batch_size, seq_len-2(Kt-1), num_nodes, output_dim2]
        x = self.norm_layer(x)
        return tf.nn.dropout(x, 1 - self.dropout)



class STLastLayer(Layer):
    def __init__(self, seq_len, n, dim, act_func='GLU', scope='last_layer', **kwargs):
        super(STLastLayer, self).__init__(**kwargs)
        self.n = n
        self.dim = dim
        with tf.variable_scope('%s_in' % scope, reuse=tf.AUTO_REUSE):
            self.temporal_layer1 = TemporalConvLayer(seq_len, dim, dim, act_func=act_func)
        with tf.variable_scope('layer_norm_%s' % scope, reuse=tf.AUTO_REUSE):
            self.norm_layer = NormLayer(n, dim)

    def _call(self, inputs):
        """
        :param inputs [batch_size, seq_len, num_nodes, input_dim].
        :return output [batch_size, 1, num_nodes, 1]
        """
        _, _, n, dim = inputs.get_shape().as_list()
        assert (self.n == n)
        assert (self.dim == dim)
        x = self.temporal_layer1(inputs)
        x = self.norm_layer(x)
        return x

class STOutputEmbeddingLayer(Layer):
    def __init__(self, seq_len, n, dim, act_func='GLU', act_func2='sigmoid', **kwargs):
        super(STOutputEmbeddingLayer, self).__init__(**kwargs)
        self.n = n
        self.dim = dim
        scope = 'output_embedding_layer'
        with tf.variable_scope('%s_in' % scope, reuse=tf.AUTO_REUSE):
            self.temporal_layer1 = TemporalConvLayer(seq_len, dim, dim, act_func=act_func)
        with tf.variable_scope('layer_norm_%s' % scope, reuse=tf.AUTO_REUSE):
            self.norm_layer = NormLayer(n, dim)
        with tf.variable_scope('%s_out' % scope, reuse=tf.AUTO_REUSE):
            self.temporal_layer2 = TemporalConvLayer(1, dim, dim, act_func=act_func2)

    def _call(self, inputs):
        """
        :param inputs [batch_size, seq_len, num_nodes, input_dim].
        :return output [batch_size, 1, num_nodes, input_dim]
        """
        _, _, n, dim = inputs.get_shape().as_list()
        assert (self.n == n)
        assert (self.dim == dim)
        x = self.temporal_layer1(inputs)
        x = self.norm_layer(x)
        x = self.temporal_layer2(x)
        return x

class STPredictLayer(Layer):
    def __init__(self, n, dim, hidden_dim, **kwargs):
        super(STPredictLayer, self).__init__(**kwargs)
        self.n = n
        self.dim = dim
        scope = 'predict_layer'
        with tf.variable_scope('%s_out' % scope, reuse=tf.AUTO_REUSE):
            self.temporal_layer2 = TemporalConvLayer(1, dim, hidden_dim, act_func='sigmoid')
        self.w = tf.get_variable(name='w_%s' % scope, shape=[1, 1, hidden_dim, 1], dtype=tf.float32)
        tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(self.w))
        self.b = tf.get_variable(name='b_%s' % scope, initializer=tf.zeros([n, 1]), dtype=tf.float32)

    def _call(self, inputs):
        """
        :param inputs [batch_size, seq_len=1, num_nodes, input_dim].
        :return (last_state, output)
                last_state [batch_size, 1, num_nodes, hidden_dim]
                output [batch_size, 1, num_nodes, 1]
        """
        _, _, n, dim = inputs.get_shape().as_list()
        assert (self.n == n)
        assert (self.dim == dim)
        last_state = self.temporal_layer2(inputs)
        x = tf.nn.conv2d(last_state, self.w, strides=[1, 1, 1, 1], padding='SAME') + self.b
        return last_state, x