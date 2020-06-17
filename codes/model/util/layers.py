# coding=utf-8
import tensorflow as tf
from tensorflow.python.ops.sparse_ops import KeywordRequired

FLAGS = tf.app.flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res

class Layer(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        layer = self.__class__.__name__.lower()
        name = kwargs.get('name')
        if not name:
            name = layer + '_' + str(get_layer_uid(layer))
        else:
            name = layer + '_' + name
        self.name = name
        self.vars = {}
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            return self._call(inputs)
    
    def add_to_weights(self):
        for key in self.vars:
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, self.vars[key])

class Dense(Layer):
    """Dense layer."""
    def __init__(self, input_dim, output_dim, dropout,
                 act=tf.nn.relu, bias=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        self.dropout = dropout
        self.act = act
        self.bias = bias

        with tf.variable_scope(self.name + '_vars', reuse=tf.AUTO_REUSE):
            self.vars['weights'] = tf.get_variable('weight', shape=[input_dim, output_dim], dtype=tf.float32)
            if self.bias:
                self.vars['bias'] = tf.get_variable('bias', shape=[output_dim], initializer=tf.zeros_initializer(), dtype=tf.float32)
        self.add_to_weights()

    def _call(self, inputs):
        """
        :param inputs [-1, input_dim]
        :return output [-1, output_dim]
        """
        x = tf.nn.dropout(inputs, keep_prob=1-self.dropout)
        # transform
        output = dot(x, self.vars['weights'])
        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)

class GraphConvolution(Layer):
    """Graph convolution layer.(Diffusion Convolution只需要选取不一样的support)"""
    def __init__(self, input_dim, output_dim, supports, dropout, agg='concat',
                 act=None, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        self.dropout = dropout # helper variable for sparse dropout
        self.act = act if act else (lambda x: x)
        self.supports = supports
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.agg = agg
        with tf.variable_scope(self.name + '_vars', reuse=tf.AUTO_REUSE):
            for i in range(len(self.supports)):
                self.vars['weights_' + str(i)] = tf.get_variable('weights_' + str(i), shape=[input_dim, output_dim], dtype=tf.float32)
        self.add_to_weights()
    def _call(self, inputs):
        """
        :param inputs [batch_size, num_nodes, input_dim]
        :return output [batch_size, num_nodes, output_dim]  (agg='add')
            or [batch_size, num_nodes, output_dim*K] (agg='concat')
        """
        assert (len(inputs.shape) == 3)
        batch_size, num_nodes, input_dim = inputs.shape
        assert (input_dim == self.input_dim)
        x = tf.nn.dropout(inputs, keep_prob=1-self.dropout)
        x = tf.reshape(x, shape=[batch_size * num_nodes, input_dim])
        x_gconvs = list()
        for i in range(len(self.supports)):
            pre_conv = dot(x, self.vars['weights_' + str(i)])
            pre_conv = tf.reshape(pre_conv, shape=[batch_size, num_nodes, self.output_dim])
            pre_conv = tf.transpose(pre_conv, perm=[1, 2, 0]) # [num_nodes, self.output_dim, batch_size]
            pre_conv = tf.reshape(pre_conv, shape=[num_nodes, -1]) # [num_nodes, self.output_dim * batch_size]
            x_gconv = dot(self.supports[i], pre_conv, sparse=True)
            x_gconv = tf.reshape(x_gconv, shape=[num_nodes, self.output_dim, batch_size])
            x_gconv = tf.transpose(x_gconv, [2, 0, 1])  # [batch_size, num_nodes, output_dim]
            x_gconvs.append(x_gconv)
        if self.agg == 'concat':
            output = tf.concat(x_gconvs, axis=-1)
        else:
            output = tf.add_n(x_gconvs)
        return self.act(output)


class AttentionHeadCheb(Layer):
    """Attention head for GAT"""
    def __init__(self, input_dim, output_dim, supports, atten_supports, kernels_num, activation=tf.nn.elu, in_drop=0.0, l2reg_rate=0.001, is_residual=False, **kwargs):
        super(AttentionHeadCheb, self).__init__(**kwargs)
        self.in_drop = in_drop
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.supports = supports
        self.atten_supports = atten_supports
        self.l2reg_rate = l2reg_rate
        self.is_residual = is_residual
        self.activation = activation
        self.kernels_num = kernels_num
        self.build()
        self.add_to_weights()

    def build(self):
        with tf.variable_scope(self.name + '_vars', reuse=tf.AUTO_REUSE):
            for i in range(len(self.supports)):
                self.vars['weights_transform_' + str(i)] = tf.get_variable('weights_transform_' + str(i), shape=[self.input_dim, self.output_dim], dtype=tf.float32)
                # for compute attention score
                for j in range(len(self.atten_supports)+1):
                    self.vars['weights_left_' + str(i) + str(j)] = tf.get_variable('weights_left_' + str(i) + str(j), shape=[self.output_dim, 1], dtype=tf.float32)
                    self.vars["weights_right_" + str(i) + str(j)] = tf.get_variable('weights_right_'+str(i) + str(j), shape=[self.output_dim, 1], dtype=tf.float32)
            if self.is_residual:
                self.vars["weights_residual"] = tf.get_variable("weights_residual", shape=[self.input_dim+self.output_dim, self.output_dim], dtype=tf.float32)

    def sparse_hadamard_product(self, sparse_matrix, a_vector, is_transpose=False):
        idx = 0
        if is_transpose:
            idx = 1
        index = sparse_matrix.indices[...,idx]
        edges = tf.gather(a_vector, index)
        edges = edges * sparse_matrix.values
        return edges

    def _call(self, inputs):
        x = inputs
        supports_num = len(self.atten_supports)
        conv_layer_num = len(self.supports)
        if self.in_drop:
            x = tf.nn.dropout(x, keep_prob=1-self.in_drop)
        output_list = []
        for k in range(conv_layer_num):
            edges_support_scores = None
            wx = dot(x, self.vars["weights_transform_" + str(k)], sparse=False)
            high_order_supports = self.supports[k]
            for i in range(supports_num):
                a_left = dot(wx, self.vars["weights_left_" + str(k) + str(i)], sparse=False)  # batch_size * num_nodes*1
                a_right = dot(wx, self.vars["weights_right_" + str(k) + str(i)], sparse=False)  # batch_size * num_nodes*1
                # edges_left = self.supports[i] * a_left
                # edges_right = self.supports[i] * tf.transpose(a_right, perm=[1, 0])
                # edges_atten = tf.sparse_add(edges_left, edges_right)
                #### the above operations is Executing in CPU, to speed up:
                a_left = tf.reshape(a_left, shape=(-1,))
                a_right = tf.reshape(a_right, shape=(-1,))
                edges_left = self.sparse_hadamard_product(self.atten_supports[i], a_left, is_transpose=False)
                edges_right = self.sparse_hadamard_product(self.atten_supports[i], a_right, is_transpose=True)
                edges_atten_values = edges_left + edges_right
                edges_atten = tf.SparseTensor(indices=self.atten_supports[i].indices, values=edges_atten_values, dense_shape=self.atten_supports[i].dense_shape)
                if i==0:
                    edges_support_scores = edges_atten
                else:
                    edges_support_scores = tf.sparse_add(edges_support_scores, edges_atten)
            # for high-order neighbors
            a_left = dot(wx, self.vars["weights_left_" + str(k) + str(supports_num)], sparse=False)  # batch_size * num_nodes*1
            a_right = dot(wx, self.vars["weights_right_" + str(k) + str(supports_num)], sparse=False)  # batch_size * num_nodes*1
            a_left = tf.reshape(a_left, shape=(-1,))
            a_right = tf.reshape(a_right, shape=(-1,))
            edges_left = self.sparse_hadamard_product(self.supports[k], a_left, is_transpose=False)
            edges_right = self.sparse_hadamard_product(self.supports[k], a_right, is_transpose=True)
            edges_atten_values = edges_left + edges_right
            edges_atten = tf.SparseTensor(indices=self.supports[k].indices, values=edges_atten_values, dense_shape=self.supports[k].dense_shape)
            all_order_scores = tf.sparse_add(edges_atten, edges_support_scores)
            edges_prob = tf.sparse_softmax(all_order_scores)
            output_k = dot(edges_prob, wx, sparse=True)
            output_list.append(output_k)
        output = tf.add_n(output_list)
        if self.is_residual:
            output_concat = tf.concat([inputs, output], axis=-1)
            output = dot(output_concat, self.vars["weights_residual"])
        return self.activation(output)


class MultiAttentionCheb(Layer):
    "multi-heads attention for GAT"
    def __init__(self, input_dim, output_dim, supports, atten_supports, kernels, heads_num, is_concat=True, activation=tf.nn.elu, in_drop=0., l2reg_rate=0.001, is_residual=False, **kwargs):
        super(MultiAttentionCheb, self).__init__(**kwargs)
        self.in_drop = in_drop
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.supports = supports
        self.atten_supports = atten_supports
        self.l2reg_rate = l2reg_rate
        self.is_residual = is_residual
        self.activation = activation
        self.heads_num = heads_num
        self.is_concat = is_concat
        self.kernels = kernels
        self.build()

    def build(self):
        with tf.variable_scope(self.name + '_vars', reuse=tf.AUTO_REUSE):
            for i in range(self.heads_num):
                params = {"name": str(i)}
                self.vars["heads_" + str(i)] = AttentionHeadCheb(input_dim=self.input_dim,
                                                            output_dim=self.output_dim,
                                                            supports=self.supports,
                                                            atten_supports=self.atten_supports,
                                                            kernels_num=self.kernels,
                                                            activation=self.activation,
                                                            in_drop=self.in_drop,
                                                            l2reg_rate=self.l2reg_rate,
                                                            is_residual=self.is_residual,**params)
    def _call(self, inputs):
        x = inputs
        head_results = []
        for i in range(self.heads_num):
            head = self.vars["heads_" + str(i)]
            re = head(x)
            head_results.append(re)
        if self.is_concat:
            output = tf.concat(head_results, axis=-1)  # dim = heads_num * output_dim
        else:
            output = tf.add_n(head_results) / self.heads_num  # dim = output_dim
        return output

