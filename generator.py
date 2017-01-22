#! -*- coding:utf-8 -*-

import os
import sys
import numpy as np
import tensorflow as tf

from util import get_weights, get_biases
from batch_normalize import batch_norm

def deconv_layer(inputs, out_shape, filter_width, filter_hight, stride, l_id):
    # ** NOTICE: weight shape is [hight, width, out_chanel, in_chanel] **
    weights = get_weights(l_id,
                          [filter_hight, filter_width, out_shape[-1], inputs.get_shape()[-1]],
                          0.1)
    
    biases = get_biases(l_id, [out_shape[-1]], 0.0)
    
    deconved = tf.nn.conv2d_transpose(inputs, weights, output_shape = out_shape,
                                      strides=[1, stride,  stride,  1])
    
    return tf.nn.bias_add(deconved, biases)

class Generator(object):
    def __init__(self, z_dim, layer_chanels):
        self.z_dim = z_dim
        self.in_dim = 4
        self.layer_chanels = layer_chanels

        self.name_scope_reshape = u'reshape_z'
        self.name_scope_deconv = u'deconvolution'

    def get_variables(self):
        t_var = tf.trainable_variables()
        ret = []
        for var in t_var:
            if self.name_scope_deconv in var.name or self.name_scope_reshape in var.name:
                ret.append(var)
        return ret
    
    def set_model(self, z, is_training):
        #self.z = tf.placeholder(tf.float32, [None, self.z_dim])

        # reshape z
        with tf.variable_scope(self.name_scope_reshape):
            w_r = get_weights('_r',
                              [self.z_dim, self.in_dim * self.in_dim * self.layer_chanels[0]],
                              0.1)
            b_r = get_biases('_r',
                             [self.in_dim * self.in_dim * self.layer_chanels[0]],
                             0.0)
            h = tf.matmul(z, w_r) + b_r
            h = batch_norm(h, 'reshape', is_training)
            h = tf.nn.relu(h)
            
        h = tf.reshape(h, [-1, self.in_dim, self.in_dim, self.layer_chanels[0]])

        # deconvolution
        with tf.variable_scope(self.name_scope_deconv):
            for i, (in_chan, out_chan) in enumerate(zip(self.layer_chanels, self.layer_chanels[1:])):
                deconved = deconv_layer(inputs = h,
                                        out_shape = [100, self.in_dim * 2 ** (i + 1), self.in_dim * 2 **(i + 1), out_chan],
                                        filter_width = 5, filter_hight = 5,
                                        stride = 2, l_id = i)
                bn_deconved = batch_norm(deconved, i, is_training)
                h = tf.nn.relu(bn_deconved)

        #return tf.nn.tanh(deconved)
        return tf.nn.sigmoid(deconved)
    
if __name__ == u'__main__':
    g = Generator(100, [1024, 512, 256, 128, 3])
    z = tf.placeholder(tf.float32, [None, 100])
    g.set_model(z)
