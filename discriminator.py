#! -*- coding:utf-8 -*-

import os
import sys
import numpy as np
import tensorflow as tf

from util import get_weights, get_biases, get_dim
from batch_normalize import batch_norm

def conv_layer(inputs, out_num, filter_width, filter_hight, stride, l_id):
    # ** NOTICE: weight shape is [hight, width, in_chanel, out_chanel] **
    weights = get_weights(l_id,
                          [filter_hight, filter_width, inputs.get_shape()[-1], out_num],
                          0.1)
    
    biases = get_biases(l_id, [out_num], 0.0)
    
    conved = tf.nn.conv2d(inputs, weights,
                          strides=[1, stride,  stride,  1],
                          padding = 'SAME')
    
    return tf.nn.bias_add(conved, biases)

class Discriminator(object):
    def __init__(self, layer_chanels):
        self.layer_chanels = layer_chanels
        
        self.name_scope_conv = u'convolution'
        self.name_scope_fc = u'full_connect'
        
    def get_variables(self):
        t_var = tf.trainable_variables()
        ret = []
        for var in t_var:
            if self.name_scope_conv in var.name or self.name_scope_fc in var.name:
                ret.append(var)
        return ret
    
    def set_model(self, figs, is_training):
        h = figs
        
        # convolution
        with tf.variable_scope(self.name_scope_conv):
            for i, (in_chan, out_chan) in enumerate(zip(self.layer_chanels, self.layer_chanels[1:])):
                conved = conv_layer(inputs = h,
                                    out_num = out_chan,
                                    filter_width = 5, filter_hight = 5,
                                    stride = 2, l_id = i)
                bn_conved = batch_norm(conved, i, is_training)
                h = tf.nn.relu(conved)

        # full connect
        dim = get_dim(h)
        h = tf.reshape(h, [-1, dim])
        
        with tf.variable_scope(self.name_scope_fc):
            weights = get_weights('fc', [dim, 1], 0.1)
            biases  = get_biases('fc', [1], 0.0)
            h = tf.matmul(h, weights) + biases
            
        return tf.nn.sigmoid(h), h
    
if __name__ == u'__main__':
    g = Discriminator([3, 64, 128, 256, 512])
    figs = tf.placeholder(tf.float32, [None, 64, 64, 3])
    g.set_model(figs)
