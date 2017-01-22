#! -*- coding:utf-8 -*-

import os
import sys

import numpy as np
import tensorflow as tf

from generator import Generator
from discriminator import Discriminator
    
class Model(object):
    def __init__(self, z_dim):
        # generator
        self.gen = Generator(z_dim, [1024, 512, 256, 128, 3])
        
        # discriminator
        self.disc = Discriminator([3, 64, 128, 256, 512])
        self.lr = 0.0001

        self.is_training = tf.placeholder(tf.bool)
        
    def set_model(self):
        # z -> gen_fig -> disc
        self.z = tf.placeholder(tf.float32, [None, 100])
        self.gen_figs = self.gen.set_model(self.z, self.is_training)
        g_sigmoid_logits, g_logits = self.disc.set_model(self.gen_figs, self.is_training)

        g_obj = tf.reduce_mean(-tf.log(1.0e-3 + g_sigmoid_logits))

        self.train_gen  = tf.train.AdamOptimizer(self.lr).minimize(g_obj, var_list = self.gen.get_variables())
        
        # for sharing variables
        tf.get_variable_scope().reuse_variables()
        
        # true_fig -> disc
        self.figs= tf.placeholder(tf.float32, [None, 64, 64, 3])        
        d_sigmoid_logits, d_logits = self.disc.set_model(self.figs, self.is_training)

        d_obj_true = tf.reduce_mean(-tf.log(1.0e-3 + d_sigmoid_logits))
        d_obj_fake = tf.reduce_mean(-tf.log(1.0e-3 + g_sigmoid_logits))
        d_obj = d_obj_true + d_obj_fake

        self.train_disc = tf.train.AdamOptimizer(self.lr).minimize(d_obj, var_list = self.disc.get_variables())
        
    def training_gen(self, sess, z_list):
        sess.run(self.train_gen,
                 feed_dict = {self.z: z_list,
                              self.is_training: True})
        
    def training_disc(self, sess, z_list, figs):
        sess.run(self.train_disc,
                 feed_dict = {self.z: z_list,
                              self.figs:figs,
                              self.is_training: True})

    def gen_fig(self, sess, z):
        ret = sess.run(self.gen_figs,
                       feed_dict = {self.z: z,
                                    self.is_training: False})
        return ret

if __name__ == u'__main__':
    model = Model(z_dim = 100)
    model.set_model()
    
