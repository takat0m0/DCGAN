#! -*- coding:utf-8 -*-

import os
import sys

import numpy as np
import tensorflow as tf

from generator import Generator
from discriminator import Discriminator
    
class Model(object):
    def __init__(self, z_dim, batch_size):
        self.z_dim = z_dim
        self.batch_size = batch_size
        
        # -- generator -----
        self.gen = Generator(z_dim, [512, 256, 128, 64,  3])

        # -- discriminator --
        self.disc = Discriminator([3, 64, 128, 256, 512])
        self.lr = 0.0002

        
    def set_model(self):
        # -- z -> gen_fig -> disc ---

        self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim])

        gen_figs = self.gen.set_model(self.z, self.batch_size, is_training = True)
        g_logits = self.disc.set_model(gen_figs, is_training = True)
        self.g_obj = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits = g_logits,
                targets = tf.ones_like(g_logits)))
        
        self.train_gen  = tf.train.AdamOptimizer(self.lr, beta1 = 0.5).minimize(self.g_obj, var_list = self.gen.get_variables())
        
        # -- for sharing variables ---
        tf.get_variable_scope().reuse_variables()
        
        # -- true_fig -> disc --------
        self.figs= tf.placeholder(tf.float32, [self.batch_size, 64, 64, 3])        

        d_logits = self.disc.set_model(self.figs, is_training = True)

        d_obj_true = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits = d_logits,
                targets = tf.ones_like(d_logits)))
        d_obj_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits = g_logits,
                targets = tf.zeros_like(g_logits)))
    
        self.d_obj = d_obj_true + d_obj_fake

        self.train_disc = tf.train.AdamOptimizer(self.lr, beta1 = 0.5).minimize(self.d_obj, var_list = self.disc.get_variables())

        # -- for figure generation -------
        self.gen_figs = self.gen.set_model(self.z, self.batch_size, is_training = False)
        
    def training_gen(self, sess, z_list):
        _, g_obj = sess.run([self.train_gen, self.g_obj],
                            feed_dict = {self.z: z_list})
        return g_obj
        
    def training_disc(self, sess, z_list, figs):
        _, d_obj = sess.run([self.train_disc, self.d_obj],
                            feed_dict = {self.z: z_list,
                              self.figs:figs})
        return d_obj
    
    def gen_fig(self, sess, z):
        ret = sess.run(self.gen_figs,
                       feed_dict = {self.z: z})
        return ret

if __name__ == u'__main__':
    model = Model(z_dim = 100)
    model.set_model()
    
