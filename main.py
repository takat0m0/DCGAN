#! -*- coding:utf-8 -*-

import os
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from Model import Model
from util import get_figs, dump_figs
        
if __name__ == u'__main__':

    # figs dir
    dir_name = u'figs'
    
    # parameter
    batch_size = 100
    epoch_num = 100
    z_dim = 100

    # make model
    print('-- make model --')
    model = Model(z_dim)
    model.set_model()

    # get_data
    print('-- get figs--')
    figs = get_figs(dir_name)
    print('num figs = {}'.format(len(figs)))
    
    # training
    print('-- begin training --')
    num_one_epoch = len(figs) //batch_size
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for epoch in range(epoch_num):
            print('epoch:{}'.format(epoch))
            for step in range(num_one_epoch):
                batch_z = np.random.uniform(-1, 1, [batch_size, z_dim]).astype(np.float32)
                batch_figs = figs[step * batch_size: (step + 1) * batch_size]
                model.training_disc(sess, batch_z, batch_figs)
                model.training_gen(sess, batch_z)
                if step%10 == 0:
                    print('   step {}'.format(step))
                    figs = model.gen_fig(sess, batch_z)
                    dump_figs(np.asarray(figs), 'sample_result')
