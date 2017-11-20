import numpy as np
import tensorflow as tf

import os
import sys
import time

import forward_model as fm
import backward_model as bm


with tf.Session() as sess:


    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model/')
    print(len(var_list))
    saver = tf.train.Saver(var_list=var_list)

    ckpt_file = "/data/ziz/jxu/save-forward" + '/params_' + "celeba" + '.ckpt'
    print('restoring parameters from', ckpt_file)
    saver.restore(sess, ckpt_file)

    test_losses = []
    for d in fm.test_data:
        feed_dict = fm.make_feed_dict(d, masks=fm.masks, is_test=True)
        l = sess.run(fm.bits_per_dim_test, feed_dict)
        test_losses.append(l)
    test_loss_gen = np.mean(test_losses)

    print("test bits_per_dim = %.4f" % (test_loss_gen))
    sys.stdout.flush()


    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model_1/')
    print(len(var_list))
    saver = tf.train.Saver(var_list=var_list)

    ckpt_file = "/data/ziz/jxu/save-test" + '/params_' + "celeba" + '.ckpt'
    print('restoring parameters from', ckpt_file)
    saver.restore(sess, ckpt_file)

    test_losses = []
    for d in bm.test_data:
        feed_dict = bm.make_feed_dict(d, masks=bm.masks, is_test=True)
        l = sess.run(bm.bits_per_dim_test, feed_dict)
        test_losses.append(l)
    test_loss_gen = np.mean(test_losses)

    print("test bits_per_dim = %.4f" % (test_loss_gen))
    sys.stdout.flush()
