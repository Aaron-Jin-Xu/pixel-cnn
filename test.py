import numpy as np
import tensorflow as tf

import os
import sys
import time

import backward_model as bm
import forward_model as fm


with tf.Session() as sess:

    ## forward model
    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model/')
    print(len(var_list))
    saver = tf.train.Saver(var_list=var_list)

    ckpt_file = "/data/ziz/jxu/save-forward" + '/params_' + "celeba" + '.ckpt'
    print('restoring parameters from', ckpt_file)
    saver.restore(sess, ckpt_file)

    d = next(fm.test_data)
    feed_dict = fm.make_feed_dict(d, masks=fm.masks, is_test=True)
    o1 = sess.run(fm.outputs, feed_dict)
    o1 = np.concatenate(o1, axis=0)
    print(o1.shape)

    # test_losses = []
    # for d in bm.test_data:
    #     feed_dict = bm.make_feed_dict(d, masks=bm.masks, is_test=True)
    #     l = sess.run(bm.bits_per_dim_test, feed_dict)
    #     test_losses.append(l)
    # test_loss_gen = np.mean(test_losses)
    #
    # print("test bits_per_dim = %.4f" % (test_loss_gen))
    # sys.stdout.flush()


    ## backward model
    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model_1/')
    print(len(var_list))
    saver = tf.train.Saver(var_list=var_list)

    ckpt_file = "/data/ziz/jxu/save-backward-rename" + '/params_' + "celeba" + '.ckpt'
    print('restoring parameters from', ckpt_file)
    saver.restore(sess, ckpt_file)

    feed_dict = bm.make_feed_dict(d, masks=bm.masks, is_test=True)
    o2 = sess.run(bm.outputs, feed_dict)
    o2 = np.concatenate(o2, axis=0)
    print(o2.shape)

    # test_losses = []
    # for d in fm.test_data:
    #     feed_dict = fm.make_feed_dict(d, masks=fm.masks, is_test=True)
    #     l = sess.run(fm.bits_per_dim_test, feed_dict)
    #     test_losses.append(l)
    # test_loss_gen = np.mean(test_losses)
    #
    # print("test bits_per_dim = %.4f" % (test_loss_gen))
    # sys.stdout.flush()
