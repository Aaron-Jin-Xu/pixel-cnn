import numpy as np
import tensorflow as tf

import forward_model as fm
import backward_model as bm

with tf.Session() as sess:
    vs = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model')
    vs1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model_1')
    print(len(vs))
    print(len(vs1))
    print([item.name for item in vs])
    print([item.name for item in vs1])
    quit()

    saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model'))
    saver1 = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model_1'))


    ckpt_file = "/data/ziz/jxu/save-forward" + '/params_' + "celeba" + '.ckpt'
    print('restoring parameters from', ckpt_file)
    saver.restore(sess, ckpt_file)

    ckpt_file = "/data/ziz/jxu/save-test" + '/params_' + "celeba" + '.ckpt'
    print('restoring parameters from', ckpt_file)
    saver1.restore(sess, ckpt_file)
