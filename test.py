import numpy as np
import tensorflow as tf

import forward_model as fm
import backward_model as bm

with tf.Session() as sess:
    saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model'))
    saver1 = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model_1'))


    ckpt_file = "/data/ziz/jxu/save-forward" + '/params_' + args.data_set + '.ckpt'
    print('restoring parameters from', ckpt_file)
    saver.restore(sess, ckpt_file)

    ckpt_file = "/data/ziz/jxu/save-test" + '/params_' + args.data_set + '.ckpt'
    print('restoring parameters from', ckpt_file)
    saver1.restore(sess, ckpt_file)

    
