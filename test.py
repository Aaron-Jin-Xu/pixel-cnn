import numpy as np
import tensorflow as tf

import os
import sys
import time

import forward_model as fm
import backward_model as bm

import pixel_cnn_pp.mask as mk
from utils import *

with tf.Session() as sess:

    ## forward model
    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model/')
    print(len(var_list))
    saver = tf.train.Saver(var_list=var_list)

    ckpt_file = "/data/ziz/jxu/save-forward" + '/params_' + "celeba" + '.ckpt'
    print('restoring parameters from', ckpt_file)
    saver.restore(sess, ckpt_file)

    ## backward model
    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model_1/')
    print(len(var_list))
    saver = tf.train.Saver(var_list=var_list)

    ckpt_file = "/data/ziz/jxu/save-backward-rename" + '/params_' + "celeba" + '.ckpt'
    print('restoring parameters from', ckpt_file)
    saver.restore(sess, ckpt_file)

    ###############################

    d = next(fm.test_data)
    # generate masks
    obs_shape = d.shape[1:]
    mgen = mk.RecNoProgressMaskGenerator(obs_shape[0], obs_shape[1])
    ms = mgen.gen(fm.args.nr_gpu * fm.args.batch_size)
    agen = mk.AllOnesMaskGenerator(obs_shape[0], obs_shape[1])
    ams = mgen.gen(fm.args.nr_gpu * fm.args.batch_size)
    for step in range(1):

        target_pixels = next_pixel(ms)

        feed_dict = fm.make_feed_dict(d, mask_values=ams, rot=False)
        o1 = sess.run(fm.outputs, feed_dict)
        o1 = np.concatenate(o1, axis=0)
        print(get_params(o1, target_pixels))

        feed_dict = bm.make_feed_dict(d, mask_values=np.rot90(ms, 2, (1,2)), rot=True)
        o2 = sess.run(bm.outputs, feed_dict)
        o2 = np.concatenate(o2, axis=0)
        print(get_params(o2, target_pixels))
