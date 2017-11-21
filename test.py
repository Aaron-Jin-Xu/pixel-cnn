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
        o1 = get_params(o1, target_pixels))

        params_to_dis(o1, fm.args.nr_logistic_mix)

        backward_ms = ms.copy()
        for idx in range(len(target_pixels)):
            p = target_pixels[idx]
            backward_ms[idx, p[0], p[1]] = 1
        backward_ms = np.rot90(ms, 2, (1,2))

        feed_dict = bm.make_feed_dict(d, mask_values=backward_ms, rot=True)
        o2 = sess.run(bm.outputs, feed_dict)
        o2 = np.concatenate(o2, axis=0)
        o2 = np.rot90(o2, 2, (1,2))
        o2 = get_params(o2, target_pixels)
