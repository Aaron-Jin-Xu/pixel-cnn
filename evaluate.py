import numpy as np
import tensorflow as tf

import os
import sys
import time

import forward_model as fm
import backward_model as bm

import pixel_cnn_pp.mask as mk
from utils import *
from PIL import Image

from evaluation import *

from configs import configs

exp_label = "celeba-hr-center"

with tf.Session() as sess:

    # restore forward model
    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model/')
    print(len(var_list))
    saver = tf.train.Saver(var_list=var_list)

    ckpt_file = fm.args.save_dir + '/params_' + fm.args.data_set + '.ckpt'
    print('restoring parameters from', ckpt_file)
    saver.restore(sess, ckpt_file)

    # restore backward model
    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model_1/')
    print(len(var_list))
    saver = tf.train.Saver(var_list=var_list)

    ckpt_file = bm.args.save_dir + '/params_' + bm.args.data_set + '.ckpt'
    print('restoring parameters from', ckpt_file)
    saver.restore(sess, ckpt_file)

    for d in fm.test_data:

        print("new batch, ", d.shape)

        # Get test images, batch_size X nr_gpu
        #d = next(fm.test_data)

        # Store original images
        # Load prior
        prior = np.load("/data/ziz/jxu/prior64.npz")["arr"]
        # prior = np.load("/data/ziz/jxu/prior-svhn.npz")["arr"]

        # generate masks
        obs_shape = d.shape[1:]
        #mgen = mk.RecNoProgressMaskGenerator(obs_shape[0], obs_shape[1])
        #mgen = mk.CircleMaskGenerator(obs_shape[0], obs_shape[1], 8)
        #mgen = mk.RectangleMaskGenerator(obs_shape[0], obs_shape[1])
        #mgen = mk.BottomMaskGenerator(obs_shape[0], obs_shape[1], 16)
        mgen = mk.HorizontalMaskGenerator(obs_shape[0], obs_shape[1], 16, 48)
        #mgen = mk.RandomNoiseMaskGenerator(obs_shape[0], obs_shape[1], 0.8)
        ms = mgen.gen(fm.args.nr_gpu * fm.args.batch_size)
        ms_ori = ms.copy()

        d = d.astype(np.float64)
        images_ori = d.copy()

        completed_images_arr = []

        for k in range(5):
            print("------------", k)

            d = images_ori.copy()
            ms = ms_ori.copy()

            # Mask the images
            d *= ms[:, :, :, None]
            agen = mk.AllOnesMaskGenerator(obs_shape[0], obs_shape[1])
            ams = agen.gen(fm.args.nr_gpu * fm.args.batch_size)

            count = 0

            while True:
                count += 1
                print(count)
                target_pixels = next_pixel(ms)
                #print(target_pixels[0])
                if target_pixels[0][0] is None:
                    break
                pr = get_prior(prior, target_pixels)
                backward_ms = ms.copy()
                for idx in range(len(target_pixels)):
                    p = target_pixels[idx]
                    backward_ms[idx, p[0], p[1]] = 1
                backward_ms = np.rot90(ms, 2, (1,2))

                # Forward model prediction
                feed_dict = fm.make_feed_dict(d, mask_values=ams, rot=False)
                o1 = sess.run(fm.outputs, feed_dict)
                o1 = np.concatenate(o1, axis=0)
                o1 = get_params(o1, target_pixels)

                # Backward model prediction
                feed_dict = bm.make_feed_dict(d, mask_values=backward_ms, rot=True)
                o2 = sess.run(bm.outputs, feed_dict)
                o2 = np.concatenate(o2, axis=0)
                o2 = np.rot90(o2, 2, (1,2))
                o2 = get_params(o2, target_pixels)

                # Sample red channel
                pars1 = params_to_dis(o1, fm.args.nr_logistic_mix)
                pars2 = params_to_dis(o2, fm.args.nr_logistic_mix)
                pars = pars1 * pars2 #/ pr[:, 0, :]
                pars[:, 0] = 0.
                pars[:, 255] = 0.
                #pars = np.power(pars, 0.5)
                pars = pars.astype(np.float64)
                pars = pars / np.sum(pars, axis=-1)[:, None]
                color_r = []
                for i in range(pars.shape[0]):
                    color_r.append(np.argmax(np.random.multinomial(1, pars[i, :])))
                color_r = np.array(color_r)

                # Sample green channel
                pars1 = params_to_dis(o1, fm.args.nr_logistic_mix, r=color_r)
                pars2 = params_to_dis(o2, fm.args.nr_logistic_mix, r=color_r)
                pars = pars1 * pars2 #/ pr[:, 1, :]
                pars[:, 0] = 0.
                pars[:, 255] = 0.
                #pars = np.power(pars, 0.5)
                pars = pars.astype(np.float64)
                pars = pars / np.sum(pars, axis=-1)[:, None]
                color_g = []
                for i in range(pars.shape[0]):
                    color_g.append(np.argmax(np.random.multinomial(1, pars[i, :])))
                color_g = np.array(color_g)

                # Sample blue channel
                pars1 = params_to_dis(o1, fm.args.nr_logistic_mix, r=color_r, g=color_g)
                pars2 = params_to_dis(o2, fm.args.nr_logistic_mix, r=color_r, g=color_g)
                pars = pars1 * pars2 #/ pr[:, 2, :]
                pars[:, 0] = 0.
                pars[:, 255] = 0.
                #pars = np.power(pars, 0.5)
                pars = pars.astype(np.float64)
                pars = pars / np.sum(pars, axis=-1)[:, None]
                color_b = []
                for i in range(pars.shape[0]):
                    color_b.append(np.argmax(np.random.multinomial(1, pars[i, :])))
                color_b = np.array(color_b)

                color = np.array([color_r, color_g, color_b]).T

                for idx in range(len(target_pixels)):
                    p = target_pixels[idx]
                    ms[idx, p[0], p[1]] = 1
                    d[idx, p[0], p[1], :] = color[idx, :]


            images_completed = d.copy()
            completed_images_arr.append(images_completed)

        images_completed = np.mean(completed_images_arr, axis=0)

        psnr = batch_psnr(images_completed, images_ori, output_mean=False)
        print(np.mean(psnr))
        print(np.std(psnr))
        print(len(psnr))

        break
