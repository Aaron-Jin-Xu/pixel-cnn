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

from configs import configs

def find_coutour(mask):
    contour = np.zeros_like(mask)
    h, w = mask.shape
    for y in range(h):
        for x in range(w):
            if mask[y, x] > 0:
                lower_bound = max(y-1, 0)
                upper_bound = min(y+1, h-1)
                left_bound = max(x-1, 0)
                right_bound = min(x+1, w-1)
                nb = mask[lower_bound:upper_bound+1, left_bound:right_bound+1]
                if np.min(nb)  == 0:
                    contour[y, x] = 1
    return contour

#display_size = (6,6)
display_size = (8,8)

exp_label = "celeba-hr-test"

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


    # Get test images, batch_size X nr_gpu
    d = next(fm.test_data)
    # Store original images
    img = Image.fromarray(tile_images(d.astype(np.uint8), size=display_size), 'RGB')
    img.save("/homes/jxu/projects/ImageInpainting/plots/original-{0}.png".format(exp_label))

    # generate masks
    obs_shape = d.shape[1:]
    #mgen = mk.RecNoProgressMaskGenerator(obs_shape[0], obs_shape[1])
    #mgen = mk.CircleMaskGenerator(obs_shape[0], obs_shape[1], 16)
    #mgen = mk.RectangleMaskGenerator(obs_shape[0], obs_shape[1])
    mgen = mk.BottomMaskGenerator(obs_shape[0], obs_shape[1], 32)
    #mgen = mk.HorizontalMaskGenerator(obs_shape[0], obs_shape[1], 16, 48)
    #mgen = mk.GridMaskGenerator(obs_shape[0], obs_shape[1], 8)
    #mgen = mk.RandomNoiseMaskGenerator(obs_shape[0], obs_shape[1], 0.8)
    ms = mgen.gen(fm.args.nr_gpu * fm.args.batch_size)
    ms_ori = ms.copy()

    # Mask the images
    d = d.astype(np.float64)
    d *= ms[:, :, :, None]
    img = Image.fromarray(tile_images(d.astype(np.uint8), size=display_size), 'RGB')
    img.save("/homes/jxu/projects/ImageInpainting/plots/masked-{0}.png".format(exp_label))
    agen = mk.AllOnesMaskGenerator(obs_shape[0], obs_shape[1])
    ams = agen.gen(fm.args.nr_gpu * fm.args.batch_size)

    # Load prior
    prior = np.load("/data/ziz/jxu/prior64.npz")["arr"]
    #prior = np.load("/data/ziz/jxu/prior-svhn.npz")["arr"]


    dis_record = []
    data_record = []
    sample_record = []

    data_record.append(d.copy())

    count = 0

    while True:
        count += 1
        print(count)

        rgb_record = []

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
        pars1 = params_to_dis(o1, fm.args.nr_logistic_mix, log_scales_shift=2.)
        pars2 = params_to_dis(o2, bm.args.nr_logistic_mix)
        pars = pars1 #* pars2 #/ pr[:, 0, :]
        pars[:, 0], pars[:, 255] = pars[:, 1], pars[:, 254]
        #pars = np.power(pars, 0.5)
        pars = pars.astype(np.float64)
        pars = pars / np.sum(pars, axis=-1)[:, None]
        rgb_record.append(np.array([pars1, pars2, pars, pr[:, 0, :]]))
        color_r = []
        for i in range(pars.shape[0]):
            #color_r.append(np.argmax(np.random.multinomial(1, pars[i, :])))
            color_r.append(np.argmax(pars[i, :]))
        color_r = np.array(color_r)

        # Sample green channel
        pars1 = params_to_dis(o1, fm.args.nr_logistic_mix, r=color_r, log_scales_shift=2.)
        pars2 = params_to_dis(o2, bm.args.nr_logistic_mix, r=color_r)
        pars = pars1 #* pars2 #/ pr[:, 1, :]
        pars[:, 0], pars[:, 255] = pars[:, 1], pars[:, 254]
        #pars = np.power(pars, 0.5)
        pars = pars.astype(np.float64)
        pars = pars / np.sum(pars, axis=-1)[:, None]
        rgb_record.append(np.array([pars1, pars2, pars, pr[:, 1, :]]))
        color_g = []
        for i in range(pars.shape[0]):
            #color_g.append(np.argmax(np.random.multinomial(1, pars[i, :])))
            color_g.append(np.argmax(pars[i, :]))
        color_g = np.array(color_g)

        # Sample blue channel
        pars1 = params_to_dis(o1, fm.args.nr_logistic_mix, r=color_r, g=color_g, log_scales_shift=2.)
        pars2 = params_to_dis(o2, bm.args.nr_logistic_mix, r=color_r, g=color_g)
        pars = pars1 #* pars2 #/ pr[:, 2, :]
        pars[:, 0], pars[:, 255] = pars[:, 1], pars[:, 254]
        #pars = np.power(pars, 0.5)
        pars = pars.astype(np.float64)
        pars = pars / np.sum(pars, axis=-1)[:, None]
        rgb_record.append(np.array([pars1, pars2, pars, pr[:, 2, :]]))
        color_b = []
        for i in range(pars.shape[0]):
            #color_b.append(np.argmax(np.random.multinomial(1, pars[i, :])))
            color_b.append(np.argmax(pars[i, :]))
        color_b = np.array(color_b)

        color = np.array([color_r, color_g, color_b]).T
        sample_record.append(color)
        #print(color)
        dis_record.append(np.array(rgb_record))

        for idx in range(len(target_pixels)):
            p = target_pixels[idx]
            ms[idx, p[0], p[1]] = 1
            d[idx, p[0], p[1], :] = color[idx, :]

        data_record.append(d.copy())

    dis_record = np.array(dis_record)
    data_record = np.array(data_record)
    #np.savez_compressed("/data/ziz/jxu/inpainting-record-{0}".format(exp_label), dis=dis_record, img=data_record, smp=sample_record, ms=ms_ori)

    # Store the completed images

    for i in range(d.shape[0]):
        contour = 1-find_coutour(ms_ori[i])[:, :, None]
        contour[contour<1] = 0.8
        d[i] *= contour

    img = Image.fromarray(tile_images(d.astype(np.uint8), size=display_size), 'RGB')
    img.save("/homes/jxu/projects/ImageInpainting/plots/complete-{0}.png".format(exp_label))
