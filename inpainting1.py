"""
Trains a Pixel-CNN++ generative model on CIFAR-10 or Tiny ImageNet data.
Uses multiple GPUs, indicated by the flag --nr-gpu

Example usage:
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_double_cnn.py --nr_gpu 4
"""

import os
import sys
import time
import json
import argparse

import numpy as np
import tensorflow as tf

import pixel_cnn_pp.nn as nn
import pixel_cnn_pp.mask as mk
import pixel_cnn_pp.plotting as plotting
from pixel_cnn_pp.model import model_spec
import data.cifar10_data as cifar10_data
import data.imagenet_data as imagenet_data
import data.celeba_data as celeba_data

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
# data I/O
parser.add_argument('-i', '--data_dir', type=str,
                    default='/tmp/pxpp/data', help='Location for the dataset')
parser.add_argument('-o', '--save_dir', type=str, default='/tmp/pxpp/save',
                    help='Location for parameter checkpoints and samples')
parser.add_argument('-d', '--data_set', type=str,
                    default='cifar', help='Can be either cifar|imagenet')
parser.add_argument('-t', '--save_interval', type=int, default=20,
                    help='Every how many epochs to write checkpoint/samples?')
parser.add_argument('-r', '--load_params', dest='load_params', action='store_true',
                    help='Restore training from previous model checkpoint?')
# model
parser.add_argument('-q', '--nr_resnet', type=int, default=5,
                    help='Number of residual blocks per stage of the model')
parser.add_argument('-n', '--nr_filters', type=int, default=160,
                    help='Number of filters to use across the model. Higher = larger model.')
parser.add_argument('-m', '--nr_logistic_mix', type=int, default=10,
                    help='Number of logistic components in the mixture. Higher = more flexible model')
parser.add_argument('-z', '--resnet_nonlinearity', type=str, default='concat_elu',
                    help='Which nonlinearity to use in the ResNet layers. One of "concat_elu", "elu", "relu" ')
parser.add_argument('-c', '--class_conditional', dest='class_conditional',
                    action='store_true', help='Condition generative model on labels?')
# optimization
parser.add_argument('-l', '--learning_rate', type=float,
                    default=0.001, help='Base learning rate')
parser.add_argument('-e', '--lr_decay', type=float, default=0.999995,
                    help='Learning rate decay, applied every step of the optimization')
parser.add_argument('-b', '--batch_size', type=int, default=12,
                    help='Batch size during training per GPU')
parser.add_argument('-a', '--init_batch_size', type=int, default=100,
                    help='How much data to use for data-dependent initialization.')
parser.add_argument('-p', '--dropout_p', type=float, default=0.5,
                    help='Dropout strength (i.e. 1 - keep_prob). 0 = No dropout, higher = more dropout.')
parser.add_argument('-x', '--max_epochs', type=int,
                    default=5000, help='How many epochs to run in total?')
parser.add_argument('-g', '--nr_gpu', type=int, default=8,
                    help='How many GPUs to distribute the training across?')
# evaluation
parser.add_argument('--polyak_decay', type=float, default=0.9995,
                    help='Exponential decay rate of the sum of previous model iterates during Polyak averaging')
# reproducibility
parser.add_argument('-s', '--seed', type=int, default=1,
                    help='Random seed to use')

parser.add_argument('-k', '--masked', dest='masked',
                    action='store_true', help='Randomly mask input images?')

parser.add_argument('-j', '--rot180', dest='rot180',
                    action='store_true', help='Rot180 the images?')

args = parser.parse_args()
print('input args:\n', json.dumps(vars(args), indent=4,
                                  separators=(',', ':')))  # pretty print args

# -----------------------------------------------------------------------------
# fix random seed for reproducibility
rng = np.random.RandomState(args.seed)
tf.set_random_seed(args.seed)

# initialize data loaders for train/test splits
if args.data_set == 'imagenet' and args.class_conditional:
    raise("We currently don't have labels for the small imagenet data set")
DataLoader = {'cifar': cifar10_data.DataLoader,
              'imagenet': imagenet_data.DataLoader,
              'celeba': celeba_data.DataLoader}[args.data_set]
#train_data = DataLoader(args.data_dir, 'train', args.batch_size * args.nr_gpu,
#                        rng=rng, shuffle=True, return_labels=args.class_conditional)
test_data = DataLoader(args.data_dir, 'valid', args.batch_size *
                       args.nr_gpu, shuffle=False, return_labels=args.class_conditional)
obs_shape = test_data.get_observation_size()  # e.g. a tuple (32,32,3)
assert len(obs_shape) == 3, 'assumed right now'

# data place holders
x_init = tf.placeholder(tf.float32, shape=(args.init_batch_size,) + obs_shape)
xs = [tf.placeholder(tf.float32, shape=(args.batch_size, ) + obs_shape)
      for i in range(args.nr_gpu)]


h_init = None
h_sample = [None] * args.nr_gpu
hs = h_sample

if args.masked:
    masks = tf.placeholder(tf.float32, shape=(args.batch_size,) + obs_shape[:-1])
else:
    masks = None

# create the model
model_opt = {'nr_resnet': args.nr_resnet, 'nr_filters': args.nr_filters,
             'nr_logistic_mix': args.nr_logistic_mix, 'resnet_nonlinearity': args.resnet_nonlinearity}
model = tf.make_template('model', model_spec)

# run once for data dependent initialization of parameters
gen_par = model(x_init, None, h_init, init=True,
                dropout_p=args.dropout_p, **model_opt)

# keep track of moving average
all_params = tf.trainable_variables()
ema = tf.train.ExponentialMovingAverage(decay=args.polyak_decay)
maintain_averages_op = tf.group(ema.apply(all_params))

# get loss gradients over multiple GPUs
grads = []
loss_gen = []
loss_gen_test = []
for i in range(args.nr_gpu):
    with tf.device('/gpu:%d' % i):
        # train
        gen_par = model(xs[i], masks, hs[i], ema=None,
                        dropout_p=args.dropout_p, **model_opt)
        loss_gen.append(nn.discretized_mix_logistic_loss(xs[i], gen_par, masks=masks))
        # gradients
        grads.append(tf.gradients(loss_gen[i], all_params))
        # test
        gen_par = model(xs[i], masks, hs[i], ema=ema, dropout_p=0., **model_opt)
        loss_gen_test.append(nn.discretized_mix_logistic_loss(xs[i], gen_par, masks=masks))

# add losses and gradients together and get training updates
tf_lr = tf.placeholder(tf.float32, shape=[])
with tf.device('/gpu:0'):
    for i in range(1, args.nr_gpu):
        loss_gen[0] += loss_gen[i]
        loss_gen_test[0] += loss_gen_test[i]
        for j in range(len(grads[0])):
            grads[0][j] += grads[i][j]
    # training op
    optimizer = tf.group(nn.adam_updates(
        all_params, grads[0], lr=tf_lr, mom1=0.95, mom2=0.9995), maintain_averages_op)

# convert loss to bits/dim
bits_per_dim = loss_gen[
    0] / (args.nr_gpu * np.log(2.) * np.prod(obs_shape) * args.batch_size)
bits_per_dim_test = loss_gen_test[
    0] / (args.nr_gpu * np.log(2.) * np.prod(obs_shape) * args.batch_size)

# sample from the model
new_x_gen = []
for i in range(args.nr_gpu):
    with tf.device('/gpu:%d' % i):
        gen_par = model(xs[i], None, h_sample[i], ema=ema, dropout_p=0, **model_opt)
        new_x_gen.append(nn.sample_from_discretized_mix_logistic(
            gen_par, args.nr_logistic_mix))


def sample_from_model(sess):
    x_gen = [np.zeros((args.batch_size,) + obs_shape, dtype=np.float32)
             for i in range(args.nr_gpu)]
    for yi in range(obs_shape[0]):
        for xi in range(obs_shape[1]):
            new_x_gen_np = sess.run(
                new_x_gen, {xs[i]: x_gen[i] for i in range(args.nr_gpu)})
            for i in range(args.nr_gpu):
                x_gen[i][:, yi, xi, :] = new_x_gen_np[i][:, yi, xi, :]
    return np.concatenate(x_gen, axis=0)


def complete(imgs, mks, sess):
    x_gen = [imgs[i] for i in range(args.nr_gpu)]
    x_mk = [mks[i] for i in range(args.nr_gpu)]

    for yi in range(20,32):
        for xi in range(32):
            new_x_gen_np = sess.run(
                new_x_gen, {xs[i]: x_gen[i] for i in range(args.nr_gpu)})
            for i in range(args.nr_gpu):
                x_gen[i][:, yi, xi, :] = new_x_gen_np[i][:, yi, xi, :]
    return np.concatenate(x_gen, axis=0)


# init & save
initializer = tf.global_variables_initializer()
saver = tf.train.Saver()

# turn numpy inputs into feed_dict for use with tensorflow

#mgen = mk.RandomMaskGenerator(obs_shape[0], obs_shape[1])
#mgen = mk.RecMaskGenerator(obs_shape[0], obs_shape[1])
mgen = mk.RecNoProgressMaskGenerator(obs_shape[0], obs_shape[1])
agen = mk.AllOnesMaskGenerator(obs_shape[0], obs_shape[1])

def make_feed_dict(data, init=False, masks=None, is_test=False):
    if type(data) is tuple:
        x, y = data
    else:
        x = data
        y = None
    if args.rot180:
        x = np.rot90(x, 2, (1,2)) #### ROT
    # input to pixelCNN is scaled from uint8 [0,255] to float in range [-1,1]
    x = np.cast[np.float32]((x - 127.5) / 127.5)
    if init:
        feed_dict = {x_init: x}
        if y is not None:
            feed_dict.update({y_init: y})
    else:
        x = np.split(x, args.nr_gpu)
        feed_dict = {xs[i]: x[i] for i in range(args.nr_gpu)}
        if masks is not None:
            if is_test:
                feed_dict[masks] = agen.gen(args.batch_size)
            else:
                feed_dict[masks] = mgen.gen(args.batch_size)
        if y is not None:
            y = np.split(y, args.nr_gpu)
            feed_dict.update({ys[i]: y[i] for i in range(args.nr_gpu)})
    return feed_dict


def ret_masked_images(imgs):
    x_gen = [imgs[i] for i in range(args.nr_gpu)]
    for yi in range(20, 32):
        for xi in range(32):
            for i in range(args.nr_gpu):
                x_gen[i][:, yi, xi, :] = 0.0
    return np.concatenate(x_gen, axis=0)

def ret_original_images(imgs):
    x_gen = [imgs[i] for i in range(args.nr_gpu)]
    return np.concatenate(x_gen, axis=0)

# //////////// perform training //////////////
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
print('starting training')
test_bpd = []
lr = args.learning_rate
with tf.Session() as sess:


    ckpt_file = args.save_dir + '/params_' + args.data_set + '.ckpt'
    print('restoring parameters from', ckpt_file)
    saver.restore(sess, ckpt_file)

    td = next(test_data)
    from PIL import Image
    for i in range(36):
        Image.fromarray(td[i]).save("/data/ziz/jxu/CelebA/celeba_cropped_test_samples/c"+str(i)+".png")
    td = np.cast[np.float32]((td - 127.5) / 127.5)
    imgs = [td[i*args.batch_size:(i+1)*args.batch_size, :, :, :] for i in range(args.nr_gpu)]

    mks = [mgen.gen(imgs[0].shape[0]) for i in range(args.nr_gpu)]


    all_imgs = ret_original_images(imgs)[:36]
    for i in range(36):
        Image.fromarray((all_imgs[i]*127.5+127.5).astype(np.uint8)).save(os.path.join(
            "/data/ziz/jxu/results/original", '%s_original_%s.png' % (args.data_set, str(i).zfill(2))))

    all_imgs = ret_masked_images(imgs)[:36]
    for i in range(36):
        Image.fromarray((all_imgs[i]*127.5+127.5).astype(np.uint8)).save(os.path.join(
            "/data/ziz/jxu/results/masked", '%s_masked_%s.png' % (args.data_set, str(i).zfill(2))))

    all_imgs = complete(imgs, mks, sess)
    for i in range(36):
        Image.fromarray((all_imgs[i]*127.5+127.5).astype(np.uint8)).save(os.path.join(
            "/data/ziz/jxu/results/complete", '%s_complete_%s.png' % (args.data_set, str(i).zfill(2))))
