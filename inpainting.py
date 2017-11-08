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


parser = argparse.ArgumentParser()
# data I/O
parser.add_argument('-i', '--data_dir', type=str,
                    default='/tmp/pxpp/data', help='Location for the dataset')
parser.add_argument('-o', '--save_dir', type=str, default='/tmp/pxpp/save',
                    help='Location for parameter checkpoints and samples')
parser.add_argument('-d', '--data_set', type=str,
                    default='cifar', help='Can be either cifar|imagenet')
parser.add_argument('-g', '--nr_gpu', type=int, default=8,
                    help='How many GPUs to distribute the training across?')

# evaluation

args = parser.parse_args()
print('input args:\n', json.dumps(vars(args), indent=4,
                                  separators=(',', ':')))  # pretty print args


# create the model
model_opt = {'nr_resnet': 3, 'nr_filters': 10,
             'nr_logistic_mix': 10, 'resnet_nonlinearity': 'concat_elu'}
model = tf.make_template('model', model_spec)

gen_par = model(x_init, None, h_init, init=True,
                dropout_p=0.5, **model_opt)

saver = tf.train.Saver()



with tf.Session() as sess:
    ckpt_file = args.save_dir + '/params_' + args.data_set + '.ckpt'
    print('restoring parameters from', ckpt_file)
    saver.restore(sess, ckpt_file)


quit()

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
# data I/O
parser.add_argument('-i', '--data_dir', type=str,
                    default='/tmp/pxpp/data', help='Location for the dataset')
parser.add_argument('-o', '--save_dir', type=str, default='/tmp/pxpp/save',
                    help='Location for parameter checkpoints and samples')
parser.add_argument('-d', '--data_set', type=str,
                    default='cifar', help='Can be either cifar|imagenet')

parser.add_argument('-r', '--load_params', dest='load_params', action='store_true',
                    help='Restore training from previous model checkpoint?')
# model
parser.add_argument('-q', '--nr_resnet', type=int, default=4,
                    help='Number of residual blocks per stage of the model')
parser.add_argument('-n', '--nr_filters', type=int, default=100,
                    help='Number of filters to use across the model. Higher = larger model.')
parser.add_argument('-m', '--nr_logistic_mix', type=int, default=10,
                    help='Number of logistic components in the mixture. Higher = more flexible model')
parser.add_argument('-z', '--resnet_nonlinearity', type=str, default='concat_elu',
                    help='Which nonlinearity to use in the ResNet layers. One of "concat_elu", "elu", "relu" ')

# optimization

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
train_data = DataLoader(args.data_dir, 'train', args.batch_size * args.nr_gpu,
                        rng=rng, shuffle=True, return_labels=False)
test_data = DataLoader(args.data_dir, 'test', args.batch_size *
                       args.nr_gpu, shuffle=False, return_labels=False)
obs_shape = train_data.get_observation_size()  # e.g. a tuple (32,32,3)
assert len(obs_shape) == 3, 'assumed right now'


# data place holders
x_init = tf.placeholder(tf.float32, shape=(args.init_batch_size,) + obs_shape)
xs = [tf.placeholder(tf.float32, shape=(args.batch_size, ) + obs_shape)
      for i in range(args.nr_gpu)]

# if the model is class-conditional we'll set up label placeholders +
# one-hot encodings 'h' to condition on

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


#initializer = tf.global_variables_initializer()
saver = tf.train.Saver()


with tf.Session() as sess:
    ckpt_file = args.save_dir + '/params_' + args.data_set + '.ckpt'
    print('restoring parameters from', ckpt_file)
    saver.restore(sess, ckpt_file)

    epoch = 0
    # generate samples from the model
    sample_x = sample_from_model(sess)
    img_tile = plotting.img_tile(sample_x[:int(np.floor(np.sqrt(
        args.batch_size * args.nr_gpu))**2)], aspect_ratio=1.0, border_color=1.0, stretch=True)
    img = plotting.plot_img(img_tile, title=args.data_set + ' samples')
    plotting.plt.savefig(os.path.join(
        args.save_dir, '%s_sample%d.png' % (args.data_set, epoch)))
    plotting.plt.close('all')
