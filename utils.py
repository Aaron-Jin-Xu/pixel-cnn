import numpy as np
import pixel_cnn_pp.mask as mk
from scipy.misc import logsumexp
from scipy.special import expit
import os


def parse_args(args, data_dir, save_dir, data_set, save_interval=10, load_params=False,
                    nr_resnet=5, nr_filters=160, nr_logistic_mix=10,
                    learning_rate=0.001, lr_decay=0.999995, batch_size=12, init_batch_size=100,
                    nr_gpu=4, polyak_decay=0.9995, masked=False, rot180=False):
    ## Default, never change
    args.class_conditional = False
    args.resnet_nonlinearity = 'concat_elu'
    args.dropout_p = 0.5
    args.max_epochs = 5000
    args.seed = 1

    ## Parse Argument
    args.data_dir = data_dir
    args.save_dir = save_dir
    args.data_set = data_set
    args.save_interval = save_interval
    args.load_params = load_params

    args.nr_resnet = nr_resnet
    args.nr_filters = nr_filters
    args.nr_logistic_mix = nr_logistic_mix
    args.learning_rate = learning_rate
    args.lr_decay = lr_decay
    args.batch_size = batch_size
    args.init_batch_size = init_batch_size
    args.nr_gpu = nr_gpu
    args.polyak_decay = polyak_decay
    args.masked = masked
    args.rot180 = rot180


def next_pixel(masks, start=None):

    assert len(masks.shape)==3, "mask shape should be (batch_size, num_row, num_col)"
    if start is not None:
        assert len(start)==masks.shape[0], "number of start points should be batch_size"
    else:
        start = [(0, 0) for i in range(masks.shape[0])]

    ret = []

    for idx in range(masks.shape[0]):
        s = start[idx]
        m = masks[idx, :, :]
        for j in range(s[0], masks.shape[1]):
            c = 0
            if j==s[0]:
                c = s[1]
            for i in range(c, masks.shape[2]):
                if m[j, i]==0:
                    ret.append((j, i))
                    break
            if len(ret)==idx+1:
                break
        if len(ret)==idx:
            ret.append((None, None))

    return ret

def backward_next_pixel(masks, start=None):
    assert len(masks.shape)==3, "mask shape should be (batch_size, num_row, num_col)"
    nrow, ncol = masks.shape[-2:]
    masks = np.rot90(masks, 2, (1,2))
    if start is not None:
        assert len(start)==masks.shape[0], "number of start points should be batch_size"
    else:
        start = [(0, 0) for i in range(masks.shape[0])]
    ret = []
    for idx in range(masks.shape[0]):
        s = start[idx]
        m = masks[idx, :, :]
        for j in range(s[0], masks.shape[1]):
            c = 0
            if j==s[0]:
                c = s[1]
            for i in range(c, masks.shape[2]):
                if m[j, i]==0:
                    ret.append((nrow-j-1, ncol-i-1))
                    break
            if len(ret)==idx+1:
                break
        if len(ret)==idx:
            ret.append((None, None))
    return ret

def get_params(pars, pixels):
    assert len(pars.shape)==4, "pars shape should be (batch_size, num_row, num_col, params)"
    assert pars.shape[0]==len(pixels), "length of argument pixels should be batch_size"
    arr = []
    for idx in range(pars.shape[0]):
        p = pixels[idx]
        arr.append(pars[idx, p[0], p[1], :])
    return np.array(arr)

def sigmoid(x):
    return expit(x)

def softplus(x):
    return np.where(x < 30., np.log(np.exp(x) + 1.), x)
    #x = np.minimum(x, 50*np.ones_like(x))
    #return np.log(np.exp(x) + 1.)

def log_softmax(x):
    m = np.amax(x, axis=-1, keepdims=True)
    return x - m - np.log(np.sum(np.exp(x-m), axis=-1, keepdims=True))

def sum_exp(x):
    return np.exp(logsumexp(x, axis=-1))
    #x = np.minimum(x, 50*np.ones_like(x))
    #return np.sum(np.exp(x), axis=-1)

def params_to_dis(params, nr_mix, r=None, g=None, b=None, MAP=False):
    ps = params.shape
    assert ps[1]==10*nr_mix, "shape of params should be (batch_size, nr_mix*10)"
    logit_probs = params[:, :nr_mix]
    l = params[:, nr_mix:].reshape([ps[0], 3, 3*nr_mix])
    means = l[:, :, :nr_mix]
    log_scales = np.maximum(l[:, :, nr_mix:2 * nr_mix], -7.)
    print(means[0])
    print(log_scales[0])
    # log_scales -= log_scales_shift
    coeffs = np.tanh(l[:, :, 2 * nr_mix:3 * nr_mix])
    #if MAP:
    #    log_scales = log_scales - 10.
    inv_stdv = np.exp(-log_scales)

    if r is None:
        arr = []
        for i in range(256):
            x = (i - 127.5) / 127.5
            centered_x = x - means[:, 0, :]
            plus_in = inv_stdv[:, 0, :] * (centered_x + 1. / 255.)
            cdf_plus = sigmoid(plus_in)
            min_in = inv_stdv[:, 0, :] * (centered_x - 1. / 255.)
            cdf_min = sigmoid(min_in)
            cdf_delta = cdf_plus - cdf_min
            log_cdf_plus = plus_in - softplus(plus_in)
            log_one_minus_cdf_min = - softplus(min_in)

            mid_in = inv_stdv[:, 0, :] * centered_x
            log_pdf_mid = mid_in - log_scales[:, 0, :] - 2. * softplus(mid_in)
            log_probs = np.where(x < -0.999, log_cdf_plus, np.where(x > 0.999, log_one_minus_cdf_min,
                                                            np.where(cdf_delta > 1e-5, np.log(np.maximum(cdf_delta, 1e-12)), log_pdf_mid - np.log(127.5))))

            if MAP:
                p = log_softmax(logit_probs)
                p = np.exp(p).astype(np.float64)
                p = p / np.sum(p, axis=-1)[:, None]
                ps = []
                for i in range(p.shape[0]):
                    ps.append(np.random.multinomial(1, p[i, :]))
                ps = np.array(ps) * 7.0 - 7.0
                log_probs = log_probs + ps # log_softmax(logit_probs)
            else:
                log_probs = log_probs + log_softmax(logit_probs)

            probs = sum_exp(log_probs)
            arr.append(probs)
        all_probs = np.array(arr).T
        return all_probs

    if g is None:
        arr = []
        r = (r - 127.5) / 127.5
        for i in range(256):
            x = (i - 127.5) / 127.5
            m2 = means[:, 1, :] + coeffs[:, 0, :] * r[:, None]
            centered_x = x - m2
            plus_in = inv_stdv[:, 1, :] * (centered_x + 1. / 255.)
            cdf_plus = sigmoid(plus_in)
            min_in = inv_stdv[:, 1, :] * (centered_x - 1. / 255.)
            cdf_min = sigmoid(min_in)
            cdf_delta = cdf_plus - cdf_min
            log_cdf_plus = plus_in - softplus(plus_in)
            log_one_minus_cdf_min = - softplus(min_in)

            mid_in = inv_stdv[:, 1, :] * centered_x
            log_pdf_mid = mid_in - log_scales[:, 1, :] - 2. * softplus(mid_in)
            log_probs = np.where(x < -0.999, log_cdf_plus, np.where(x > 0.999, log_one_minus_cdf_min,
                                                            np.where(cdf_delta > 1e-5, np.log(np.maximum(cdf_delta, 1e-12)), log_pdf_mid - np.log(127.5))))

            if MAP:
                p = log_softmax(logit_probs)
                p = np.exp(p).astype(np.float64)
                p = p / np.sum(p, axis=-1)[:, None]
                ps = []
                for i in range(p.shape[0]):
                    ps.append(np.random.multinomial(1, p[i, :]))
                ps = np.array(ps) * 7.0 - 7.0
                log_probs = log_probs + ps # log_softmax(logit_probs)
            else:
                log_probs = log_probs + log_softmax(logit_probs)

            probs = sum_exp(log_probs)
            arr.append(probs)
        all_probs = np.array(arr).T
        return all_probs

    if b is None:
        arr = []
        r = (r - 127.5) / 127.5
        g = (g - 127.5) / 127.5
        for i in range(256):
            x = (i - 127.5) / 127.5
            m3 = means[:, 2, :] + coeffs[:, 1, :] * r[:, None] + coeffs[:, 2, :] * g[:, None]
            centered_x = x - m3
            plus_in = inv_stdv[:, 2, :] * (centered_x + 1. / 255.)
            cdf_plus = sigmoid(plus_in)
            min_in = inv_stdv[:, 2, :] * (centered_x - 1. / 255.)
            cdf_min = sigmoid(min_in)
            cdf_delta = cdf_plus - cdf_min
            log_cdf_plus = plus_in - softplus(plus_in)
            log_one_minus_cdf_min = - softplus(min_in)

            mid_in = inv_stdv[:, 2, :] * centered_x
            log_pdf_mid = mid_in - log_scales[:, 2, :] - 2. * softplus(mid_in)
            log_probs = np.where(x < -0.999, log_cdf_plus, np.where(x > 0.999, log_one_minus_cdf_min,
                                                            np.where(cdf_delta > 1e-5, np.log(np.maximum(cdf_delta, 1e-12)), log_pdf_mid - np.log(127.5))))
            if MAP:
                p = log_softmax(logit_probs)
                p = np.exp(p).astype(np.float64)
                p = p / np.sum(p, axis=-1)[:, None]
                ps = []
                for i in range(p.shape[0]):
                    ps.append(np.random.multinomial(1, p[i, :]))
                ps = np.array(ps) * 7.0 - 7.0
                log_probs = log_probs + ps # log_softmax(logit_probs)
            else:
                log_probs = log_probs + log_softmax(logit_probs)

            probs = sum_exp(log_probs)
            arr.append(probs)
        all_probs = np.array(arr).T
        return all_probs


def combine_forward_backward(pars_f, pars_b):
    print(pars_f.shape)
    print(pars_b.shape)

    quit()

def tile_images(imgs, size=(6, 6)):
    imgs = imgs[:size[0]*size[1], :, :, :]
    img_h, img_w = imgs.shape[1], imgs.shape[2]
    all_images = np.zeros((img_h*size[0], img_w*size[1], 3), np.uint8)
    for j in range(size[0]):
        for i in range(size[1]):
            all_images[img_h*j:img_h*(j+1), img_w*i:img_w*(i+1), :] = imgs[j*size[0]+i, :, :, :]
    return all_images

def get_prior(prior, target_pixels):
    arr = []
    for p in target_pixels:
        arr.append(prior[:, p[0], p[1], :].T)
    return np.array(arr)

from scipy.stats import entropy

def KL_divergence(p, q):
    return entropy(p, q)
