import numpy as np
import pixel_cnn_pp.mask as mk
from scipy.misc import logsumexp
from scipy.special import expit

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
    return np.where(x < 50., np.log(np.exp(x) + 1.), x)
    #x = np.minimum(x, 50*np.ones_like(x))
    #return np.log(np.exp(x) + 1.)

def log_softmax(x):
    m = np.amax(x, axis=-1, keepdims=True)
    return x - m - np.log(np.sum(np.exp(x-m), axis=-1, keepdims=True))

def sum_exp(x):
    return np.exp(logsumexp(x, axis=-1))
    #x = np.minimum(x, 50*np.ones_like(x))
    #return np.sum(np.exp(x), axis=-1)

def params_to_dis(params, nr_mix, r=None, g=None, b=None):
    ps = params.shape
    assert ps[1]==10*nr_mix, "shape of params should be (batch_size, nr_mix*10)"
    logit_probs = params[:, :nr_mix]
    l = params[:, nr_mix:].reshape([ps[0], 3, 3*nr_mix])
    means = l[:, :, :nr_mix]
    log_scales = np.maximum(l[:, :, nr_mix:2 * nr_mix], -7.)
    coeffs = np.tanh(l[:, :, 2 * nr_mix:3 * nr_mix])

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

            print(log_cdf_plus.max())
            print(log_one_minus_cdf_min.max())
            print(np.log(np.maximum(cdf_delta, 1e-12)).max())
            log_probs = np.where(x < -0.999, log_cdf_plus, np.where(x > 0.999, log_one_minus_cdf_min,
                                                            np.where(cdf_delta > 1e-5, np.log(np.maximum(cdf_delta, 1e-12)), log_pdf_mid - np.log(127.5))))
            log_probs = log_probs + log_softmax(logit_probs)
            probs = sum_exp(log_probs)
            arr.append(probs)
        all_probs = np.array(arr)
        print(all_probs.shape)
        print(all_probs.sum(0))
        print(all_probs[:, np.argmax(all_probs.sum(0))])
        quit()
            #arr.append(cdf_delta.mean(1))
        return np.array(arr)
