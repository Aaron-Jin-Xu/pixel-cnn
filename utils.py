import numpy as np
import pixel_cnn_pp.mask as mk

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
    return 1. / (1. + np.exp(-x))

def softplus(x):
    return np.log(np.exp(x) + 1.)

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
        for i in range(1, 255):
            x = (i - 127.5) / 127.5
            centered_x = x - means[:, 0, :]
            plus_in = inv_stdv[:, 0, :] * (centered_x + 1. / 255.)
            cdf_plus = sigmoid(plus_in)
            min_in = inv_stdv[:, 0, :] * (centered_x - 1. / 255.)
            cdf_min = sigmoid(min_in)
            cdf_delta = cdf_plus - cdf_min
            log_cdf_plus = plus_in - softplus(plus_in)
            log_one_minus_cdf_min = - softplus(min_in)
            log_probs = np.where(x < -0.999, log_cdf_plus, np.where(x > 0.999, log_one_minus_cdf_min, np.log(cdf_delta)))
            print(log_probs.shape)# + log_prob_from_logits(logit_probs)
            quit()
            #arr.append(cdf_delta.mean(1))
        return np.array(arr)



    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * centered_x


    log_probs = tf.select(x < -0.999, log_cdf_plus, tf.select(x > 0.999, log_one_minus_cdf_min, tf.log(cdf_delta)))
