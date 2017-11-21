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
