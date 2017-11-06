import numpy as np

class MaskGenerator(object):

    def __init__(self, h, w):
        self.h = h
        self.w = w

    def gen(self, n):
        return np.ones((n, h, w))


class CentralMaskGenerator(MaskGenerator):

    def __init__(self, h, w, ratio):
        super().__init__(h, w)
        self.ratio = ratio

    def gen(self, n):
        mask = np.ones((n, self.h, self.w))
        h_offset = int(self.h * (1-self.ratio) / 2.)
        w_offset = int(self.w * (1-self.ratio) / 2.)
        h_delta = int(self.h * self.ratio)
        w_delta = int(self.w * self.ratio)
        mask[:, h_offset:h_offset+h_delta, w_offset:w_offset+w_delta] = 0
        return mask
