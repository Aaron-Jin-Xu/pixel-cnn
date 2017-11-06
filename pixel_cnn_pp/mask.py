import numpy as np

class MaskGenerator(object):

    def __init__(self, h, w, rng=None):
        self.h = h
        self.w = w
        if rng is None:
            rng = np.random.RandomState(None)
        self.rng = rng

    def gen(self, n):
        return np.ones((n, h, w))

class AllOnesMaskGenerator(MaskGenerator):

    def __init__(self, h, w, rng=None):
        super().__init__(h, w, rng)

    def gen(self, n):
        return np.ones((n, self.h, self.w))


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

class RandomMaskGenerator(MaskGenerator):

    def __init__(self, h, w, rng=None):
        super().__init__(h, w, rng)

    def gen_par(self):
        mh = self.rng.randint(low=1, high=self.h/2+1)
        mw = self.rng.randint(low=1, high=self.w/2+1)
        pgh = self.rng.randint(low=0, high=mh)
        pgw =  self.rng.randint(low=0, high=mw)
        oh = self.rng.randint(low=1, high=self.h - mh)
        ow = self.rng.randint(low=1, high=self.w - mw)
        return mh, mw, pgh, pgw, oh, ow

    def gen(self, n):
        mask = np.ones((n, self.h, self.w))
        for i in range(n):
            missing_h, missing_w, progress_h, progress_w, offset_h, offset_w = self.gen_par()
            print(missing_h, missing_w, progress_h, progress_w, offset_h, offset_w)
            missing = np.zeros((missing_h, missing_w))
            missing[:progress_h, :] = 1
            missing[progress_h, :progress_w] = 1
            mask[i, offset_h:offset_h+missing_h, offset_w:offset_w+missing_w] = missing
        return mask
