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
            missing = np.zeros((missing_h, missing_w))
            missing[:progress_h, :] = 1
            missing[progress_h, :progress_w] = 1
            mask[i, offset_h:offset_h+missing_h, offset_w:offset_w+missing_w] = missing
        return mask


class RecMaskGenerator(MaskGenerator):

    def __init__(self, h, w, rng=None):
        super().__init__(h, w, rng)

    def gen_par(self, seed=None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        mh = int(self.h * 0.4)
        mw = int(self.w * 0.4)
        pgh = self.rng.randint(low=0, high=mh)
        pgw =  self.rng.randint(low=0, high=mw)
        oh = self.rng.randint(low=1, high=self.h - mh)
        ow = self.rng.randint(low=1, high=self.w - mw)
        return mh, mw, pgh, pgw, oh, ow

    def gen(self, n):
        mask = np.ones((n, self.h, self.w))
        for i in range(n):
            missing_h, missing_w, progress_h, progress_w, offset_h, offset_w = self.gen_par()
            missing = np.zeros((missing_h, missing_w))
            missing[:progress_h, :] = 1
            missing[progress_h, :progress_w] = 1
            mask[i, offset_h:offset_h+missing_h, offset_w:offset_w+missing_w] = missing
        return np.rot90(mask, 2, (1,2))



class RecNoProgressMaskGenerator(RecMaskGenerator):

    def gen(self, n):
        mask = np.ones((n, self.h, self.w))
        for i in range(n):
            missing_h, missing_w, progress_h, progress_w, offset_h, offset_w = self.gen_par(i)
            progress_w, progress_h = 0, 0
            missing = np.zeros((missing_h, missing_w))
            missing[:progress_h, :] = 1
            missing[progress_h, :progress_w] = 1
            mask[i, offset_h:offset_h+missing_h, offset_w:offset_w+missing_w] = missing
        return np.rot90(mask, 2, (1,2))




class RectangleMaskGenerator(MaskGenerator):

    def __init__(self, h, w, rng=None):
        super().__init__(h, w, rng)

    def gen_par(self, seed=None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        #mh = self.rng.randint(low=int(self.h * 0.2), high=int(self.h*0.8))
        #mw = self.rng.randint(low=int(self.w * 0.2), high=int(self.w*0.8))
        mh = int(self.h * 0.4)
        mw = int(self.w * 0.4)
        pgh = self.rng.randint(low=0, high=mh)
        pgw =  self.rng.randint(low=0, high=mw)
        oh = self.rng.randint(low=1, high=self.h - mh)
        ow = self.rng.randint(low=1, high=self.w - mw)
        return mh, mw, pgh, pgw, oh, ow

    def gen(self, n):
        mask = np.ones((n, self.h, self.w))
        for i in range(n):
            missing_h, missing_w, progress_h, progress_w, offset_h, offset_w = self.gen_par(i)
            progress_w, progress_h = 0, 0
            missing = np.zeros((missing_h, missing_w))
            missing[:progress_h, :] = 1
            missing[progress_h, :progress_w] = 1
            mask[i, offset_h:offset_h+missing_h, offset_w:offset_w+missing_w] = missing
        return np.rot90(mask, 2, (1,2))




# https://stackoverflow.com/questions/8647024/how-to-apply-a-disc-shaped-mask-to-a-numpy-array
def cmask(index, radius, array):
    a, b = index
    nx, ny = array.shape
    y, x = np.ogrid[-a:nx-a,-b:ny-b]
    mask = x*x + y*y <= radius*radius
    return mask

class CircleMaskGenerator(MaskGenerator):

    def __init__(self, height, width, radius):
        assert radius < min(height, width) // 2
        self.height = height
        self.width = width
        self.radius = radius


    def gen(self, n):
        masks = []
        for i in range(n):
            c_y = np.random.randint(self.radius+1, self.height-self.radius-1)
            c_x = np.random.randint(self.radius+1, self.width-self.radius-1)
            m = cmask((c_y, c_x), self.radius, np.ones((self.height, self.width)))
            masks.append(m)
        masks = np.array(masks)
        return 1-masks
