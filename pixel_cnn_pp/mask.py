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




class RectangleInProgressMaskGenerator(MaskGenerator):

    def __init__(self, h, w, forward=False, rng=None):
        super().__init__(h, w, rng)
        self.forward = forward

    def gen_par(self, seed=None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        mh = self.rng.randint(low=int(self.h * 0.1), high=int(self.h * 0.4))
        mw = self.rng.randint(low=int(self.w * 0.1), high=int(self.w * 0.4))
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
        if self.forward:
            return mask
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



class BottomMaskGenerator(MaskGenerator):

    def __init__(self, height, width, mask_height):
        self.height = height
        self.width = width
        self.mask_height = mask_height


    def gen(self, n):
        masks = np.ones((n, self.height, self.width))
        masks[:, -self.mask_height:, :] = 0
        return masks


class HorizontalMaskGenerator(MaskGenerator):

    def __init__(self, height, width, upper_bound, lower_bound):
        self.height = height
        self.width = width
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound


    def gen(self, n):
        masks = np.ones((n, self.height, self.width))
        #masks[:, self.upper_bound:self.lower_bound, self.upper_bound:self.lower_bound] = 0
        masks[:, self.upper_bound:self.lower_bound, :] = 0
        return masks

class RandomNoiseMaskGenerator(MaskGenerator):

    def __init__(self, height, width, lossy_ratio=0.5):
        self.height = height
        self.width = width
        self.lossy_ratio = lossy_ratio

    def gen(self, n):
        masks = np.ones((n, self.height, self.width))
        idx_arr = [(y, x) for y in range(self.height) for x in range(self.width)]
        for i in range(n):
            idxs = np.random.choice(range(len(idx_arr)), size=int(self.height*self.width*self.lossy_ratio), replace=False)
            for d in idxs:
                d = idx_arr[d]
                masks[i, d[0], d[1]] = 0
        return masks


class HalfMaskGenerator(MaskGenerator):

    def __init__(self, height, width):
        self.height = height
        self.width = width

    def gen(self, n):
        masks = np.ones((n, self.height, self.width))
        choices = ['upper', 'lower', 'left', 'right']
        #c = np.random.randint(4, size=n)
        c = np.arange(n)
        c = c % 4
        for i in range(n):
            if choices[c[i]]=='upper':
                masks[i][:self.height//2, :] = 0
            elif choices[c[i]]=='lower':
                masks[i][-self.height//2:, :] = 0
            elif choices[c[i]]=='left':
                masks[i][:, :self.width//2] = 0
            elif choices[c[i]]=='right':
                masks[i][:, -self.width//2:] = 0
        return masks

class RightMaskGenerator(MaskGenerator):

    def __init__(self, height, width, scale=0.5):
        self.height = height
        self.width = width
        self.scale = scale

    def gen(self, n):
        masks = np.ones((n, self.height, self.width))
        w = int(self.width * self.scale)
        masks[:, 16:48, -w:] = 0
        return masks

class CenterMaskGenerator(MaskGenerator):


    def __init__(self, height, width, scale):
        self.height = height
        self.width = width
        self.scale = scale

    def gen(self, n):
        masks = np.ones((n, self.height, self.width))
        h = int(self.height * self.scale)
        w = int(self.width * self.scale)
        h_start = (self.height - h) // 2
        w_start = (self.width - w) // 2
        masks[:, h_start:h_start+h, w_start:w_start+w] = 0
        return masks

class RectangleMaskGenerator(MaskGenerator):

    def __init__(self, height, width, h_start, h_end, w_start, w_end):
        self.height = height
        self.width = width
        self.h_start = h_start
        self.h_end = h_end
        self.w_start = w_start
        self.w_end = w_end

    def gen(self, n):
        masks = np.ones((n, self.height, self.width))
        masks[:, self.h_start:self.h_end, self.w_start:self.w_end] = 0
        return masks

class GridMaskGenerator(MaskGenerator):

    def __init__(self, height, width, grid_size):
        self.height = height
        self.width = width
        self.grid_size = grid_size

    def gen(self, n):
        grid_size = self.grid_size
        masks = np.ones((n, self.height, self.width))
        for y in range(self.height//grid_size):
            for x in range(self.width//grid_size):
                if (y * (self.height//grid_size) + x + y) % 2 ==0:
                    masks[:, y*grid_size:(y+1)*grid_size, x*grid_size:(x+1)*grid_size] = 0
        return masks
