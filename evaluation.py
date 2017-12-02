import numpy as np
import math

## https://github.com/aizvorski/video-quality/blob/master/psnr.py
def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    #if mse == 0:
    #    return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def batch_psnr(imgs1, imgs2, output_mean=True):
    assert imgs1.shape[0]==imgs2.shape[0], "batch size of imgs1 and imgs2 should be the same"
    batch_size = imgs1.shape[0]
    v = [psnr(imgs1[i], imgs2[i]) for i in range(batch_size)]
    if output_mean:
        return np.mean(v)
    return v


def evaluate(original_imgs, completed_imgs):
    return batch_psnr(original_imgs, completed_imgs)
