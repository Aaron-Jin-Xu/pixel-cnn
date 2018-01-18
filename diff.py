import numpy as np
from PIL import Image
from utils import *
from evaluation import *

def find_contour(mask):
    contour = np.zeros_like(mask)
    h, w = mask.shape
    for y in range(h):
        for x in range(w):
            if mask[y, x] > 0:
                lower_bound = max(y-1, 0)
                upper_bound = min(y+1, h-1)
                left_bound = max(x-1, 0)
                right_bound = min(x+1, w-1)
                nb = mask[lower_bound:upper_bound+1, left_bound:right_bound+1]
                if np.min(nb)  == 0:
                    contour[y, x] = 1
    return contour

def tile_plot(imgs, file_path="../plots/test.png", display_size=None):
    if display_size is None:
        s = int(np.sqrt(imgs.shape[0]))
        display_size = (s, s)
    img = Image.fromarray(tile_images(imgs.astype(np.uint8), size=display_size), 'RGB')
    img.save(file_path)

#mgen = mk.CenterMaskGenerator(64, 64, 0.5)
mgen = mk.CrossMaskGenerator(64, 64, (28, 38, 2, 62), (5, 59, 28, 36))
mask = mgen.gen(1)[0]
contour = find_contour(mask)[:, :, None]

data = np.load("psnr-cross-gan.npz")
all_completed = data['comp']
ground_truth = data['ori']
delta = np.abs(np.mean(all_completed, axis=0) - ground_truth)
delta += contour * 100
tile_plot(delta, "../plots1/celeba-cross-gan.png")
